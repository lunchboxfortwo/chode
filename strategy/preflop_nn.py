"""
Neural preflop strategy — continuous stack-depth GTO solver.

A single neural network replaces all discrete tabular policies,
providing smooth, stack-depth-continuous strategy queries for any
(n_players, stack_bb, hand, position, history) combination.

Architecture:
  Input:  [hand_oh(169), position_oh(8), history_enc(30), n_players_oh(5), stack_bb_norm(1)]
  Hidden: 4 × 512 ReLU
  Output: [fold, call, bet, squeeze, allin] probabilities (softmax)

Training: ReBeL-style MCCFR with neural policy/value networks.
  - External-sampling MCCFR traverses the game tree
  - At each decision node, the network provides the current strategy
  - After traversal, regrets are used to update the network
  - The network generalizes across stack depths, so training at any
    depth improves the strategy at ALL depths simultaneously.

Usage:
  from strategy.preflop_nn import PreflopNN
  nn = PreflopNN()           # loads latest checkpoint
  probs = nn.query(n=5, bb=45, pidx=2, hand=(0,5), hist=['bet','call'])
  # → {'fold': 0.12, 'call': 0.58, 'bet': 0.30}
"""

import os
import sys
import math
import struct
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_training.preflop_fixed_train import (
    State, legal_actions, apply_action, is_terminal, payoff,
    position_names, _hand_cat, _pack_hand_cat, _POSITION_MAP,
    N_ACTIONS, BB, SB,
)

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

N_HAND_CATS = 169      # 13×13 hand grid (suited/pair/offsuit)
N_POSITIONS = 8         # max positions (UTG1..BB)
MAX_HIST_LEN = 10       # max action history length
N_ACTION_TYPES = 5      # fold/call/bet/squeeze/allin
N_PLAYER_SLOTS = 5      # 2p..6p
N_ACTIONS_TOTAL = 5     # fold(0), call(1), bet(2), squeeze(3), allin(4)
INPUT_DIM = N_HAND_CATS + N_POSITIONS + MAX_HIST_LEN + N_PLAYER_SLOTS + 1
OUTPUT_DIM = N_ACTIONS_TOTAL

MODEL_DIR = Path(__file__).parent.parent / "data" / "preflop_nn"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Action names for serialization
ACTION_NAMES = ["fold", "call", "bet", "squeeze", "allin"]

# ─── Feature encoding ────────────────────────────────────────────────────────

def _hand_index(c1: int, c2: int) -> int:
    """Map two hole cards to a 0-168 canonical hand index."""
    r1, r2 = c1 // 4, c2 // 4
    s1, s2 = c1 % 4, c2 % 4
    # Canonical: higher rank first
    if r1 < r2:
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    elif r1 == r2 and s1 > s2:
        s1, s2 = s2, s1

    suited = (s1 == s2)
    if r1 == r2:
        # Pair: rows on the 13x13 grid (0..12 for AA..22)
        return r1 * 13 + r1
    elif suited:
        # Suited: upper triangle (r1 > r2)
        return r1 * 13 + r2
    else:
        # Offsuit: lower triangle
        return r2 * 13 + r1


def encode_features(
    n_players: int,
    stack_bb: float,
    pidx: int,
    hand: tuple[int, int],
    history: list[str],
) -> torch.Tensor:
    """Encode a preflop decision state into a fixed-size feature vector."""
    features = np.zeros(INPUT_DIM, dtype=np.float32)
    offset = 0

    # 1. Hand one-hot (169 dims)
    hidx = _hand_index(hand[0], hand[1])
    features[offset + hidx] = 1.0
    offset += N_HAND_CATS

    # 2. Position one-hot (8 dims)
    pos_name = position_names(n_players)[pidx]
    all_positions = ["UTG1", "UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
    pos_idx = all_positions.index(pos_name) if pos_name in all_positions else 7
    features[offset + pos_idx] = 1.0
    offset += N_POSITIONS

    # 3. Action history encoding (10 dims, each 0-4)
    action_map = {"fold": 0, "call": 1, "bet": 2, "squeeze": 3, "allin": 4}
    for i in range(MAX_HIST_LEN):
        if i < len(history):
            features[offset + i] = action_map.get(history[i], 0) / 4.0
    offset += MAX_HIST_LEN

    # 4. n_players one-hot (5 dims: 2-6)
    np_idx = min(max(n_players - 2, 0), 4)
    features[offset + np_idx] = 1.0
    offset += N_PLAYER_SLOTS

    # 5. Stack depth (1 dim, normalized to [0, 1] using log scale)
    features[offset] = math.log(max(stack_bb, 1.0)) / math.log(200.0)
    offset += 1

    assert offset == INPUT_DIM, f"Feature encoding mismatch: {offset} != {INPUT_DIM}"
    return torch.from_numpy(features)


# ─── Network ──────────────────────────────────────────────────────────────────

class PreflopNet(nn.Module):
    """
    Neural network for preflop strategy prediction.

    Input:  encoded (hand, position, history, n_players, stack_bb)
    Output: unnormalized log-probabilities for [fold, call, bet, squeeze, allin]
    """

    def __init__(self, hidden_dim: int = 512, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(INPUT_DIM, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, OUTPUT_DIM)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Initialize with small weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (batch, 5) unnormalized log-probabilities
            value:  (batch, 1) estimated state value for the acting player
        """
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h)

    def predict(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            probs: (batch, 5) probability distribution over actions
            value: (batch, 1) estimated state value
        """
        logits, value = self.forward(x)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)
        probs = F.softmax(logits, dim=-1)
        return probs, value


# ─── Public API ───────────────────────────────────────────────────────────────

class PreflopNN:
    """
    High-level interface for neural preflop strategy queries.

    Handles model loading, caching, and strategy lookup.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.net = PreflopNet().to(self.device)
        self.net.eval()
        self._latest_step = -1
        self._latest_checkpoint()

    def _latest_checkpoint(self):
        """Load the latest checkpoint from disk."""
        ckpts = sorted(MODEL_DIR.glob("preflop_nn_*.pt"))
        ckpts = [c for c in ckpts if "_optimizer" not in c.name]
        if ckpts:
            latest = ckpts[-1]
            state = torch.load(latest, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state["model"])
            self._latest_step = state.get("step", 0)
            logger.info(f"Loaded NN preflop checkpoint: {latest.name} (step {self._latest_step:,})")
        else:
            logger.info("No NN preflop checkpoint found — using random init")

    def _check_reload(self):
        """Reload if a newer checkpoint exists."""
        ckpts = sorted(MODEL_DIR.glob("preflop_nn_*.pt"))
        ckpts = [c for c in ckpts if "_optimizer" not in c.name]
        if ckpts:
            latest = ckpts[-1]
            # Extract step from filename
            try:
                step = int(latest.stem.split("_")[-1])
            except ValueError:
                return
            if step > self._latest_step:
                state = torch.load(latest, map_location=self.device, weights_only=True)
                self.net.load_state_dict(state["model"])
                self._latest_step = step
                logger.info(f"Auto-reloaded NN checkpoint: {latest.name} (step {step:,})")

    def query(
        self,
        n: int,
        bb: float,
        pidx: int,
        hand: tuple[int, int],
        hist: list[str] | None = None,
        legal_actions: list[int] | None = None,
    ) -> dict[str, float]:
        """
        Query the neural preflop strategy.

        Args:
            n:       number of players (2-6)
            bb:      stack depth in big blinds (continuous)
            pidx:    player index (0-based)
            hand:    tuple of two card indices (0-51)
            hist:    action history (e.g. ['bet', 'call'])
            legal_actions: list of legal action indices (0-4), auto-detected if None

        Returns:
            dict mapping action name → probability
        """
        if hist is None:
            hist = []

        self._check_reload()

        with torch.no_grad():
            x = encode_features(n, bb, pidx, hand, hist).unsqueeze(0).to(self.device)

            # Legal action mask
            if legal_actions is None:
                # Infer from game state — build a State and query
                legal_actions = self._infer_legal(n, bb, pidx, hist)

            mask = torch.zeros(1, OUTPUT_DIM, dtype=torch.bool, device=self.device)
            for a in legal_actions:
                mask[0, a] = True

            probs, value = self.net.predict(x, mask)
            probs = probs[0].cpu().numpy()

        return {ACTION_NAMES[a]: float(probs[a]) for a in legal_actions}

    def _infer_legal(
        self, n: int, bb: float, pidx: int, hist: list[str]
    ) -> list[int]:
        """Infer legal actions by replaying the game history."""
        s = State(n_players=n, stack_bb=int(bb))
        # Replay history
        action_map = {"fold": 0, "call": 1, "bet": 2, "squeeze": 3, "allin": 4}
        for a_name in hist:
            a = action_map[a_name]
            s = apply_action(s, a)
        return legal_actions(s)

    def query_chart(
        self, n: int, bb: float, pidx: int, history: list[str]
    ) -> dict:
        """
        Query strategy for all 169 hands (full chart grid).

        Returns dict compatible with the /charts API:
        {
            "n_players": int,
            "stack_bb": float,
            "position": str,
            "history": list[str],
            "label": str,
            "hands": {hand_str: {action: prob, ...}, ...},
            "n_hands_decoded": 169,
        }
        """
        # Build legal action mask from game state
        s = State(n_players=n, stack_bb=int(bb))
        action_map = {"fold": 0, "call": 1, "bet": 2, "squeeze": 3, "allin": 4}
        for a_name in history:
            a = action_map[a_name]
            s = apply_action(s, a)
        la = legal_actions(s)
        mask = torch.zeros(1, OUTPUT_DIM, dtype=torch.bool, device=self.device)
        for a in la:
            mask[0, a] = True

        pos = position_names(n)[pidx]

        # Batch all 169 hand combinations
        hands = {}
        batch_features = []
        hand_keys = []

        for r1 in range(13):
            for r2 in range(13):
                # Reconstruct two cards for this hand category
                c1, c2 = _hand_to_cards(r1, r2)
                feat = encode_features(n, bb, pidx, (c1, c2), history)
                batch_features.append(feat)
                hand_keys.append(f"{r1},{r2}")

        # Batch forward pass
        with torch.no_grad():
            batch = torch.stack(batch_features).to(self.device)
            mask_batch = mask.expand(len(batch_features), -1)
            probs, values = self.net.predict(batch, mask_batch)
            probs = probs.cpu().numpy()

        for i, (r1, r2) in enumerate(
            (r1, r2) for r1 in range(13) for r2 in range(13)
        ):
            key = f"{r1},{r2}"
            hands[key] = {
                ACTION_NAMES[a]: float(probs[i, a]) for a in la
            }

        # Build label
        label = _build_label(n, bb, pidx, history)

        return {
            "n_players": n,
            "stack_bb": bb,
            "position": pos,
            "history": history,
            "label": label,
            "hands": hands,
            "n_hands_decoded": 169,
            "source": "nn",
        }

    def save(self, step: int):
        """Save model checkpoint."""
        path = MODEL_DIR / f"preflop_nn_{step:08d}.pt"
        torch.save({"model": self.net.state_dict(), "step": step}, path)
        logger.info(f"Saved NN preflop checkpoint: {path.name}")

    def load(self, path: str | Path):
        """Load model from a specific path."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state["model"])
        self.net.eval()


def _hand_to_cards(r1: int, r2: int) -> tuple[int, int]:
    """Convert hand grid coordinates (r1, r2) to two card indices.

    Chart grid convention: 0=Ace, 1=King, ..., 12=Two
    Card index convention: rank*4 + suit, where 0=Two, 12=Ace
    So we must invert: solver_rank = 12 - chart_rank
    """
    sr1 = 12 - r1  # Convert chart rank to solver rank
    sr2 = 12 - r2
    # Use suit encoding: pair=same suit, suited=both hearts, offsuit=heart+spade
    if r1 == r2:
        # Pair
        return sr1 * 4, sr1 * 4 + 1
    elif r1 < r2:
        # Upper triangle = suited in chart convention (chart r1 < r2 means higher card first)
        return sr1 * 4, sr2 * 4
    else:
        # Lower triangle = offsuit
        return sr1 * 4, sr2 * 4 + 1


def _build_label(n: int, bb: float, pidx: int, history: list[str]) -> str:
    """Build a human-readable label for a preflop spot."""
    pos = position_names(n)[pidx]
    if not history:
        return f"{pos} RFI ({n}p {bb:.0f}bb)"

    # Describe the spot by action history
    h = [a.lower() for a in history]
    # Count raises (bet/squeeze/allin) in history
    raises = sum(1 for a in h if a in ("bet", "squeeze", "allin"))
    calls = sum(1 for a in h if a == "call")
    folds = sum(1 for a in h if a == "fold")

    if raises == 1 and calls == 0 and folds == 0:
        return f"{pos} facing RFI ({n}p {bb:.0f}bb)"
    if raises == 1 and calls >= 1 and folds == 0 and n >= 3:
        return f"{pos} squeeze spot ({n}p {bb:.0f}bb)"
    if raises == 2 and calls == 0 and folds == 0:
        return f"{pos} facing 3bet ({n}p {bb:.0f}bb)"
    if raises == 2 and calls >= 1 and folds == 0:
        return f"{pos} facing 3bet+call ({n}p {bb:.0f}bb)"
    if raises == 3 and calls == 0 and folds == 0:
        return f"{pos} facing 4bet ({n}p {bb:.0f}bb)"
    if raises >= 4:
        return f"{pos} facing {raises+1}bet ({n}p {bb:.0f}bb)"
    # Fallback: describe actions
    desc = " → ".join(h)
    return f"{pos} after {desc} ({n}p {bb:.0f}bb)"


# ─── Checkpoint info ─────────────────────────────────────────────────────────

def nn_status() -> dict:
    """Return status info about the NN preflop model."""
    ckpts = sorted(MODEL_DIR.glob("preflop_nn_*.pt"))
    ckpts = [c for c in ckpts if "_optimizer" not in c.name]
    latest_step = 0
    latest_name = None
    if ckpts:
        state = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
        latest_step = state.get("step", 0)
        latest_name = ckpts[-1].name

    # Training progress
    training = None
    prog_path = MODEL_DIR / "training_progress.json"
    if prog_path.exists():
        try:
            import json as _json
            training = _json.loads(prog_path.read_text())
        except Exception:
            pass

    return {
        "available": len(ckpts) > 0,
        "n_checkpoints": len(ckpts),
        "latest_checkpoint": latest_name,
        "step": latest_step,
        "params": sum(p.numel() for p in PreflopNet().parameters()),
        "model_dir": str(MODEL_DIR),
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "training": training,
    }
