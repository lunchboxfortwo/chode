"""Neural postflop solver — replaces tabular NPZ/PKL lookup with a single NN.

Architecture:
  Input:  [hand(169), position(8), street(3), texture(64),
           n_players(5), pot_size(1), stack_ratio(1), facing_size(1),
           agg_actions(4), action_history(12)]  = 268 features
  Hidden: 4 × 1024 ReLU
  Output: [fold, check/call, bet_small, bet_large, raise, allin] + value

Board abstraction uses 64 canonical texture IDs from board_abstraction.py,
collapsing 2.1M unique boards into 64 texture classes.
"""
import os, glob, time, logging
import numpy as np

log = logging.getLogger(__name__)

CHODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR = os.path.join(CHODE, "data", "postflop_nn")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ACTIONS = 6  # fold, check/call, bet_small, bet_large, raise, allin
ACTION_NAMES = ["fold", "check_call", "bet_small", "bet_large", "raise", "allin"]
FEATURE_DIM = 268  # see encode_features()

# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------
def encode_features(
    hand_cat: int = 0,       # 0-168
    position: int = 0,       # 0-7
    street: int = 0,         # 0=flop, 1=turn, 2=river
    texture_id: int = 0,     # 0-63
    n_players: int = 2,      # 2-5
    pot_size: float = 1.0,   # in BB
    stack_ratio: float = 1.0,# stack/pot
    facing_size: float = 0.0,# facing bet / pot
    agg_actions: int = 0,    # 0-3 aggressive actions this street
    action_history: list = None,  # list of action indices (0-5)
) -> np.ndarray:
    """Encode game state into a fixed-size feature vector."""
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    # Hand category: one-hot (169)
    if 0 <= hand_cat < 169:
        feat[hand_cat] = 1.0

    # Position: one-hot (8)
    off = 169
    if 0 <= position < 8:
        feat[off + position] = 1.0

    # Street: one-hot (3)
    off = 177
    if 0 <= street < 3:
        feat[off + street] = 1.0

    # Board texture: one-hot (64)
    off = 180
    if 0 <= texture_id < 64:
        feat[off + texture_id] = 1.0

    # Number of players: one-hot (5)
    off = 244
    if 0 <= n_players - 2 < 5:
        feat[off + n_players - 2] = 1.0

    # Continuous features (4)
    off = 249
    feat[off] = min(pot_size / 100.0, 1.0)       # normalized pot size
    feat[off + 1] = min(stack_ratio / 10.0, 1.0)  # normalized stack ratio
    feat[off + 2] = min(facing_size / 3.0, 1.0)   # normalized facing size
    feat[off + 3] = agg_actions / 3.0              # normalized aggression

    # Action history: last 12 actions (multi-hot)
    off = 253
    if action_history:
        for i, a in enumerate(action_history[-12:]):
            feat[off + i] = a / (N_ACTIONS - 1)  # normalize to 0-1

    return feat


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def _build_net():
    """Build the postflop network."""
    import torch, torch.nn as nn

    class PostflopNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(FEATURE_DIM, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(512, N_ACTIONS)
            self.value_head = nn.Linear(512, 1)

        def forward(self, x):
            h = self.net(x)
            logits = self.policy_head(h)
            value = self.value_head(h)
            return logits, value

    return PostflopNet()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class PostflopNN:
    """Neural postflop strategy engine."""

    def __init__(self):
        self.net = None
        self._ckpt_step = -1
        self._last_check = 0

    def _ensure_loaded(self):
        """Lazy-load the latest checkpoint."""
        if self.net is not None and time.time() - self._last_check < 30:
            return  # Check at most every 30s

        self._last_check = time.time()
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "postflop_nn_*.pt")))
        ckpts = [c for c in ckpts if "_optimizer" not in os.path.basename(c)]
        if not ckpts:
            return

        # Get step from filename
        latest = ckpts[-1]
        step = int(os.path.basename(latest).replace("postflop_nn_", "").replace(".pt", ""))

        if step <= self._ckpt_step and self.net is not None:
            return  # Already loaded this or newer

        import torch
        if self.net is None:
            self.net = _build_net()

        state = torch.load(latest, map_location="cpu", weights_only=False)
        self.net.load_state_dict(state["model_state"])
        self.net.eval()
        self._ckpt_step = step
        log.info("Loaded postflop NN checkpoint step %d", step)

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self.net is not None

    def query(
        self,
        hand_cat: int,
        position: int,
        street: int,
        texture_id: int,
        n_players: int,
        pot_size: float,
        stack_ratio: float,
        facing_size: float,
        agg_actions: int,
        action_history: list = None,
        legal_actions: list = None,
    ) -> dict:
        """Query the neural postflop strategy.

        Returns dict mapping action names to probabilities.
        """
        import torch

        self._ensure_loaded()
        if self.net is None:
            # Fallback: uniform over legal actions
            if legal_actions:
                p = 1.0 / len(legal_actions)
                return {ACTION_NAMES[a]: p for a in legal_actions if a < N_ACTIONS}
            return {"check_call": 1.0}

        feat = encode_features(
            hand_cat=hand_cat, position=position, street=street,
            texture_id=texture_id, n_players=n_players,
            pot_size=pot_size, stack_ratio=stack_ratio,
            facing_size=facing_size, agg_actions=agg_actions,
            action_history=action_history,
        )

        with torch.no_grad():
            x = torch.from_numpy(feat).unsqueeze(0)
            logits, value = self.net(x)
            probs = torch.softmax(logits[0], dim=-1).numpy()

        # Mask illegal actions
        if legal_actions is not None:
            mask = np.zeros(N_ACTIONS, dtype=np.float32)
            for a in legal_actions:
                if 0 <= a < N_ACTIONS:
                    mask[a] = 1.0
            probs = probs * mask
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                # Fallback to uniform over legal
                for a in legal_actions:
                    if 0 <= a < N_ACTIONS:
                        probs[a] = 1.0 / len(legal_actions)

        result = {}
        for i, name in enumerate(ACTION_NAMES):
            if probs[i] > 0.001:
                result[name] = round(float(probs[i]), 4)

        return result


# Singleton
_instance = None

def get_postflop_nn() -> PostflopNN:
    global _instance
    if _instance is None:
        _instance = PostflopNN()
    return _instance


def nn_status() -> dict:
    """Return NN status for the progress page."""
    nn = get_postflop_nn()
    # Only consider model checkpoints, not optimizer state files
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "postflop_nn_*.pt")))
    ckpts = [c for c in ckpts if "_optimizer" not in os.path.basename(c)]

    latest_step = 0
    latest_name = None
    n_params = 0

    if ckpts:
        latest = ckpts[-1]
        latest_name = os.path.basename(latest)
        latest_step = int(
            os.path.basename(latest).replace("postflop_nn_", "").replace(".pt", "")
        )
        n_params = sum(p.numel() for p in _build_net().parameters())

    # Training progress
    training = None
    prog_path = os.path.join(CKPT_DIR, "training_progress.json")
    if os.path.exists(prog_path):
        try:
            import json as _json
            with open(prog_path) as f:
                training = _json.load(f)
        except Exception:
            pass

    return {
        "available": nn.available,
        "n_checkpoints": len(ckpts),
        "latest_checkpoint": latest_name,
        "step": latest_step,
        "params": n_params,
        "training": training,
    }
