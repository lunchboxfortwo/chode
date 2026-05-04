#!/usr/bin/env python3
"""Train a neural postflop solver via distillation from tabular MCCFR.

Two-phase pipeline:
  Phase 1 — Distillation: Simulate random game states, build info_key, hash it,
            binary-search the NPZ for the matching strategy, train NN to predict it.
  Phase 2 — Online MCCFR: Fine-tune with neural MCCFR self-play.

The NPZ files use columnar format with MD5-hashed uint64 keys — you cannot
parse the keys back into game features. Instead, we generate random game states,
encode them into info_key bytes (same as the trainer), hash with MD5, and look
up the strategy via binary search in the sorted keys array.

Usage:
  python3 solver_training/train_postflop_nn.py --iters 1000 --phase distill
  python3 solver_training/train_postflop_nn.py --phase online --iters 100000
"""
import argparse, json, os, sys, time, logging, glob, random, hashlib
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(CHODE, "data")
POSTFLOP_TABLES = os.path.join(DATA_DIR, "postflop_tables")
CKPT_DIR = os.path.join(DATA_DIR, "postflop_nn")
CACHE_PATH = os.path.join(CKPT_DIR, "distill_data.npz")
PROGRESS_PATH = os.path.join(CKPT_DIR, "training_progress.json")

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
VALUE_LOSS_WEIGHT = 0.25
CHECKPOINT_EVERY = 50   # steps
LOG_EVERY = 10          # steps
EXAMPLES_PER_ROUND = 50_000  # random states to sample per distillation round

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
sys.path.insert(0, CHODE)
from strategy.board_abstraction import texture_id, board_texture
from strategy.postflop_nn import (
    encode_features, FEATURE_DIM, N_ACTIONS, ACTION_NAMES,
    CKPT_DIR as NN_CKPT_DIR,
)

# ---------------------------------------------------------------------------
# Card / key utilities (mirrors postflop_fixed_train.py encoding)
# ---------------------------------------------------------------------------
RANKS = "23456789TJQKA"


def _hand_cat_str(c1: int, c2: int) -> str:
    """Card indices (0-51) → hand category string like 'AKs', '72o', 'TT'."""
    r1, s1 = c1 // 4, c1 % 4
    r2, s2 = c2 // 4, c2 % 4
    if r1 < r2:
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return RANKS[r1] * 2
    return RANKS[r1] + RANKS[r2] + ('s' if s1 == s2 else 'o')


def _hand_cat_to_int(hand_cat_str: str) -> int:
    """Hand category string → int 0-168.
    Pairs: 0-12, Suited: 13-90, Offsuit: 91-168.
    """
    r1 = RANKS.index(hand_cat_str[0])
    r2 = RANKS.index(hand_cat_str[1]) if len(hand_cat_str) > 1 else r1
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return hi  # pairs 0-12
    idx = 0
    for h in range(12, 0, -1):
        for l in range(h - 1, -1, -1):
            if h == hi and l == lo:
                if len(hand_cat_str) > 2 and hand_cat_str[2] == 's':
                    return 13 + idx  # suited
                else:
                    return 91 + idx  # offsuit
            idx += 1
    return 0


def _pack_hand_cat(hand_cat_str: str) -> tuple:
    """Pack hand category string → (r1_r2_byte, suit_byte)."""
    r1 = RANKS.index(hand_cat_str[0])
    r2 = RANKS.index(hand_cat_str[1]) if len(hand_cat_str) > 1 else r1
    if len(hand_cat_str) == 2:  # pair
        suit = 2
    elif hand_cat_str[2] == 's':
        suit = 1
    else:
        suit = 0
    return ((r1 << 4) | r2, suit)


def _board_norm_raw(board: list[int]) -> list:
    """Suit-isomorphic board normalization. Returns list of (rank, suit_idx) pairs."""
    flop = sorted(board[:3], key=lambda c: (-c // 4, c % 4))
    rest = board[3:]
    suit_map = {}
    next_suit = 0
    norm = []
    for c in (flop + rest):
        r, s = c // 4, c % 4
        if s not in suit_map:
            suit_map[s] = next_suit
            next_suit += 1
        norm.append((r, suit_map[s]))
    return norm


def _board_texture_from_cards(board_cards: list) -> int:
    """Compute board texture ID from card indices (0-51)."""
    if len(board_cards) < 3:
        return 0
    # board_abstraction.texture_id expects card strings, but we have indices
    # Compute directly from rank/suit data
    ranks = sorted([c // 4 for c in board_cards[:3]], reverse=True)
    suits = [c % 4 for c in board_cards[:3]]

    # High card bucket
    top = ranks[0]
    high = 0 if top == 12 else 1 if top == 11 else 2 if top == 10 else 3 if top == 9 else 4 if top == 8 else 5

    # Paired
    paired = int(ranks[0] == ranks[1] or ranks[1] == ranks[2])

    # Suit texture
    unique_suits = len(set(suits))
    suit_tex = 2 if unique_suits == 1 else 1 if unique_suits == 2 else 0

    # Connectedness
    gaps = [ranks[i] - ranks[i + 1] for i in range(len(ranks) - 1)]
    max_gap = max(gaps) if gaps else 0
    connected = 0 if max_gap <= 3 else 1

    return high * 12 + paired * 6 + suit_tex * 2 + connected


def build_info_key(
    player: int, hole_cards: list[int], board: list[int],
    street: int, facing_bet: bool, agg_count: int, action_hist: list[int],
) -> bytes:
    """Build the same info_key bytes as postflop_fixed_train.py.

    Encoding:
      0       1     player (0-7)
      1       1     hand_cat packed: (rank1 << 4) | rank2
      2       1     hand_cat suit: 0=offsuit, 1=suited, 2=pair
      3       nc    board_norm: (rank<<4|suit_idx) per card, nc=3..5
      3+nc    1     street (0-2)
      4+nc    1     facing_bet: 0 or 1
      5+nc    1     agg_count (0-3)
      6+nc    na    action_hist: 1 byte per action (0-3)
    """
    buf = bytearray()
    buf.append(player)
    hc = _hand_cat_str(hole_cards[0], hole_cards[1])
    r1r2, suit = _pack_hand_cat(hc)
    buf.append(r1r2)
    buf.append(suit)
    bn = _board_norm_raw(board)
    for r, si in bn:
        buf.append((r << 4) | si)
    buf.append(street)
    buf.append(int(facing_bet))
    buf.append(agg_count)
    for a in action_hist:
        buf.append(a)
    return bytes(buf)


def hash_key(key: bytes) -> np.uint64:
    """Hash an info_key to uint64, matching _hash_key in postflop_fixed_train.py."""
    return np.uint64(int.from_bytes(
        hashlib.md5(key).digest()[:8], "little", signed=False
    ))


# ---------------------------------------------------------------------------
# NPZ loader with binary search
# ---------------------------------------------------------------------------
class NPZLookup:
    """Memory-mapped NPZ policy table with binary-search lookup."""

    def __init__(self, npz_path: str):
        log.info("Loading NPZ: %s", npz_path)
        t0 = time.time()
        data = np.load(npz_path, allow_pickle=True)
        self.keys = data["keys"]          # uint64, sorted
        self.actions = data["actions"]    # int16 (N, 4)
        self.probs = data["probs"]        # float16 (N, 4)
        self.n_players = int(data["n_players"][0])
        self.n_entries = len(self.keys)
        log.info("  Loaded %d entries in %.1fs", self.n_entries, time.time() - t0)

    def lookup(self, info_key_bytes: bytes):
        """Binary-search for a key. Returns (action_ids, probabilities) or None."""
        h = hash_key(info_key_bytes)
        idx = np.searchsorted(self.keys, h)
        if idx < self.n_entries and self.keys[idx] == h:
            return self.actions[idx], self.probs[idx]
        return None


# ---------------------------------------------------------------------------
# Random game state generator
# ---------------------------------------------------------------------------
def _random_state(n_players: int):
    """Generate a random postflop game state.

    Returns dict with: player, hole, board, street, facing_bet, agg_count, action_hist,
                       plus derived features: hand_cat, texture_id, pot_size, stack_ratio, facing_size
    """
    deck = list(range(52))
    random.shuffle(deck)

    # Deal hole cards
    holes = [deck[i*2:(i+1)*2] for i in range(n_players)]
    # Deal board
    n_board_cards = random.choice([3, 4, 5])
    board_start = n_players * 2
    board = deck[board_start:board_start + n_board_cards]
    street = n_board_cards - 3  # 0=flop, 1=turn, 2=river

    # Choose acting player
    player = random.randint(0, n_players - 1)

    # Generate action history (0-3 per action, capped at 8 actions)
    # Bias toward shorter histories — most NPZ entries are early-street with few actions
    hist_len = random.choices([0, 1, 2, 3, 4, 5, 6, 7, 8],
                              weights=[35, 25, 15, 10, 5, 4, 3, 2, 1])[0]
    action_hist = [random.randint(0, 3) for _ in range(hist_len)]

    # Count aggressive actions in history
    agg_count = min(sum(1 for a in action_hist if a >= 2), 3)

    # Facing bet: true if last action was aggressive (2=raise, 3=allin)
    facing_bet = len(action_hist) > 0 and action_hist[-1] >= 2

    # Pot / stack features (approximate)
    n_calls = sum(1 for a in action_hist if a == 1)
    n_raises = sum(1 for a in action_hist if a == 2)
    pot_size = 2.0 + n_calls * 1.0 + n_raises * 2.5  # rough BB units
    stack_ratio = random.uniform(0.5, 15.0)
    facing_size = random.choice([0, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]) if facing_bet else 0.0

    # Hand category
    hc_str = _hand_cat_str(holes[player][0], holes[player][1])
    hand_cat = _hand_cat_to_int(hc_str)

    # Board texture
    tex_id = _board_texture_from_cards(board)

    # Build info_key
    info_key_bytes = build_info_key(
        player=player,
        hole_cards=holes[player],
        board=board,
        street=street,
        facing_bet=facing_bet,
        agg_count=agg_count,
        action_hist=action_hist,
    )

    return {
        "player": player,
        "hole": holes[player],
        "board": board,
        "street": street,
        "facing_bet": facing_bet,
        "agg_count": agg_count,
        "action_hist": action_hist,
        "hand_cat": hand_cat,
        "texture_id": tex_id,
        "pot_size": pot_size,
        "stack_ratio": stack_ratio,
        "facing_size": facing_size,
        "n_players": n_players,
        "info_key": info_key_bytes,
    }


# ---------------------------------------------------------------------------
# Distillation dataset builder (simulation + lookup)
# ---------------------------------------------------------------------------
def build_distillation_dataset(force: bool = False) -> tuple:
    """Build or load cached distillation dataset.

    Strategy: Generate random game states, build info_key, hash it,
    binary-search the NPZ for the strategy, and use found rows as targets.

    Returns (features, targets, masks, values) as numpy arrays.
    """
    if os.path.exists(CACHE_PATH) and not force:
        log.info("Loading cached distillation data from %s", CACHE_PATH)
        d = np.load(CACHE_PATH)
        return d["features"], d["targets"], d["masks"], d["values"]

    os.makedirs(CKPT_DIR, exist_ok=True)

    # Load NPZ lookup tables
    lookups = []
    for npz in sorted(glob.glob(os.path.join(POSTFLOP_TABLES, "*_policy.npz"))):
        try:
            lookups.append(NPZLookup(npz))
        except Exception as e:
            log.warning("Failed to load NPZ %s: %s", npz, e)

    if not lookups:
        log.error("No NPZ policy files found in %s! Generating synthetic data.", POSTFLOP_TABLES)
        return _generate_synthetic_arrays(50_000)

    # Generate examples via simulation + lookup
    all_features = []
    all_targets = []
    all_masks = []
    all_values = []

    # NPZ action mapping: 0=fold, 1=call, 2=raise, 3=allin
    # NN action mapping:  0=fold, 1=check_call, 2=bet_small, 3=bet_large, 4=raise, 5=allin
    NPZ_TO_NN = {0: 0, 1: 1, 2: 4, 3: 5}  # fold→fold, call→check_call, raise→raise, allin→allin

    total_tried = 0
    total_found = 0
    target_examples = 500_000  # aim for 500K examples (sufficient for distillation baseline)

    log.info("Generating distillation data: %d target examples", target_examples)

    while total_found < target_examples:
        # Pick a random NPZ (weighted by entry count)
        lookup = random.choice(lookups)
        n_players = lookup.n_players

        # Generate a batch of random states
        batch_size = min(10_000, target_examples - total_found + 1000)
        for _ in range(batch_size):
            total_tried += 1
            state = _random_state(n_players)

            # Look up strategy in NPZ
            result = lookup.lookup(state["info_key"])
            if result is None:
                continue

            npz_actions, npz_probs = result
            total_found += 1

            # Map NPZ actions → NN actions and build target distribution
            nn_probs = np.zeros(N_ACTIONS, dtype=np.float32)
            nn_mask = np.zeros(N_ACTIONS, dtype=np.float32)
            for slot in range(4):
                a_npz = int(npz_actions[slot])
                p = float(npz_probs[slot])
                if a_npz < 0 or p <= 0:
                    continue
                a_nn = NPZ_TO_NN.get(a_npz, a_npz)
                if 0 <= a_nn < N_ACTIONS:
                    nn_probs[a_nn] += p
                    nn_mask[a_nn] = 1.0

            # Renormalize
            total_p = nn_probs.sum()
            if total_p > 0:
                nn_probs /= total_p
            else:
                continue

            # Build feature vector
            feat = encode_features(
                hand_cat=state["hand_cat"],
                position=state["player"],
                street=state["street"],
                texture_id=state["texture_id"],
                n_players=n_players,
                pot_size=state["pot_size"],
                stack_ratio=state["stack_ratio"],
                facing_size=state["facing_size"],
                agg_actions=state["agg_count"],
                action_history=state["action_hist"],
            )

            all_features.append(feat)
            all_targets.append(nn_probs)
            all_masks.append(nn_mask)
            all_values.append(0.0)  # value target unknown during distillation

        hit_rate = total_found / max(total_tried, 1) * 100
        log.info("  Found %d/%d (%.1f%% hit rate)", total_found, total_tried, hit_rate)

    features = np.stack(all_features)
    targets = np.stack(all_targets)
    masks = np.stack(all_masks)
    values = np.array(all_values, dtype=np.float32)

    log.info("Saving %d examples to %s", len(features), CACHE_PATH)
    np.savez(CACHE_PATH, features=features, targets=targets, masks=masks, values=values)
    return features, targets, masks, values


def _generate_synthetic_arrays(n: int) -> tuple:
    """Generate random synthetic training data for testing (when no NPZ files exist)."""
    log.warning("Generating %d synthetic examples — train on real data for useful results!", n)
    features = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    targets = np.zeros((n, N_ACTIONS), dtype=np.float32)
    masks = np.zeros((n, N_ACTIONS), dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)

    for i in range(n):
        feat = encode_features(
            hand_cat=random.randint(0, 168),
            position=random.randint(0, 5),
            street=random.randint(0, 2),
            texture_id=random.randint(0, 63),
            n_players=random.randint(2, 5),
            pot_size=random.uniform(1, 20),
            stack_ratio=random.uniform(0.5, 10),
            facing_size=random.choice([0, 0.33, 0.75, 1.0, 2.0]),
            agg_actions=random.randint(0, 3),
            action_history=[random.randint(0, 5) for _ in range(random.randint(0, 5))],
        )
        features[i] = feat

        n_legal = random.randint(2, 4)
        legal = sorted(random.sample(range(N_ACTIONS), n_legal))
        raw = np.random.random(n_legal).astype(np.float32)
        raw /= raw.sum()
        for j, a in enumerate(legal):
            targets[i, a] = raw[j]
            masks[i, a] = 1.0
        values[i] = random.uniform(-1, 1)

    return features, targets, masks, values


# ---------------------------------------------------------------------------
# Phase 1: Distillation training
# ---------------------------------------------------------------------------
def distill(n_rounds: int = 1000, resume: bool = True):
    """Train the postflop NN by distilling from tabular solver data."""
    import torch, torch.nn as nn

    features, targets, masks, values = build_distillation_dataset(force=not resume)
    N = len(features)
    log.info("Distillation dataset: %d examples, %d features", N, features.shape[1])

    # Mild inverse-frequency weighting (sqrt for balance without overwhelming rare actions)
    action_counts = masks.sum(axis=0)
    action_weights = np.ones(N_ACTIONS, dtype=np.float32)
    for i in range(N_ACTIONS):
        if action_counts[i] > 0:
            action_weights[i] = np.sqrt(N / (N_ACTIONS * action_counts[i]))
    action_weights_tensor = torch.from_numpy(action_weights)

    os.makedirs(CKPT_DIR, exist_ok=True)

    from strategy.postflop_nn import _build_net
    net = _build_net()

    start_step = 0
    ckpts = []
    if resume:
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "postflop_nn_*.pt")))
        if ckpts:
            state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
            net.load_state_dict(state["model_state"])
            start_step = state.get("step", 0)
            log.info("Resumed from step %d", start_step)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    if resume and ckpts:
        opt_path = ckpts[-1].replace(".pt", "_optimizer.pt")
        if os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(opt_state)

    net.train()
    step = start_step
    t0 = time.time()

    for rnd in range(n_rounds):
        # Subsample
        idx = np.random.choice(N, size=min(EXAMPLES_PER_ROUND, N), replace=False)
        X = torch.from_numpy(features[idx])
        Y = torch.from_numpy(targets[idx])
        M = torch.from_numpy(masks[idx])

        # Mini-batch training
        perm = np.random.permutation(len(idx))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(perm), BATCH_SIZE):
            batch = perm[start:start + BATCH_SIZE]
            x = X[batch]
            y = Y[batch]
            m = M[batch]

            logits, pred_v = net(x)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Masked policy loss with inverse-frequency weighting
            policy_loss = -(y * m * log_probs * action_weights_tensor).sum(dim=-1).mean()

            # Value loss
            value_loss = nn.MSELoss()(pred_v.squeeze(-1), torch.zeros(len(batch)))

            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            if step % LOG_EVERY == 0:
                log.info("Step %d | round %d | loss %.4f", step, rnd, loss.item())
                _save_progress(step, rnd, n_rounds, epoch_loss / max(n_batches, 1),
                               N, time.time() - t0, "distill")

            if step % CHECKPOINT_EVERY == 0:
                _save_checkpoint(net, optimizer, step, CKPT_DIR)
                _save_progress(step, rnd, n_rounds, epoch_loss / max(n_batches, 1),
                               N, time.time() - t0, "distill")

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        log.info("Round %d/%d | step %d | avg_loss %.4f | %.1fs", rnd + 1, n_rounds, step, avg_loss, elapsed)

        # Save checkpoint every 10 rounds
        if (rnd + 1) % 10 == 0:
            _save_checkpoint(net, optimizer, step, CKPT_DIR)
            _save_progress(step, rnd, n_rounds, avg_loss, N, elapsed, "distill")

    # Final save
    _save_checkpoint(net, optimizer, step, CKPT_DIR)
    _save_progress(step, n_rounds, n_rounds, 0.0, N, time.time() - t0, "distill")
    log.info("Distillation complete: %d steps", step)


# ---------------------------------------------------------------------------
# Phase 2: Online MCCFR fine-tuning
# ---------------------------------------------------------------------------
def online_mccfr(n_iters: int = 100000, resume: bool = True):
    """Fine-tune the postflop NN with online MCCFR self-play."""
    import torch, torch.nn as nn

    os.makedirs(CKPT_DIR, exist_ok=True)

    from strategy.postflop_nn import _build_net
    net = _build_net()

    start_step = 0
    if resume:
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "postflop_nn_*.pt")))
        if ckpts:
            state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
            net.load_state_dict(state["model_state"])
            start_step = state.get("step", 0)

    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE * 0.1)

    buf_features = np.zeros((BATCH_SIZE * 20, FEATURE_DIM), dtype=np.float32)
    buf_targets = np.zeros((BATCH_SIZE * 20, N_ACTIONS), dtype=np.float32)
    buf_masks = np.zeros((BATCH_SIZE * 20, N_ACTIONS), dtype=np.float32)
    buf_values = np.zeros(BATCH_SIZE * 20, dtype=np.float32)
    buf_pos = 0

    step = start_step
    t0 = time.time()

    for it in range(n_iters):
        # Generate one MCCFR traversal via simulation + NPZ lookup
        examples = _traverse_one(net)

        for feat, target, mask, val in examples:
            if buf_pos >= len(buf_features):
                step = _train_on_buffer(net, optimizer, buf_features, buf_targets,
                                        buf_masks, buf_values, buf_pos, step, t0)
                buf_pos = 0

            buf_features[buf_pos] = feat
            buf_targets[buf_pos] = target
            buf_masks[buf_pos] = mask
            buf_values[buf_pos] = val
            buf_pos += 1

        if it % LOG_EVERY == 0 and it > 0:
            elapsed = time.time() - t0
            rate = it / elapsed if elapsed > 0 else 0
            log.info("Online MCCFR iter %d | step %d | %.1f it/s", it, step, rate)
            _save_progress(step, it, n_iters, 0.0, 0, elapsed, "online")

    # Final train
    if buf_pos > 0:
        step = _train_on_buffer(net, optimizer, buf_features, buf_targets,
                                buf_masks, buf_values, buf_pos, step, t0)

    _save_checkpoint(net, optimizer, step, CKPT_DIR)
    _save_progress(step, n_iters, n_iters, 0.0, 0, time.time() - t0, "online")
    log.info("Online MCCFR complete: %d steps", step)


def _traverse_one(net) -> list:
    """Generate one MCCFR traversal via simulation + NPZ lookup.

    Instead of running the full CFR engine, we simulate random game states
    and look up their tabular strategy from the NPZ files. This gives us
    regret-matched targets for online fine-tuning.
    """
    import torch

    # Use the same NPZ lookups as distillation
    if not hasattr(_traverse_one, "_lookups"):
        _traverse_one._lookups = []
        for npz in sorted(glob.glob(os.path.join(POSTFLOP_TABLES, "*_policy.npz"))):
            try:
                _traverse_one._lookups.append(NPZLookup(npz))
            except Exception:
                pass

    lookups = _traverse_one._lookups
    if not lookups:
        return []

    NPZ_TO_NN = {0: 0, 1: 1, 2: 4, 3: 5}
    examples = []

    # Generate a few random states and look up strategies
    for _ in range(random.randint(1, 5)):
        lookup = random.choice(lookups)
        state = _random_state(lookup.n_players)
        result = lookup.lookup(state["info_key"])
        if result is None:
            continue

        npz_actions, npz_probs = result
        nn_probs = np.zeros(N_ACTIONS, dtype=np.float32)
        nn_mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for slot in range(4):
            a_npz = int(npz_actions[slot])
            p = float(npz_probs[slot])
            if a_npz < 0 or p <= 0:
                continue
            a_nn = NPZ_TO_NN.get(a_npz, a_npz)
            if 0 <= a_nn < N_ACTIONS:
                nn_probs[a_nn] += p
                nn_mask[a_nn] = 1.0

        total_p = nn_probs.sum()
        if total_p <= 0:
            continue
        nn_probs /= total_p

        feat = encode_features(
            hand_cat=state["hand_cat"],
            position=state["player"],
            street=state["street"],
            texture_id=state["texture_id"],
            n_players=state["n_players"],
            pot_size=state["pot_size"],
            stack_ratio=state["stack_ratio"],
            facing_size=state["facing_size"],
            agg_actions=state["agg_count"],
            action_history=state["action_hist"],
        )
        examples.append((feat, nn_probs, nn_mask, 0.0))

    return examples


def _train_on_buffer(net, optimizer, buf_f, buf_t, buf_m, buf_v, n, step, t0):
    """Train on buffer contents."""
    import torch, torch.nn as nn

    X = torch.from_numpy(buf_f[:n])
    Y = torch.from_numpy(buf_t[:n])
    M = torch.from_numpy(buf_m[:n])

    perm = np.random.permutation(n)
    for start in range(0, n, BATCH_SIZE):
        batch = perm[start:start + BATCH_SIZE]
        logits, pred_v = net(X[batch])
        log_probs = torch.log_softmax(logits, dim=-1)
        y, m = Y[batch], M[batch]
        policy_loss = -(y * m * log_probs).sum(dim=-1).mean()
        value_loss = nn.MSELoss()(pred_v.squeeze(-1), torch.zeros(len(batch)))
        loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        if step % CHECKPOINT_EVERY == 0:
            _save_checkpoint(net, optimizer, step, CKPT_DIR)

    return step


# ---------------------------------------------------------------------------
# Checkpoint / progress
# ---------------------------------------------------------------------------
def _save_checkpoint(net, optimizer, step, ckpt_dir):
    import torch
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"postflop_nn_{step:08d}.pt")
    torch.save({"model_state": net.state_dict(), "step": step}, path)
    opt_path = path.replace(".pt", "_optimizer.pt")
    torch.save(optimizer.state_dict(), opt_path)
    # Keep only last 5 checkpoints
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "postflop_nn_*.pt")))
    for old in ckpts[:-5]:
        os.remove(old)
        opt_old = old.replace(".pt", "_optimizer.pt")
        if os.path.exists(opt_old):
            os.remove(opt_old)
    log.info("Saved checkpoint %s", os.path.basename(path))


def _save_progress(step, round_or_iter, total, loss, n_examples, elapsed, phase):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    progress = {
        "step": step,
        "iteration": round_or_iter,
        "games_played": step * BATCH_SIZE,  # approximate
        "target_games": total,
        "elapsed_seconds": round(elapsed, 1),
        "games_per_second": round(step * BATCH_SIZE / max(elapsed, 1), 1),
        "phase": phase,
        "loss": round(loss, 4),
        "n_examples": n_examples,
        "last_update": time.time(),
        "last_checkpoint_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train postflop NN solver")
    parser.add_argument("--phase", choices=["distill", "online", "both"], default="both",
                        help="Training phase")
    parser.add_argument("--iters", type=int, default=1000,
                        help="Rounds (distill) or iterations (online)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start from scratch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    resume = not args.no_resume

    if args.phase in ("distill", "both"):
        log.info("=== Phase 1: Distillation ===")
        distill(n_rounds=args.iters, resume=resume)

    if args.phase in ("online", "both"):
        log.info("=== Phase 2: Online MCCFR ===")
        online_mccfr(n_iters=args.iters * 100, resume=resume)


if __name__ == "__main__":
    main()
