"""
Preflop charts — stack-depth-aware GTO strategy query engine.

Loads chart-friendly PKL files produced by the fixed-tree trainer and interpolates
between the two nearest stack-depth buckets for correct strategy at any depth.

Stack buckets (in BB): 30, 50, 75, 100, 150, 200

Effective stack formula (multiway):
  eff = 0.45 * (vs_BB) + 0.25 * (vs_SB) + 0.20 * avg(other stacks behind) + 0.10 * min(all opponents)

Bet sizing (position-aware, per spec):
  Open: 3bb from BTN/SB; 2.5bb from all other positions.
  3bet IP: 9bb fixed.  3bet OOP: 12bb fixed.
  Squeeze IP: (3 + n_callers) * open_size.  Squeeze OOP: (4 + n_callers) * open_size.
  4bet IP: 2.3x the 3bet.  4bet OOP: 2.8x the 3bet.
  5bet: all-in only.  No 6bet branches.
"""
import json
import logging
import pickle
import random
from pathlib import Path

from config import RANGES_DIR

logger = logging.getLogger(__name__)

TABLES_DIR = Path(__file__).parent.parent / "data" / "preflop_tables"

# BB stack buckets in ascending order
STACK_BUCKETS = [30, 50, 75, 100, 150, 200]

# Positions that use 3bb open (vs 2.5bb for others)
_BTN_SB = {"btn", "sb", "bu", "button", "small blind", "dealer"}

# Position ordering per table size (same as trainer)
POSITION_ORDER = {
    2: ["SB", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    6: ["LJ", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG1", "UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
}

RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {r: i for i, r in enumerate(RANK_ORDER, 2)}


# ─── Hand category (from preflop.py, now merged) ────────────────────────────

_range_cache: dict = {}


def _load_range(position: str) -> dict:
    key = position.lower()
    if key not in _range_cache:
        path = RANGES_DIR / f"{key}.json"
        with open(path) as f:
            _range_cache[key] = json.load(f)
    return _range_cache[key]


def hand_category(c1: str, c2: str) -> str:
    """Convert two card strings (e.g. 'Ah', 'Ks') to 169-hand category like 'AKs'."""
    r1, s1 = c1[0].upper(), c1[1].lower()
    r2, s2 = c2[0].upper(), c2[1].lower()
    if RANK_VALUE.get(r1, 0) < RANK_VALUE.get(r2, 0):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return f"{r1}{r2}"
    return f"{r1}{r2}s" if s1 == s2 else f"{r1}{r2}o"


def should_open(position: str, c1: str, c2: str) -> bool:
    data = _load_range(position)
    cat = hand_category(c1, c2)
    return cat in data.get("open", [])


def preflop_action(position: str, c1: str, c2: str, facing_raise: bool = False, raise_position: str = "btn") -> str:
    """Returns 'raise', 'call', or 'fold'. Uses range charts as fallback."""
    data = _load_range(position)
    cat = hand_category(c1, c2)

    if facing_raise:
        if position == "bb":
            defense_key = f"vs_{raise_position}_open_call"
            if cat in data.get("vs_open_3bet", []):
                return "raise"
            if cat in data.get(defense_key, data.get("vs_btn_open_call", [])):
                return "call"
            return "fold"
        else:
            if cat in data.get("vs_raise_3bet", []):
                return "raise"
            if cat in data.get("vs_raise_call", []):
                return "call"
            return "fold"

    if cat in data.get("open", []):
        return "raise"
    return "fold"


# ─── Bet sizing (per spec) ───────────────────────────────────────────────────

def open_raise_size(stack: int, bb: int, position: str = "") -> int:
    """Position-aware open raise size. BTN/SB: 3bb. Others: 2.5bb."""
    if position.lower() in _BTN_SB:
        return min(3 * bb, stack)
    return min(int(2.5 * bb), stack)


def three_bet_size(last_raise: int, stack: int, is_ip: bool = True, bb: int = 100) -> int:
    """IP=9bb, OOP=12bb per spec. Falls back to 3× for non-standard depths."""
    if bb in (30, 50, 75, 100, 150, 200):
        size = (9 * bb) if is_ip else (12 * bb)
        return min(size, stack)
    return min(last_raise * 3, stack)


def four_bet_size(last_raise: int, stack: int, is_ip: bool = True, bb: int = 100) -> int:
    """IP=2.3× facing, OOP=2.8× facing per spec. Falls back to 2.5× for non-standard depths."""
    if bb in (30, 50, 75, 100, 150, 200):
        size = int(2.3 * last_raise) if is_ip else int(2.8 * last_raise)
        return min(size, stack)
    return min(int(last_raise * 2.5), stack)


# ─── Effective stack computation ─────────────────────────────────────────────

def effective_stack_bb(
    my_stack: int,
    stacks: list[int],
    player_idx: int,
    bb: int,
    n_players: int,
) -> float:
    """Compute effective stack in BB using spec weighted formula.

    45% vs BB, 25% vs SB, 20% avg eff vs players behind, 10% shortest
    dangerous stack behind. Positions list is in action order: first-to-act
    first, SB second-to-last, BB last.
    """
    if not stacks or bb <= 0:
        return 100.0

    n = len(stacks)
    if n <= 1 or player_idx >= n:
        return my_stack / bb

    # HU: simple min
    if n == 2:
        return min(my_stack, stacks[1 - player_idx]) / bb

    # SB is at index n-2, BB at index n-1 in the action-order positions list
    sb_idx = n - 2
    bb_idx = n - 1
    sb_stack = stacks[sb_idx]
    bb_stack = stacks[bb_idx]

    # Players "behind" hero = those who act after hero (higher indices)
    behind_indices = [i for i in range(player_idx + 1, n) if i != sb_idx and i != bb_idx]
    behind_stacks = [stacks[i] for i in behind_indices]

    avg_behind = (sum(min(my_stack, s) for s in behind_stacks) / len(behind_stacks)) if behind_stacks else 0
    # Shortest dangerous stack behind = shortest stack of non-blind players acting after hero
    min_dangerous = min((min(my_stack, s) for s in behind_stacks), default=0)

    eff_chips = (
        0.45 * min(my_stack, bb_stack) +
        0.25 * min(my_stack, sb_stack) +
        0.20 * avg_behind +
        0.10 * min_dangerous
    )
    return eff_chips / bb


# ─── Bucket selection and interpolation ──────────────────────────────────────

def _nearest_buckets(eff_bb: float) -> tuple[int, int, float]:
    """Return (lo_bucket, hi_bucket, blend) for linear interpolation."""
    if eff_bb <= STACK_BUCKETS[0]:
        return STACK_BUCKETS[0], STACK_BUCKETS[0], 0.0
    if eff_bb >= STACK_BUCKETS[-1]:
        return STACK_BUCKETS[-1], STACK_BUCKETS[-1], 0.0

    for i in range(len(STACK_BUCKETS) - 1):
        lo, hi = STACK_BUCKETS[i], STACK_BUCKETS[i + 1]
        if lo <= eff_bb <= hi:
            blend = (eff_bb - lo) / (hi - lo)
            return lo, hi, blend

    return STACK_BUCKETS[-1], STACK_BUCKETS[-1], 0.0


def _interpolate_probs(lo_probs: dict, hi_probs: dict, blend: float) -> dict:
    """Linearly interpolate two action probability dicts."""
    if lo_probs is None and hi_probs is None:
        return None
    if lo_probs is None:
        return hi_probs
    if hi_probs is None or blend == 0.0:
        return lo_probs
    if blend == 1.0:
        return hi_probs

    all_keys = set(lo_probs) | set(hi_probs)
    merged = {}
    for k in all_keys:
        lo_p = lo_probs.get(k, 0.0)
        hi_p = hi_probs.get(k, 0.0)
        if isinstance(lo_p, tuple):
            lo_p = lo_p[1]
        if isinstance(hi_p, tuple):
            hi_p = hi_p[1]
        merged[k] = (1.0 - blend) * lo_p + blend * hi_p

    total = sum(merged.values())
    if total > 0:
        merged = {k: v / total for k, v in merged.items()}
    return merged


# ─── Policy cache (loads from chart-friendly PKLs) ───────────────────────────

_policy_cache: dict[tuple[int, int], dict | None] = {}


def _load_policy(n_players: int, stack_bb: int) -> dict | None:
    """Load a chart-friendly PKL. Returns dict[(hand, pidx, hist_tuple), dict[str,float]] or None."""
    key = (n_players, stack_bb)
    if key in _policy_cache:
        return _policy_cache[key]

    suffix = "" if stack_bb == 100 else f"_{stack_bb}bb"
    path = TABLES_DIR / f"{n_players}p{suffix}_preflop_policy.pkl"

    if not path.exists() or path.stat().st_size == 0:
        _policy_cache[key] = None
        return None

    try:
        with open(path, "rb") as f:
            policy = pickle.load(f)
        _policy_cache[key] = policy
        return policy
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        _policy_cache[key] = None
        return None


def position_to_player_idx(position: str, n_players: int) -> int:
    """Convert a position name to the trainer's player index.

    The trainer indexes positions in first-to-act order:
      6-max: LJ=0, HJ=1, CO=2, BTN=3, SB=4, BB=5
    The engine uses a different relative-from-button scheme.
    This function bridges the two so bots query the correct chart slot.
    """
    return _position_to_idx(position, n_players)


def _position_to_idx(position: str, n_players: int) -> int:
    """Convert position name to player index using the trainer's ordering."""
    labels = POSITION_ORDER.get(n_players, POSITION_ORDER[6])
    pos_upper = position.upper()
    for i, label in enumerate(labels):
        if label == pos_upper:
            return i
    # Common aliases
    aliases = {"DEALER": "BTN", "UTG+1": "UTG1", "MP": "LJ", "MP1": "LJ", "MP2": "HJ", "EP": "UTG"}
    pos_mapped = aliases.get(pos_upper, pos_upper)
    for i, label in enumerate(labels):
        if label == pos_mapped:
            return i
    return 0  # fallback


def _query_policy(
    policy: dict,
    hand_cat: str,
    player_idx: int,
    action_history: list[str],
) -> dict[str, float] | None:
    """Look up a single policy entry. Returns {action: prob} or None."""
    hist_tuple = tuple(action_history)
    entry = policy.get((hand_cat, player_idx, hist_tuple))
    if entry is not None:
        return entry
    # Try without history (RFI spot has empty tuple)
    return None


# ─── Public API ───────────────────────────────────────────────────────────────

def action_probs(
    hole_cards: list[str],
    player_idx: int,
    action_history: list[str],
    n_players: int,
    stacks: list[int],
    bb: int,
    position: str = "",
) -> dict[str, float] | None:
    """
    Return {action: prob} or None if no solver data is available.

    Converts hole cards to 169-hand category, finds the nearest stack buckets,
    and interpolates between them.
    """
    # Convert cards to hand category
    if len(hole_cards) < 2:
        return None
    hand_cat = hand_category(hole_cards[0], hole_cards[1])

    # Compute effective stack
    eff_bb = effective_stack_bb(
        stacks[player_idx] if player_idx < len(stacks) else bb * 100,
        stacks, player_idx, bb, n_players,
    )

    lo_bb, hi_bb, blend = _nearest_buckets(eff_bb)

    lo_policy = _load_policy(n_players, lo_bb)
    hi_policy = _load_policy(n_players, hi_bb) if hi_bb != lo_bb else lo_policy

    lo_entry = _query_policy(lo_policy, hand_cat, player_idx, action_history) if lo_policy else None
    hi_entry = _query_policy(hi_policy, hand_cat, player_idx, action_history) if hi_policy and hi_bb != lo_bb else None

    if lo_entry is None and hi_entry is None:
        # Try nearest available bucket
        for bb_try in sorted(STACK_BUCKETS, key=lambda b: abs(b - eff_bb)):
            fb_policy = _load_policy(n_players, bb_try)
            if fb_policy:
                entry = _query_policy(fb_policy, hand_cat, player_idx, action_history)
                if entry is not None:
                    return entry
        return None

    return _interpolate_probs(lo_entry, hi_entry, blend)


def sample_action(
    hole_cards: list[str],
    player_idx: int,
    action_history: list[str],
    n_players: int,
    stacks: list[int],
    bb: int,
    position: str = "",
) -> str | None:
    """Sample a single action from the stack-adjusted GTO strategy."""
    probs = action_probs(hole_cards, player_idx, action_history, n_players, stacks, bb, position)
    if not probs:
        return None
    keys = list(probs.keys())
    weights = [probs[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]
