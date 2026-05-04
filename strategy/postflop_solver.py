"""
Postflop GTO solver — multi-street, multi-player state reconstruction.

Supports two storage formats (tried in order):
  1. Compact  — {n}p_postflop_policy.npz  (~400-700 MB, loads in ~5-15 s)
  2. Full pkl — {n}p_postflop_solver.pkl  (~11-12 GB,  loads in ~60-120 s)

OpenSpiel HU game player mapping:
  blind = SB BB  →  player 0 = SB = IP (acts second postflop)
                     player 1 = BB = OOP (acts first, firstPlayer=2)
  So: position='oop' → player_idx=1, position='ip' → player_idx=0

OpenSpiel 3p game player mapping:
  blind = SB BB 0  →  player 0 = SB (acts first, firstPlayer=1)
                        player 1 = BB
                        player 2 = BTN = IP (acts last)
  So: position='oop' → player_idx=0, position='ip' → player_idx=2
"""
import os
import sys
import pickle
import random
import hashlib
import logging
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.solver import monte_carlo_equity, river_exact_equity
from strategy.board_abstraction import board_texture, bet_fraction
# postflop_net removed — neural distillation not yet implemented

logger = logging.getLogger(__name__)

TABLES_DIR = Path(__file__).parent.parent / "data" / "postflop_tables"

RANKS = "23456789TJQKA"
SUITS = "cdhs"

# Required solvers — both must load before games start
_REQUIRED = (2, 3)

# Per-player-count solver cache
_solvers: dict[int, dict] = {}   # {n: {"game": ..., "policy": ..., "compact": ...}}
_load_lock = threading.Lock()

# Fixed-tree compact tables (loaded on demand; keyed by info_key tuples, not hashes)
_fixed_tree: dict[int, dict | None] = {}  # {n_players: policy_dict or None}


# ─── Compact format helpers ───────────────────────────────────────────────────

def _hash_key(info_state_str: str) -> int:
    return int.from_bytes(
        hashlib.md5(info_state_str.encode()).digest()[:8], "little", signed=False
    )


def _query_compact(compact: dict, state, player_idx: int):
    """Look up action probabilities from compact numpy policy. Returns {action: prob} or None."""
    info_key = state.information_state_string(player_idx)
    h = np.uint64(_hash_key(info_key))
    keys_arr = compact["keys"]
    idx = np.searchsorted(keys_arr, h)
    if idx >= len(keys_arr) or keys_arr[idx] != h:
        return None  # state not seen during training → fallback
    acts  = compact["actions"][idx]
    probs = compact["probs"][idx]
    valid = acts >= 0
    return {int(a): float(p) for a, p in zip(acts[valid], probs[valid])}


# ─── Fixed-tree compact table helpers ────────────────────────────────────────

def _load_fixed_tree(n: int) -> dict | None:
    """Load fixed-tree policy, trying NPZ first (compact, fast), then PKL fallback."""
    if n in _fixed_tree:
        return _fixed_tree[n]

    # Path 1: NPZ compact format (from streaming extract_and_save)
    npz_path = TABLES_DIR / f"{n}p_postflop_fixed_policy.npz"
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=False)
            # Validate metadata matches expected player count
            if "n_players" in data and int(data["n_players"][0]) != n:
                logger.warning(f"NPZ n_players={data['n_players'][0]} != expected {n}, skipping")
                _fixed_tree[n] = None
                return None
            compact = {
                "keys":    data["keys"],
                "actions": data["actions"],
                "probs":   data["probs"],
            }
            # Store as a special marker so _query_fixed_tree knows it's NPZ, not a dict
            _fixed_tree[n] = ("npz", compact)
            logger.info(f"Loaded {n}p fixed-tree postflop NPZ ({len(compact['keys']):,} entries)")
            return _fixed_tree[n]
        except Exception as e:
            logger.warning(f"Failed to load fixed-tree NPZ for {n}p: {e}")

    # Path 2: PKL format (legacy, uses tuple keys — may have key mismatch)
    path = TABLES_DIR / f"{n}p_postflop_fixed_policy.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                policy = pickle.load(f)
            _fixed_tree[n] = ("pkl", policy)
            logger.info(f"Loaded {n}p fixed-tree postflop PKL ({len(policy):,} entries)")
            return _fixed_tree[n]
        except Exception as e:
            logger.warning(f"Failed to load fixed-tree policy for {n}p: {e}")

    _fixed_tree[n] = None
    return None


_FT_ACTION_MAP = {"fold": 0, "check": 1, "call": 1, "bet": 2, "raise": 2, "allin": 3}


def _fixed_key_bytes(hole: list[str], board: list[str], player_order_idx: int,
                     action_history: list[str],
                     prev_street_actions: list[list[str]] | None = None) -> bytes:
    """Build a compact bytes key matching postflop_fixed_train.py's info_key().
    
    Encoding (matches trainer exactly):
      player, hand_cat_packed(r1r2, suit), board_norm_raw per card, 
      street, facing_bet, agg_count, action_hist bytes.
    """
    buf = bytearray()
    buf.append(player_order_idx)

    # Hand category
    r1 = RANKS.index(hole[0][0].upper())
    s1 = SUITS.index(hole[0][1].lower())
    r2 = RANKS.index(hole[1][0].upper())
    s2 = SUITS.index(hole[1][1].lower())
    if r1 < r2:
        r1, r2, s1, s2 = r2, r1, s2, s1
    # Pack hand_cat the same way as _pack_hand_cat in the trainer
    r1r2 = (r1 << 4) | r2
    if r1 == r2:
        suit = 2
    elif s1 == s2:
        suit = 1
    else:
        suit = 0
    buf.append(r1r2)
    buf.append(suit)

    # Board normalization (same as _board_norm_raw in trainer)
    board_ints = [RANKS.index(c[0].upper()) * 4 + SUITS.index(c[1].lower()) for c in board]
    flop = sorted(board_ints[:3], key=lambda c: (-c // 4, c % 4))
    rest = board_ints[3:]
    cards = flop + rest
    suit_map: dict = {}
    next_suit = 0
    for c in cards:
        r, s = c // 4, c % 4
        if s not in suit_map:
            suit_map[s] = next_suit
            next_suit += 1
        buf.append((r << 4) | suit_map[s])

    # Street
    street = max(0, len(board) - 3)
    buf.append(street)

    # Facing bet and agg count from current street actions
    all_actions: list[str] = []
    for street_acts in (prev_street_actions or []):
        all_actions.extend(street_acts)
    all_actions.extend(action_history)

    # Reconstruct facing_bet and agg_count from current street only.
    # Key insight: in multi-player, a call does NOT clear the facing bet.
    # The bet stays live until ALL active players have matched it and the
    # street ends. So facing_bet = True whenever there's been any aggressive
    # action on the current street. It's only False on the opening action
    # (no bet to match) or after all checks.
    facing_bet = False
    agg_count = 0
    for a in action_history:
        al = a.lower()
        if al in ("bet", "raise", "allin"):
            facing_bet = True
            agg_count += 1
        # Note: "call" does NOT clear facing_bet. In multi-player,
        # P0 bets → P1 calls → P2 still faces the bet (facing_bet=True).
        # Only the street-end clears it, which the runtime handles by
        # resetting action_history to [] for the new street.

    buf.append(int(facing_bet))
    buf.append(agg_count)

    # Action history as byte values (matches trainer)
    hist_ints = [_FT_ACTION_MAP.get(a.lower(), 1) for a in all_actions]
    for a in hist_ints:
        buf.append(a)

    return bytes(buf)


def _fixed_key(hole: list[str], board: list[str], player_order_idx: int,
               action_history: list[str],
               prev_street_actions: list[list[str]] | None = None) -> tuple:
    """Build the info_key tuple matching postflop_fixed_train.py's info_key()."""
    # Hand category
    r1 = RANKS.index(hole[0][0].upper())
    s1 = SUITS.index(hole[0][1].lower())
    r2 = RANKS.index(hole[1][0].upper())
    s2 = SUITS.index(hole[1][1].lower())
    if r1 < r2:
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        hand_cat = RANKS[r1] * 2
    else:
        hand_cat = RANKS[r1] + RANKS[r2] + ('s' if s1 == s2 else 'o')

    # Board normalization: flop sorted by rank desc, suit asc (canonical, matches _board_norm)
    board_ints = [RANKS.index(c[0].upper()) * 4 + SUITS.index(c[1].lower()) for c in board]
    flop  = sorted(board_ints[:3], key=lambda c: (-c // 4, c % 4))
    rest  = board_ints[3:]
    cards = flop + rest
    suit_map: dict = {}
    norm = []
    for c in cards:
        r, s = c // 4, c % 4
        if s not in suit_map:
            suit_map[s] = len(suit_map)
        norm.append((r, suit_map[s]))
    board_norm = tuple(norm)

    street = max(0, len(board) - 3)   # 0=flop, 1=turn, 2=river

    # Full action history across all streets (fixed-tree hist accumulates from root)
    all_actions: list[str] = []
    for street_acts in (prev_street_actions or []):
        all_actions.extend(street_acts)
    all_actions.extend(action_history)
    hist_ints = tuple(_FT_ACTION_MAP.get(a.lower(), 1) for a in all_actions)

    # Reconstruct facing_bet and agg_count from current street only.
    # Key insight: in multi-player, a call does NOT clear the facing bet.
    # The bet stays live until ALL active players have matched it and the
    # street ends. So facing_bet = True whenever there's been any aggressive
    # action on the current street.
    facing_bet = False
    agg_count = 0
    for a in action_history:
        al = a.lower()
        if al in ("bet", "raise", "allin"):
            facing_bet = True
            agg_count += 1
        # Note: "call" does NOT clear facing_bet. See _fixed_key_bytes for details.

    return (player_order_idx, hand_cat, board_norm, street, facing_bet, agg_count, hist_ints)


def _query_fixed_tree(n: int, hole: list[str], board: list[str],
                      player_order_idx: int, action_history: list[str],
                      prev_street_actions: list[list[str]] | None = None) -> dict | None:
    """Query the fixed-tree compact policy. Returns {action_int: prob} or None."""
    loaded = _load_fixed_tree(n)
    if not loaded:
        return None

    # NPZ format: hash the bytes key and do binary search
    if isinstance(loaded, tuple) and loaded[0] == "npz":
        compact = loaded[1]
        key_bytes = _fixed_key_bytes(hole, board, player_order_idx,
                                     action_history, prev_street_actions)
        h = np.uint64(int.from_bytes(
            hashlib.md5(key_bytes).digest()[:8], "little", signed=False
        ))
        keys_arr = compact["keys"]
        idx = np.searchsorted(keys_arr, h)
        if idx >= len(keys_arr) or keys_arr[idx] != h:
            return None  # state not seen during training
        acts = compact["actions"][idx]
        probs = compact["probs"][idx]
        valid = acts >= 0
        return {int(a): float(p) for a, p in zip(acts[valid], probs[valid]) if p > 0}

    # PKL format: look up by tuple key (legacy)
    if isinstance(loaded, tuple) and loaded[0] == "pkl":
        policy = loaded[1]
        key = _fixed_key(hole, board, player_order_idx, action_history, prev_street_actions)
        entry = policy.get(key)
        if entry is None:
            return None
        # Handle both dict and flat-tuple formats
        if isinstance(entry, dict):
            total = sum(entry.values())
            if total <= 0:
                return None
            return {a: p / total for a, p in entry.items()}
        elif isinstance(entry, tuple):
            # Flat tuple format: (action, prob, action, prob, ...)
            result = {}
            for i in range(0, len(entry), 2):
                result[entry[i]] = entry[i + 1]
            total = sum(result.values())
            if total <= 0:
                return None
            return {a: p / total for a, p in result.items()}

    return None


# ─── Loader ───────────────────────────────────────────────────────────────────

def _load_worker(n_players: int):
    """Load postflop solver for n_players. Only loads compact/extracted formats
    suitable for the server — never the full training checkpoint."""
    compact_path = TABLES_DIR / f"{n_players}p_postflop_policy.npz"
    fixed_npz_path = TABLES_DIR / f"{n_players}p_postflop_fixed_policy.npz"
    fixed_pkl_path = TABLES_DIR / f"{n_players}p_postflop_fixed_policy.pkl"

    try:
        # Compact NPZ format (from old extract_and_save / OpenSpiel solver)
        if compact_path.exists():
            size_mb = compact_path.stat().st_size // (1024 ** 2)
            logger.info(f"Loading {n_players}p compact postflop policy ({size_mb} MB)…")
            data = np.load(compact_path, allow_pickle=False)
            compact = {
                "keys":    data["keys"],
                "probs":   data["probs"],
                "actions": data["actions"],
            }
            _solvers[n_players] = {"game": None, "policy": None, "compact": compact}
            logger.info(f"{n_players}p compact postflop policy ready")
            return

        # Fixed-tree NPZ format (streaming extraction — preferred over PKL)
        if fixed_npz_path.exists():
            _load_fixed_tree(n_players)  # loads NPZ into _fixed_tree cache
            _solvers[n_players] = {"game": None, "policy": None, "compact": None}
            logger.info(f"{n_players}p fixed-tree postflop policy ready (NPZ)")
            return

        # Fixed-tree PKL format (legacy fallback)
        if fixed_pkl_path.exists():
            _load_fixed_tree(n_players)
            _solvers[n_players] = {"game": None, "policy": None, "compact": None}
            logger.info(f"{n_players}p fixed-tree postflop policy ready (PKL)")
            return

        logger.info(f"{n_players}p postflop solver not found — using MC equity fallback")

    except Exception as e:
        logger.warning(f"Failed to load {n_players}p postflop solver: {e}")


def preload_postflop_solvers():
    """Load 2p then 3p sequentially in a single background thread.
    Sequential loading avoids simultaneous IO pressure on a memory-heavy machine."""
    def _worker():
        for n in _REQUIRED:
            if n not in _solvers:
                _load_worker(n)
    threading.Thread(target=_worker, daemon=True, name="postflop-preload").start()


def are_solvers_ready() -> bool:
    """True once every required solver has either loaded or confirmed missing."""
    return all(n in _solvers for n in _REQUIRED)


def _available(n: int) -> bool:
    """Return True if solver n is loaded. Does NOT trigger on-demand loading —
    use preload_postflop_solvers() at startup instead."""
    return n in _solvers


# ─── State reconstruction ────────────────────────────────────────────────────

def _card_to_int(card_str: str) -> int:
    rank = card_str[0].upper()
    suit = card_str[1].lower()
    return RANKS.index(rank) * 4 + SUITS.index(suit)


def _dummy_cards(exclude: set, count: int) -> list[int]:
    result = []
    for i in range(52):
        if i not in exclude and len(result) < count:
            result.append(i)
    return result


def _apply_card(state, card_int: int, used: set):
    legal = [a for a, _ in state.chance_outcomes()]
    if card_int in legal:
        state.apply_action(card_int)
        return
    for alt in legal:
        if alt not in used:
            state.apply_action(alt)
            used.add(alt)
            return
    state.apply_action(legal[0])


def _replay_street(state, actions: list[str]) -> bool:
    for a_str in actions:
        if state.is_terminal() or state.is_chance_node():
            return False
        action_int = _map_action(state, a_str)
        if action_int is None:
            return False
        state.apply_action(action_int)
    return True


def _reconstruct_state(
    game,
    hole: list[str],
    board: list[str],
    player_idx: int,
    n_players: int,
    action_history: list[str],
    prev_street_actions: list[list[str]] = None,
):
    prev = prev_street_actions or []
    my_ints = [_card_to_int(c) for c in hole]
    board_ints = [_card_to_int(c) for c in board]
    used = set(my_ints) | set(board_ints)

    n_dummy = (n_players - 1) * 2
    dummy = iter(_dummy_cards(used, n_dummy))

    state = game.new_initial_state()

    hole_seq: list[int] = []
    for p in range(n_players):
        if p == player_idx:
            hole_seq += my_ints
        else:
            hole_seq += [next(dummy), next(dummy)]

    for card_int in hole_seq:
        if not state.is_chance_node():
            return None
        _apply_card(state, card_int, used)

    for card_int in board_ints[:3]:
        if not state.is_chance_node():
            return None
        _apply_card(state, card_int, used)

    if len(board) >= 4:
        if not _replay_street(state, prev[0] if len(prev) > 0 else []):
            return None
        if state.is_terminal() or not state.is_chance_node():
            return None
        _apply_card(state, board_ints[3], used)

    if len(board) == 5:
        if not _replay_street(state, prev[1] if len(prev) > 1 else []):
            return None
        if state.is_terminal() or not state.is_chance_node():
            return None
        _apply_card(state, board_ints[4], used)

    if not _replay_street(state, action_history):
        return None
    if state.is_terminal() or state.is_chance_node():
        return None
    if state.current_player() != player_idx:
        return None

    return state


def _map_action(state, action_str: str):
    a = action_str.lower()
    legal = state.legal_actions()
    labels = [state.action_to_string(state.current_player(), x).lower() for x in legal]

    if "fold"  in a:                  target = "fold"
    elif "allin" in a:                target = "allin"
    elif "raise" in a or "bet" in a:  target = "bet"
    elif "check" in a:                target = "check"
    else:                             target = "call"

    for ai, label in zip(legal, labels):
        if target in label:
            return ai
    for ai, label in zip(legal, labels):
        if "check" in label or "call" in label:
            return ai
    return legal[0] if legal else None


def _norm(label: str) -> str:
    l = label.lower()
    if "fold"  in l: return "fold"
    if "allin" in l: return "allin"
    if "bet"   in l: return "bet"
    if "raise" in l: return "raise"
    if "check" in l: return "check"
    if "call"  in l: return "call"
    return l


# ─── Player index mapping ─────────────────────────────────────────────────────

def _player_idx(order_idx: int, n_players: int) -> int:
    """Map postflop action order (0=OOP/first, 1=middle, 2=IP/last) to OpenSpiel player index.

    2p game: player 0=SB=IP, player 1=BB=OOP  →  OOP(0)→1, IP(1)→0
    3p game: player 0=SB(first), 1=BB, 2=BTN(last)  →  order index == player index
    """
    if n_players == 2:
        return 1 if order_idx == 0 else 0
    elif n_players == 3:
        return min(order_idx, 2)
    return 0


# ─── Public API ──────────────────────────────────────────────────────────────

def sample_postflop_action(
    hole: list[str],
    board: list[str],
    player_order_idx: int,
    action_history: list[str],
    n_active: int = 2,
    prev_street_actions: list[list[str]] = None,
) -> tuple[str | None, dict | None]:
    """Sample a postflop action from the solver.

    Returns (action_label, raw_probs) where raw_probs is
    {action_int: prob} from the solver, or None if no solver data.
    """
    n_solver = min(n_active, 3)
    if len(board) < 3:
        return None, None

    # Trim action history for >3 player games mapped onto 3p solver
    if n_active > n_solver and len(action_history) > n_solver - 1:
        action_history = action_history[-(n_solver - 1):]

    # ── Path 1: Fixed-tree compact policy (no OpenSpiel dependency) ──
    ft = _query_fixed_tree(n_solver, hole, board, player_order_idx,
                           action_history, prev_street_actions)
    if ft is not None:
        # ft is {action_int: prob} with action ints: 0=fold, 1=check/call, 2=bet/raise, 3=allin
        keys = list(ft.keys())
        weights = list(ft.values())
        if sum(weights) > 0:
            chosen_int = random.choices(keys, weights=weights, k=1)[0]
            label_map = {0: "fold", 1: "call", 2: "bet", 3: "allin"}
            return label_map.get(chosen_int, "call"), ft

    # ── Path 2: OpenSpiel-based solver (compact NPZ or full pkl) ──
    if not _available(n_solver):
        return None, None

    solver_data = _solvers[n_solver]
    g       = solver_data["game"]
    policy  = solver_data["policy"]
    compact = solver_data["compact"]
    pidx    = _player_idx(player_order_idx, n_solver)

    # Need OpenSpiel game object for this path
    if g is None:
        return None, None

    try:
        state = _reconstruct_state(g, hole, board, pidx, n_solver,
                                   action_history, prev_street_actions)
        if state is None:
            return None, None

        if compact is not None:
            raw = _query_compact(compact, state, pidx)
        else:
            raw = policy.action_probabilities(state)

        if raw is None or not raw:
            return None, None

        keys = list(raw.keys())
        weights = list(raw.values())
        if sum(weights) == 0:
            return None, None

        chosen_int = random.choices(keys, weights=weights, k=1)[0]
        label = state.action_to_string(pidx, chosen_int)
        result = _norm(label)
        return ("check" if result == "call" and not any(
            "call" in state.action_to_string(pidx, a).lower() and
            state.action_to_string(pidx, a).lower() != "check/call"
            for a in state.legal_actions()
        ) else result), raw

    except Exception as e:
        logger.debug(f"Postflop solver query failed: {e}")
        return None, None


def solve_postflop_gto(
    hole: list[str],
    board: list[str],
    pot: int,
    stack: int,
    player_order_idx: int = 0,
    action_history: list[str] = None,
    to_call: int = 0,
    n_opponents: int = 1,
    prev_street_actions: list[list[str]] = None,
    n_active: int = 2,
    street: str = "flop",
) -> dict:
    if action_history is None:
        action_history = []
    if prev_street_actions is None:
        prev_street_actions = []

    # Always derive n_opponents from n_active so multiway equity is correct.
    # The caller's n_opponents param is ignored in favour of the live count.
    n_opponents = max(1, n_active - 1)
    is_river = (street == "river")

    def _equity():
        if is_river:
            return river_exact_equity(hole, board, n_opponents=n_opponents)
        return monte_carlo_equity(hole, board, n_opponents=n_opponents, n_sims=200)

    sampled, raw_probs = sample_postflop_action(
        hole, board, player_order_idx, action_history,
        n_active=n_active, prev_street_actions=prev_street_actions,
    )

    if sampled is not None:
        if sampled == "call" and to_call == 0:
            sampled = "check"
        equity = _equity()
        amount = _size_bet(sampled, board, street, equity, pot, stack, to_call)
        strat = "MCCFR postflop" if not is_river else "MCCFR+exact river"
        return {"action": sampled, "amount": amount, "equity": equity, "strategy": strat}

    equity = _equity()
    result = _equity_fallback(equity, board, street, pot, stack, to_call)
    result["strategy"] = "exact river" if is_river else "MC equity"
    return result


# ─── Sizing ───────────────────────────────────────────────────────────────────

def _raise_to(to_call: int, pot: int) -> int:
    """Fixed raise formula: call + 75% × (pot after calling)."""
    return to_call + int((pot + to_call) * 0.75)


def _size_bet(action: str, board: list[str], street: str,
              equity: float, pot: int, stack: int, to_call: int) -> int:
    if action in ("check", "fold"): return 0
    if action == "call":            return min(to_call, stack)
    if action == "allin":           return stack

    if to_call > 0:
        # Facing a bet: use fixed raise formula (call + 75% × pot_after_call)
        return min(_raise_to(to_call, pot), stack)

    # Opening bet: 33% or 75% pot depending on equity / street
    # High equity or late street → larger size (75%)
    if equity >= 0.65 or street == "river":
        return min(int(pot * 0.75), stack)
    return min(int(pot * 0.33), stack)


def _equity_fallback(equity: float, board: list[str], street: str,
                     pot: int, stack: int, to_call: int) -> dict:
    """Equity-based fallback using fixed 33%/75% pot sizings per spec.
    No overbets, no SPR-sensitive jam logic."""
    if to_call > 0:
        if equity >= 0.65:
            # Use fixed raise formula: raise-to = call + 75% × (pot after calling)
            pot_after_call = pot + 2 * to_call  # pot after hero calls
            raise_to = to_call + int(pot_after_call * 0.75)
            raise_amt = min(raise_to, stack)
            return {"action": "raise", "amount": raise_amt, "equity": equity}
        if equity >= 0.38:
            return {"action": "call", "amount": min(to_call, stack), "equity": equity}
        return {"action": "fold", "amount": 0, "equity": equity}

    if equity >= 0.55:
        # Use fixed 75% pot sizing (large bet)
        bet_amt = min(int(pot * 0.75), stack)
        if equity >= 0.70:
            bet_amt = min(int(pot * 0.33), stack)  # small bet for strong but not premium
        if bet_amt > 0:
            return {"action": "bet", "amount": bet_amt, "equity": equity}
    return {"action": "check", "amount": 0, "equity": equity}
