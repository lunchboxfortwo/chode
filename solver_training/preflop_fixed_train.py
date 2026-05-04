"""
Preflop MCCFR solver with fixed human-realistic game tree.

Enforces the full preflop spec:
  - Position-aware RFI sizing: BTN/SB open 3bb, all others open 2.5bb
  - No open-limping from ANY position (everyone folds or raises first in)
  - 3-bet sizing: 9bb in position, 12bb out of position
  - Squeeze sizing: IP (3 + n_callers) × open_size, OOP (4 + n_callers) × open_size
  - 4-bet sizing: 2.3× facing 3-bet in position, 2.8× out of position
  - All 5-bets are all-in; no non-all-in 5-bets; no 6-bet branches
  - Rake: 3% of pot, capped at 2×BB, applied at all terminals
  - Position labels by table size (2–8)
  - 169-hand abstraction for information-state keys

Action categories (N_ACTIONS=5):
  0  fold
  1  call / check
  2  raise-A  — RFI at level 0, 3-bet at level 1, 4-bet at level 2
  3  raise-B  — squeeze at level 1 (only when cold callers exist)
  4  all-in

Usage:
    python3 solver_training/preflop_fixed_train.py --players 2 --stack-bb 100 --iters 50000000
    python3 solver_training/preflop_fixed_train.py --players 6 --stack-bb 100 --iters 100000000 --resume
"""

import os
import sys
import random
import struct
import pickle
import argparse
import time
import json
import tempfile
import hashlib
import logging
import signal

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "preflop_tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BB        = 100   # big blind in chips
SB        = 50    # small blind in chips
RAKE_RATE = 0.03
RAKE_CAP  = 2     # × BB chips
N_ACTIONS = 5     # fold, call, raise-A, raise-B, allin

RANKS = "23456789TJQKA"
SUITS = "cdhs"

# ─── Fast hand evaluator (phevaluator, C extension) ──────────────────────────
# phevaluator uses the SAME card encoding as our code (suit*13 + rank),
# and is ~5x faster than treys (pure Python).
# Fall back to treys if phevaluator is unavailable.
try:
    from phevaluator import evaluate_cards as _pheval
    _HAS_PHEVAL = True
except ImportError:
    _HAS_PHEVAL = False

# ─── Precomputed caches ──────────────────────────────────────────────────────
_CARD_BIT = [1 << c for c in range(52)]

# Hand category cache: (card0, card1) -> (r1r2_byte, suit_byte)
_HAND_CAT_CACHE: dict[tuple[int, int], tuple[int, int]] | None = None

def _build_hand_cat_cache():
    global _HAND_CAT_CACHE
    if _HAND_CAT_CACHE is not None:
        return
    cache = {}
    for c1 in range(52):
        r1, s1 = c1 // 4, c1 % 4
        for c2 in range(52):
            if c1 == c2:
                continue
            r2, s2 = c2 // 4, c2 % 4
            # Use local copies for canonical ordering (don't mutate r1/r2)
            cr1, cs1, cr2, cs2 = r1, s1, r2, s2
            if cr1 < cr2:
                cr1, cr2, cs1, cs2 = cr2, cr1, cs2, cs1
            if cr1 == cr2:
                suit_byte = 2
            elif cs1 == cs2:
                suit_byte = 1
            else:
                suit_byte = 0
            r1r2_byte = (cr1 << 4) | cr2
            cache[(c1, c2)] = (r1r2_byte, suit_byte)
    _HAND_CAT_CACHE = cache

# Position index cache: n_players -> pos_name -> index
_POS_IDX_CACHE: dict[int, dict[str, int]] = {}

def _get_pos_idx(n_players: int, pos_name: str) -> int:
    if n_players not in _POS_IDX_CACHE:
        names = position_names(n_players)
        _POS_IDX_CACHE[n_players] = {n: i for i, n in enumerate(names)}
    return _POS_IDX_CACHE[n_players][pos_name]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Treys evaluator (lazy init) ─────────────────────────────────────────────
_EVAL = None
_TREYS_CARD = None

def _init_treys():
    global _EVAL, _TREYS_CARD
    if _EVAL is not None:
        return
    try:
        from treys import Card as TCard, Evaluator as TEval
        _EVAL = TEval()
        _TREYS_CARD = [TCard.new(_card_str(c)) for c in range(52)]
    except ImportError:
        pass


# ─── System helpers ───────────────────────────────────────────────────────────

def _set_oom_score(score: int = 200):
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(str(score))
    except Exception:
        pass


def _free_ram_gb() -> float:
    """Return available RAM in GB, accounting for swap exhaustion risk."""
    mem_available = swap_free = swap_total = 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1]) / (1024 ** 2)
                elif line.startswith("SwapFree:"):
                    swap_free = int(line.split()[1]) / (1024 ** 2)
                elif line.startswith("SwapTotal:"):
                    swap_total = int(line.split()[1]) / (1024 ** 2)
    except Exception:
        return 0.0
    # Add swap_free to get total reclaimable memory. When swap is exhausted,
    # Count RAM at full value, swap at 50% (swap is slower and less reliable
    # under pressure — OOM killer can still fire when RAM+swap are exhausted).
    # Pure MemAvailable is the real RAM headroom; swap extends it partially.
    return mem_available + 0.5 * swap_free


_sigterm_solver = [None]
_sigterm_path   = [None]
_sigterm_prog   = [None]


def _sigterm_handler(signum, frame):
    logger.info("[SIGNAL] SIGTERM received — saving checkpoint before exit...")
    if _sigterm_solver[0] is not None:
        try:
            _save_ckpt(_sigterm_solver[0], _sigterm_path[0], _sigterm_prog[0])
            logger.info(f"  Saved: {_sigterm_path[0]}  ({_sigterm_prog[0].get('iterations_done', 0):,} iters)")
        except Exception as e:
            logger.error(f"  Checkpoint failed: {e}")
    sys.exit(0)


def _register_sigterm_handler():
    """Register SIGTERM handler. Only call from __main__, not on import."""
    signal.signal(signal.SIGTERM, _sigterm_handler)


# ─── Card utilities ───────────────────────────────────────────────────────────

def _card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def _hand_cat(h: list[int]) -> str:
    """Canonical 169-hand category: 'AA', 'AKs', 'AKo', etc."""
    c1, c2 = h[0], h[1]
    r1, s1 = c1 // 4, c1 % 4
    r2, s2 = c2 // 4, c2 % 4
    if r1 < r2:
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return RANKS[r1] * 2
    return RANKS[r1] + RANKS[r2] + ('s' if s1 == s2 else 'o')


def _pack_hand_cat(hand_cat: str) -> tuple:
    """Pack hand category string to (r1_r2_byte, suit_byte)."""
    r1 = RANKS.index(hand_cat[0])
    r2 = RANKS.index(hand_cat[1]) if len(hand_cat) > 1 else r1
    if len(hand_cat) == 2:  # pair
        suit = 2
    elif hand_cat[2] == 's':
        suit = 1
    else:
        suit = 0
    return ((r1 << 4) | r2, suit)


def _deck_minus(exclude: set[int]) -> list[int]:
    deck = [c for c in range(52) if c not in exclude]
    random.shuffle(deck)
    return deck


# ─── Position system ──────────────────────────────────────────────────────────

# Position names by table size, indexed by player position (0-based).
# Player 0 is always the first to act preflop (UTG for 7p+, LJ for 6p, etc.).
# The last two positions are always SB and BB.
_POSITION_MAP = {
    2: ["SB", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    6: ["LJ", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG1", "UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
}


def position_names(n_players: int) -> list[str]:
    return _POSITION_MAP[n_players]


_IP_VS_CACHE: dict[tuple[str, str, int], bool] = {}

def _postflop_order(n_players: int) -> list[str]:
    """
    Postflop acting order (first to last).
    SB acts first, BB second, then early→late positions, BTN last.
    In 2p: SB/BTN acts first, BB acts last.
    """
    names = position_names(n_players)
    if n_players == 2:
        return ["SB", "BB"]
    # SB, BB, then everyone else in the same order they sit (preflop first-to-act onward)
    sb_idx = names.index("SB")
    bb_idx = names.index("BB")
    others = [n for i, n in enumerate(names) if i not in (sb_idx, bb_idx)]
    return ["SB", "BB"] + others


def _is_ip_vs(my_pos: str, vs_pos: str, n_players: int) -> bool:
    """
    Is my_pos 'in position' (acts after) vs_pos postflop?
    Cached for repeated lookups.
    """
    key = (my_pos, vs_pos, n_players)
    result = _IP_VS_CACHE.get(key)
    if result is not None:
        return result
    order = _postflop_order(n_players)
    result = order.index(my_pos) > order.index(vs_pos)
    _IP_VS_CACHE[key] = result
    return result


# ─── Bet sizing ───────────────────────────────────────────────────────────────

def _rfi_size(position: str, bb: int) -> int:
    """RFI (raise first in) sizing: BTN/SB=3bb, others=2.5bb."""
    if position in ("BTN", "SB"):
        return 3 * bb
    return int(2.5 * bb)


def _three_bet_size(is_ip: bool, bb: int) -> int:
    """3-bet sizing: 9bb IP, 12bb OOP."""
    return 9 * bb if is_ip else 12 * bb


def _squeeze_size(is_ip: bool, n_callers: int, open_size: int) -> int:
    """Squeeze sizing: IP (3+n_callers)×open, OOP (4+n_callers)×open."""
    mult = (3 + n_callers) if is_ip else (4 + n_callers)
    return mult * open_size


def _four_bet_size(is_ip: bool, facing_size: int) -> int:
    """4-bet sizing: 2.3× IP, 2.8× OOP (of the facing 3-bet/squeeze size)."""
    return int(2.3 * facing_size) if is_ip else int(2.8 * facing_size)


# ─── Game state ───────────────────────────────────────────────────────────────

class State:
    """
    Preflop game state for the fixed-tree solver.

    Key invariants:
      - facing_size: the total investment by the last aggressor. All other
        players must match this amount (invested[p] >= facing_size) or fold.
      - raise_level: 0=unopened, 1=open(RFI), 2=3bet/squeeze, 3=4bet, 4=5bet+
      - responded: set of players who have responded to the current bet level.
        Reset when someone raises. Betting is complete when all active players
        have responded and matched facing_size (or are all-in).
    """
    __slots__ = [
        'n_players', 'stack_bb', 'bb',
        'hole', 'stacks', 'invested', 'pot',
        'acting', 'folded', 'responded',
        'raise_level', 'facing_size',
        'open_size', 'open_pos', 'n_cold_callers', 'last_raiser_pos',
    ]

    def __init__(self, n_players: int, stack_bb: int = 100):
        self.n_players = n_players
        self.stack_bb  = stack_bb
        self.bb        = BB
        buy_in = stack_bb * BB

        # Post blinds
        names = position_names(n_players)
        sb_idx = names.index("SB")
        bb_idx = names.index("BB")

        self.stacks   = [buy_in] * n_players
        self.invested = [0] * n_players
        self.stacks[sb_idx] -= SB;  self.invested[sb_idx] = SB
        self.stacks[bb_idx] -= BB;  self.invested[bb_idx] = BB
        self.pot = SB + BB

        self.folded   = [False] * n_players
        self.responded = set()  # who has responded to current facing_size

        # Betting context
        self.raise_level       = 0
        self.facing_size       = BB     # the BB is the initial "bet" to match
        self.open_size         = 0      # original open size (chips)
        self.open_pos          = ""     # original opener's position name
        self.n_cold_callers    = 0      # callers of the open before a squeezer
        self.last_raiser_pos   = ""     # position name of last raiser

        # First to act preflop
        self.acting = 0  # index 0 is always first preflop position
        self.hole = [[] for _ in range(n_players)]

    def copy(self):
        s = State.__new__(State)
        s.n_players        = self.n_players
        s.stack_bb         = self.stack_bb
        s.bb               = self.bb
        s.hole             = [h[:] for h in self.hole]
        s.stacks           = self.stacks[:]
        s.invested         = self.invested[:]
        s.pot              = self.pot
        s.acting           = self.acting
        s.folded           = self.folded[:]
        s.responded        = set(self.responded)
        s.raise_level      = self.raise_level
        s.facing_size      = self.facing_size
        s.open_size        = self.open_size
        s.open_pos         = self.open_pos
        s.n_cold_callers   = self.n_cold_callers
        s.last_raiser_pos  = self.last_raiser_pos
        return s

    def active(self) -> list[int]:
        return [p for p in range(self.n_players) if not self.folded[p]]

    def next_active(self, from_p: int) -> int:
        for off in range(1, self.n_players + 1):
            p = (from_p + off) % self.n_players
            if not self.folded[p]:
                return p
        return from_p

    def pos(self, p: int) -> str:
        return position_names(self.n_players)[p]

    def to_call(self, p: int) -> int:
        """Additional chips player p must put in to match facing_size."""
        return max(self.facing_size - self.invested[p], 0)

    def _is_bb(self, p: int) -> bool:
        return self.pos(p) == "BB"


# ─── Game logic ───────────────────────────────────────────────────────────────

def legal_actions(s: State) -> list[int]:
    """
    Compute legal actions for the current acting player.

    At raise_level=0 (unopened pot):
      - ALL positions: fold, RFI (raise-A), all-in
      - NO call/limp allowed from any position (including SB and BB)
      - Rationale: we only allow raise-or-fold when first to enter;
        if everyone folds, BB wins by default (terminal).

    At raise_level=1 (facing an open):
      - fold, call, raise-A (3-bet), possibly raise-B (squeeze), all-in
      - Squeeze (raise-B) only available when n_cold_callers > 0

    At raise_level=2 (facing 3-bet or squeeze):
      - fold, call, raise-A (4-bet), all-in

    At raise_level>=3 (facing 4-bet or higher):
      - fold, call, all-in  (5-bet = all-in only, no sizing choices)
    """
    p = s.acting
    stack = s.stacks[p]
    pos = s.pos(p)
    n = s.n_players
    tc = s.to_call(p)

    if s.raise_level == 0:
        # ── Unopened: fold, RFI ──
        # NO call/limp from any position.
        # All-in only available when stacks are shallow enough that
        # the fixed RFI sizing would effectively commit the player.
        # Threshold: stack < 3× RFI size (raise commits >33% of stack)
        rfi = _rfi_size(pos, s.bb)
        if rfi >= stack + s.invested[p]:
            # Can't afford RFI → only fold or shove
            return [0, 4]
        shallow = stack < 3 * rfi
        if shallow:
            return [0, 2, 4]
        return [0, 2]

    elif s.raise_level == 1:
        # ── Facing an open: fold, call, 3-bet, squeeze, all-in (shallow) ──
        acts = [0, 1]  # fold, call

        # 3-bet sizing (raise-A)
        is_ip = _is_ip_vs(pos, s.last_raiser_pos, n)
        three_bet_total = _three_bet_size(is_ip, s.bb)  # total chips to put in
        three_bet_add = three_bet_total - s.invested[p]  # additional chips needed
        if three_bet_add > tc and three_bet_add < stack:
            # Can 3-bet (and it's a raise, not just a call)
            # Also verify the total doesn't exceed stack
            if three_bet_total < stack + s.invested[p]:
                acts.append(2)

        # Squeeze sizing (raise-B) — only if cold callers exist and size differs from 3-bet
        if s.n_cold_callers > 0 and s.open_size > 0:
            sq_total = _squeeze_size(is_ip, s.n_cold_callers, s.open_size)
            sq_add = sq_total - s.invested[p]
            if sq_add > tc and sq_add < stack and sq_total != three_bet_total:
                if sq_total < stack + s.invested[p]:
                    acts.append(3)

        # All-in only when shallow-stacked (3-bet commits >33% of stack)
        # or when 3-bet size exceeds stack (can't raise normally)
        if stack > tc:
            shallow = three_bet_total >= stack + s.invested[p] or stack < 3 * three_bet_add
            if shallow:
                acts.append(4)

        return acts

    elif s.raise_level == 2:
        # ── Facing 3-bet/squeeze: fold, call, 4-bet, all-in (shallow) ──
        acts = [0, 1]  # fold, call

        # 4-bet sizing (raise-A)
        is_ip = _is_ip_vs(pos, s.last_raiser_pos, n)
        four_bet_total = _four_bet_size(is_ip, s.facing_size)
        four_bet_add = four_bet_total - s.invested[p]
        if four_bet_add > tc and four_bet_add < stack:
            if four_bet_total < stack + s.invested[p]:
                acts.append(2)

        # All-in only when shallow-stacked (4-bet commits >33% of stack)
        # or when 4-bet size exceeds stack
        if stack > tc:
            shallow = four_bet_total >= stack + s.invested[p] or stack < 3 * four_bet_add
            if shallow:
                acts.append(4)

        return acts

    else:
        # ── Facing 4-bet+: fold, call, all-in (5-bet = all-in only) ──
        acts = [0, 1]  # fold, call
        if stack > tc:
            acts.append(4)  # 5-bet all-in
        return acts


def apply_action(s: State, action: int) -> State:
    """Apply action and return a new state. Does not mutate s."""
    s = s.copy()
    p = s.acting
    pos = s.pos(p)
    n = s.n_players
    tc = s.to_call(p)

    if action == 0:
        # ── Fold ──
        s.folded[p] = True
        s.responded.add(p)
        s.acting = s.next_active(p)

    elif action == 1:
        # ── Call ──
        amt = min(tc, s.stacks[p])
        s.pot += amt
        s.stacks[p] -= amt
        s.invested[p] += amt
        s.responded.add(p)
        # Track cold callers at raise_level=1
        if s.raise_level == 1 and s.invested[p] == s.facing_size:
            s.n_cold_callers += 1
        s.acting = s.next_active(p)

    elif action == 2:
        # ── raise-A: RFI / 3-bet / 4-bet ──
        if s.raise_level == 0:
            # RFI
            total = _rfi_size(pos, s.bb)
            add = total - s.invested[p]
            add = min(add, s.stacks[p])
            s.pot += add
            s.stacks[p] -= add
            s.invested[p] += add
            s.facing_size = s.invested[p]
            s.raise_level = 1
            s.open_size = total
            s.open_pos = pos
            s.last_raiser_pos = pos
            s.n_cold_callers = 0
            s.responded = {p}

        elif s.raise_level == 1:
            # 3-bet
            is_ip = _is_ip_vs(pos, s.last_raiser_pos, n)
            total = _three_bet_size(is_ip, s.bb)
            add = total - s.invested[p]
            add = min(add, s.stacks[p])
            s.pot += add
            s.stacks[p] -= add
            s.invested[p] += add
            s.facing_size = s.invested[p]
            s.raise_level = 2
            s.last_raiser_pos = pos
            s.responded = {p}

        elif s.raise_level == 2:
            # 4-bet
            is_ip = _is_ip_vs(pos, s.last_raiser_pos, n)
            total = _four_bet_size(is_ip, s.facing_size)
            add = total - s.invested[p]
            add = min(add, s.stacks[p])
            s.pot += add
            s.stacks[p] -= add
            s.invested[p] += add
            s.facing_size = s.invested[p]
            s.raise_level = 3
            s.last_raiser_pos = pos
            s.responded = {p}

        s.acting = s.next_active(p)

    elif action == 3:
        # ── raise-B: squeeze (only at raise_level=1) ──
        if s.raise_level == 1 and s.n_cold_callers > 0:
            is_ip = _is_ip_vs(pos, s.last_raiser_pos, n)
            total = _squeeze_size(is_ip, s.n_cold_callers, s.open_size)
            add = total - s.invested[p]
            add = min(add, s.stacks[p])
            s.pot += add
            s.stacks[p] -= add
            s.invested[p] += add
            s.facing_size = s.invested[p]
            s.raise_level = 2
            s.last_raiser_pos = pos
            s.responded = {p}

        s.acting = s.next_active(p)

    elif action == 4:
        # ── All-in ──
        amt = s.stacks[p]
        s.pot += amt
        s.invested[p] += amt
        s.stacks[p] = 0
        s.responded.add(p)

        if amt > tc:
            # All-in is a raise
            s.facing_size = s.invested[p]
            s.raise_level += 1
            s.last_raiser_pos = pos
            s.responded = {p}

        s.acting = s.next_active(p)

    return s


def _betting_complete(s: State) -> bool:
    """
    Preflop betting is complete when all active non-all-in players
    have responded to the current bet level and matched facing_size.
    """
    for p in s.active():
        if p not in s.responded:
            return False
        if s.stacks[p] > 0 and s.invested[p] < s.facing_size:
            return False
    return True


def is_terminal(s: State) -> bool:
    # Fast path: count active players without building a list
    n = s.n_players
    n_active = 0
    for p in range(n):
        if not s.folded[p]:
            n_active += 1
            if n_active > 1:
                break
    if n_active <= 1:
        return True
    # Inline _betting_complete
    for p in range(n):
        if s.folded[p]:
            continue
        if p not in s.responded:
            return False
        if s.stacks[p] > 0 and s.invested[p] < s.facing_size:
            return False
    return True
    return False


def payoff(s: State) -> list[float]:
    """
    Chip gain/loss for each player.
    Rake: 3% of pot, capped at 2×BB, applied at all terminals.
    Uses phevaluator (C extension, ~5x faster) when available.
    """
    rake = min(s.pot * RAKE_RATE, RAKE_CAP * BB)
    n = s.n_players

    # Count active without allocating a list
    active = []
    for p in range(n):
        if not s.folded[p]:
            active.append(p)

    if len(active) <= 1:
        winner = active[0]
        gains = [-s.invested[p] for p in range(n)]
        gains[winner] += s.pot - rake
        return gains

    # Showdown — deal a random board and evaluate
    gains = [-s.invested[p] for p in range(n)]

    # Build deck using bitmask (faster than set difference)
    exclude_mask = 0
    for h in s.hole:
        exclude_mask |= _CARD_BIT[h[0]] | _CARD_BIT[h[1]]
    deck = [c for c in range(52) if not (exclude_mask & _CARD_BIT[c])]
    board = random.sample(deck, 5)

    if _HAS_PHEVAL:
        # phevaluator: same card encoding, 5x faster than treys
        best_rank = 999999
        winners = []
        for p in active:
            h = s.hole[p]
            rank = _pheval(board[0], board[1], board[2], board[3], board[4],
                           h[0], h[1])
            if rank < best_rank:
                best_rank = rank
                winners = [p]
            elif rank == best_rank:
                winners.append(p)
    else:
        # Fallback to treys
        _init_treys()
        if _EVAL is not None:
            board_treys = [_TREYS_CARD[c] for c in board]
            scores = {}
            for p in active:
                hole_treys = [_TREYS_CARD[c] for c in s.hole[p]]
                scores[p] = _EVAL.evaluate(board_treys, hole_treys)
            best = min(scores.values())
            winners = [p for p, sc in scores.items() if sc == best]
        else:
            winners = [random.choice(active)]

    share = (s.pot - rake) / len(winners)
    for w in winners:
        gains[w] += share
    return gains


# ─── Information state key ────────────────────────────────────────────────────

def info_key(p: int, s: State, action_hist: list) -> bytes:
    """
    Compact information state key for MCCFR regret table.
    Uses precomputed hand_cat cache and struct.pack for speed.

    Encoding (all fixed-width):
      Offset  Size  Field
      0       1     player index (0-7)
      1       1     hand_cat: (rank1 << 4) | rank2
      2       1     hand_cat suit: 0=offsuit, 1=suited, 2=pair
      3       1     n_players (2-8)
      4       1     stack_bb // 10 (3=30bb, 5=50bb, 7=75bb→7, 10=100bb, 15=150bb, 20=200bb)
      5       1     raise_level (0-4)
      6       1     position index (0-7)
      7       1     n_cold_callers (0-7)
      8       na    action_hist: 1 byte per action (0-4)
    """
    _build_hand_cat_cache()
    h = s.hole[p]
    r1r2, suit_byte = _HAND_CAT_CACHE[(h[0], h[1])]
    pos_idx = _get_pos_idx(s.n_players, s.pos(p))

    header = struct.pack('BBBBBBBB',
        p, r1r2, suit_byte,
        s.n_players,
        min(s.stack_bb // 10, 255),
        min(s.raise_level, 255),
        pos_idx,
        min(s.n_cold_callers, 255),
    )

    if action_hist:
        return header + bytes(action_hist)
    return header


# ─── External Sampling MCCFR Solver ──────────────────────────────────────────

class Solver:
    """
    Flat-array external-sampling MCCFR solver with compact bytes keys.
    Same architecture as the postflop fixed-tree solver.
    """

    _INIT_CAP = 65_536

    def __init__(self, n_players: int = 2, stack_bb: int = 100, *, capacity: int = 0):
        self.n_players = n_players
        self.stack_bb  = stack_bb
        self.iterations = 0
        cap = capacity or self._INIT_CAP
        self._capacity  = cap
        self._n         = 0
        self._key_index = {}
        self._regrets   = np.zeros((cap, N_ACTIONS), dtype=np.float32)
        self._strat_sum = np.zeros((cap, N_ACTIONS), dtype=np.float32)

    def _grow(self, min_cap: int):
        new_cap = self._capacity
        while new_cap < min_cap:
            new_cap *= 2
        if new_cap <= self._capacity:
            return
        old_r, old_s = self._regrets, self._strat_sum
        self._regrets   = np.zeros((new_cap, N_ACTIONS), dtype=np.float32)
        self._strat_sum = np.zeros((new_cap, N_ACTIONS), dtype=np.float32)
        self._regrets[:self._n]   = old_r[:self._n]
        self._strat_sum[:self._n] = old_s[:self._n]
        self._capacity = new_cap

    def _ensure(self, key: bytes) -> int:
        idx = self._key_index.get(key)
        if idx is not None:
            return idx
        if self._n >= self._capacity:
            self._grow(self._n + 1)
        idx = self._n
        self._n += 1
        self._key_index[key] = idx
        return idx

    @property
    def n_info_states(self) -> int:
        return self._n

    def _strategy(self, key: bytes, legal: list[int]) -> np.ndarray:
        """Regret-matching+ strategy. Hybrid Python/numpy for small arrays."""
        idx = self._key_index.get(key)
        strat = np.zeros(N_ACTIONS, dtype=np.float32)
        if idx is not None:
            r = self._regrets[idx]
            total = 0.0
            for a in legal:
                v = r[a]
                if v > 0:
                    total += v
            if total > 0:
                for a in legal:
                    v = r[a]
                    strat[a] = (v if v > 0 else 0.0) / total
                return strat
        inv = 1.0 / len(legal)
        for a in legal:
            strat[a] = inv
        return strat

    def _deal(self) -> State:
        deck = list(range(52))
        random.shuffle(deck)
        n = self.n_players
        state = State(n, self.stack_bb)
        for p in range(n):
            state.hole[p] = [deck[p * 2], deck[p * 2 + 1]]
        return state

    def run_iteration(self):
        state = self._deal()
        for player in range(self.n_players):
            self._traverse(state, player, [])

    def _apply_action_mut(self, s: State, action: int):
        """
        Apply action to state IN-PLACE. Returns an undo tuple for _undo_action.
        Avoids the expensive State.copy() per action.
        """
        p = s.acting
        pos = s.pos(p)
        n = s.n_players
        tc = s.to_call(p)

        # Save undo data
        undo = (
            action, p,
            s.stacks[p], s.invested[p], s.pot,
            s.folded[p], set(s.responded),
            s.raise_level, s.facing_size,
            s.open_size, s.open_pos, s.n_cold_callers, s.last_raiser_pos,
            s.acting,
        )

        if action == 0:
            s.folded[p] = True
            s.responded.add(p)
            s.acting = s.next_active(p)

        elif action == 1:
            amt = min(tc, s.stacks[p])
            s.pot += amt
            s.stacks[p] -= amt
            s.invested[p] += amt
            s.responded.add(p)
            if s.raise_level == 1 and s.invested[p] == s.facing_size:
                s.n_cold_callers += 1
            s.acting = s.next_active(p)

        elif action == 2:
            if s.raise_level == 0:
                total = _rfi_size(pos, s.bb)
                add = total - s.invested[p]
                add = min(add, s.stacks[p])
                s.pot += add; s.stacks[p] -= add; s.invested[p] += add
                s.facing_size = s.invested[p]; s.raise_level = 1
                s.open_size = total; s.open_pos = pos; s.last_raiser_pos = pos
                s.n_cold_callers = 0; s.responded = {p}
            elif s.raise_level == 1:
                is_ip = _is_ip_vs(pos, s.last_raiser_pos, n)
                total = _three_bet_size(is_ip, s.bb)
                add = total - s.invested[p]
                add = min(add, s.stacks[p])
                s.pot += add; s.stacks[p] -= add; s.invested[p] += add
                s.facing_size = s.invested[p]; s.raise_level = 2
                s.last_raiser_pos = pos; s.responded = {p}
            elif s.raise_level == 2:
                is_ip = _is_ip_vs(pos, s.last_raiser_pos, n)
                total = _four_bet_size(is_ip, s.facing_size)
                add = total - s.invested[p]
                add = min(add, s.stacks[p])
                s.pot += add; s.stacks[p] -= add; s.invested[p] += add
                s.facing_size = s.invested[p]; s.raise_level = 3
                s.last_raiser_pos = pos; s.responded = {p}
            s.acting = s.next_active(p)

        elif action == 3:
            if s.raise_level == 1 and s.n_cold_callers > 0:
                is_ip = _is_ip_vs(pos, s.last_raiser_pos, n)
                total = _squeeze_size(is_ip, s.n_cold_callers, s.open_size)
                add = total - s.invested[p]
                add = min(add, s.stacks[p])
                s.pot += add; s.stacks[p] -= add; s.invested[p] += add
                s.facing_size = s.invested[p]; s.raise_level = 2
                s.last_raiser_pos = pos; s.responded = {p}
            s.acting = s.next_active(p)

        elif action == 4:
            amt = s.stacks[p]
            s.pot += amt; s.invested[p] += amt; s.stacks[p] = 0
            s.responded.add(p)
            if amt > tc:
                s.facing_size = s.invested[p]; s.raise_level += 1
                s.last_raiser_pos = pos; s.responded = {p}
            s.acting = s.next_active(p)

        return undo

    def _undo_action(self, s: State, undo: tuple):
        """Undo a mutable action, restoring state to before _apply_action_mut."""
        (action, p, stack_p, invested_p, pot,
         folded_p, responded, raise_level, facing_size,
         open_size, open_pos, n_cc, last_raiser_pos,
         acting) = undo
        s.stacks[p] = stack_p
        s.invested[p] = invested_p
        s.pot = pot
        s.folded[p] = folded_p
        s.responded = responded
        s.raise_level = raise_level
        s.facing_size = facing_size
        s.open_size = open_size
        s.open_pos = open_pos
        s.n_cold_callers = n_cc
        s.last_raiser_pos = last_raiser_pos
        s.acting = acting

    def _traverse(self, s: State, updating: int, hist: list) -> float:
        if is_terminal(s):
            return payoff(s)[updating]

        p     = s.acting
        legal = legal_actions(s)
        if not legal:
            return 0.0
        key   = info_key(p, s, hist)
        strat = self._strategy(key, legal)

        if p == updating:
            values = np.zeros(N_ACTIONS, dtype=np.float32)
            for a in legal:
                undo = self._apply_action_mut(s, a)
                values[a] = self._traverse(s, updating, hist + [a])
                self._undo_action(s, undo)
            ev = sum(strat[a] * values[a] for a in legal)

            idx = self._ensure(key)
            r = self._regrets[idx]
            for a in legal:
                r[a] += values[a] - ev

            ss = self._strat_sum[idx]
            total = 0.0
            for a in legal:
                v = r[a]
                if v > 0:
                    total += v
            if total > 0:
                for a in legal:
                    v = r[a]
                    ss[a] += (v if v > 0 else 0.0) / total
            else:
                inv = 1.0 / len(legal)
                for a in legal:
                    ss[a] += inv
            return ev
        else:
            # Sample one action (external sampling)
            cum = 0.0
            r = random.random()
            chosen = legal[-1]
            for a in legal:
                cum += strat[a]
                if r < cum:
                    chosen = a
                    break

            idx = self._ensure(key)
            ss = self._strat_sum[idx]
            for a in legal:
                ss[a] += strat[a]

            undo = self._apply_action_mut(s, chosen)
            val = self._traverse(s, updating, hist + [chosen])
            self._undo_action(s, undo)
            return val

    def average_policy(self) -> dict:
        policy = {}
        for key, idx in self._key_index.items():
            ss = self._strat_sum[idx]
            total = ss.sum()
            if total <= 0:
                continue
            probs = {a: float(ss[a] / total) for a in range(N_ACTIONS) if ss[a] > 0}
            policy[key] = probs
        return policy


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _save_ckpt(solver: Solver, path: str, progress: dict):
    import shutil
    free_gb = shutil.disk_usage(os.path.dirname(path) or ".").free / (1024 ** 3)
    if free_gb < 5.0:
        raise IOError(f"Disk space too low ({free_gb:.1f} GB free) — refusing to checkpoint")

    n = solver._n
    rev = {v: k for k, v in solver._key_index.items()}
    keys = [rev[i] for i in range(n)]

    snapshot = {
        "format":     "preflop-v1",
        "n_players":  solver.n_players,
        "stack_bb":   solver.stack_bb,
        "iterations": solver.iterations,
        "keys":       keys,
        "regrets":    solver._regrets[:n].copy(),
        "strat_sums": solver._strat_sum[:n].copy(),
    }

    dir_ = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".tmp_", suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise

    pp = path.replace(".pkl", ".progress.json")
    progress["checkpoint_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    fd2, tmp2 = tempfile.mkstemp(dir=os.path.dirname(pp) or ".", prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd2, "w") as f:
            json.dump(progress, f)
        os.replace(tmp2, pp)
    except Exception:
        try: os.unlink(tmp2)
        except OSError: pass
        raise


def _load_solver(path: str) -> Solver:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and obj.get("format") == "preflop-v1":
        keys    = obj["keys"]
        regrets = obj["regrets"]
        strats  = obj["strat_sums"]
        n = len(keys)
        solver = Solver(obj["n_players"], obj["stack_bb"], capacity=max(n, Solver._INIT_CAP))
        solver.iterations = obj.get("iterations", 0)
        solver._n = n
        solver._key_index = {k: i for i, k in enumerate(keys)}
        solver._regrets[:n]   = regrets
        solver._strat_sum[:n] = strats
        return solver

    raise ValueError(f"Unknown checkpoint format in {path}")


def _write_progress_only(path: str, progress: dict):
    pp = path.replace(".pkl", ".progress.json")
    progress["checkpoint_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dir_ = os.path.dirname(pp) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(progress, f, indent=2)
        os.replace(tmp, pp)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def _load_progress(path: str) -> dict:
    pp = path.replace(".pkl", ".progress.json")
    if os.path.exists(pp):
        try:
            with open(pp) as f:
                return json.load(f)
        except Exception:
            pass
    return {"iterations_done": 0}


# ─── Hand-category reverse map (for chart extraction) ────────────────────────

_HAND_MAP: dict[tuple[int, int], str] | None = None

def _build_hand_map() -> dict[tuple[int, int], str]:
    """Reverse map: (r1r2_byte, suit_byte) -> hand string like 'AA', 'AKs', 'QTo'."""
    m: dict[tuple[int, int], str] = {}
    for r1 in range(13):
        for r2 in range(r1 + 1, 13):
            # Suited: same suit for both cards
            hc = _hand_cat([r1 * 4 + 0, r2 * 4 + 0])
            r1r2, sb = _pack_hand_cat(hc)
            m[(r1r2, sb)] = hc  # use _hand_cat output for correct label
            # Offsuit: different suits
            hc = _hand_cat([r1 * 4 + 0, r2 * 4 + 1])
            r1r2, sb = _pack_hand_cat(hc)
            m[(r1r2, sb)] = hc  # use _hand_cat output for correct label
        hc = _hand_cat([r1 * 4 + 0, r1 * 4 + 1])
        r1r2, sb = _pack_hand_cat(hc)
        m[(r1r2, sb)] = hc
    return m


# ─── Compact policy extraction ────────────────────────────────────────────────

def _hash_key(key) -> np.uint64:
    """Hash a bytes key to uint64 for NPZ storage."""
    if isinstance(key, bytes):
        return np.uint64(int.from_bytes(
            hashlib.md5(key).digest()[:8], "little", signed=False
        ))
    return np.uint64(int.from_bytes(
        hashlib.md5(str(key).encode()).digest()[:8], "little", signed=False
    ))


def extract_and_save(solver: Solver, n_players: int, stack_bb: int):
    """
    Stream policy extraction directly from flat arrays — no intermediate dict.

    Previous version called solver.average_policy() which built a giant Python dict
    that could OOM on large info state counts. New version iterates flat arrays
    directly and builds compact NPZ + chart-friendly PKL in a memory-efficient way.
    """
    n = solver._n
    if n == 0:
        logger.warning("Policy empty — nothing to save")
        return

    suffix = "" if stack_bb == 100 else f"_{stack_bb}bb"
    logger.info(f"  Extracting policy from {n:,} info states (streaming, no dict)...")

    # Build reverse map: row index → key
    rev = {v: k for k, v in solver._key_index.items()}

    # ── NPZ: compact hash-based format ────────────────────────────────────
    keys_out  = np.empty(n, dtype=np.uint64)
    acts_out  = np.full((n, 5), -1, dtype=np.int16)
    probs_out = np.zeros((n, 5), dtype=np.float16)
    n_valid = 0

    for i in range(n):
        ss = solver._strat_sum[i]
        total = ss.sum()
        if total <= 0:
            continue

        # Compute strategy from positive regrets (regret matching)
        r = solver._regrets[i]
        pos = np.maximum(r, 0.0)
        pos_total = pos.sum()
        if pos_total > 0:
            strat = pos / pos_total
        else:
            nz = (ss > 0).sum()
            if nz == 0:
                continue
            strat = np.where(ss > 0, 1.0 / nz, 0.0)

        # Build probability dict for dominance simplification
        raw_probs = {}
        for a in range(N_ACTIONS):
            if ss[a] > 0:
                raw_probs[a] = float(strat[a])

        # No dominance simplification for preflop (5 actions, not 4 — different thresholds)
        # Just normalize
        ptotal = sum(raw_probs.values())
        if ptotal <= 0:
            continue

        key = rev[i]
        keys_out[n_valid] = _hash_key(key)

        for j, (a, p) in enumerate(sorted(raw_probs.items())):
            if j >= 5:
                break
            acts_out[n_valid, j] = a
            probs_out[n_valid, j] = p / ptotal

        n_valid += 1

        if n_valid % 1_000_000 == 0:
            logger.info(f"    ... {n_valid:,} entries processed ({i:,}/{n:,} info states)")

    if n_valid > 0:
        keys_out  = keys_out[:n_valid]
        acts_out  = acts_out[:n_valid]
        probs_out = probs_out[:n_valid]

        order = np.argsort(keys_out)
        keys_out  = keys_out[order]
        acts_out  = acts_out[order]
        probs_out = probs_out[order]

        npz_path = os.path.join(OUTPUT_DIR, f"{n_players}p{suffix}_preflop_fixed_policy.npz")
        np.savez_compressed(npz_path, keys=keys_out, actions=acts_out, probs=probs_out,
                            n_players=np.array([n_players]), stack_bb=np.array([stack_bb]),
                            max_actions=np.array([5]))
        logger.info(f"  NPZ: {npz_path}  ({os.path.getsize(npz_path)//1024//1024} MB, {n_valid:,} entries)")

    # ── Chart-friendly PKL: only for small info state counts ──────────────
    # Preflop info state counts are usually manageable (<10M), so we can
    # build the chart dict. But guard against OOM for large player counts.
    if n <= 10_000_000:
        logger.info(f"  Building chart PKL ({n:,} info states)...")
        global _HAND_MAP
        if _HAND_MAP is None:
            _HAND_MAP = _build_hand_map()

        _ACTION_MAP = {0: "fold", 1: "call", 2: "bet", 3: "squeeze", 4: "allin"}
        chart: dict[tuple[str, int, tuple[str, ...]], dict[str, float]] = {}
        for i in range(n):
            ss = solver._strat_sum[i]
            total = ss.sum()
            if total <= 0:
                continue
            r = solver._regrets[i]
            pos = np.maximum(r, 0.0)
            pos_total = pos.sum()
            if pos_total > 0:
                strat = pos / pos_total
            else:
                nz = (ss > 0).sum()
                if nz == 0:
                    continue
                strat = np.where(ss > 0, 1.0 / nz, 0.0)
            raw_probs = {a: float(strat[a]) for a in range(N_ACTIONS) if ss[a] > 0}
            ptotal = sum(raw_probs.values())
            if ptotal <= 0:
                continue

            key = rev[i]
            player_idx = key[0]
            hand_str = _HAND_MAP.get((key[1], key[2]))
            if hand_str is None:
                continue
            hist_ints = list(key[8:])
            hist_strs = tuple(_ACTION_MAP.get(a, f"?{a}") for a in hist_ints)
            chart_val: dict[str, float] = {}
            for a_int, prob in raw_probs.items():
                a_name = _ACTION_MAP.get(a_int, f"?{a_int}")
                chart_val[a_name] = chart_val.get(a_name, 0.0) + prob / ptotal
            chart[(hand_str, player_idx, hist_strs)] = chart_val

        chart_path = os.path.join(OUTPUT_DIR, f"{n_players}p{suffix}_preflop_policy.pkl")
        with open(chart_path, "wb") as f:
            pickle.dump(chart, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"  Chart PKL:  {chart_path}  ({os.path.getsize(chart_path)//1024} KB, {len(chart):,} entries)")
    else:
        logger.info(f"  Skipping chart PKL ({n:,} info states > 10M — would OOM). NPZ is sufficient.")


# ─── Training loop ────────────────────────────────────────────────────────────

def train(n_players: int, stack_bb: int, n_iters: int,
          checkpoint_every: int = 500_000,
          resume: bool = False, ram_floor_gb: float = 4.0):
    _set_oom_score(200)

    suffix = "" if stack_bb == 100 else f"_{stack_bb}bb"
    path = os.path.join(OUTPUT_DIR, f"{n_players}p{suffix}_preflop_fixed_solver.pkl")

    import glob
    for _orphan in glob.glob(os.path.join(os.path.dirname(path), ".tmp_*")):
        try:
            if (time.time() - os.path.getmtime(_orphan)) > 600:
                os.unlink(_orphan)
                logger.info(f"[cleanup] removed orphan {_orphan}")
        except OSError:
            pass

    progress = _load_progress(path)
    done_before = progress.get("iterations_done", 0)

    logger.info(f"\n{'='*60}")
    logger.info(f"Preflop fixed-tree solver — {n_players}p {stack_bb}bb  ({n_iters:,} iters)")
    logger.info(f"RAM floor: {ram_floor_gb:.1f} GB free required")
    logger.info(f"{'='*60}")

    if resume and os.path.exists(path) and os.path.getsize(path) > 0:
        logger.info(f"Resuming from {path} ({done_before:,} iters done)")
        solver = _load_solver(path)
    else:
        solver = Solver(n_players, stack_bb)
        done_before = 0
        logger.info("Starting fresh solver")

    _sigterm_solver[0] = solver
    _sigterm_path[0]   = path
    _sigterm_prog[0]   = progress

    t0 = time.time()
    for i in range(1, n_iters + 1):
        solver.run_iteration()

        if i % 100 == 0:
            free = _free_ram_gb()
            progress["iterations_done"] = done_before + i
            progress["n_players"] = n_players
            progress["stack_bb"] = stack_bb
            _write_progress_only(path, progress)
            if free < ram_floor_gb:
                logger.info(f"[RAM] {free:.1f} GB free — saving checkpoint and exiting")
                _save_ckpt(solver, path, progress)
                sys.exit(0)

        if i % 10_000 == 0:
            elapsed = time.time() - t0
            rate    = i / elapsed
            remain  = (n_iters - i) / rate
            free    = _free_ram_gb()
            logger.info(
                f"  iter {i:>8,}/{n_iters:,}  (total {done_before+i:,})  |  "
                f"{rate:,.0f} it/s  |  ~{remain/60:.1f} min  |  "
                f"{solver.n_info_states:,} info states  |  RAM {free:.1f} GB free"
            )

        if i % checkpoint_every == 0 or i == n_iters:
            progress["iterations_done"] = done_before + i
            progress["n_players"]       = n_players
            progress["stack_bb"]        = stack_bb
            _save_ckpt(solver, path, progress)
            logger.info(f"  checkpoint @ {progress['iterations_done']:,} iters")

    logger.info(f"\nDone in {(time.time()-t0)/60:.1f} min. Extracting policy...")
    extract_and_save(solver, n_players, stack_bb)
    return solver


# Target iterations for NashConv < 0.5 bb/hand (GTO convergence targets)
# Derived from MCCFR O(1/√T) convergence projection with measured NashConv
# and iteration rates as of 2026-05-03.
_CONVERGENCE_TARGETS = {
    (2, 100): 1_700_000_000,   # 1.7B  (NC 2.86 → 0.5, ~5 days)
    (2, 30):  500_000_000,     # 500M  (extrapolated from 2p 100bb)
    (3, 30):  800_000_000,     # 800M  (NC 2.55 → 0.5, ~5 days)
    (4, 30):  2_900_000_000,   # 2.9B  (NC 4.87 → 0.5, ~33 days)
    (5, 30):  5_900_000_000,   # 5.9B  (NC 6.98 → 0.5, ~105 days)
    (6, 30):  22_000_000_000,  # 22B   (NC est ~13.6 → 0.5, ~589 days)
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players",          type=int,   default=6)
    parser.add_argument("--stack-bb",         type=int,   default=100,
                        help="Effective stack in BB (30/50/75/100/150/200)")
    parser.add_argument("--iters",            type=int,   default=None,
                        help="Total iterations (default: convergence target for config)")
    parser.add_argument("--checkpoint-every", type=int,   default=500_000)
    parser.add_argument("--resume",           action="store_true")
    parser.add_argument("--ram-floor-gb",     type=float, default=8.0)
    args = parser.parse_args()

    n_iters = args.iters
    if n_iters is None:
        n_iters = _CONVERGENCE_TARGETS.get(
            (args.players, args.stack_bb), 50_000_000
        )

    train(args.players, args.stack_bb, n_iters,
          args.checkpoint_every, args.resume, args.ram_floor_gb)


if __name__ == "__main__":
    main()
