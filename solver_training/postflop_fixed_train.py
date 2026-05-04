"""
Postflop CFR+ solver with simplified fixed-size game tree.

Action categories (N_ACTIONS=4):
  0  fold
  1  check / call
  2  small:  bet 33% pot (opening)  OR  raise(call + 75%×pot_after_call) (facing bet)
  3  large:  bet 75% pot (opening)  OR  allin (facing bet when raise would go to/over stack)

Bet cap: max AGG_CAP=3 aggressive actions per street. After cap: only fold(0), call(1), allin(3).

Rake: 3% of pot, capped at 2×BB. Applied to ALL postflop terminals (fold and showdown),
      since any hand that reaches the flop triggers rake.

Usage:
    python3 solver_training/postflop_fixed_train.py --players 2 --iters 2000000
    python3 solver_training/postflop_fixed_train.py --players 2 --iters 2000000 --resume
"""
import os
import sys
import random
import pickle
import argparse
import time
import json
import tempfile
import hashlib
import logging
import signal
import threading

import numpy as np
from treys import Card as TCard, Evaluator as TEval

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "postflop_tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BB        = 100
SB        = 50
BUY_IN    = 10_000
RAKE_RATE = 0.03
RAKE_CAP  = 2       # × BB chips
N_ACTIONS = 4
AGG_CAP   = 3
MIN_BET   = BB      # minimum bet size

RANKS = "23456789TJQKA"
SUITS = "cdhs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

_EVAL = TEval()


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
    # Count RAM at full value, swap at 50% (swap is slower and less reliable
    # under pressure — OOM killer can still fire when RAM+swap are exhausted).
    # Pure MemAvailable is the real RAM headroom; swap extends it partially.
    return mem_available + 0.5 * swap_free


# Mutable refs so SIGTERM handler can reach live solver state.
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


# Signal handler registered only in __main__ to avoid killing uvicorn on import


# ─── Card utilities ───────────────────────────────────────────────────────────

def _card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def _to_treys(c: int) -> int:
    s = _card_str(c)
    return TCard.new(s[0].upper() + s[1])


def _eval_hand(hole: list[int], board: list[int]) -> int:
    """Lower rank = better hand."""
    return _EVAL.evaluate([_to_treys(c) for c in board], [_to_treys(c) for c in hole])


def _deck_minus(exclude: set[int]) -> list[int]:
    deck = [c for c in range(52) if c not in exclude]
    random.shuffle(deck)
    return deck


# ─── Game state ───────────────────────────────────────────────────────────────

class State:
    """
    Postflop game state.

    facing_size: the total amount invested by the last aggressor.
                A player facing a bet must match this amount to call.
                Call amount for player p = facing_size - invested[p].
                When facing_size == 0, no bet is active (opening action).

    acted: set of players who have acted in the current betting round.
           A bet/raise resets this set (opens new round).
           The street ends when len(acted) == len(active) AND facing_size == 0
           (all active players have checked, or all have matched the bet and acted).
    """
    __slots__ = [
        'hole', 'board', 'pot', 'stacks', 'invested',
        'street', 'acting', 'facing_size', 'agg_count',
        'folded', 'n_players', 'acted',
    ]

    def __init__(self, hole, board, pot, stacks, invested,
                 n_players=2, street=0, acting=0):
        self.hole      = hole           # list of [c1, c2]
        self.board     = board
        self.pot       = pot
        self.stacks    = stacks
        self.invested  = invested       # total chips invested per player
        self.street    = street
        self.acting    = acting
        self.facing_size = 0            # total investment to match (0 = no bet)
        self.agg_count  = 0
        self.folded    = [False] * n_players
        self.n_players  = n_players
        self.acted     = set()          # players who acted in current betting round

    def copy(self):
        s = State.__new__(State)
        s.hole      = [h[:] for h in self.hole]
        s.board     = self.board[:]
        s.pot       = self.pot
        s.stacks    = self.stacks[:]
        s.invested  = self.invested[:]
        s.street    = self.street
        s.acting    = self.acting
        s.facing_size = self.facing_size
        s.agg_count  = self.agg_count
        s.folded    = self.folded[:]
        s.n_players  = self.n_players
        s.acted     = set(self.acted)
        return s

    def active(self):
        return [p for p in range(self.n_players) if not self.folded[p]]

    def next_active(self, from_p: int) -> int:
        for off in range(1, self.n_players + 1):
            p = (from_p + off) % self.n_players
            if not self.folded[p]:
                return p
        return from_p


# ─── Game logic ───────────────────────────────────────────────────────────────

def _bet_small(pot: int) -> int:
    return max(int(pot * 0.33), MIN_BET)


def _bet_large(pot: int) -> int:
    return max(int(pot * 0.75), MIN_BET)


def _call_amt(s: State, p: int) -> int:
    """Additional chips player p must put in to call the current bet."""
    return max(s.facing_size - s.invested[p], 0)


def _raise_total(facing_size: int, invested_p: int, pot: int) -> int:
    """Total investment for player p after a raise.
    
    Raise sizing: call + 75% × (pot after calling).
    This is the player's TOTAL invested amount after the raise, not the raise increment.
    """
    call_amt = facing_size - invested_p
    return facing_size + int((pot + call_amt) * 0.75)


def legal_actions(s: State) -> list[int]:
    p = s.acting
    stack = s.stacks[p]
    if s.facing_size == 0:
        # Opening action: check / bet-small / bet-large / (implicit allin via large)
        if s.agg_count >= AGG_CAP or stack == 0:
            return [1]  # check only
        acts = [1, 2, 3]
        # If small bet ≥ stack, collapse small and large into just allin
        if _bet_small(s.pot) >= stack:
            acts = [1, 3]
        return acts
    else:
        # Facing bet/raise
        call_amt = _call_amt(s, p)
        acts = [0, 1]  # fold, call
        if s.agg_count < AGG_CAP:
            raise_inv = _raise_total(s.facing_size, s.invested[p], s.pot)
            # Can raise if we have more chips than the call amount
            # and the raise total doesn't exceed our stack (otherwise it's an allin)
            if stack > call_amt and raise_inv - s.invested[p] < stack:
                acts.append(2)  # fixed raise
        if stack > call_amt:
            acts.append(3)  # allin (always available if we have chips beyond the call)
        return acts


def apply_action(s: State, action: int) -> State:
    """Returns a NEW state (s is not mutated).
    
    Key invariant: facing_size is the total investment that must be matched.
    After any aggressive action (bet/raise), facing_size = invested[aggressor].
    Call amount for player p = facing_size - invested[p].
    Street ends when all active players have acted AND facing_size == 0 (all checked)
    OR all active players have matched facing_size and acted (bet called around).
    """
    s = s.copy()
    p = s.acting

    if s.facing_size == 0:
        # Opening action (no bet to match)
        if action == 1:
            # Check
            s.acted.add(p)
            s.acting = s.next_active(p)
            if len(s.acted) >= len(s.active()):
                # All active players checked — end street
                s = _end_street(s)
        elif action == 2:
            # Bet small (33% pot)
            bet_amt = min(_bet_small(s.pot), s.stacks[p])
            _put_in(s, p, bet_amt)
            s.facing_size = s.invested[p]  # aggressor sets the new facing_size
            s.agg_count += 1
            s.acted   = {p}   # reset: p opened new betting round
            s.acting  = s.next_active(p)
        elif action == 3:
            # Bet large (75% pot) or allin
            bet_amt = min(_bet_large(s.pot), s.stacks[p])
            if bet_amt >= s.stacks[p]:
                bet_amt = s.stacks[p]  # allin
            _put_in(s, p, bet_amt)
            s.facing_size = s.invested[p]
            s.agg_count += 1
            s.acted   = {p}
            s.acting  = s.next_active(p)
    else:
        # Facing bet/raise
        call_amt = _call_amt(s, p)

        if action == 0:
            # Fold
            s.folded[p] = True
            # If only one player left, they win — no further action needed
            if len(s.active()) <= 1:
                s.acting = s.next_active(p)
                return s
            # After a fold, check if all remaining active players have acted.
            # E.g., 3p: P0 bets, P1 calls, P2 folds → P0 and P1 already acted → end street.
            if s.acted >= set(s.active()):
                s = _end_street(s)
            else:
                s.acting = s.next_active(p)
        elif action == 1:
            # Call — match the facing_size
            amt = min(call_amt, s.stacks[p])
            _put_in(s, p, amt)
            s.acted.add(p)
            s.acting  = s.next_active(p)
            # Street ends when all active players have acted after the last raise.
            # Since facing_size > 0, the round closes when everyone has called/matched.
            if len(s.acted) >= len(s.active()):
                s = _end_street(s)
        elif action == 2:
            # Fixed raise: call + 75% × (pot after calling)
            raise_inv = _raise_total(s.facing_size, s.invested[p], s.pot)
            # Chips to add = raise total - already invested
            amt_add = min(raise_inv - s.invested[p], s.stacks[p])
            _put_in(s, p, amt_add)
            s.facing_size = s.invested[p]  # new facing_size = raiser's total investment
            s.agg_count += 1
            s.acted   = {p}   # reset: p opened new betting round
            s.acting  = s.next_active(p)
        elif action == 3:
            # Allin — put entire remaining stack in
            amt = s.stacks[p]
            _put_in(s, p, amt)
            s.agg_count += 1
            s.acted   = {p}
            # If allin exceeds current facing_size, it becomes a raise
            if s.invested[p] > s.facing_size:
                s.facing_size = s.invested[p]
            s.acting = s.next_active(p)
            # If all players are allin, run out board
            if all(s.stacks[q] == 0 for q in s.active()):
                s = _run_out(s)

    return s


def _put_in(s: State, p: int, amt: int):
    s.pot       += amt
    s.stacks[p] -= amt
    s.invested[p] += amt


def _end_street(s: State) -> State:
    """End current street, reset betting state, deal next card or mark terminal."""
    s.facing_size = 0
    s.agg_count   = 0
    s.acted       = set()
    s.acting      = 0  # OOP acts first
    s.street     += 1
    return s


def _run_out(s: State) -> State:
    """All players allin — deal remaining board cards."""
    exclude = set()
    for h in s.hole:
        exclude.update(h)
    exclude.update(s.board)
    deck = _deck_minus(exclude)
    while len(s.board) < 5:
        s.board.append(deck.pop())
    s.street = 3   # past river → terminal
    return s


def is_terminal(s: State) -> bool:
    if len(s.active()) <= 1:
        return True
    if s.street > 2:
        return True
    return False


def needs_card(s: State) -> bool:
    """True if we need to deal the next street card before continuing."""
    # street=0 → flop dealt, need turn after flop betting
    # street=1 → turn dealt, need river after turn betting
    # We only enter a new street AFTER _end_street increments street.
    # So street=1 means we need the turn card, street=2 means river card.
    return s.street in (1, 2) and len(s.board) < (3 + s.street)


def payoff(s: State) -> list[float]:
    """
    Chip gain/loss for each player (relative to their stack at session start).
    Rake 3% of pot, capped at 2×BB, applied since hand reached postflop.
    """
    rake = min(s.pot * RAKE_RATE, RAKE_CAP * BB)
    active = s.active()

    if len(active) == 1:
        winner = active[0]
        gains = [-s.invested[p] for p in range(s.n_players)]
        gains[winner] += s.pot - rake
        return gains

    # Showdown — need complete 5-card board
    board5 = _complete_board(s)
    scores  = {p: _eval_hand(s.hole[p], board5) for p in active}
    best    = min(scores.values())
    winners = [p for p, sc in scores.items() if sc == best]

    gains = [-s.invested[p] for p in range(s.n_players)]
    share  = (s.pot - rake) / len(winners)
    for w in winners:
        gains[w] += share
    return gains


def _complete_board(s: State) -> list[int]:
    exclude = set()
    for h in s.hole:
        exclude.update(h)
    exclude.update(s.board)
    deck = _deck_minus(exclude)
    board = s.board[:]
    while len(board) < 5:
        board.append(deck.pop())
    return board


# ─── Information state key ────────────────────────────────────────────────────

def _hand_cat(h: list[int]) -> str:
    c1, c2 = h[0], h[1]
    r1, s1 = c1 // 4, c1 % 4
    r2, s2 = c2 // 4, c2 % 4
    if r1 < r2:
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return RANKS[r1] * 2
    return RANKS[r1] + RANKS[r2] + ('s' if s1 == s2 else 'o')


def _board_norm(board: list[int]) -> tuple:
    """Suit-isomorphic board key. Flop sorted canonical: rank desc, suit asc."""
    flop = sorted(board[:3], key=lambda c: (-c // 4, c % 4))
    rest = board[3:]
    suit_map: dict[int, int] = {}
    norm = []
    for c in (flop + rest):
        r, s = c // 4, c % 4
        if s not in suit_map:
            suit_map[s] = len(suit_map)
        norm.append((r, suit_map[s]))
    return tuple(norm)


def _pack_hand_cat(hand_cat: str) -> tuple:
    """Pack hand category string to (r1_r2_byte, suit_byte).
    r1_r2_byte: (rank1 << 4) | rank2   where rank 2..A → 0..12
    suit_byte: 0=offsuit, 1=suited, 2=pair
    """
    r1 = RANKS.index(hand_cat[0])
    r2 = RANKS.index(hand_cat[1]) if len(hand_cat) > 1 else r1
    if len(hand_cat) == 2:  # pair like "22"
        suit = 2
    elif hand_cat[2] == 's':
        suit = 1
    else:
        suit = 0
    return ((r1 << 4) | r2, suit)


def info_key(p: int, s: State, action_hist: list) -> bytes:
    """
    Compact information state key for MCCFR regret table.

    Returns a bytes object (~15-25 bytes) instead of a nested tuple (~400-600 bytes).
    This saves ~5 GB for 10M info states because:
      - bytes: sys.getsizeof ~50 bytes (obj header + data)
      - tuple: sys.getsizeof ~500+ bytes (outer + inner tuples + str + bools)

    Encoding (all fixed-width, no delimiters needed):
      Offset  Size  Field
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
    buf.append(p)
    hc = _hand_cat(s.hole[p])
    r1r2, suit = _pack_hand_cat(hc)
    buf.append(r1r2)
    buf.append(suit)
    # board_norm: list of (rank, suit_idx) pairs
    bn = _board_norm_raw(s.board)
    for r, si in bn:
        buf.append((r << 4) | si)
    buf.append(s.street)
    buf.append(int(s.facing_size > 0))
    buf.append(s.agg_count)
    for a in action_hist:
        buf.append(a)
    return bytes(buf)


def _board_norm_raw(board: list[int]) -> list:
    """Like _board_norm but returns list of (rank, suit_idx) pairs — avoids tuple overhead."""
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


# ─── External Sampling MCCFR ─────────────────────────────────────────────────

class Solver:
    """
    Flat-array external-sampling MCCFR solver with compact bytes keys.

    Memory layout:
        _key_index:  dict[bytes, int]   — packed info-key → row index (CPython C impl, fast)
        _regrets:    np.float32 (capacity, N_ACTIONS)  — contiguous regret table
        _strat_sum:  np.float32 (capacity, N_ACTIONS)  — contiguous strategy-sum table
        _n:          int — number of used rows
        _capacity:   int — allocated rows (grows 1.5× when full)

    Memory per info state at ~400M entries:
        _key_index:  ~86B (bytes key ~42B + int val 28B + dict slot ~16B) → ~34.4 GB
        _regrets:    16B → ~6.4 GB (with 1.5× growth → 82% utilization)
        _strat_sum:  16B → ~6.4 GB
        ─────────────────────
        Total:       ~47 GB — FITS in 62 GB RAM with 15 GB headroom

    Previous dict approach with 2× growth used ~67 GB (doesn't fit).
    The 1.5× growth saves ~7.4 GB in wasted array capacity.
    Robin Hood hash table saved ~24 GB more but was 40× slower in Python.
    """

    _INIT_CAP = 65_536  # ~1 MB initial allocation, grows 1.5× as needed

    def __init__(self, n_players: int = 2, *, capacity: int = 0):
        self.n_players = n_players
        self.iterations = 0
        cap = capacity or self._INIT_CAP
        self._capacity  = cap
        self._n         = 0
        self._key_index = {}                          # bytes → row index
        self._regrets   = np.zeros((cap, N_ACTIONS), dtype=np.float32)
        self._strat_sum = np.zeros((cap, N_ACTIONS), dtype=np.float32)

    # ── grow ──────────────────────────────────────────────────────────────────

    def _grow(self, min_cap: int):
        """Grow capacity by 1.5× until >= min_cap. Saves ~7 GB vs 2× growth at 400M entries."""
        new_cap = self._capacity
        while new_cap < min_cap:
            new_cap = int(new_cap * 1.5) + 1
        if new_cap <= self._capacity:
            return
        old_r = self._regrets
        old_s = self._strat_sum
        self._regrets   = np.zeros((new_cap, N_ACTIONS), dtype=np.float32)
        self._strat_sum = np.zeros((new_cap, N_ACTIONS), dtype=np.float32)
        self._regrets[:self._n]   = old_r[:self._n]
        self._strat_sum[:self._n] = old_s[:self._n]
        self._capacity = new_cap

    def _ensure(self, key: bytes) -> int:
        """Return row index for key, allocating a new row if needed."""
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

    # ── strategy (regret matching) ────────────────────────────────────────────

    def _strategy(self, key: bytes, legal: list[int]) -> np.ndarray:
        """Regret matching over legal actions."""
        idx = self._key_index.get(key)
        strat = np.zeros(N_ACTIONS, dtype=np.float32)
        if idx is not None:
            r = self._regrets[idx]
            pos = np.maximum(r, 0.0)
            total = pos[legal].sum()
            if total > 0:
                for a in legal:
                    strat[a] = pos[a] / total
                return strat
        for a in legal:
            strat[a] = 1.0 / len(legal)
        return strat

    # ── dealing ───────────────────────────────────────────────────────────────

    def _deal(self) -> State:
        deck = list(range(52))
        random.shuffle(deck)
        n = self.n_players
        hole = [deck[p*2 : p*2+2] for p in range(n)]
        flop = deck[n*2 : n*2+3]
        blinds   = [SB, BB] + [0] * max(0, n - 2)
        stacks   = [BUY_IN - b for b in blinds]
        pot      = sum(blinds)
        invested = blinds[:]
        return State(hole, flop, pot, stacks, invested, n_players=n, street=0, acting=0)

    def _deal_card(self, s: State) -> int:
        exclude = set()
        for h in s.hole:
            exclude.update(h)
        exclude.update(s.board)
        return random.choice([c for c in range(52) if c not in exclude])

    # ── MCCFR traversal ────────────────────────────────────────────────────────

    def run_iteration(self):
        state = self._deal()
        for player in range(self.n_players):
            self._traverse(state, player, [])

    def _traverse(self, s: State, updating: int, hist: list) -> float:
        """Returns counterfactual value for `updating` player."""
        if is_terminal(s):
            return payoff(s)[updating]

        if needs_card(s):
            card = self._deal_card(s)
            ns = s.copy()
            ns.board.append(card)
            return self._traverse(ns, updating, hist)

        p     = s.acting
        legal = legal_actions(s)
        key   = info_key(p, s, hist)
        strat = self._strategy(key, legal)

        if p == updating:
            values = np.zeros(N_ACTIONS, dtype=np.float32)
            for a in legal:
                ns = apply_action(s, a)
                values[a] = self._traverse(ns, updating, hist + [a])
            ev = sum(strat[a] * values[a] for a in legal)

            idx = self._ensure(key)
            r = self._regrets[idx]
            for a in legal:
                r[a] += values[a] - ev

            ss = self._strat_sum[idx]
            pos = np.maximum(r, 0.0)
            total = pos[legal].sum()
            if total > 0:
                for a in legal:
                    ss[a] += pos[a] / total
            else:
                for a in legal:
                    ss[a] += 1.0 / len(legal)
            return ev

        else:
            probs = [strat[a] for a in legal]
            a = random.choices(legal, weights=probs, k=1)[0]
            idx = self._ensure(key)
            ss = self._strat_sum[idx]
            for la in legal:
                ss[la] += strat[la]
            return self._traverse(apply_action(s, a), updating, hist + [a])

    # ── policy extraction ─────────────────────────────────────────────────────

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


# ─── Dominance simplification ─────────────────────────────────────────────────

DOMINANCE_FREQ = 0.80    # collapse if one action has ≥80% mass
DOMINANCE_HARD = 0.95    # always collapse if ≥95%
DOMINANCE_EV_THRESHOLD = 0.005  # max EV loss as fraction of pot

def _simplify(probs: dict, pot: int = 0, regrets: np.ndarray | None = None) -> dict:
    """Apply 80% dominance: collapse to pure action if safe.

    Spec: collapse when ≥80% freq AND EV loss < 0.5% pot.

    If regrets array is provided, uses regret-based EV loss estimate:
      EV loss ≈ sum of positive regrets for non-dominant actions / total positive regrets × pot
    This is much tighter than the conservative (1-freq)×pot bound.

    Without regrets, falls back to the conservative bound (effectively requires 99.5%+ freq).
    """
    if not probs:
        return probs
    items = list(probs.items())
    best_a, best_p = max(items, key=lambda x: x[1])
    if best_p >= DOMINANCE_HARD:
        return {best_a: 1.0}
    if best_p >= DOMINANCE_FREQ:
        if regrets is not None:
            # Regret-based EV loss: non-dominant positive regret / total positive regret × pot
            pos_r = np.maximum(regrets, 0.0)
            pos_total = pos_r.sum()
            if pos_total > 0:
                non_dom_regret = pos_total - pos_r[best_a] if best_a < len(pos_r) else pos_total
                ev_loss = (non_dom_regret / pos_total)  # fraction of pot
                if ev_loss <= DOMINANCE_EV_THRESHOLD:
                    return {best_a: 1.0}
        else:
            # Conservative bound: (1 - freq) × pot ≤ threshold × pot
            ev_loss_bound = 1.0 - best_p
            if ev_loss_bound <= DOMINANCE_EV_THRESHOLD:
                return {best_a: 1.0}
    return probs


# ─── Compact NPZ extraction ───────────────────────────────────────────────────

def _hash_key(key) -> np.uint64:
    """Hash a key (bytes or tuple) to a uint64 for NPZ storage."""
    if isinstance(key, bytes):
        # bytes keys: hash directly (fast, no string conversion)
        return np.uint64(int.from_bytes(
            hashlib.md5(key).digest()[:8], "little", signed=False
        ))
    # Legacy tuple keys
    return np.uint64(int.from_bytes(
        hashlib.md5(str(key).encode()).digest()[:8], "little", signed=False
    ))


def extract_and_save(solver: Solver, n_players: int):
    """
    Stream policy extraction directly from flat arrays — no intermediate dict.
    
    Previous version called solver.average_policy() which built a dict[bytes, dict[int, float]]
    with 222M+ entries (~40 GB RAM). This caused OOM kills on top of the solver's ~30 GB.
    
    New version iterates the solver's flat _regrets/_strat_sum arrays directly,
    building compact NPZ arrays in batches. Peak overhead: ~1 GB (vs 40 GB before).
    """
    n = solver._n
    if n == 0:
        logger.warning("Policy empty — nothing to save")
        return

    logger.info(f"  Extracting policy from {n:,} info states (streaming, no dict)...")

    # ── NPZ: stream from flat arrays ──────────────────────────────────────
    # Pre-allocate arrays for the worst case (all info states have non-zero strategy)
    keys_out  = np.empty(n, dtype=np.uint64)
    acts_out  = np.full((n, 4), -1, dtype=np.int16)
    probs_out = np.zeros((n, 4), dtype=np.float16)
    n_valid = 0

    # Build ordered key list: row index → key (list uses ~40% less memory than dict)
    rev = [None] * n
    for k, v in solver._key_index.items():
        rev[v] = k

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
            # Uniform over actions that had strategy sum
            nz = (ss > 0).sum()
            if nz == 0:
                continue
            strat = np.where(ss > 0, 1.0 / nz, 0.0)

        # Build probability dict for simplification
        raw_probs = {}
        for a in range(N_ACTIONS):
            if ss[a] > 0:
                raw_probs[a] = float(strat[a])

        probs = _simplify(raw_probs, regrets=r)
        ptotal = sum(probs.values())
        if ptotal <= 0:
            continue

        # Hash key
        key = rev[i]
        keys_out[n_valid] = _hash_key(key)

        # Fill 4-slot arrays
        for j, (a, p) in enumerate(sorted(probs.items())):
            if j >= 4:
                break
            acts_out[n_valid, j] = a
            probs_out[n_valid, j] = p / ptotal

        n_valid += 1

        if n_valid % 1_000_000 == 0:
            logger.info(f"    ... {n_valid:,} entries processed ({i:,}/{n:,} info states)")

    if n_valid == 0:
        logger.warning("Policy empty after extraction — nothing to save")
        return

    # Trim to actual size
    keys_out  = keys_out[:n_valid]
    acts_out  = acts_out[:n_valid]
    probs_out = probs_out[:n_valid]

    # Sort by key for binary search at runtime
    order = np.argsort(keys_out)
    keys_out  = keys_out[order]
    acts_out  = acts_out[order]
    probs_out = probs_out[order]

    npz_path = os.path.join(OUTPUT_DIR, f"{n_players}p_postflop_fixed_policy.npz")
    np.savez_compressed(npz_path, keys=keys_out, actions=acts_out, probs=probs_out,
                        n_players=np.array([n_players]), max_actions=np.array([4]))
    logger.info(f"  NPZ: {npz_path}  ({os.path.getsize(npz_path)//1024//1024} MB, {n_valid:,} entries)")

    # ── PKL: lightweight bytes→list format (no nested dicts) ──────────────
    # Old PKL used dict[bytes, dict[int, float]] — huge because each inner dict
    # has ~56 bytes of Python object overhead × 222M entries ≈ 12 GB just for dicts.
    # New format: dict[bytes, tuple] where tuple is (action, prob, action, prob, ...)
    # This avoids inner dict overhead. Also skip if info states > 10M (still too big).
    if n <= 10_000_000:
        logger.info(f"  Building PKL ({n:,} info states)...")
        policy = {}
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
            raw_probs = {}
            for a in range(N_ACTIONS):
                if ss[a] > 0:
                    raw_probs[a] = float(strat[a])
            probs = _simplify(raw_probs, regrets=r)
            if sum(probs.values()) <= 0:
                continue
            key = rev[i]
            # Store as flat tuple instead of dict to save memory
            policy[key] = tuple(v for pair in sorted(probs.items()) for v in pair)
        pkl_path = os.path.join(OUTPUT_DIR, f"{n_players}p_postflop_fixed_policy.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(policy, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"  PKL: {pkl_path}  ({os.path.getsize(pkl_path)//1024//1024} MB)")
    else:
        logger.info(f"  Skipping PKL ({n:,} info states > 10M — would OOM). NPZ is sufficient.")


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _save_ckpt(solver: Solver, path: str, progress: dict):
    """
    Atomic checkpoint write — flat-array format with compact bytes keys.

    On-disk layout (format='flat-v3'):
        {
          'format': 'flat-v3',
          'n_players': int,
          'iterations': int,
          'keys':       List[bytes]       # compact packed keys (~15-25B each)
          'regrets':    np.ndarray((N, N_ACTIONS), float32)
          'strat_sums': np.ndarray((N, N_ACTIONS), float32)
        }

    The Robin Hood hash table is NOT saved — it's rebuilt on load from the
    keys list. This saves ~7 GB on disk and avoids serializing numpy arrays
    that would be stale on the next load anyway.
    """
    import shutil
    free_gb = shutil.disk_usage(os.path.dirname(path) or ".").free / (1024 ** 3)
    if free_gb < 5.0:
        raise IOError(f"Disk space too low ({free_gb:.1f} GB free) — refusing to checkpoint")

    n = solver._n
    # Build ordered key list: row index → key (list uses ~40% less memory than dict)
    keys = [None] * n
    for k, v in solver._key_index.items():
        keys[v] = k

    snapshot = {
        "format":     "flat-v3",
        "n_players":  solver.n_players,
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
    dir_ = os.path.dirname(pp) or "."
    fd, tmp_pp = tempfile.mkstemp(dir=dir_, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(progress, f)
        os.replace(tmp_pp, pp)
    except Exception:
        try: os.unlink(tmp_pp)
        except OSError: pass
        raise


def _tuple_key_to_bytes(key: tuple) -> bytes:
    """Convert old-style tuple info key to compact bytes format.
    Used when loading legacy flat-v1 checkpoints with tuple keys."""
    buf = bytearray()
    buf.append(key[0])                           # player
    hc = key[1]                                   # hand category string
    r1r2, suit = _pack_hand_cat(hc)
    buf.append(r1r2)
    buf.append(suit)
    for r, s in key[2]:                           # board_norm tuple of (rank, suit_idx)
        buf.append((r << 4) | s)
    buf.append(key[3])                             # street
    buf.append(int(key[4]))                        # facing_bet bool
    buf.append(key[5])                             # agg_count
    for a in key[6]:                               # action_hist tuple
        buf.append(a)
    return bytes(buf)


def _load_solver(path: str) -> Solver:
    """
    Load a Solver from disk, handling four formats:
      - Legacy format: pickled Solver instance with dict internals (pre-2026-05-01)
      - flat-v1 format: pickled dict with tuple keys (pre-2026-05-01)
      - flat-v2 format: pickled dict with bytes keys
      - flat-v3 format: pickled dict with bytes keys (1.5× growth)
    All formats load into the dict + flat-array internal representation.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # Helper: load keys into a new Solver using the Python dict
    def _build_solver(keys_list, regrets_arr, strats_arr, n_players, iters):
        n = len(keys_list)
        solver = Solver(n_players, capacity=max(n, Solver._INIT_CAP))
        solver.iterations = iters
        solver._n = n
        solver._regrets[:n]   = regrets_arr
        solver._strat_sum[:n] = strats_arr
        # Build dict index (CPython C dict — fast lookups during training)
        for i, key in enumerate(keys_list):
            if isinstance(key, tuple):
                key = _tuple_key_to_bytes(key)
            solver._key_index[key] = i
            if i > 0 and i % 5_000_000 == 0:
                logger.info(f"    ... {i:,}/{n:,} keys indexed")
        logger.info(f"  Dict index built: {n:,} entries")
        return solver

    # Legacy: full Solver instance pickled directly (had dict[tuple, np.ndarray] internals)
    if isinstance(obj, Solver):
        n = len(obj._regrets)
        keys = list(obj._regrets.keys())
        regrets = np.stack([obj._regrets[k] for k in keys])
        zero4 = np.zeros(N_ACTIONS, dtype=np.float32)
        strats = np.stack([obj._strat_sum.get(k, zero4) for k in keys])
        del obj._regrets
        del obj._strat_sum
        return _build_solver(keys, regrets, strats, obj.n_players, obj.iterations)

    # flat-v1: dict with tuple keys
    if isinstance(obj, dict) and obj.get("format") == "flat-v1":
        return _build_solver(
            obj["keys"], obj["regrets"], obj["strat_sums"],
            obj["n_players"], obj.get("iterations", 0))

    # flat-v2: dict with bytes keys
    if isinstance(obj, dict) and obj.get("format") == "flat-v2":
        return _build_solver(
            obj["keys"], obj["regrets"], obj["strat_sums"],
            obj["n_players"], obj.get("iterations", 0))

    # flat-v3: dict with bytes keys (current format — same on-disk layout as v2)
    if isinstance(obj, dict) and obj.get("format") == "flat-v3":
        return _build_solver(
            obj["keys"], obj["regrets"], obj["strat_sums"],
            obj["n_players"], obj.get("iterations", 0))

    raise ValueError(f"Unknown checkpoint format in {path}: type={type(obj).__name__}")


def _write_progress_only(path: str, progress: dict):
    """
    Cheap progress update: writes ONLY the .progress.json sidecar, atomically.
    Skips the heavy solver pickle. Safe to call every 100 iters.
    """
    progress_path = path.replace(".pkl", ".progress.json")
    progress["checkpoint_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dir_ = os.path.dirname(progress_path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(progress, f, indent=2)
        os.replace(tmp, progress_path)
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


# ─── Training loop ────────────────────────────────────────────────────────────

def train(n_players: int, n_iters: int, checkpoint_every: int = 100_000,
          resume: bool = False, ram_floor_gb: float = 4.0):
    _set_oom_score(200)

    path     = os.path.join(OUTPUT_DIR, f"{n_players}p_postflop_fixed_solver.pkl")
    # Sweep orphan .tmp_* files from prior killed runs (>10 min old)
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
    logger.info(f"Postflop fixed-tree solver — {n_players}p  ({n_iters:,} iters)")
    logger.info(f"RAM floor: {ram_floor_gb:.1f} GB free required")
    logger.info(f"{'='*60}")

    if resume and os.path.exists(path) and os.path.getsize(path) > 0:
        logger.info(f"Resuming from {path} ({done_before:,} iters done)")
        solver = _load_solver(path)
    else:
        solver = Solver(n_players)
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
            _write_progress_only(path, progress)
            if free < ram_floor_gb:
                logger.info(f"[RAM] {free:.1f} GB free, below floor {ram_floor_gb:.1f} GB — saving checkpoint and exiting")
                _save_ckpt(solver, path, progress)
                sys.exit(0)

        if i % 2_000 == 0:
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
            _save_ckpt(solver, path, progress)
            logger.info(f"  checkpoint @ {progress['iterations_done']:,} iters")

    logger.info(f"\nDone in {(time.time()-t0)/60:.1f} min. Extracting policy...")
    extract_and_save(solver, n_players)
    return solver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players",          type=int,   default=2)
    parser.add_argument("--iters",            type=int,   default=2_000_000)
    parser.add_argument("--checkpoint-every", type=int,   default=100_000)
    parser.add_argument("--resume",           action="store_true")
    parser.add_argument("--ram-floor-gb",     type=float, default=4.0)
    args = parser.parse_args()
    train(args.players, args.iters, args.checkpoint_every, args.resume, args.ram_floor_gb)


if __name__ == "__main__":
    main()
