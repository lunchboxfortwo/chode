"""
Postflop equity engine.
- Primary: Monte Carlo equity via treys (real hand evaluation, ~750 sims)
- Optional: TexasSolver-Console subprocess for full GTO tree solve
- Decision thresholds translate equity into bet/check/fold actions
"""
import random
import subprocess
import tempfile
import os
import resource
from pathlib import Path
from functools import lru_cache

from treys import Card as TCard, Evaluator, Deck as TDeck

from config import SOLVER_PATH, SOLVER_THREADS, SOLVER_RAM_GB, SOLVER_TIME_LIMIT

_evaluator = Evaluator()


# ─── Card conversion ──────────────────────────────────────────────────────────

def _to_treys(card_str: str) -> int:
    """Convert 'Ah' → treys int. treys uses lowercase suits, uppercase ranks."""
    rank = card_str[0].upper()
    suit = card_str[1].lower()
    # treys rank chars: 'A','K','Q','J','T','9'...'2'
    # treys suit chars: 's','h','d','c'
    return TCard.new(rank + suit)


def _treys_deck_minus(exclude: list[str]) -> list[int]:
    excluded = {_to_treys(c) for c in exclude}
    full = TDeck()
    full.shuffle()
    return [c for c in full.cards if c not in excluded]


# ─── Monte Carlo equity ───────────────────────────────────────────────────────

def monte_carlo_equity(
    hole: list[str],
    board: list[str],
    n_opponents: int = 1,
    n_sims: int = 750,
) -> float:
    """
    True equity vs n_opponents random hands via Monte Carlo.
    Returns a [0, 1] win rate (ties counted as 0.5).
    """
    try:
        my_hand = [_to_treys(c) for c in hole]
        known_board = [_to_treys(c) for c in board]
        board_needed = 5 - len(known_board)

        wins = 0.0

        for _ in range(n_sims):
            remaining = _treys_deck_minus(hole + board)
            random.shuffle(remaining)

            cursor = 0
            # Deal opponent hands
            opp_hands = []
            for _ in range(n_opponents):
                opp_hands.append(remaining[cursor:cursor + 2])
                cursor += 2

            # Complete board
            sim_board = known_board + remaining[cursor:cursor + board_needed]

            my_score = _evaluator.evaluate(sim_board, my_hand)
            best_opp = min(_evaluator.evaluate(sim_board, oh) for oh in opp_hands)

            if my_score < best_opp:
                wins += 1.0
            elif my_score == best_opp:
                wins += 0.5

        return wins / n_sims

    except Exception:
        return _fallback_heuristic(hole, board)


# ─── Public interface ─────────────────────────────────────────────────────────

def solve_postflop(
    hole: list[str],
    board: list[str],
    pot: int,
    effective_stack: int,
    hero_position: str = "btn",
    villain_position: str = "bb",
) -> dict:
    """
    Returns {'action': 'bet'|'check'|'fold', 'amount': int, 'equity': float}
    Uses TexasSolver if built, otherwise Monte Carlo equity.
    """
    if _texas_solver_available():
        return _run_texas_solver(hole, board, pot, effective_stack,
                                  hero_position, villain_position)

    equity = monte_carlo_equity(hole, board)
    return _equity_to_decision(equity, pot, effective_stack)


# ─── Decision logic ───────────────────────────────────────────────────────────

def _equity_to_decision(equity: float, pot: int, stack: int) -> dict:
    """Translate real equity into a bet/check/fold action with sizing."""
    if equity >= 0.70:
        # Strong hand — bet 75% pot
        amount = min(int(pot * 0.75), stack)
        return {"action": "bet", "amount": amount, "equity": equity}
    if equity >= 0.55:
        # Decent hand — bet 45% pot
        amount = min(int(pot * 0.45), stack)
        return {"action": "bet", "amount": amount, "equity": equity}
    if equity >= 0.38:
        # Marginal — check/call
        return {"action": "check", "amount": 0, "equity": equity}
    # Weak — fold if facing aggression, check if free
    return {"action": "fold", "amount": 0, "equity": equity}


# ─── TexasSolver subprocess (optional) ───────────────────────────────────────

def _texas_solver_available() -> bool:
    return Path(SOLVER_PATH).exists()


def _range_for_position(position: str) -> str:
    ranges = {
        "btn": "AA:1,KK:1,QQ:1,JJ:1,TT:1,99:1,88:0.75,AKs:1,AQs:1,AJs:1,ATs:1,KQs:1,QJs:1,JTs:1",
        "bb":  "AA:1,KK:1,QQ:1,JJ:1,TT:1,99:1,88:1,77:1,66:1,AKs:1,AQs:1,AJs:1,ATs:1,A9s:1,KQs:1",
        "sb":  "AA:1,KK:1,QQ:1,JJ:1,TT:1,99:1,AKs:1,AQs:1,AJs:1,KQs:1,QJs:1",
        "default": "AA:1,KK:1,QQ:1,JJ:1,TT:1,AKs:1,AQs:1,AJs:1,AKo:1",
    }
    return ranges.get(position.lower(), ranges["default"])


def _set_solver_limits():
    ram_bytes = SOLVER_RAM_GB * 1024 ** 3
    try:
        resource.setrlimit(resource.RLIMIT_AS, (ram_bytes, ram_bytes))
    except Exception:
        pass


def _run_texas_solver(hole, board, pot, stack, hero_pos, villain_pos):
    board_str = " ".join(board[:3]) if len(board) >= 3 else " ".join(board)
    config_lines = [
        f"set_pot {pot}",
        f"set_effective_stack {stack}",
        f"set_board {board_str}",
        f"set_range_ip {_range_for_position(hero_pos)}",
        f"set_range_oop {_range_for_position(villain_pos)}",
        "set_bet_sizes 0.5,1",
        "set_raise_sizes 2.5",
        "set_allin_threshold 1.5",
        "solve",
        "show_result",
    ]
    equity = monte_carlo_equity(hole, board)  # still compute real equity

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(config_lines))
            cfg_path = f.name

        result = subprocess.run(
            [str(SOLVER_PATH), cfg_path],
            capture_output=True, text=True,
            timeout=SOLVER_TIME_LIMIT * 2,
            preexec_fn=_set_solver_limits,
        )
        os.unlink(cfg_path)
        return _parse_solver_output(result.stdout, equity, pot, stack)
    except Exception:
        return _equity_to_decision(equity, pot, stack)


def _parse_solver_output(output: str, equity: float, pot: int, stack: int) -> dict:
    bet_freq = 0.0
    for line in output.lower().splitlines():
        if "bet" in line and "%" in line:
            try:
                bet_freq = float(line.split("%")[0].split()[-1]) / 100
            except ValueError:
                pass

    if bet_freq > 0 and random.random() < bet_freq:
        return {"action": "bet", "amount": min(int(pot * 0.6), stack), "equity": equity}
    return {"action": "check", "amount": 0, "equity": equity}


# ─── Fallback (no treys) ──────────────────────────────────────────────────────

RANK_VALUE = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
              "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def _fallback_heuristic(hole: list[str], board: list[str]) -> float:
    """Last-resort estimate if treys fails for any reason."""
    all_cards = hole + board
    ranks = [RANK_VALUE.get(c[0].upper(), 0) for c in all_cards]
    suits = [c[1].lower() for c in all_cards]

    rank_counts: dict = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    suit_counts: dict = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1

    max_of_a_kind = max(rank_counts.values(), default=1)
    n_flush = max(suit_counts.values(), default=0)
    pairs = sum(1 for v in rank_counts.values() if v == 2)
    top_hole = max((RANK_VALUE.get(c[0].upper(), 0) for c in hole), default=0)

    if max_of_a_kind >= 4: return 0.97
    if max_of_a_kind == 3 and pairs >= 1: return 0.93
    if n_flush >= 5: return 0.90
    if max_of_a_kind == 3: return 0.78
    if pairs >= 2: return 0.68
    if pairs == 1: return 0.45 + (top_hole / 14) * 0.15
    if n_flush >= 4: return 0.35
    return (top_hole / 14) * 0.28
