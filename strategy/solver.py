"""Postflop equity engine — Monte Carlo via treys + fallback heuristic."""

import random
from treys import Card as TCard, Evaluator, Deck as TDeck

_evaluator = Evaluator()
_FULL_DECK: list[int] = TDeck().cards[:]


def _to_treys(card_str: str) -> int:
    """Convert 'Ah' → treys int."""
    return TCard.new(card_str[0].upper() + card_str[1].lower())


# ─── Monte Carlo equity ───────────────────────────────────────────────────────

def monte_carlo_equity(
    hole: list[str],
    board: list[str],
    n_opponents: int = 1,
    n_sims: int = 300,
) -> float:
    """True equity vs n_opponents random hands via Monte Carlo."""
    try:
        my_hand = [_to_treys(c) for c in hole]
        known_board = [_to_treys(c) for c in board]
        board_needed = 5 - len(known_board)

        excluded = set(my_hand) | set(known_board)
        remaining = [c for c in _FULL_DECK if c not in excluded]

        wins = 0.0
        for _ in range(n_sims):
            random.shuffle(remaining)
            cursor = 0
            opp_hands = []
            for _ in range(n_opponents):
                opp_hands.append(remaining[cursor:cursor + 2])
                cursor += 2
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


def river_exact_equity(
    hole: list[str],
    board: list[str],
    n_opponents: int = 1,
) -> float:
    """Exact equity on the river (enumerate 1-opp combos, MC for 2+)."""
    if len(board) != 5:
        return monte_carlo_equity(hole, board, n_opponents=n_opponents, n_sims=300)
    if n_opponents >= 2:
        return monte_carlo_equity(hole, board, n_opponents=n_opponents, n_sims=1500)
    try:
        from itertools import combinations
        my_hand = [_to_treys(c) for c in hole]
        board_treys = [_to_treys(c) for c in board]
        excluded = set(my_hand) | set(board_treys)
        remaining = [c for c in _FULL_DECK if c not in excluded]
        my_score = _evaluator.evaluate(board_treys, my_hand)
        wins = 0.0
        total = 0
        for opp_hand in combinations(remaining, 2):
            opp_score = _evaluator.evaluate(board_treys, list(opp_hand))
            if my_score < opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5
            total += 1
        return wins / total if total > 0 else 0.5
    except Exception:
        return monte_carlo_equity(hole, board, n_opponents=n_opponents, n_sims=300)


# ─── Fallback heuristic ──────────────────────────────────────────────────────

RANK_VALUE = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
              "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def _fallback_heuristic(hole: list[str], board: list[str]) -> float:
    """Last-resort estimate if treys fails."""
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
