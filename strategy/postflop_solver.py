"""
Postflop GTO solver — state reconstruction query interface.

Loads data/postflop_tables/postflop_solver.pkl and queries the MCCFR
average policy by reconstructing an OpenSpiel state with the actual
hole cards, board, and action history.

Falls back to Monte Carlo equity when the solver isn't available.
"""
import os
import sys
import pickle
import random
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.solver import monte_carlo_equity

logger = logging.getLogger(__name__)

SOLVER_PATH = Path(__file__).parent.parent / "data" / "postflop_tables" / "postflop_solver.pkl"

RANKS = "23456789TJQKA"
SUITS = "cdhs"

_solver = None
_policy = None
_game   = None
_loaded = False


def _load():
    global _solver, _policy, _game, _loaded
    if _loaded:
        return
    _loaded = True
    if not SOLVER_PATH.exists():
        logger.info("Postflop solver not found — using MC equity fallback")
        return
    try:
        from solver_training.postflop_train import build_postflop_game
        with open(SOLVER_PATH, "rb") as f:
            _solver = pickle.load(f)
        _game   = build_postflop_game()
        _policy = _solver.average_policy()
        logger.info("Postflop solver loaded")
    except Exception as e:
        logger.warning(f"Failed to load postflop solver: {e}")


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


def _reconstruct_state(hole: list[str], board: list[str],
                       player_idx: int, action_history: list[str]):
    """
    Reconstruct an OpenSpiel postflop state for player_idx with the
    given hole cards, board (3-5 cards), and prior betting actions.

    player_idx: 0 = OOP (acts first postflop), 1 = IP
    action_history: list of omega-style strings for actions taken
                    THIS street before this player.
    """
    if _game is None:
        return None

    my_ints = [_card_to_int(c) for c in hole]
    board_ints = [_card_to_int(c) for c in board]
    used = set(my_ints) | set(board_ints)
    dummy = iter(_dummy_cards(used, 2))  # opponent gets 2 dummy hole cards

    state = _game.new_initial_state()

    # Deal phase: hole cards first (p0 then p1), then board cards
    # Player 0 gets cards at deal slots 0-1, player 1 at slots 2-3
    deal_sequence = []
    if player_idx == 0:
        deal_sequence += my_ints          # p0 = our cards
        deal_sequence += [next(dummy), next(dummy)]  # p1 = dummy
    else:
        deal_sequence += [next(dummy), next(dummy)]  # p0 = dummy
        deal_sequence += my_ints          # p1 = our cards

    # Then board: flop (3), and optionally turn (1), river (1)
    deal_sequence += board_ints

    for card_int in deal_sequence:
        if not state.is_chance_node():
            break
        legal = [a for a, _ in state.chance_outcomes()]
        if card_int in legal:
            state.apply_action(card_int)
        else:
            # Conflict (dummy card clashed) — pick any legal card not in used
            for alt in legal:
                if alt not in used:
                    state.apply_action(alt)
                    used.add(alt)
                    break
            else:
                state.apply_action(legal[0])

    # Replay betting actions up to this player's turn
    omega_actions = list(action_history)
    while omega_actions and not state.is_terminal() and not state.is_chance_node():
        if state.current_player() == player_idx:
            break
        a_str = omega_actions.pop(0)
        action_int = _map_action(state, a_str)
        if action_int is None:
            return None
        state.apply_action(action_int)

    return state


def _map_action(state, action_str: str):
    """Map an omega action string to the nearest legal OpenSpiel action."""
    a = action_str.lower()
    legal = state.legal_actions()
    labels = [state.action_to_string(state.current_player(), x).lower() for x in legal]

    if "fold" in a:
        target = "fold"
    elif "allin" in a:
        target = "allin"
    elif "raise" in a or "bet" in a:
        target = "bet"
    elif "check" in a:
        target = "check"
    else:
        target = "call"

    for action_int, label in zip(legal, labels):
        if target in label:
            return action_int
    # fallback: check or call
    for action_int, label in zip(legal, labels):
        if "check" in label or "call" in label:
            return action_int
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


def sample_postflop_action(
    hole: list[str],
    board: list[str],
    player_idx: int,           # 0=OOP, 1=IP
    action_history: list[str], # actions taken this street before this player
) -> str | None:
    """
    Sample one action from the GTO policy.
    Returns 'fold'/'check'/'call'/'bet'/'raise'/'allin' or None on miss.
    """
    _load()
    if _policy is None or len(board) < 3:
        return None

    try:
        state = _reconstruct_state(hole, board, player_idx, action_history)
        if state is None or state.is_terminal() or state.is_chance_node():
            return None
        if state.current_player() != player_idx:
            return None

        raw = _policy.action_probabilities(state)
        keys = list(raw.keys())
        weights = list(raw.values())
        if sum(weights) == 0:
            return None

        chosen_int = random.choices(keys, weights=weights, k=1)[0]
        label = state.action_to_string(player_idx, chosen_int)
        return _norm(label)

    except Exception as e:
        logger.debug(f"Postflop solver query failed: {e}")
        return None


def solve_postflop_gto(
    hole: list[str],
    board: list[str],
    pot: int,
    stack: int,
    position: str = "ip",      # 'ip' or 'oop'
    action_history: list[str] = None,
    to_call: int = 0,
    n_opponents: int = 1,
) -> dict:
    """
    Returns {'action': str, 'amount': int, 'equity': float}.
    Tries the MCCFR solver first, then falls back to MC equity thresholds.
    """
    if action_history is None:
        action_history = []

    equity = monte_carlo_equity(hole, board, n_opponents=n_opponents)

    player_idx = 1 if position == "ip" else 0
    sampled = sample_postflop_action(hole, board, player_idx, action_history)

    if sampled is not None:
        amount = _size_for_action(sampled, pot, stack, to_call)
        return {"action": sampled, "amount": amount, "equity": equity}

    return _equity_fallback(equity, pot, stack, to_call)


def _size_for_action(action: str, pot: int, stack: int, to_call: int) -> int:
    if action in ("check", "fold"):   return 0
    if action == "call":              return min(to_call, stack)
    if action == "allin":             return stack
    return min(int(pot * 0.67), stack)


def _equity_fallback(equity: float, pot: int, stack: int, to_call: int) -> dict:
    if to_call > 0:
        if equity >= 0.65:
            return {"action": "raise",  "amount": min(to_call * 3, stack), "equity": equity}
        if equity >= 0.38:
            return {"action": "call",   "amount": min(to_call, stack),      "equity": equity}
        return     {"action": "fold",   "amount": 0,                        "equity": equity}
    if equity >= 0.70:
        return {"action": "bet",   "amount": min(int(pot * 0.75), stack), "equity": equity}
    if equity >= 0.55:
        return {"action": "bet",   "amount": min(int(pot * 0.45), stack), "equity": equity}
    return     {"action": "check", "amount": 0,                           "equity": equity}
