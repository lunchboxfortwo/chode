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


def _apply_card(state, card_int: int, used: set):
    """Apply a specific card to a chance node, with fallback if conflicted."""
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


def _replay_street(state, actions: list[str], stop_at_player: int = -1):
    """
    Apply a sequence of betting actions to advance through a completed street.
    Stops early if stop_at_player is reached (used for current street).
    Returns False if the state becomes terminal/invalid mid-replay.
    """
    for a_str in actions:
        if state.is_terminal() or state.is_chance_node():
            return False
        if stop_at_player >= 0 and state.current_player() == stop_at_player:
            return True
        action_int = _map_action(state, a_str)
        if action_int is None:
            return False
        state.apply_action(action_int)
    return True


def _reconstruct_state(
    hole: list[str],
    board: list[str],
    player_idx: int,
    action_history: list[str],
    prev_street_actions: list[list[str]] = None,
):
    """
    Reconstruct an OpenSpiel postflop state at any street.

    player_idx         : 0=OOP, 1=IP
    board              : 3 (flop), 4 (turn), or 5 (river) cards
    action_history     : actions taken THIS street before player_idx's turn
    prev_street_actions: [flop_actions] for turn; [flop_actions, turn_actions] for river
    """
    if _game is None:
        return None

    prev = prev_street_actions or []
    my_ints = [_card_to_int(c) for c in hole]
    board_ints = [_card_to_int(c) for c in board]
    used = set(my_ints) | set(board_ints)
    dummy = iter(_dummy_cards(used, 2))

    state = _game.new_initial_state()

    # ── 1. Deal hole cards ────────────────────────────────────────────────
    if player_idx == 0:
        hole_seq = my_ints + [next(dummy), next(dummy)]
    else:
        hole_seq = [next(dummy), next(dummy)] + my_ints

    for card_int in hole_seq:
        if not state.is_chance_node():
            return None
        _apply_card(state, card_int, used)

    # ── 2. Deal flop (always present) ────────────────────────────────────
    for card_int in board_ints[:3]:
        if not state.is_chance_node():
            return None
        _apply_card(state, card_int, used)

    # ── 3. Turn / river: replay prior streets to advance through them ─────
    if len(board) >= 4:
        # Replay flop betting in full (prev[0]) to reach turn card deal
        flop_actions = prev[0] if len(prev) > 0 else []
        if not _replay_street(state, flop_actions):
            return None
        # The flop round may not have ended yet if actions were incomplete —
        # in that case we can't reach the turn.
        if state.is_terminal():
            return None
        # Deal turn card (next chance node after flop betting ends)
        if not state.is_chance_node():
            return None
        _apply_card(state, board_ints[3], used)

    if len(board) == 5:
        # Replay turn betting in full (prev[1]) to reach river card deal
        turn_actions = prev[1] if len(prev) > 1 else []
        if not _replay_street(state, turn_actions):
            return None
        if state.is_terminal():
            return None
        if not state.is_chance_node():
            return None
        _apply_card(state, board_ints[4], used)

    # ── 4. Replay current street up to player_idx's turn ──────────────────
    _replay_street(state, action_history, stop_at_player=player_idx)

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
    player_idx: int,
    action_history: list[str],
    prev_street_actions: list[list[str]] = None,
) -> str | None:
    """
    Sample one action from the GTO policy.
    Returns 'fold'/'check'/'call'/'bet'/'raise'/'allin' or None on miss.
    """
    _load()
    if _policy is None or len(board) < 3:
        return None

    try:
        state = _reconstruct_state(hole, board, player_idx, action_history, prev_street_actions)
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
    position: str = "ip",
    action_history: list[str] = None,
    to_call: int = 0,
    n_opponents: int = 1,
    prev_street_actions: list[list[str]] = None,
) -> dict:
    """
    Returns {'action': str, 'amount': int, 'equity': float}.
    Tries the MCCFR solver first, then falls back to MC equity thresholds.
    """
    if action_history is None:
        action_history = []

    equity = monte_carlo_equity(hole, board, n_opponents=n_opponents)

    player_idx = 1 if position == "ip" else 0
    sampled = sample_postflop_action(hole, board, player_idx, action_history, prev_street_actions)

    if sampled is not None:
        if sampled == "call" and to_call == 0:
            sampled = "check"
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
