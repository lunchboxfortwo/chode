"""
GTO preflop solver integration.
Loads a trained MCCFR solver (pkl) and queries it for action probabilities
given a player's hole cards, position, and the action history so far.

Card encoding (OpenSpiel universal_poker):
  card_int = rank_idx * 4 + suit_idx
  rank_idx: 0='2', 1='3', ..., 12='A'  (RANKS = "23456789TJQKA")
  suit_idx: 0='c', 1='d', 2='h', 3='s'

Dealing order: all cards to player 0 first, then player 1, etc.
  positions [p*2, p*2+1] in the chance sequence belong to player p.
"""
import os
import sys
import pickle
import random
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

TABLES_DIR = Path(__file__).parent.parent / "data" / "preflop_tables"
RANKS = "23456789TJQKA"
SUITS = "cdhs"

# OpenSpiel action indices for universal_poker preflop
ACTION_FOLD  = 0
ACTION_CALL  = 1
ACTION_BET   = 2
ACTION_ALLIN = 3


# ─── Card conversion ──────────────────────────────────────────────────────────

def _card_to_int(card_str: str) -> int:
    """'Ah' → OpenSpiel card int."""
    rank = card_str[0].upper()
    suit = card_str[1].lower()
    return RANKS.index(rank) * 4 + SUITS.index(suit)


def _int_to_card(card_int: int) -> str:
    """OpenSpiel card int → 'Ah'."""
    return RANKS[card_int // 4] + SUITS[card_int % 4]


def _dummy_cards(exclude: set, count: int) -> list[int]:
    """Return `count` card ints not in exclude set."""
    result = []
    for i in range(52):
        if i not in exclude and len(result) < count:
            result.append(i)
    return result


# ─── Solver class ─────────────────────────────────────────────────────────────

class PreflopSolver:
    """
    Wraps a trained MCCFR solver for a specific player count.
    Thread-safe for reads (policy queries are stateless).
    """

    def __init__(self, n_players: int = 6):
        self.n_players = n_players
        self._solver = None
        self._policy = None
        self._game = None
        self._load(n_players)

    def _load(self, n_players: int):
        path = TABLES_DIR / f"{n_players}p_solver.pkl"
        if not path.exists():
            logger.warning(f"Preflop solver not found: {path}. Using JSON range fallback.")
            return

        try:
            from solver_training.preflop_train import build_preflop_game
            import pyspiel

            with open(path, "rb") as f:
                self._solver = pickle.load(f)

            self._game = build_preflop_game(n_players)
            self._policy = self._solver.average_policy()
            logger.info(f"Loaded {n_players}p preflop solver from {path}")
        except Exception as e:
            logger.warning(f"Failed to load preflop solver: {e}")
            self._solver = None

    @property
    def available(self) -> bool:
        return self._policy is not None

    def action_probs(
        self,
        hole_cards: list[str],
        player_idx: int,
        action_history: list[str],
    ) -> dict | None:
        """
        Returns {action_label: probability} or None if solver unavailable.

        action_history: list of Omega Poker action strings for actions taken
            THIS betting round before this player, in order.
            e.g. ['fold', 'raise', 'fold', 'fold']
        """
        if not self.available:
            return None

        try:
            state = self._reconstruct_state(hole_cards, player_idx, action_history)
            if state is None or state.is_terminal() or state.current_player() != player_idx:
                return None

            raw = self._policy.action_probabilities(state)
            result = {}
            for action_int, prob in raw.items():
                label = state.action_to_string(player_idx, action_int)
                result[_label_to_key(label)] = (action_int, prob)
            return result

        except Exception as e:
            logger.debug(f"Solver query failed: {e}")
            return None

    def sample_action(
        self,
        hole_cards: list[str],
        player_idx: int,
        action_history: list[str],
    ) -> str | None:
        """
        Sample a single action label from the GTO mixed strategy.
        Returns one of: 'fold', 'call', 'bet', 'allin', or None on failure.
        """
        probs = self.action_probs(hole_cards, player_idx, action_history)
        if probs is None:
            return None

        keys = list(probs.keys())
        weights = [probs[k][1] for k in keys]
        chosen = random.choices(keys, weights=weights, k=1)[0]
        return chosen

    # ─── State reconstruction ─────────────────────────────────────────────

    def _reconstruct_state(self, hole_cards, player_idx, action_history):
        """
        Build an OpenSpiel state that matches the given hole cards and
        action history for player_idx.
        """
        my_ints = [_card_to_int(c) for c in hole_cards]
        used = set(my_ints)
        dummy = iter(_dummy_cards(used, self.n_players * 2))

        state = self._game.new_initial_state()

        # Deal phase: n_players × 2 chance nodes
        # Order: (player0_card1, player0_card2, player1_card1, player1_card2, ...)
        for p in range(self.n_players):
            for slot in range(2):
                if p == player_idx:
                    card_int = my_ints[slot]
                else:
                    card_int = next(dummy)
                if state.is_chance_node():
                    state.apply_action(card_int)

        # Apply prior betting actions
        omega_actions = list(action_history)
        while omega_actions and not state.is_terminal():
            if state.current_player() == player_idx:
                break  # it's now this player's turn
            a_str = omega_actions.pop(0)
            action_int = _omega_to_openspiel_action(state, a_str)
            if action_int is None:
                return None
            state.apply_action(action_int)

        return state


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _label_to_key(label: str) -> str:
    """'player=2 move=Fold' → 'fold'"""
    if "Fold" in label:  return "fold"
    if "Call" in label:  return "call"
    if "Bet"  in label:  return "bet"
    if "AllIn" in label: return "allin"
    return label.lower()


def _omega_to_openspiel_action(state, action_str: str) -> int | None:
    """
    Map an Omega Poker action string to the nearest legal OpenSpiel action.
    """
    a = action_str.lower()
    legal = state.legal_actions()
    labels = [state.action_to_string(state.current_player(), x) for x in legal]

    if "fold"  in a:
        target = "Fold"
    elif "allin" in a:
        target = "AllIn"
    elif "raise" in a or "bet" in a:
        target = "Bet"
    else:
        target = "Call"

    for action_int, label in zip(legal, labels):
        if target in label:
            return action_int

    # Fallback: Call if exact match not found
    for action_int, label in zip(legal, labels):
        if "Call" in label:
            return action_int
    return None


# ─── Module-level cache (one solver per player count) ─────────────────────────

_solvers: dict[int, PreflopSolver] = {}


def get_solver(n_players: int) -> PreflopSolver:
    if n_players not in _solvers:
        _solvers[n_players] = PreflopSolver(n_players)
    return _solvers[n_players]
