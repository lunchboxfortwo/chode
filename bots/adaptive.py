import random
from bots.base import BaseBot, Action
from strategy.preflop import preflop_action, hand_category, open_raise_size, three_bet_size
from strategy.postflop_solver import solve_postflop_gto
from strategy.tracker import OpponentTracker
from strategy.preflop_solver import get_solver

RANK_VALUE = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
              "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


class AdaptiveBot(BaseBot):
    """
    Tracks human stats and node-locks its strategy to exploit detected leaks.
    - High VPIP opponent → 3-bet wider with bluffs
    - Calling station (low fold-to-cbet) → bet thinner for value, bluff less
    - Passive player (low AF) → check-raise more
    - Tight (low VPIP) → widen value range, bluff less
    """

    def __init__(self, name: str, seat: int, tracker: OpponentTracker, human_seat: int):
        super().__init__(name, seat)
        self.tracker = tracker
        self.human_seat = human_seat

    def _stats(self):
        return self.tracker.get(self.human_seat)

    def decide_preflop(self, position, stack, pot, to_call, facing_raise, raise_position, last_raise, bb,
                       action_sequence=None, player_idx=0) -> Action:
        stats = self._stats()
        c1, c2 = self.hole_cards
        cat = hand_category(c1, c2)

        # Get GTO baseline from MCCFR solver
        base = "fold"
        solver = get_solver(self.n_players)
        if solver.available and action_sequence is not None:
            sampled = solver.sample_action(self.hole_cards, player_idx, action_sequence)
            if sampled is not None:
                base = sampled  # 'fold', 'call', 'bet', 'allin'
        if base == "fold":
            base = preflop_action(position, c1, c2, facing_raise, raise_position)

        if facing_raise:
            # If human folds a lot to 3-bets → 3-bet bluff wider
            if stats.fold_to_3bet_pct > 0.65:
                bluff_range = {"A5s", "A4s", "A3s", "A2s", "76s", "65s", "54s", "K9s", "Q9s"}
                if cat in bluff_range and random.random() < 0.7:
                    amt = three_bet_size(last_raise, stack)
                    return Action("raise", min(amt, stack))
            # If human is a calling station → tighten 3-bet to value only
            if stats.fold_to_3bet_pct < 0.35:
                value_only = {"AA", "KK", "QQ", "AKs", "AKo"}
                if cat in value_only:
                    amt = three_bet_size(last_raise, stack)
                    return Action("raise", min(amt, stack))
                if base == "call":
                    return Action("call", min(to_call, stack))
                return Action("fold")

        if base in ("raise", "bet", "allin"):
            if base == "allin":
                return Action("allin", stack)
            amt = open_raise_size(stack, bb)
            if stats.vpip < 0.20:
                amt = min(bb * 2, stack)
            return Action("raise", min(amt, stack))
        if base == "call":
            return Action("call", min(to_call, stack))
        return Action("fold")

    def decide_postflop(self, board, position, stack, pot, to_call, is_first_to_act,
                        action_sequence=None) -> Action:
        stats = self._stats()
        pos = "oop" if is_first_to_act else "ip"
        result = solve_postflop_gto(self.hole_cards, board, pot, stack,
                                    position=pos, action_history=action_sequence or [], to_call=to_call)
        equity = result["equity"]

        # Exploit: human folds to cbets → bet more often
        cbet_exploit = stats.fold_to_cbet_pct > 0.60

        # Exploit: calling station → bet thinner, never bluff
        is_station = stats.fold_to_cbet_pct < 0.30

        if to_call > 0:
            if is_station:
                # Only continue with strong hands
                if equity >= 0.55:
                    if equity >= 0.70:
                        return Action("raise", min(to_call * 3, stack))
                    return Action("call", min(to_call, stack))
                return Action("fold")
            if equity >= 0.45:
                return Action("call", min(to_call, stack))
            return Action("fold")

        # Not facing a bet
        if is_station:
            if equity >= 0.50:
                bet = int(pot * 0.55)
                return Action("bet", min(bet, stack))
            return Action("check")

        if cbet_exploit or equity >= 0.45:
            size = 0.75 if equity >= 0.65 else 0.45
            bet = int(pot * size)
            if bet > 0:
                return Action("bet", min(bet, stack))
        return Action("check")
