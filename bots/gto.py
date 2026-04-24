import random
from bots.base import BaseBot, Action
from strategy.preflop import preflop_action, open_raise_size, three_bet_size
from strategy.solver import solve_postflop


class GTOBot(BaseBot):
    """Follows solver outputs as closely as possible."""

    def decide_preflop(self, position, stack, pot, to_call, facing_raise, raise_position, last_raise, bb) -> Action:
        c1, c2 = self.hole_cards
        decision = preflop_action(position, c1, c2, facing_raise, raise_position)

        if decision == "raise":
            if facing_raise:
                amt = three_bet_size(last_raise, stack)
            else:
                amt = open_raise_size(stack, bb)
            return Action("raise", min(amt, stack))
        if decision == "call":
            return Action("call", min(to_call, stack))
        return Action("fold")

    def decide_postflop(self, board, position, stack, pot, to_call, is_first_to_act) -> Action:
        result = solve_postflop(self.hole_cards, board, pot, stack, position)
        action = result["action"]

        if action == "fold" and to_call == 0:
            action = "check"

        if action in ("bet", "raise") and result["amount"] > 0:
            if to_call > 0:
                # Facing a bet: decide call or raise
                if result["equity"] >= 0.65:
                    raise_amt = min(to_call * 3, stack)
                    return Action("raise", raise_amt)
                if result["equity"] >= 0.35:
                    return Action("call", min(to_call, stack))
                return Action("fold")
            return Action("bet", min(result["amount"], stack))

        if action == "check":
            if to_call > 0:
                if result["equity"] >= 0.40:
                    return Action("call", min(to_call, stack))
                return Action("fold")
            return Action("check")

        if to_call == 0:
            return Action("check")
        return Action("fold")
