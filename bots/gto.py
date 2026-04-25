import random
from bots.base import BaseBot, Action
from strategy.preflop import preflop_action, open_raise_size, three_bet_size
from strategy.solver import solve_postflop
from strategy.preflop_solver import get_solver


class GTOBot(BaseBot):
    """Follows solver outputs as closely as possible."""

    def decide_preflop(self, position, stack, pot, to_call, facing_raise, raise_position, last_raise, bb,
                       action_sequence=None, player_idx=0) -> Action:
        c1, c2 = self.hole_cards

        # Try MCCFR solver first
        solver = get_solver(6)
        if solver.available and action_sequence is not None:
            sampled = solver.sample_action(self.hole_cards, player_idx, action_sequence)
            if sampled is not None:
                return _solver_action_to_action(sampled, to_call, last_raise, stack, bb, facing_raise)

        # Fallback to JSON range tables
        decision = preflop_action(position, c1, c2, facing_raise, raise_position)
        if decision == "raise":
            amt = three_bet_size(last_raise, stack) if facing_raise else open_raise_size(stack, bb)
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


def _solver_action_to_action(sampled: str, to_call: int, last_raise: int, stack: int, bb: int, facing_raise: bool) -> Action:
    """Map solver action label to Action with correct sizing."""
    if sampled == "fold":
        return Action("fold")
    if sampled == "call":
        return Action("call", min(to_call, stack))
    if sampled == "allin":
        return Action("allin", stack)
    # bet / raise
    if facing_raise:
        amt = three_bet_size(last_raise, stack)
    else:
        amt = open_raise_size(stack, bb)
    return Action("raise", min(amt, stack))
