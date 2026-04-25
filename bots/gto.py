import random
from bots.base import BaseBot, Action
from strategy.preflop import preflop_action, open_raise_size, three_bet_size
from strategy.postflop_solver import solve_postflop_gto
from strategy.preflop_solver import get_solver


class GTOBot(BaseBot):
    """Follows solver outputs as closely as possible."""

    def decide_preflop(self, position, stack, pot, to_call, facing_raise, raise_position, last_raise, bb,
                       action_sequence=None, player_idx=0) -> Action:
        c1, c2 = self.hole_cards

        # Try MCCFR solver first
        solver = get_solver(self.n_players)
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

    def decide_postflop(self, board, position, stack, pot, to_call, is_first_to_act,
                        action_sequence=None, prev_street_actions=None) -> Action:
        pos = "oop" if is_first_to_act else "ip"
        result = solve_postflop_gto(
            self.hole_cards, board, pot, stack,
            position=pos, action_history=action_sequence or [], to_call=to_call,
            prev_street_actions=prev_street_actions or [],
        )
        action = result["action"]
        equity = result["equity"]

        if action == "fold" and to_call == 0:
            action = "check"

        if action in ("bet", "raise"):
            if to_call > 0:
                if equity >= 0.65:
                    return Action("raise", min(to_call * 3, stack))
                if equity >= 0.35:
                    return Action("call", min(to_call, stack))
                return Action("fold")
            amt = result.get("amount", int(pot * 0.67))
            return Action("bet", min(amt, stack))

        if action == "allin":
            return Action("allin", stack)

        if action in ("check", "fold"):
            if to_call > 0:
                if equity >= 0.40:
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
