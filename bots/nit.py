from bots.base import BaseBot, Action
from strategy.preflop_charts import hand_category, open_raise_size, three_bet_size

NIT_RANGE = {
    "AA", "KK", "QQ", "JJ", "TT",
    "AKs", "AKo",
}


class NitBot(BaseBot):
    """Only plays top ~5% of hands. Folds to any aggression."""

    def decide_preflop(self, position, stack, pot, to_call, facing_raise, raise_position, last_raise, bb,
                       action_sequence=None, player_idx=0, stacks=None) -> Action:
        c1, c2 = self.hole_cards
        cat = hand_category(c1, c2)

        if cat not in NIT_RANGE:
            return Action("fold", strategy_note="nit/fold")

        if facing_raise:
            if cat in ("AA", "KK", "AKs", "AKo"):
                amt = min(three_bet_size(last_raise, stack), stack)
                return Action("raise", amt, strategy_note="nit/3bet-premium")
            return Action("call", min(to_call, stack), strategy_note="nit/call-premium")

        amt = min(open_raise_size(stack, bb, position), stack)
        return Action("raise", amt, strategy_note="nit/open-premium")

    def decide_postflop(self, board, position, stack, pot, to_call, is_first_to_act, action_sequence=None, prev_street_actions=None, n_active=2, postflop_player_idx=0) -> Action:
        c1, c2 = self.hole_cards
        cat = hand_category(c1, c2)

        is_premium = cat in NIT_RANGE

        if to_call > 0:
            if is_premium:
                return Action("call", min(to_call, stack), strategy_note="nit/call-premium")
            return Action("fold", strategy_note="nit/fold")

        if is_premium:
            bet = min(int(pot * 0.5), stack)
            return Action("bet", bet, strategy_note="nit/bet-premium")
        return Action("check", strategy_note="nit/check")
