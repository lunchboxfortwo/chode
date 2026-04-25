from bots.base import BaseBot, Action
from strategy.preflop import hand_category

NIT_RANGE = {
    "AA", "KK", "QQ", "JJ", "TT",
    "AKs", "AKo",
}


class NitBot(BaseBot):
    """Only plays top ~5% of hands. Folds to any aggression."""

    def decide_preflop(self, position, stack, pot, to_call, facing_raise, raise_position, last_raise, bb,
                       action_sequence=None, player_idx=0) -> Action:
        c1, c2 = self.hole_cards
        cat = hand_category(c1, c2)

        if cat not in NIT_RANGE:
            return Action("fold")

        if facing_raise:
            if cat in ("AA", "KK", "AKs", "AKo"):
                amt = min(last_raise * 3, stack)
                return Action("raise", amt)
            return Action("call", min(to_call, stack))

        amt = min(bb * 3, stack)
        return Action("raise", amt)

    def decide_postflop(self, board, position, stack, pot, to_call, is_first_to_act, action_sequence=None, prev_street_actions=None) -> Action:
        c1, c2 = self.hole_cards
        cat = hand_category(c1, c2)

        # Nit only bets premium made hands, folds otherwise
        is_premium = cat in NIT_RANGE

        if to_call > 0:
            # Nit folds to any aggression unless holding a premium hand
            if is_premium:
                return Action("call", min(to_call, stack))
            return Action("fold")

        if is_premium:
            bet = min(int(pot * 0.5), stack)
            return Action("bet", bet)
        return Action("check")
