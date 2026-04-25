import random
from bots.base import BaseBot, Action
from strategy.preflop import hand_category

RANK_VALUE = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
              "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def _top_card_rank(c1: str, c2: str) -> int:
    return max(RANK_VALUE.get(c1[0].upper(), 0), RANK_VALUE.get(c2[0].upper(), 0))


class WhaleBot(BaseBot):
    """High VPIP, large bluffs, non-solver compliant. 50%+ VPIP."""

    VPIP_THRESHOLD = 0.50  # plays top ~50% of hands
    BLUFF_FREQ = 0.45

    def decide_preflop(self, position, stack, pot, to_call, facing_raise, raise_position, last_raise, bb,
                       action_sequence=None, player_idx=0) -> Action:
        c1, c2 = self.hole_cards
        cat = hand_category(c1, c2)
        top_rank = _top_card_rank(c1, c2)

        # Always play pairs, suited aces, broadways
        is_pair = len(cat) == 2
        is_suited_ace = cat.startswith("A") and cat.endswith("s")
        is_broadway = all(RANK_VALUE.get(r, 0) >= 10 for r in [c1[0].upper(), c2[0].upper()])
        is_connector = abs(RANK_VALUE.get(c1[0].upper(), 0) - RANK_VALUE.get(c2[0].upper(), 0)) == 1

        will_play = is_pair or is_suited_ace or is_broadway or is_connector or random.random() < 0.25

        if not will_play:
            if facing_raise and to_call > bb * 4:
                return Action("fold")
            return Action("fold")

        # Whale loves to raise big
        raise_roll = random.random()
        if facing_raise:
            if raise_roll < 0.55:
                amt = min(last_raise * 4, stack)
                return Action("raise", amt)
            return Action("call", min(to_call, stack))
        else:
            if raise_roll < 0.70:
                amt = min(bb * 4, stack)
                return Action("raise", amt)
            return Action("call", min(bb, stack))

    def decide_postflop(self, board, position, stack, pot, to_call, is_first_to_act, action_sequence=None) -> Action:
        c1, c2 = self.hole_cards
        top_rank = _top_card_rank(c1, c2)
        bluff = random.random() < self.BLUFF_FREQ

        if to_call > 0:
            if top_rank >= 10 or bluff:
                if random.random() < 0.40 and to_call < pot * 0.5:
                    raise_amt = min(to_call * 3, stack)
                    return Action("raise", raise_amt)
                return Action("call", min(to_call, stack))
            return Action("fold")

        # First to act or facing check
        if bluff or top_rank >= 11:
            bet_size = random.choice([int(pot * 0.75), int(pot * 1.0), int(pot * 1.5)])
            return Action("bet", min(bet_size, stack))
        return Action("check")
