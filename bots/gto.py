import random
from bots.base import BaseBot, Action
from strategy.preflop_nn import PreflopNN
from strategy.postflop_nn import PostflopNN
from strategy.board_abstraction import texture_id

# ─── Shared NN singletons (lazy-loaded, reused across bots) ────────────────
_preflop_nn: PreflopNN | None = None
_postflop_nn: PostflopNN | None = None


def _get_preflop_nn() -> PreflopNN:
    global _preflop_nn
    if _preflop_nn is None:
        _preflop_nn = PreflopNN()
    return _preflop_nn


def _get_postflop_nn() -> PostflopNN:
    global _postflop_nn
    if _postflop_nn is None:
        _postflop_nn = PostflopNN()
    return _postflop_nn


# ─── Card utilities ─────────────────────────────────────────────────────────

# Map card string like "Ah" → integer 0-51 (suit: clubs=0, diamonds=1, hearts=2, spades=3)
_SUIT_MAP = {"c": 0, "d": 1, "h": 2, "s": 3}
_RANK_MAP = {
    "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6,
    "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12,
}


def _card_to_int(card: str) -> int:
    """Convert 'Ah' → int 0-51."""
    rank = _RANK_MAP[card[0].upper()]
    suit = _SUIT_MAP[card[1].lower()]
    return rank * 4 + suit


def _hand_category(c1: int, c2: int) -> int:
    """Map two card indices (0-51) to 0-168 hand category index.

    169 categories: 13 pairs + 78 suited + 78 offsuit.
    Pairs: 0-12 (22,33,...,AA)
    Suited: 13-90 (higher-lower suited combos)
    Offsuit: 91-168 (higher-lower offsuit combos)
    """
    r1, s1 = c1 // 4, c1 % 4
    r2, s2 = c2 // 4, c2 % 4
    if r1 < r2:
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    # Now r1 >= r2, and if r1==r2 then s1 <= s2

    if r1 == r2:
        # Pair: 0-12
        return r1
    elif s1 == s2:
        # Suited: 13 + r1*... mapping
        idx = 0
        for hi in range(12, 0, -1):
            for lo in range(hi - 1, -1, -1):
                if hi == r1 and lo == r2:
                    return 13 + idx
                idx += 1
        return 13  # fallback
    else:
        # Offsuit
        idx = 0
        for hi in range(12, 0, -1):
            for lo in range(hi - 1, -1, -1):
                if hi == r1 and lo == r2:
                    return 91 + idx
                idx += 1
        return 91  # fallback


def _position_to_idx(position: str, n_players: int) -> int:
    """Convert position name to 0-based index for the given table size."""
    POS_MAP = {
        2: {"sb": 0, "bb": 1},
        3: {"btn": 0, "sb": 1, "bb": 2},
        4: {"co": 0, "btn": 1, "sb": 2, "bb": 3},
        5: {"hj": 0, "co": 1, "btn": 2, "sb": 3, "bb": 4},
        6: {"lj": 0, "hj": 1, "co": 2, "btn": 3, "sb": 4, "bb": 5},
        7: {"utg": 0, "lj": 1, "hj": 2, "co": 3, "btn": 4, "sb": 5, "bb": 6},
        8: {"utg1": 0, "utg": 1, "lj": 2, "hj": 3, "co": 4, "btn": 5, "sb": 6, "bb": 7},
    }
    mapping = POS_MAP.get(n_players, POS_MAP[6])
    return mapping.get(position.lower(), 0)


def _map_action_sequence(seq: list[str]) -> list[str]:
    """Map game engine action labels to NN action history tokens.

    Engine uses: 'fold', 'call', 'check', 'raise', 'bet', 'allin'
    NN uses: 'fold', 'call', 'bet', 'squeeze', 'allin'
    """
    result = []
    for a in seq:
        a_low = a.lower()
        if a_low in ("fold", "call", "allin"):
            result.append(a_low)
        elif a_low in ("raise", "bet"):
            # Distinguish bet vs squeeze: 3rd+ aggressive action = squeeze
            n_agg = sum(1 for x in result if x in ("bet", "squeeze", "allin"))
            if n_agg >= 2:
                result.append("squeeze")
            else:
                result.append("bet")
        elif a_low == "check":
            result.append("call")  # check ≈ call-0
        else:
            result.append("call")
    return result


# ─── Sizing helpers ─────────────────────────────────────────────────────────

def _open_raise_size(stack: int, bb: int, position: str) -> int:
    """Standard open-raise sizing: 2.5bb from late, 3bb from early."""
    if position.lower() in ("btn", "co", "sb"):
        return min(int(bb * 2.5), stack)
    return min(int(bb * 3), stack)


def _three_bet_size(last_raise: int, stack: int) -> int:
    """Standard 3bet sizing: ~3x the raise."""
    return min(last_raise * 3, stack)


class GTOBot(BaseBot):
    """Follows neural network GTO strategy."""

    def decide_preflop(self, position, stack, pot, to_call, facing_raise,
                       raise_position, last_raise, bb,
                       action_sequence=None, player_idx=0, stacks=None) -> Action:
        n = self.n_players
        bb_depth = stack / bb if bb > 0 else 30
        pidx = _position_to_idx(position, n)
        hand = (_card_to_int(self.hole_cards[0]), _card_to_int(self.hole_cards[1]))

        # Map action sequence to NN format
        hist = _map_action_sequence(action_sequence or [])

        try:
            nn = _get_preflop_nn()
            probs = nn.query(n, bb_depth, pidx, hand, hist)
        except Exception:
            probs = {}

        if probs:
            # Sample from the distribution
            actions = list(probs.keys())
            weights = list(probs.values())
            # Threshold: only consider actions with >5% probability
            viable = [(a, p) for a, p in zip(actions, weights) if p > 0.05]
            if viable:
                viable_actions, viable_weights = zip(*viable)
                chosen = random.choices(viable_actions, weights=viable_weights, k=1)[0]
            else:
                chosen = actions[0] if actions else "fold"

            note = f"NN {'/'.join(f'{a}:{p:.0%}' for a, p in sorted(probs.items(), key=lambda x: -x[1])[:3])}"

            if chosen == "fold":
                if to_call == 0:
                    return Action("check", strategy_note=note)
                return Action("fold", strategy_note=note)
            if chosen == "call":
                return Action("call", min(to_call, stack), strategy_note=note)
            if chosen == "allin":
                return Action("allin", stack, strategy_note=note)
            # bet or squeeze → raise
            if facing_raise:
                amt = _three_bet_size(last_raise, stack)
            else:
                amt = _open_raise_size(stack, bb, position)
            return Action("raise", min(amt, stack), strategy_note=note)

        # Fallback: simple heuristic
        if to_call == 0:
            amt = _open_raise_size(stack, bb, position)
            return Action("raise", min(amt, stack), strategy_note="heuristic")
        if to_call <= stack * 0.05:
            return Action("call", min(to_call, stack), strategy_note="heuristic")
        return Action("fold", strategy_note="heuristic")

    def decide_postflop(self, board, position, stack, pot, to_call,
                        is_first_to_act, action_sequence=None,
                        prev_street_actions=None, n_active=2,
                        postflop_player_idx=0) -> Action:
        n = self.n_players

        # Encode hand
        c1 = _card_to_int(self.hole_cards[0])
        c2 = _card_to_int(self.hole_cards[1])
        hand_cat = _hand_category(c1, c2)

        # Position
        pos_idx = _position_to_idx(position, n)

        # Street
        if len(board) <= 3:
            street = 0  # flop
        elif len(board) == 4:
            street = 1  # turn
        else:
            street = 2  # river

        # Board texture
        tex_id = texture_id(board) if len(board) >= 3 else 0

        # Pot and stack features
        bb = 100  # default big blind
        pot_bb = pot / bb if bb > 0 else 1.0
        spr = stack / pot if pot > 0 else 10.0
        facing_size = to_call / pot if pot > 0 and to_call > 0 else 0.0

        # Aggression count
        agg_actions = 0
        if action_sequence:
            agg_actions = sum(1 for a in action_sequence if a.lower() in ("raise", "bet", "allin"))
        agg_actions = min(agg_actions, 3)

        # Action history
        nn_history = []
        if action_sequence:
            for a in action_sequence:
                a_low = a.lower()
                if a_low == "fold": nn_history.append(0)
                elif a_low in ("check", "call"): nn_history.append(1)
                elif a_low == "bet": nn_history.append(2)
                elif a_low == "raise": nn_history.append(4)
                elif a_low == "allin": nn_history.append(5)
                else: nn_history.append(1)

        try:
            nn = _get_postflop_nn()
            probs = nn.query(
                hand_cat=hand_cat,
                position=pos_idx,
                street=street,
                texture_id=tex_id,
                n_players=n,
                pot_size=pot_bb,
                stack_ratio=spr,
                facing_size=facing_size,
                agg_actions=agg_actions,
                action_history=nn_history,
            )
        except Exception:
            probs = {}

        if probs:
            actions = list(probs.keys())
            weights = list(probs.values())
            viable = [(a, p) for a, p in zip(actions, weights) if p > 0.05]
            if viable:
                viable_actions, viable_weights = zip(*viable)
                chosen = random.choices(viable_actions, weights=viable_weights, k=1)[0]
            else:
                chosen = actions[0] if actions else "check_call"

            note = f"NN {'/'.join(f'{a}:{p:.0%}' for a, p in sorted(probs.items(), key=lambda x: -x[1])[:3])}"

            if chosen == "fold":
                if to_call == 0:
                    return Action("check", strategy_note=note)
                return Action("fold", strategy_note=note)
            if chosen == "check_call":
                if to_call == 0:
                    return Action("check", strategy_note=note)
                return Action("call", min(to_call, stack), strategy_note=note)
            if chosen == "bet_small":
                amt = min(int(pot * 0.33), stack)
                return Action("bet", amt, strategy_note=note)
            if chosen == "bet_large":
                amt = min(int(pot * 0.75), stack)
                return Action("bet", amt, strategy_note=note)
            if chosen == "raise":
                if to_call > 0:
                    amt = min(to_call + int((pot + to_call) * 0.75), stack)
                else:
                    amt = min(int(pot * 0.67), stack)
                return Action("raise", amt, strategy_note=note)
            if chosen == "allin":
                return Action("allin", stack, strategy_note=note)

        # Fallback: equity-based heuristic
        if to_call == 0:
            return Action("check", strategy_note="heuristic")
        return Action("fold", strategy_note="heuristic")
