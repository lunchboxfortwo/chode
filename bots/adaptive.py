"""
AdaptiveBot — GTO baseline with node-locking exploitation overlay.

Strategy:
  Preflop : MCCFR solver probabilities, re-weighted by opponent deviation
            from GTO baseline frequencies.
  Postflop: Same — solver probabilities adjusted by tracked stats per texture.

Node-locking approximation:
  For each exploitable situation, compute the opponent's observed frequency
  and compare to a GTO baseline. The deviation drives a multiplier on our
  action probabilities: if they fold too much to cbets, increase our bet
  probability; if they call too much, decrease our bluff frequency.
"""
import random
from bots.base import BaseBot, Action
from strategy.preflop_charts import preflop_action, hand_category
from strategy.preflop_charts import open_raise_size, three_bet_size
from strategy.preflop_charts import position_to_player_idx
from strategy.postflop_solver import solve_postflop_gto
from strategy.tracker import OpponentTracker
from strategy.preflop_charts import action_probs as charts_action_probs
from strategy.board_abstraction import board_texture

RANK_VALUE = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
              "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

# GTO baseline frequencies for comparison
GTO_FOLD_TO_CBET_DRY = 0.50
GTO_FOLD_TO_CBET_WET = 0.38
GTO_FOLD_TO_3BET     = 0.55
GTO_CHECK_RAISE      = 0.08
GTO_RIVER_FOLD       = 0.45
GTO_LIMP_PCT         = 0.05   # rarely limp in full GTO
GTO_CALL_3BET        = 0.30
GTO_POS_AWARENESS    = 2.0    # LP VPIP / EP VPIP ratio for balanced player
GTO_TURN_PROBE       = 0.25


class AdaptiveBot(BaseBot):
    """
    Tracks opponent stats per situation and re-weights GTO probabilities
    to exploit detected deviations from equilibrium.
    """

    def __init__(self, name: str, seat: int, tracker: OpponentTracker, human_seat: int):
        super().__init__(name, seat)
        self.tracker = tracker
        self.human_seat = human_seat

    def _stats(self):
        return self.tracker.get(self.human_seat)

    # ─── Preflop ──────────────────────────────────────────────────────────

    def decide_preflop(self, position, stack, pot, to_call, facing_raise, raise_position, last_raise, bb,
                       action_sequence=None, player_idx=0, stacks=None) -> Action:
        stats = self._stats()
        c1, c2 = self.hole_cards
        cat = hand_category(c1, c2)
        _stacks = stacks or [stack] * self.n_players

        # Get GTO probabilities (stack-depth-aware, interpolates between trained buckets)
        gto_probs = None
        if action_sequence is not None:
            chart_idx = position_to_player_idx(position, self.n_players)
            gto_probs = charts_action_probs(
                self.hole_cards, chart_idx, action_sequence,
                self.n_players, _stacks, bb, position=position,
            )

        if gto_probs is not None:
            # Apply exploitation weights to GTO probabilities
            weights = self._exploit_preflop_weights(gto_probs, stats, facing_raise, cat)
            sampled = _weighted_sample(weights)
            if sampled is not None:
                a = _preflop_action_to_action(sampled, to_call, last_raise, stack, bb, facing_raise, position)
                a.strategy_note = "adaptive+exploit"
                return a

        # Fallback: JSON ranges with exploitation overlays
        base = preflop_action(position, c1, c2, facing_raise, raise_position)

        if facing_raise:
            ftb = stats.fold_to_3bet_pct
            # Opponent calls 3-bets too loosely → 3-bet pure value only
            c3b = stats.call_3bet_pct
            if c3b > GTO_CALL_3BET + 0.15:
                if cat in {"AA","KK","QQ","JJ","AKs","AKo"}:
                    return Action("raise", min(three_bet_size(last_raise, stack), stack), strategy_note="adaptive/3b-value-only")
                # Don't bluff vs station
                if base == "raise" and cat not in {"AA","KK","QQ","JJ","AKs","AKo"}:
                    return Action("call", min(to_call, stack), strategy_note="adaptive/3b-station-adjust") if to_call > 0 else Action("fold", strategy_note="adaptive/3b-station-adjust")
            # Opponent folds too much to 3-bets → expand bluff 3-bets
            if ftb > GTO_FOLD_TO_3BET + 0.12:
                bluff_range = {"A5s","A4s","A3s","A2s","76s","65s","54s","K9s","Q9s","J9s"}
                deviation = (ftb - GTO_FOLD_TO_3BET) / (1 - GTO_FOLD_TO_3BET)
                if cat in bluff_range and random.random() < deviation:
                    return Action("raise", min(three_bet_size(last_raise, stack), stack), strategy_note="adaptive/bluff-exploit")
            # Opponent never folds to 3-bets → tighten to pure value
            if ftb < GTO_FOLD_TO_3BET - 0.18:
                if cat in {"AA","KK","QQ","AKs","AKo"}:
                    return Action("raise", min(three_bet_size(last_raise, stack), stack), strategy_note="adaptive/value-tighten")
                return Action("call", min(to_call, stack), strategy_note="adaptive/value-tighten") if base != "fold" else Action("fold", strategy_note="adaptive/value-tighten")

        # Limp exploitation: opponent limps frequently → steal with wider range
        limp_dev = stats.limp_pct - GTO_LIMP_PCT
        if not facing_raise and limp_dev > 0.10 and position.upper() in ("CO", "BTN", "HJ"):
            steal_range = {"A2s","A3s","A4s","A5s","K7s","K8s","K9s","KTo","Q8s","Q9s","J9s","T9s","98s","87s"}
            if cat in steal_range and random.random() < min(limp_dev * 2, 0.70):
                amt = open_raise_size(stack, bb, position)
                return Action("raise", min(amt, stack), strategy_note="adaptive/limp-steal")

        # Positional awareness exploit: loose EP opener → call down; tight BTN → widen steal
        pos_aware = stats.positional_awareness
        if not facing_raise and position.upper() == "BTN":
            if pos_aware < 1.3:
                # Low positional awareness: their EP ranges are too wide → respect more
                pass  # don't over-isolate
            elif pos_aware > 2.5:
                # High positional awareness gap: EP opens are very strong → fold more to them
                pass
        if not facing_raise and raise_position and position.upper() == "BTN":
            # If raiser is in EP and they're position-aware, respect the range
            if raise_position.upper() in ("UTG", "UTG+1") and pos_aware > 2.0:
                if base == "call" and cat not in {"AA","KK","QQ","JJ","TT","AKs","AKo","AQs"}:
                    return Action("fold", strategy_note="adaptive/pos-aware-fold")

        if base == "raise":
            amt = open_raise_size(stack, bb, position)
            if stats.vpip < 0.18:
                amt = min(bb * 2, stack)
            return Action("raise", min(amt, stack), strategy_note="adaptive/range")
        if base == "call":
            return Action("call", min(to_call, stack), strategy_note="adaptive/range")
        return Action("fold", strategy_note="adaptive/range")

    def _exploit_preflop_weights(self, gto_probs, stats, facing_raise, cat) -> dict:
        """
        Re-weight GTO action probs based on opponent's deviation from equilibrium.
        Returns {action_key: adjusted_weight}.
        """
        weights = {k: v for k, v in gto_probs.items()}  # already {action: prob}

        if facing_raise:
            dev = stats.fold_to_3bet_pct - GTO_FOLD_TO_3BET
            if abs(dev) > 0.10:
                for k in weights:
                    if k in ("bet", "raise", "allin"):
                        weights[k] *= (1 + dev * 1.5)  # fold too much → raise more
                    elif k == "fold":
                        weights[k] *= max(0.1, 1 - dev)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    # ─── Postflop ─────────────────────────────────────────────────────────

    def decide_postflop(self, board, position, stack, pot, to_call, is_first_to_act,
                        action_sequence=None, prev_street_actions=None, n_active=2,
                        postflop_player_idx=0) -> Action:
        stats = self._stats()

        result = solve_postflop_gto(
            self.hole_cards, board, pot, stack,
            player_order_idx=postflop_player_idx,
            action_history=action_sequence or [],
            to_call=to_call,
            prev_street_actions=prev_street_actions or [],
            n_active=n_active,
            street=_board_to_street(board),
        )

        action = result["action"]
        equity = result["equity"]
        amount = result.get("amount", 0)

        base_strategy = result.get("strategy", "MC equity")

        # Apply exploitation overlay on top of solver recommendation
        action, amount, exploit_note = self._exploit_postflop(
            action, amount, equity, stats, board, pot, stack, to_call, is_first_to_act
        )

        note = f"adaptive/{exploit_note}" if exploit_note else f"adaptive/{base_strategy}"
        return Action(action, amount, strategy_note=note)

    def _exploit_postflop(self, action, amount, equity, stats, board, pot, stack, to_call, is_first_to_act):
        """
        Adjust solver recommendation based on opponent's known postflop tendencies.
        Returns (action, amount, exploit_note).
        """
        if not board:
            return action, amount, ""

        t = board_texture(board[:3])
        is_dry = t["suit"] == 0 and t["connected"] == 1

        fold_to_cbet = (stats.fold_to_cbet_dry if is_dry else stats.fold_to_cbet_wet)
        gto_baseline = GTO_FOLD_TO_CBET_DRY if is_dry else GTO_FOLD_TO_CBET_WET
        cbet_dev = fold_to_cbet - gto_baseline  # positive = folds too much

        # ── OOP betting (cbet or leading) ────────────────────────────────
        if is_first_to_act and to_call == 0:
            if cbet_dev > 0.15 and action == "check":
                if random.random() < cbet_dev:
                    frac = 0.45 if equity < 0.45 else 0.65
                    return "bet", min(int(pot * frac), stack), "cbet-exploit"

            if cbet_dev < -0.15 and action == "bet":
                if equity < 0.50 and random.random() < abs(cbet_dev):
                    return "check", 0, "bluff-suppress"

        # ── Facing a bet ──────────────────────────────────────────────────
        if to_call > 0 and not is_first_to_act:
            river_dev = stats.river_fold_to_bet_pct - GTO_RIVER_FOLD
            if _board_to_street(board) == "river" and river_dev > 0.15:
                if action == "call" and equity < 0.45 and random.random() < river_dev:
                    raise_amt = min(int(pot * 0.80), stack)
                    return "raise", raise_amt, "river-bluff-exploit"

        # ── Check-raise exploitation ──────────────────────────────────────
        if is_first_to_act and to_call > 0:
            cr_dev = stats.check_raise_pct - GTO_CHECK_RAISE
            if cr_dev < -0.05 and action in ("fold", "call"):
                if equity >= 0.35 and action == "fold":
                    return "call", min(to_call, stack), "cr-exploit"

        # ── Turn probe exploitation ───────────────────────────────────────
        # Opponent probes the turn often (OOP bets turn after check-check flop)
        # → widen our calling range when facing a turn lead
        if _board_to_street(board) == "turn" and to_call > 0 and not is_first_to_act:
            probe_dev = stats.turn_probe_pct - GTO_TURN_PROBE
            if probe_dev > 0.15 and action == "fold" and equity >= 0.28:
                if random.random() < probe_dev:
                    return "call", min(to_call, stack), "probe-exploit"

        return action, amount, ""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _board_to_street(board: list[str]) -> str:
    if len(board) <= 3: return "flop"
    if len(board) == 4: return "turn"
    return "river"


def _weighted_sample(weights: dict) -> str | None:
    if not weights:
        return None
    keys = list(weights.keys())
    vals = [weights[k] for k in keys]
    total = sum(vals)
    if total == 0:
        return None
    return random.choices(keys, weights=[v / total for v in vals], k=1)[0]


def _preflop_action_to_action(sampled: str, to_call: int, last_raise: int,
                               stack: int, bb: int, facing_raise: bool,
                               position: str = "") -> Action:
    if sampled == "fold":   return Action("fold")
    if sampled == "call":   return Action("call", min(to_call, stack))
    if sampled == "allin":  return Action("allin", stack)
    amt = three_bet_size(last_raise, stack) if facing_raise else open_raise_size(stack, bb, position)
    return Action("raise", min(amt, stack))
