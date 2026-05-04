"""
Core game engine — UI-agnostic. Emits events via a callback so any
frontend (CLI, WebSocket, test harness) can consume them.
"""
import random
import sys
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BUY_IN, SMALL_BLIND, BIG_BLIND, DEFAULT_BOT_CONFIGS
from bots.base import Action
from bots.gto import GTOBot
from bots.whale import WhaleBot
from bots.nit import NitBot
from bots.adaptive import AdaptiveBot
from engine.history import HandHistoryWriter
from strategy.tracker import OpponentTracker

RANKS = "23456789TJQKA"
SUITS = "cdhs"

# Position names by player count, keyed by relative index from button (0=BTN)
_POSITIONS = {
    2: {0: "BTN", 1: "BB"},
    3: {0: "BTN", 1: "SB", 2: "BB"},
    4: {0: "BTN", 1: "SB", 2: "BB", 3: "CO"},
    5: {0: "BTN", 1: "SB", 2: "BB", 3: "HJ", 4: "CO"},
    6: {0: "BTN", 1: "SB", 2: "BB", 3: "LJ", 4: "HJ", 5: "CO"},
    7: {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "LJ", 5: "HJ", 6: "CO"},
    8: {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "UTG1", 5: "LJ", 6: "HJ", 7: "CO"},
    9: {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "UTG1", 5: "UTG2", 6: "LJ", 7: "HJ", 8: "CO"},
}

_EP_POSITIONS = {"UTG", "UTG+1", "UTG+2"}
_MP_POSITIONS = {"LJ", "HJ"}
_LP_POSITIONS = {"CO", "BTN"}


def _pos_bucket(position: str) -> str:
    p = position.upper()
    if p in _EP_POSITIONS:
        return "ep"
    if p in _MP_POSITIONS:
        return "mp"
    if p in _LP_POSITIONS:
        return "lp"
    return "blinds"



def _make_deck() -> list[str]:
    deck = [r + s for r in RANKS for s in SUITS]
    random.shuffle(deck)
    return deck


def _make_bot(bot_type: str, name: str, seat: int, tracker: OpponentTracker, human_seat: int):
    if bot_type == "gto":
        return GTOBot(name, seat)
    if bot_type == "whale":
        return WhaleBot(name, seat)
    if bot_type == "nit":
        return NitBot(name, seat)
    if bot_type == "adaptive":
        return AdaptiveBot(name, seat, tracker, human_seat)
    return GTOBot(name, seat)


@dataclass
class SeatState:
    seat: int
    name: str
    is_human: bool
    stack: int = BUY_IN
    hole_cards: list = field(default_factory=list)
    current_bet: int = 0
    total_in: int = 0
    folded: bool = False
    all_in: bool = False
    position: str = ""

    def is_active(self) -> bool:
        return not self.folded and self.stack > 0

    def to_dict(self, reveal: bool = False) -> dict:
        show = self.is_human or reveal
        return {
            "seat": self.seat,
            "name": self.name,
            "is_human": self.is_human,
            "stack": self.stack,
            "current_bet": self.current_bet,
            "folded": self.folded,
            "all_in": self.all_in,
            "position": self.position,
            "hole_cards": self.hole_cards if show else (
                ["??", "??"] if self.hole_cards else []
            ),
            "hole_cards_revealed": self.hole_cards,
        }


class PokerGame:
    def __init__(
        self,
        human_name: str = "Hero",
        event_cb: Optional[Callable] = None,
        bot_configs: Optional[list[dict]] = None,  # [{"name": ..., "type": ...}, ...]
        bot_delay: float = 1.0,
    ):
        self.human_name = human_name
        self.event_cb = event_cb or (lambda e, d: None)
        self.bot_delay = bot_delay
        self.tracker = OpponentTracker()
        self.hand_num = 0
        self.button = 0
        self.history = HandHistoryWriter()

        self.god_mode: bool = False  # set after construction by server if requested
        configs = bot_configs if bot_configs is not None else DEFAULT_BOT_CONFIGS

        # Seat 0 = human, 1..N = bots
        self.seats: list[SeatState] = [SeatState(0, human_name, is_human=True)]
        for i, cfg in enumerate(configs, start=1):
            self.seats.append(SeatState(i, cfg["name"], is_human=False))

        n_players = len(self.seats)
        self.bots = {}
        for i, cfg in enumerate(configs, start=1):
            bot = _make_bot(cfg["type"], cfg["name"], i, self.tracker, 0)
            bot.n_players = n_players
            self.bots[i] = bot

        # Register human name for persistent tracker (loads prior session data)
        self.tracker.set_player_name(0, human_name)

        # P&L history: [{hand_num, stack, position, net}]
        self._start_stack: int = BUY_IN
        self.pl_history: list[dict] = []

        # Hand state
        self.board: list[str] = []
        self.pot: int = 0
        self.street: str = "preflop"
        self.deck: list[str] = []
        self.current_actor: Optional[int] = None
        self.to_call: int = 0
        self.min_raise: int = BIG_BLIND
        self.last_raiser: Optional[int] = None
        self.last_raise_amount: int = 0
        self.raise_position: str = "btn"
        self.action_log: list[str] = []
        self.waiting_for_human: bool = False
        self._human_action_queue: Optional[Action] = None
        self._human_turn_probe_opp: bool = False

    # ─── Public interface ────────────────────────────────────────────────

    def game_state(self) -> dict:
        return {
            "hand_num": self.hand_num,
            "pot": self.pot,
            "board": self.board,
            "street": self.street,
            "seats": [s.to_dict(reveal=self.god_mode) for s in self.seats],
            "current_actor": self.current_actor,
            "to_call": self.to_call,
            "min_raise": self.min_raise,
            "action_log": list(self.action_log[-20:]),
            "waiting_for_human": self.waiting_for_human,
            "button": self.button,
            "n_players": len(self.seats),
            "pl_history": self.pl_history[-200:],   # capped for wire size
            "player_stats": self.tracker.stats_dict(0),
        }

    def submit_human_action(self, action_type: str, amount: int = 0):
        """Called by the web layer when the human clicks an action button."""
        if self.waiting_for_human and self._human_action_queue is None:
            self._human_action_queue = Action(action_type, amount)

    def start_hand(self):
        self.hand_num += 1
        self.action_log = []
        self.board = []
        self.pot = 0
        self.street = "preflop"
        self.deck = _make_deck()

        for s in self.seats:
            s.hole_cards = []
            s.current_bet = 0
            s.total_in = 0
            s.folded = s.stack == 0  # busted players sit out for this hand
            s.all_in = False

        self._street_history: dict[str, list[str]] = {}
        self._assign_positions()
        self._emit("hand_start", {"hand_num": self.hand_num, "button": self.button})
        self._record_history_header()
        self._deal_hole_cards()
        self._run_preflop()

    def _run_preflop(self):
        self.street = "preflop"
        self._post_blinds()
        self._emit("street", {"street": "preflop", "board": []})
        self._betting_round(preflop=True)
        if self._one_left():
            self._award_pot()
            return
        self._deal_flop()

    def _deal_flop(self):
        self.street = "flop"
        self._collect_bets()
        flop = [self.deck.pop(), self.deck.pop(), self.deck.pop()]
        self.board.extend(flop)
        self.history.street("FLOP", flop)
        self._emit("street", {"street": "flop", "board": self.board})
        self._betting_round()
        self._street_history['flop'] = list(self._street_action_seq)
        if self._one_left():
            self._award_pot()
            return
        self._deal_turn()

    def _deal_turn(self):
        self.street = "turn"
        self._collect_bets()
        turn = self.deck.pop()
        self.board.append(turn)
        self.history.street("TURN", [turn])
        self._emit("street", {"street": "turn", "board": self.board})
        self._betting_round()
        self._street_history['turn'] = list(self._street_action_seq)
        if self._one_left():
            self._award_pot()
            return
        self._deal_river()

    def _deal_river(self):
        self.street = "river"
        self._collect_bets()
        river = self.deck.pop()
        self.board.append(river)
        self.history.street("RIVER", [river])
        self._emit("street", {"street": "river", "board": self.board})
        self._betting_round()
        if self._one_left():
            self._award_pot()
            return
        self._showdown()

    # ─── Betting round ───────────────────────────────────────────────────

    # Max raises per street before bots must call or fold (human can always raise).
    # Prevents infinite re-raise loops between bots.
    _BOT_RAISE_CAP = 5

    def _betting_round(self, preflop: bool = False):
        self.to_call = BIG_BLIND if preflop else 0
        self.min_raise = BIG_BLIND * 2
        self.last_raiser = None
        self.last_raise_amount = BIG_BLIND
        self._street_action_seq: list[str] = []

        # Skip only when nobody can act at all, or the sole remaining player
        # has already matched the bet (postflop check or preflop call).
        can_act = [s for s in self.seats if not s.folded and not s.all_in and s.stack > 0]
        if len(can_act) == 0:
            return
        if len(can_act) == 1 and can_act[0].current_bet >= self.to_call:
            return

        order = self._action_order(preflop)
        n = len(order)
        if n == 0:
            return
        street_raises = 0  # total raises this street (used for bot raise cap)

        acted = set()
        if preflop:
            # SB is "acted" because they already posted; current_bet < to_call keeps them in still_to_act.
            # BB is NOT in acted so they always get the option to raise even if everyone just limps.
            acted = {self.seats[self._sb_idx()].seat}

        pos = 0  # ring-buffer pointer into order[]
        for _ in range(max(200, n * (n + 20))):
            active_seats = [s for s in order if not s.folded and not s.all_in]

            still_to_act = [s for s in active_seats if s.seat not in acted or s.current_bet < self.to_call]
            if not still_to_act:
                break

            # Advance pos to the next player in still_to_act (preserving action order)
            still_seats = {s.seat for s in still_to_act}
            seat = None
            for offset in range(n):
                candidate = order[(pos + offset) % n]
                if candidate.seat in still_seats:
                    seat = candidate
                    pos = (pos + offset + 1) % n
                    break

            if seat is None:
                break

            action = self._get_action(seat, preflop)

            # Clamp bot raises to call once the raise cap is hit (human can always raise)
            if (not seat.is_human
                    and action.type in ("raise", "bet")
                    and street_raises >= self._BOT_RAISE_CAP):
                call_amt = min(self.to_call - seat.current_bet, seat.stack)
                action = Action("call", call_amt)

            acted.add(seat.seat)
            note = getattr(action, "strategy_note", "")

            if action.type == "fold":
                seat.folded = True
                self._street_action_seq.append("fold")
                self._log_action(seat.name, "folds")
                self.history.action(seat.name, "folds", strategy_note=note)
                self.tracker.record_action(seat.seat, "fold")
                self._emit("action", {
                    "player": seat.name, "action": "fold", "amount": 0,
                    "strategy_note": note,
                })
                if self._one_left():
                    break

            elif action.type in ("call", "check"):
                if action.type == "call":
                    call_amt = min(self.to_call - seat.current_bet, seat.stack)
                    seat.stack -= call_amt
                    seat.current_bet += call_amt
                    seat.total_in += call_amt
                    self.pot += call_amt
                    if seat.stack == 0:
                        seat.all_in = True
                    self._street_action_seq.append("call")
                    self._log_action(seat.name, f"calls ${call_amt:,}")
                    self.history.action(seat.name, "calls", call_amt, strategy_note=note)
                    self.tracker.record_action(seat.seat, "call")
                    self._emit("action", {
                        "player": seat.name, "action": "call", "amount": call_amt,
                        "strategy_note": note,
                    })
                    if seat.is_human:
                        bucket = _pos_bucket(seat.position)
                        self.tracker.record_vpip(0)
                        self.tracker.record_vpip_pos(0, bucket)
                        if preflop:
                            raise_count = self._street_action_seq.count("raise")
                            if raise_count == 0 and self.to_call == BIG_BLIND:
                                # calling BB with no prior raise = limp
                                self.tracker.record_limp(0)
                            elif raise_count >= 2:
                                # calling a 3-bet (or higher)
                                self.tracker.record_call_3bet(0)
                else:
                    self._street_action_seq.append("check")
                    self._log_action(seat.name, "checks")
                    self.history.action(seat.name, "checks", strategy_note=note)
                    self._emit("action", {
                        "player": seat.name, "action": "check", "amount": 0,
                        "strategy_note": note,
                    })

            elif action.type in ("raise", "bet", "allin"):
                amt = action.amount
                if action.type == "allin":
                    # total commitment this street = already in + remaining chips
                    amt = seat.current_bet + seat.stack
                # Enforce min-raise unless the player is going all-in
                can_min_raise = seat.current_bet + seat.stack >= self.min_raise
                if action.type != "allin" and can_min_raise and amt < self.min_raise:
                    amt = self.min_raise
                # Guard: treat sub-call raise as a call
                if amt < self.to_call:
                    amt = self.to_call
                additional = amt - seat.current_bet
                additional = max(0, min(additional, seat.stack))
                seat.stack -= additional
                seat.total_in += additional
                self.pot += additional
                self.last_raise_amount = max(BIG_BLIND, amt - self.to_call)
                self.to_call = amt
                self.min_raise = self.to_call + self.last_raise_amount
                self.last_raiser = seat.seat
                self.raise_position = seat.position.lower()
                seat.current_bet = amt
                if seat.stack == 0:
                    seat.all_in = True
                self._street_action_seq.append("allin" if action.type == "allin" else "raise")
                street_raises += 1
                label = "raises to" if self.to_call > 0 else "bets"
                self._log_action(seat.name, f"{label} ${amt:,}")
                self.history.action(seat.name, label, amt, strategy_note=note)
                self.tracker.record_action(seat.seat, "raise")
                self._emit("action", {
                    "player": seat.name, "action": label, "amount": amt,
                    "strategy_note": note,
                })
                if seat.is_human:
                    bucket = _pos_bucket(seat.position)
                    self.tracker.record_vpip(0)
                    self.tracker.record_vpip_pos(0, bucket)
                    self.tracker.record_pfr(0)
                    if getattr(self, "_human_turn_probe_opp", False):
                        self.tracker.record_turn_probe(0)
                        self._human_turn_probe_opp = False
                # Discard all other active players from acted so they must respond
                for s in order:
                    if s.seat != seat.seat and not s.folded and not s.all_in:
                        acted.discard(s.seat)

    def _get_action(self, seat: SeatState, preflop: bool) -> Action:
        if seat.is_human:
            return self._get_human_action(seat, preflop)
        self._emit("bot_acting", {"name": seat.name, "seat": seat.seat})
        if self.bot_delay > 0:
            import time as _time
            _time.sleep(self.bot_delay)
        return self._get_bot_action(seat, preflop)

    def _get_human_action(self, seat: SeatState, preflop: bool) -> Action:
        # Record opportunity stats before prompting
        if preflop:
            raise_count = self._street_action_seq.count("raise")
            if raise_count == 0 and self.to_call == BIG_BLIND:
                self.tracker.record_limp_opp(0)
            elif raise_count >= 2:
                self.tracker.record_3bet_call_opp(0)
        elif self.street == "turn" and self.last_raiser is None:
            # Turn probe opportunity: flop went check-check and human acts first on turn
            flop_seq = self._street_history.get("flop", [])
            if flop_seq and all(a == "check" for a in flop_seq):
                self.tracker.record_turn_probe_opp(0)
                self._human_turn_probe_opp = True
            else:
                self._human_turn_probe_opp = False
        else:
            self._human_turn_probe_opp = False

        self._human_action_queue = None  # discard any stale value before waiting
        self.current_actor = seat.seat
        self.waiting_for_human = True
        self._emit("your_turn", {
            "to_call": max(0, self.to_call - seat.current_bet),
            "pot": self.pot,
            "stack": seat.stack,
            "min_raise": self.min_raise,
            "can_check": self.to_call <= seat.current_bet,
            "board": self.board,
            "hole_cards": seat.hole_cards,
            "state": self.game_state(),
        })
        import time
        timeout = 300
        elapsed = 0
        while self._human_action_queue is None and elapsed < timeout:
            time.sleep(0.1)
            elapsed += 0.1

        self.waiting_for_human = False
        action = self._human_action_queue or Action("fold")
        self._human_action_queue = None
        self.current_actor = None
        return action

    def _get_bot_action(self, seat: SeatState, preflop: bool) -> Action:
        bot = self.bots[seat.seat]
        bot.set_cards(seat.hole_cards)
        to_call = max(0, self.to_call - seat.current_bet)
        facing_raise = self.last_raiser is not None

        if preflop:
            n = len(self.seats)
            rel = (seat.seat - self.button) % n
            player_idx = rel if n == 2 else (rel - 1) % n
            return bot.decide_preflop(
                position=seat.position.lower(),
                stack=seat.stack,
                pot=self.pot,
                to_call=to_call,
                facing_raise=facing_raise,
                raise_position=self.raise_position,
                last_raise=self.to_call,
                bb=BIG_BLIND,
                action_sequence=list(self._street_action_seq),
                player_idx=player_idx,
                stacks=[s.stack for s in self.seats],
            )
        else:
            prev = []
            if self.street in ('turn', 'river'):
                prev.append(self._street_history.get('flop', []))
            if self.street == 'river':
                prev.append(self._street_history.get('turn', []))
            n_active = sum(1 for s in self.seats if not s.folded and not s.all_in)
            # Determine this player's index in the postflop action order (0=OOP/first, 1=middle, 2=IP/last)
            postflop_order = self._action_order(preflop=False)
            active_postflop = [s for s in postflop_order if not s.folded and not s.all_in]
            postflop_player_idx = next(
                (i for i, s in enumerate(active_postflop) if s.seat == seat.seat), 0
            )
            return bot.decide_postflop(
                board=self.board,
                position=seat.position.lower(),
                stack=seat.stack,
                pot=self.pot,
                to_call=to_call,
                is_first_to_act=(postflop_player_idx == 0),
                action_sequence=list(self._street_action_seq),
                prev_street_actions=prev,
                n_active=n_active,
                postflop_player_idx=postflop_player_idx,
            )

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _assign_positions(self):
        n = len(self.seats)
        pos_map = _POSITIONS.get(n, {})
        for i, seat in enumerate(self.seats):
            rel = (i - self.button) % n
            seat.position = pos_map.get(rel, f"P{rel}")

    def _sb_idx(self) -> int:
        n = len(self.seats)
        if n == 2:
            return self.button  # BTN = SB in HU
        return (self.button + 1) % n

    def _bb_idx(self) -> int:
        n = len(self.seats)
        if n == 2:
            return (self.button + 1) % n
        return (self.button + 2) % n

    def _post_blinds(self):
        sb_seat = self.seats[self._sb_idx()]
        bb_seat = self.seats[self._bb_idx()]

        sb_amt = min(SMALL_BLIND, sb_seat.stack)
        sb_seat.stack -= sb_amt
        sb_seat.current_bet = sb_amt
        sb_seat.total_in = sb_amt
        self.pot += sb_amt

        bb_amt = min(BIG_BLIND, bb_seat.stack)
        bb_seat.stack -= bb_amt
        bb_seat.current_bet = bb_amt
        bb_seat.total_in = bb_amt
        self.pot += bb_amt

        self.to_call = bb_amt
        self.history.post_blinds(sb_seat.name, bb_seat.name, sb_amt, bb_amt)
        self._emit("blinds", {
            "sb": sb_seat.name, "bb": bb_seat.name, "pot": self.pot,
            "seats": [s.to_dict(reveal=False) for s in self.seats],
        })

    def _deal_hole_cards(self):
        for seat in self.seats:
            if seat.stack == 0:
                continue
            seat.hole_cards = [self.deck.pop(), self.deck.pop()]
            self.bots.get(seat.seat) and self.bots[seat.seat].set_cards(seat.hole_cards)
            if seat.is_human:
                self.tracker.record_hand_dealt(seat.seat)
                self.tracker.record_hand_dealt_pos(seat.seat, _pos_bucket(seat.position))

        human_cards = self.seats[0].hole_cards
        self.history.hole_cards(self.human_name, human_cards[0], human_cards[1])
        self._emit("deal", {
            "hole_cards": human_cards,
            "seats": [s.to_dict(reveal=self.god_mode) for s in self.seats],
        })

    def _action_order(self, preflop: bool) -> list[SeatState]:
        n = len(self.seats)
        if preflop:
            # HU: BTN/SB acts first preflop; 3-handed: BTN acts first
            if n <= 3:
                start = self.button
            else:
                start = (self.button + 3) % n  # UTG
        else:
            if n == 2:
                start = (self.button + 1) % n  # BB acts first postflop
            else:
                start = (self.button + 1) % n  # SB first postflop
        return [self.seats[(start + i) % n] for i in range(n) if self.seats[(start + i) % n].stack > 0]

    def _one_left(self) -> bool:
        return sum(1 for s in self.seats if not s.folded) <= 1

    def _collect_bets(self):
        for s in self.seats:
            s.current_bet = 0

    def _side_pots(self) -> list[tuple[int, list]]:
        """
        Returns [(amount, [eligible_seats]), ...] from main pot to side pots.
        Each pot can only be won by players who contributed at least that level.
        """
        contenders = [s for s in self.seats if not s.folded]
        levels = sorted(set(s.total_in for s in self.seats if s.total_in > 0))
        pots, prev = [], 0
        for level in levels:
            size = sum(min(s.total_in, level) - min(s.total_in, prev) for s in self.seats)
            eligible = [s for s in contenders if s.total_in >= level]
            if size > 0 and eligible:
                pots.append((size, eligible))
            prev = level
        return pots

    def _award_pot(self, winner: Optional[SeatState] = None):
        if winner is None:
            remaining = [s for s in self.seats if not s.folded]
            winner = remaining[0] if remaining else self.seats[0]

        winner.stack += self.pot
        self._log_action(winner.name, f"wins ${self.pot:,}")
        self.history.collected(winner.name, self.pot)
        self.history.summary(self.pot, self.board, [(winner.name, self.pot)])
        self.history.flush_hand()

        # Record P&L snapshot for the human player
        human = self.seats[0]
        self.pl_history.append({
            "hand_num": self.hand_num,
            "stack": human.stack,
            "net": human.stack - self._start_stack,
            "position": human.position,
        })

        # Persist tracker after every hand
        self.tracker.save(0)

        self._emit("winner", {
            "player": winner.name,
            "amount": self.pot,
            "hand_desc": "",
            "seats": [s.to_dict(reveal=self.god_mode) for s in self.seats],
            "pl_history": self.pl_history[-200:],
            "player_stats": self.tracker.stats_dict(0),
        })
        self.pot = 0
        self.button = (self.button + 1) % len(self.seats)
        self._check_game_over()

    def _showdown(self):
        """Reveal all hands and award pot(s) using treys. Handles side pots."""
        try:
            from treys import Evaluator, Card as TreysCard
            evaluator = Evaluator()
            board_treys = [TreysCard.new(c[0] + c[1].lower()) for c in self.board]
            scores: dict[int, int] = {}
            for s in self.seats:
                if not s.folded:
                    try:
                        hand = [TreysCard.new(c[0] + c[1].lower()) for c in s.hole_cards]
                        scores[s.seat] = evaluator.evaluate(board_treys, hand)
                    except Exception:
                        scores[s.seat] = 9999
            lower_is_better = True
        except Exception:
            from strategy.solver import _fallback_heuristic
            scores = {s.seat: _fallback_heuristic(s.hole_cards, self.board)
                      for s in self.seats if not s.folded}
            lower_is_better = False  # higher is better for fallback heuristic

        reveal_data = [
            {"name": s.name, "cards": s.hole_cards}
            for s in self.seats if not s.folded
        ]
        self._emit("showdown", {"players": reveal_data, "board": self.board})

        # Distribute each pot to the best eligible hand
        pots = self._side_pots()
        summary_awards: list[tuple[str, int]] = []
        for pot_size, eligible in pots:
            if lower_is_better:
                winner = min(eligible, key=lambda s: scores.get(s.seat, 9999))
            else:
                winner = max(eligible, key=lambda s: scores.get(s.seat, 0))
            winner.stack += pot_size
            self._log_action(winner.name, f"wins ${pot_size:,}")
            self.history.collected(winner.name, pot_size)
            summary_awards.append((winner.name, pot_size))

        self.history.summary(self.pot, self.board, summary_awards)
        self.history.flush_hand()

        # Use the main pot winner for the winner event
        main_winner_name = summary_awards[0][0] if summary_awards else self.seats[0].name
        human = self.seats[0]
        self.pl_history.append({
            "hand_num": self.hand_num,
            "stack": human.stack,
            "net": human.stack - self._start_stack,
            "position": human.position,
        })
        self.tracker.save(0)
        self._emit("winner", {
            "player": main_winner_name,
            "amount": self.pot,
            "hand_desc": "",
            "seats": [s.to_dict(reveal=self.god_mode) for s in self.seats],
            "pl_history": self.pl_history[-200:],
            "player_stats": self.tracker.stats_dict(0),
        })
        self.pot = 0
        self.button = (self.button + 1) % len(self.seats)
        self._check_game_over()

    def _check_game_over(self):
        human = self.seats[0]
        bots_alive = [s for s in self.seats[1:] if s.stack > 0]
        if human.stack == 0:
            self._emit("game_over", {"reason": "bust", "winner": None})
        elif not bots_alive:
            self._emit("game_over", {"reason": "all_bots_bust", "winner": human.name})

    def _record_history_header(self):
        stacks = {s.name: s.stack for s in self.seats}
        self.history.begin_hand(self.hand_num, self.button, stacks)

    def _log_action(self, player: str, msg: str):
        self.action_log.append(f"{player}: {msg}")

    def _emit(self, event: str, data: dict):
        self.event_cb(event, data)
