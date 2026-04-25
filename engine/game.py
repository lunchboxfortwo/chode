"""
Core game engine — UI-agnostic. Emits events via a callback so any
frontend (CLI, WebSocket, test harness) can consume them.
"""
import random
import sys
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

# Ensure project root is on path when imported from web server
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BUY_IN, SMALL_BLIND, BIG_BLIND, POSITIONS_6MAX, BOT_NAMES, BOT_TYPES
from bots.base import Action
from bots.gto import GTOBot
from bots.whale import WhaleBot
from bots.nit import NitBot
from bots.adaptive import AdaptiveBot
from engine.history import HandHistoryWriter
from strategy.tracker import OpponentTracker

RANKS = "23456789TJQKA"
SUITS = "cdhs"

_POS_TO_SPIEL_IDX = {"SB": 0, "BB": 1, "UTG": 2, "HJ": 3, "CO": 4, "BTN": 5}


def _position_to_openspiel_idx(position: str) -> int:
    return _POS_TO_SPIEL_IDX.get(position.upper(), 2)


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

    def to_dict(self) -> dict:
        return {
            "seat": self.seat,
            "name": self.name,
            "is_human": self.is_human,
            "stack": self.stack,
            "current_bet": self.current_bet,
            "folded": self.folded,
            "all_in": self.all_in,
            "position": self.position,
            "hole_cards": self.hole_cards if self.is_human else (
                ["??", "??"] if self.hole_cards else []
            ),
            "hole_cards_revealed": self.hole_cards,
        }


class PokerGame:
    def __init__(self, human_name: str = "Hero", event_cb: Optional[Callable] = None):
        self.human_name = human_name
        self.event_cb = event_cb or (lambda e, d: None)
        self.tracker = OpponentTracker()
        self.hand_num = 0
        self.button = 0  # seat index of button
        self.history = HandHistoryWriter()

        # Seats: 0 = human, 1-5 = bots
        self.seats: list[SeatState] = [
            SeatState(0, human_name, is_human=True)
        ]
        for i, (name, btype) in enumerate(zip(BOT_NAMES, BOT_TYPES), start=1):
            self.seats.append(SeatState(i, name, is_human=False))

        n_players = 1 + len(BOT_NAMES)
        self.bots = {}
        for i, (name, btype) in enumerate(zip(BOT_NAMES, BOT_TYPES), start=1):
            bot = _make_bot(btype, name, i, self.tracker, 0)
            bot.n_players = n_players
            self.bots[i] = bot

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

    # ─── Public interface ────────────────────────────────────────────────

    def game_state(self) -> dict:
        return {
            "hand_num": self.hand_num,
            "pot": self.pot,
            "board": self.board,
            "street": self.street,
            "seats": [s.to_dict() for s in self.seats],
            "current_actor": self.current_actor,
            "to_call": self.to_call,
            "min_raise": self.min_raise,
            "action_log": list(self.action_log[-20:]),
            "waiting_for_human": self.waiting_for_human,
            "button": self.button,
        }

    def submit_human_action(self, action_type: str, amount: int = 0):
        """Called by the web layer when the human clicks an action button."""
        self._human_action_queue = Action(action_type, amount)

    def start_hand(self):
        self.hand_num += 1
        self.action_log = []
        self.board = []
        self.pot = 0
        self.street = "preflop"
        self.deck = _make_deck()

        # Reset per-hand seat state
        for s in self.seats:
            s.hole_cards = []
            s.current_bet = 0
            s.total_in = 0
            s.folded = False
            s.all_in = False

        # Bust check + rebuy bots
        for s in self.seats[1:]:
            if s.stack == 0:
                s.stack = BUY_IN
                self._emit("rebuy", {"name": s.name, "stack": BUY_IN})

        self._street_history: dict[str, list[str]] = {}  # completed street action seqs
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

    def _betting_round(self, preflop: bool = False):
        active = [s for s in self.seats if not s.folded and not s.all_in]
        if len(active) <= 1:
            return

        self.to_call = BIG_BLIND if preflop else 0
        self.min_raise = BIG_BLIND * 2
        self.last_raiser = None
        self.last_raise_amount = BIG_BLIND
        self._street_action_seq: list[str] = []  # ordered action history this street

        # Determine action order
        order = self._action_order(preflop)

        # Track who still needs to act (tracks raises)
        acted = set()
        if preflop:
            # SB and BB already acted (posted)
            sb_seat = self.seats[self._sb_idx()].seat
            bb_seat = self.seats[self._bb_idx()].seat
            acted = {sb_seat, bb_seat}

        i = 0
        while True:
            active_seats = [s for s in order if not s.folded and not s.all_in]
            if len(active_seats) <= 1:
                break

            seat = order[i % len(order)]
            i += 1

            if seat.folded or seat.all_in:
                if i > len(order) * 3:
                    break
                continue

            # Check if everyone has acted and no pending call
            still_to_act = [s for s in active_seats if s.seat not in acted or s.current_bet < self.to_call]
            if not still_to_act:
                break

            if seat.seat not in [s.seat for s in still_to_act]:
                if len(still_to_act) == 0:
                    break
                # Skip to next player that needs to act
                found = False
                for candidate in still_to_act:
                    if candidate.seat == order[i % len(order)].seat:
                        found = True
                        break
                if not found:
                    i_next = next((j for j, s in enumerate(order) if s.seat == still_to_act[0].seat), None)
                    if i_next is not None:
                        i = i_next
                continue

            action = self._get_action(seat, preflop)
            acted.add(seat.seat)

            if action.type == "fold":
                seat.folded = True
                self._street_action_seq.append("fold")
                self._log_action(seat.name, "folds")
                self.history.action(seat.name, "folds")
                self.tracker.record_action(seat.seat, "fold")
                self._emit("action", {"player": seat.name, "action": "fold", "amount": 0})
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
                    self.history.action(seat.name, "calls", call_amt)
                    self.tracker.record_action(seat.seat, "call")
                    self._emit("action", {"player": seat.name, "action": "call", "amount": call_amt})
                    if seat.is_human:
                        self.tracker.record_vpip(0)
                else:
                    self._street_action_seq.append("check")
                    self._log_action(seat.name, "checks")
                    self.history.action(seat.name, "checks")
                    self._emit("action", {"player": seat.name, "action": "check", "amount": 0})

            elif action.type in ("raise", "bet", "allin"):
                amt = action.amount
                if action.type == "allin":
                    amt = seat.stack
                additional = amt - seat.current_bet
                additional = min(additional, seat.stack)
                seat.stack -= additional
                seat.total_in += additional
                self.pot += additional
                self.last_raise_amount = amt - self.to_call
                self.to_call = amt
                self.min_raise = self.to_call + self.last_raise_amount
                self.last_raiser = seat.seat
                seat.current_bet = amt
                if seat.stack == 0:
                    seat.all_in = True
                self._street_action_seq.append("allin" if action.type == "allin" else "raise")
                label = "raises to" if self.to_call > 0 else "bets"
                self._log_action(seat.name, f"{label} ${amt:,}")
                self.history.action(seat.name, f"{label}", amt)
                self.tracker.record_action(seat.seat, "raise")
                self._emit("action", {"player": seat.name, "action": label, "amount": amt})
                if seat.is_human:
                    self.tracker.record_vpip(0)
                    self.tracker.record_pfr(0)
                # Everyone else needs to re-act
                for s in active_seats:
                    if s.seat != seat.seat:
                        acted.discard(s.seat)

    def _get_action(self, seat: SeatState, preflop: bool) -> Action:
        if seat.is_human:
            return self._get_human_action(seat, preflop)
        return self._get_bot_action(seat, preflop)

    def _get_human_action(self, seat: SeatState, preflop: bool) -> Action:
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
        # Blocking wait for web layer to call submit_human_action
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
                player_idx=_position_to_openspiel_idx(seat.position),
            )
        else:
            prev = []
            if self.street in ('turn', 'river'):
                prev.append(self._street_history.get('flop', []))
            if self.street == 'river':
                prev.append(self._street_history.get('turn', []))
            return bot.decide_postflop(
                board=self.board,
                position=seat.position.lower(),
                stack=seat.stack,
                pot=self.pot,
                to_call=to_call,
                is_first_to_act=(self.last_raiser is None),
                action_sequence=list(self._street_action_seq),
                prev_street_actions=prev,
            )

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _assign_positions(self):
        n = len(self.seats)
        pos_map = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO"}
        for i, seat in enumerate(self.seats):
            rel = (i - self.button) % n
            seat.position = pos_map.get(rel, str(rel))

    def _sb_idx(self) -> int:
        return (self.button + 1) % len(self.seats)

    def _bb_idx(self) -> int:
        return (self.button + 2) % len(self.seats)

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
        self._emit("blinds", {"sb": sb_seat.name, "bb": bb_seat.name, "pot": self.pot})

    def _deal_hole_cards(self):
        for seat in self.seats:
            seat.hole_cards = [self.deck.pop(), self.deck.pop()]
            self.bots.get(seat.seat) and self.bots[seat.seat].set_cards(seat.hole_cards)
            if seat.is_human:
                self.tracker.record_hand_dealt(seat.seat)

        human_cards = self.seats[0].hole_cards
        self.history.hole_cards(self.human_name, human_cards[0], human_cards[1])
        self._emit("deal", {
            "hole_cards": human_cards,
            "seats": [s.to_dict() for s in self.seats],
        })

    def _action_order(self, preflop: bool) -> list[SeatState]:
        n = len(self.seats)
        if preflop:
            start = (self.button + 3) % n  # UTG first
        else:
            start = (self.button + 1) % n  # SB first
        order = []
        for i in range(n):
            order.append(self.seats[(start + i) % n])
        return order

    def _one_left(self) -> bool:
        return sum(1 for s in self.seats if not s.folded) <= 1

    def _collect_bets(self):
        for s in self.seats:
            s.current_bet = 0

    def _award_pot(self, winner: Optional[SeatState] = None):
        if winner is None:
            remaining = [s for s in self.seats if not s.folded]
            winner = remaining[0] if remaining else self.seats[0]

        winner.stack += self.pot
        self._log_action(winner.name, f"wins ${self.pot:,}")
        self.history.collected(winner.name, self.pot)
        self.history.summary(self.pot, self.board, [(winner.name, self.pot)])
        self.history.flush_hand()
        self._emit("winner", {
            "player": winner.name,
            "amount": self.pot,
            "hand_desc": "",
            "seats": [s.to_dict() for s in self.seats],
        })
        self.pot = 0
        self.button = (self.button + 1) % len(self.seats)
        self._check_game_over()

    def _showdown(self):
        """Reveal all hands and award pot using simple rank-based heuristic."""
        from strategy.solver import _fallback_heuristic as _hand_strength_heuristic
        remaining = [(s, _hand_strength_heuristic(s.hole_cards, self.board))
                     for s in self.seats if not s.folded]
        remaining.sort(key=lambda x: x[1], reverse=True)
        winner_seat, equity = remaining[0]

        # Reveal cards
        reveal_data = [
            {"name": s.name, "cards": s.hole_cards, "equity": eq}
            for s, eq in remaining
        ]
        self._emit("showdown", {"players": reveal_data, "board": self.board})
        self._award_pot(winner_seat)

    def _check_game_over(self):
        human = self.seats[0]
        bots_alive = [s for s in self.seats[1:] if s.stack > 0]
        if human.stack == 0:
            self._emit("game_over", {"reason": "bust", "winner": None})
        elif not bots_alive:
            self._emit("game_over", {"reason": "all_bots_bust", "winner": human.name})

    def _record_history_header(self):
        stacks = {s.name: s.stack for s in self.seats}
        btn_seat = self.button
        self.history.begin_hand(self.hand_num, btn_seat, stacks)

    def _log_action(self, player: str, msg: str):
        self.action_log.append(f"{player}: {msg}")

    def _emit(self, event: str, data: dict):
        self.event_cb(event, data)
