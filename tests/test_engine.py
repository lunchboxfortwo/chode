"""
Unit + integration tests for the game engine.

Run with:  python3 -m pytest tests/test_engine.py -v
"""
import sys, os, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from config import BUY_IN, SMALL_BLIND, BIG_BLIND
from engine.game import PokerGame, _pos_bucket, _POSITIONS, SeatState
from strategy.tracker import TRACKER_DIR


# ─── Helpers ─────────────────────────────────────────────────────────────────

_TEST_HUMAN = "__testplayer__"

@pytest.fixture(autouse=True)
def _clean_test_tracker():
    """Remove the test player's tracker file before and after each test."""
    path = TRACKER_DIR / f"{_TEST_HUMAN}.json"
    path.unlink(missing_ok=True)
    yield
    path.unlink(missing_ok=True)


def _make_game(n_bots: int = 1, bot_type: str = "gto", human_name: str = _TEST_HUMAN) -> PokerGame:
    configs = [{"name": f"Bot{i}", "type": bot_type} for i in range(1, n_bots + 1)]
    return PokerGame(human_name=human_name, bot_configs=configs)


def _run_hand_autofold(game: PokerGame, timeout: float = 10.0) -> list[tuple]:
    """Run one hand; human auto-folds whenever prompted. Returns events list."""
    events = []
    game.event_cb = lambda ev, d: events.append((ev, d))

    done = threading.Event()
    def _runner():
        game.start_hand()
        done.set()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()

    deadline = time.monotonic() + timeout
    while not done.is_set():
        if game.waiting_for_human:
            game.submit_human_action("fold")
        time.sleep(0.02)
        if time.monotonic() > deadline:
            raise TimeoutError("hand did not complete")

    done.wait(timeout=1)
    return events


def _run_hands(game: PokerGame, n: int, timeout: float = 30.0,
               extra_cb=None) -> list[tuple]:
    """Run n hands auto-folding for human. Returns events list.
    extra_cb, if given, is called in addition to collecting events."""
    events = []
    def _cb(ev, d):
        events.append((ev, d))
        if extra_cb:
            extra_cb(ev, d)
    game.event_cb = _cb

    done = threading.Event()
    def _runner():
        for _ in range(n):
            game.start_hand()
        done.set()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()

    deadline = time.monotonic() + timeout
    while not done.is_set():
        if game.waiting_for_human:
            game.submit_human_action("fold")
        time.sleep(0.02)
        if time.monotonic() > deadline:
            raise TimeoutError("hands did not complete")

    done.wait(timeout=2)
    return events


# ─── Position helpers ─────────────────────────────────────────────────────────

class TestPosBucket:
    def test_ep(self):
        for p in ("UTG", "UTG+1", "UTG+2"):
            assert _pos_bucket(p) == "ep"

    def test_mp(self):
        for p in ("LJ", "HJ"):
            assert _pos_bucket(p) == "mp"

    def test_lp(self):
        for p in ("CO", "BTN"):
            assert _pos_bucket(p) == "lp"

    def test_blinds(self):
        for p in ("SB", "BB"):
            assert _pos_bucket(p) == "blinds"

    def test_case_insensitive(self):
        assert _pos_bucket("btn") == "lp"
        assert _pos_bucket("utg") == "ep"


class TestPositionMaps:
    def test_all_player_counts_covered(self):
        for n in range(2, 10):
            assert n in _POSITIONS, f"missing position map for {n} players"

    def test_position_counts_match_players(self):
        for n, pos_map in _POSITIONS.items():
            assert len(pos_map) == n, f"{n}-player map has {len(pos_map)} entries"

    def test_btn_always_idx_0(self):
        for n, pos_map in _POSITIONS.items():
            assert pos_map[0] == "BTN", f"{n}p: index 0 should be BTN"

    def test_2p_positions(self):
        assert _POSITIONS[2] == {0: "BTN", 1: "BB"}

    def test_6p_positions(self):
        m = _POSITIONS[6]
        assert m[5] == "CO"
        assert m[4] == "HJ"
        assert m[3] == "LJ"

    def test_8p_has_utg_plus1(self):
        assert "UTG1" in _POSITIONS[8].values()


# ─── Position assignment ──────────────────────────────────────────────────────

class TestAssignPositions:
    def test_6p_button_0(self):
        g = _make_game(5)
        g.button = 0
        g._assign_positions()
        assert g.seats[0].position == "BTN"
        assert g.seats[1].position == "SB"
        assert g.seats[2].position == "BB"

    def test_6p_button_rotates(self):
        g = _make_game(5)
        g.button = 2
        g._assign_positions()
        assert g.seats[2].position == "BTN"
        assert g.seats[3].position == "SB"
        assert g.seats[4].position == "BB"
        assert g.seats[0].position == "HJ"
        assert g.seats[1].position == "CO"

    def test_2p_positions(self):
        g = _make_game(1)
        g.button = 0
        g._assign_positions()
        assert g.seats[0].position == "BTN"
        assert g.seats[1].position == "BB"


# ─── Blind posting ───────────────────────────────────────────────────────────

class TestBlinds:
    def test_2p_blinds_from_button(self):
        g = _make_game(1)
        g.button = 0
        g._assign_positions()
        for s in g.seats:
            s.stack = BUY_IN
            s.current_bet = 0
            s.total_in = 0
        g.pot = 0
        g._post_blinds()
        # HU: BTN=SB posts SB, BB posts BB
        assert g.pot == SMALL_BLIND + BIG_BLIND

    def test_6p_pot_after_blinds(self):
        g = _make_game(5)
        g.button = 0
        g._assign_positions()
        for s in g.seats:
            s.stack = BUY_IN
            s.current_bet = 0
            s.total_in = 0
        g.pot = 0
        g._post_blinds()
        assert g.pot == SMALL_BLIND + BIG_BLIND

    def test_blinds_deducted_from_stacks(self):
        g = _make_game(5)
        g.button = 0
        g._assign_positions()
        for s in g.seats:
            s.stack = BUY_IN
            s.current_bet = 0
            s.total_in = 0
        g.pot = 0
        g._post_blinds()
        sb = g.seats[g._sb_idx()]
        bb = g.seats[g._bb_idx()]
        assert sb.stack == BUY_IN - SMALL_BLIND
        assert bb.stack == BUY_IN - BIG_BLIND


# ─── SB/BB index correctness ─────────────────────────────────────────────────

class TestBlindIndexes:
    def test_hu_sb_is_button(self):
        g = _make_game(1)
        g.button = 0
        assert g._sb_idx() == 0  # BTN=SB in HU

    def test_hu_bb_is_other(self):
        g = _make_game(1)
        g.button = 0
        assert g._bb_idx() == 1

    def test_6p_sb_is_button_plus1(self):
        g = _make_game(5)
        g.button = 0
        assert g._sb_idx() == 1

    def test_6p_bb_is_button_plus2(self):
        g = _make_game(5)
        g.button = 0
        assert g._bb_idx() == 2

    def test_wraps_correctly(self):
        g = _make_game(5)
        g.button = 5  # last seat
        assert g._sb_idx() == 0
        assert g._bb_idx() == 1


# ─── Chip conservation ───────────────────────────────────────────────────────

class TestChipConservation:
    def test_total_chips_constant_after_one_hand_2p(self):
        # Use 2-player (no heavy solver) to keep test fast
        g = _make_game(1)
        initial_total = sum(s.stack for s in g.seats)
        _run_hand_autofold(g)
        final_total = sum(s.stack for s in g.seats)
        assert final_total == initial_total

    def test_total_chips_constant_after_three_hands(self):
        g = _make_game(1)
        initial_total = sum(s.stack for s in g.seats)
        events = _run_hands(g, 3)
        # Account for bot rebuys: each rebuy injects BUY_IN chips
        rebuy_count = sum(1 for ev, _ in events if ev == "rebuy")
        final_total = sum(s.stack for s in g.seats)
        assert final_total == initial_total + rebuy_count * BUY_IN

    def test_2p_chip_conservation(self):
        g = _make_game(1)
        initial_total = sum(s.stack for s in g.seats)
        _run_hand_autofold(g)
        final_total = sum(s.stack for s in g.seats)
        assert final_total == initial_total

    def test_3p_chip_conservation(self):
        g = _make_game(2)
        initial_total = sum(s.stack for s in g.seats)
        _run_hand_autofold(g)
        final_total = sum(s.stack for s in g.seats)
        assert final_total == initial_total


# ─── P&L history ─────────────────────────────────────────────────────────────

class TestPLHistory:
    def test_pl_history_populated_after_hand(self):
        g = _make_game(1)
        _run_hand_autofold(g)
        assert len(g.pl_history) == 1

    def test_pl_history_has_required_keys(self):
        g = _make_game(1)
        _run_hand_autofold(g)
        entry = g.pl_history[0]
        assert "hand_num" in entry
        assert "stack" in entry
        assert "net" in entry
        assert "position" in entry

    def test_pl_history_hand_num_correct(self):
        g = _make_game(1)
        _run_hands(g, 3)
        assert [e["hand_num"] for e in g.pl_history] == [1, 2, 3]

    def test_pl_history_position_is_valid(self):
        g = _make_game(1)
        _run_hands(g, 4)
        valid_positions = set(_POSITIONS[2].values())
        for entry in g.pl_history:
            assert entry["position"] in valid_positions, f"unexpected position: {entry['position']}"

    def test_net_reflects_stack_change(self):
        g = _make_game(1)
        _run_hand_autofold(g)
        entry = g.pl_history[0]
        assert entry["net"] == entry["stack"] - BUY_IN


# ─── Winner event emitted ─────────────────────────────────────────────────────

class TestWinnerEvent:
    def test_winner_event_emitted(self):
        g = _make_game(1)
        events = _run_hand_autofold(g)
        winner_events = [d for ev, d in events if ev == "winner"]
        assert len(winner_events) == 1

    def test_winner_event_has_pl_history(self):
        g = _make_game(1)
        events = _run_hand_autofold(g)
        d = next(d for ev, d in events if ev == "winner")
        assert "pl_history" in d
        assert len(d["pl_history"]) >= 1

    def test_winner_event_has_player_stats(self):
        g = _make_game(1)
        events = _run_hand_autofold(g)
        d = next(d for ev, d in events if ev == "winner")
        assert "player_stats" in d
        assert "hands_dealt" in d["player_stats"]

    def test_winner_pot_positive(self):
        g = _make_game(1)
        events = _run_hand_autofold(g)
        d = next(d for ev, d in events if ev == "winner")
        assert d["amount"] > 0


# ─── Tracker integration ─────────────────────────────────────────────────────

class TestTrackerIntegration:
    def test_hands_dealt_increments(self):
        g = _make_game(1)
        _run_hands(g, 3)
        assert g.tracker.get(0).hands_dealt == 3

    def test_positional_hands_dealt_fills(self):
        g = _make_game(5, "whale")
        _run_hands(g, 6)
        s = g.tracker.get(0)
        total_pos = s.hands_ep + s.hands_mp + s.hands_lp + s.hands_blinds
        assert total_pos == s.hands_dealt

    def test_hand_dealt_pos_matches_pl_history(self):
        g = _make_game(5, "whale")
        _run_hands(g, 6)
        s = g.tracker.get(0)
        # Count positions from pl_history (tracker has no prior disk data for __testplayer__)
        from engine.game import _pos_bucket
        buckets = [_pos_bucket(e["position"]) for e in g.pl_history]
        assert buckets.count("ep") == s.hands_ep
        assert buckets.count("mp") == s.hands_mp
        assert buckets.count("lp") == s.hands_lp
        assert buckets.count("blinds") == s.hands_blinds


# ─── Button rotation ─────────────────────────────────────────────────────────

class TestButtonRotation:
    def test_button_advances_each_hand(self):
        g = _make_game(5, "whale")
        n = len(g.seats)
        buttons = []
        def _capture(ev, d):
            if ev == "hand_start":
                buttons.append(d["button"])
        _run_hands(g, n, extra_cb=_capture)
        assert len(buttons) == n
        assert buttons == list(range(n))

    def test_2p_button_rotates(self):
        g = _make_game(1)
        buttons = []
        def _capture(ev, d):
            if ev == "hand_start":
                buttons.append(d["button"])
        _run_hands(g, 4, extra_cb=_capture)
        assert buttons == [0, 1, 0, 1]


# ─── Showdown ─────────────────────────────────────────────────────────────────

class TestShowdown:
    def _eval_winner(self, h1_cards, h2_cards, board):
        """Use treys directly to get expected winner."""
        from treys import Evaluator, Card
        ev = Evaluator()
        board_t = [Card.new(c[0] + c[1].lower()) for c in board]
        s1 = ev.evaluate(board_t, [Card.new(c[0] + c[1].lower()) for c in h1_cards])
        s2 = ev.evaluate(board_t, [Card.new(c[0] + c[1].lower()) for c in h2_cards])
        return 1 if s1 < s2 else 2  # lower = better in treys

    def test_royal_flush_beats_straight(self):
        winner = self._eval_winner(
            ["Ah", "Kh"],          # royal flush with board
            ["Jd", "Jc"],          # two pair
            ["Qh", "Jh", "Th", "2c", "3d"],
        )
        assert winner == 1

    def test_full_house_beats_flush(self):
        winner = self._eval_winner(
            ["Ac", "Ad"],          # aces full
            ["2h", "5h"],          # flush
            ["As", "Kh", "Kd", "7h", "9h"],
        )
        assert winner == 1

    def test_lower_kicker_loses(self):
        winner = self._eval_winner(
            ["Ac", "Kd"],          # AK
            ["Ac", "Qd"],          # AQ (same A on board)
            ["Ad", "Jh", "Td", "2c", "3s"],
        )
        assert winner == 1


# ─── OpenSpiel player_idx formula ────────────────────────────────────────────

def _player_idx(seat: int, button: int, n: int) -> int:
    """Mirror of the formula in game.py _get_bot_action."""
    rel = (seat - button) % n
    return rel if n == 2 else (rel - 1) % n


class TestPlayerIdx:
    """The player_idx passed to the preflop solver must match OpenSpiel's
    player numbering: SB=0, BB=1, UTG=2, ..., BTN=n-1."""

    def _check_all_seats(self, n: int, button: int):
        pos_map = _POSITIONS[n]
        expected = {"SB": 0, "BB": 1}
        # UTG through CO are 2..n-2; BTN is n-1
        for rel, pos in pos_map.items():
            if pos == "BTN":
                expected[pos] = n - 1
            elif pos not in ("SB", "BB"):
                expected[pos] = rel - 1  # rel goes 0=BTN,1=SB,2=BB,3=UTG... so UTG rel=3→idx=2

        for seat in range(n):
            rel = (seat - button) % n
            pos = pos_map[rel]
            idx = _player_idx(seat, button, n)
            assert idx == expected[pos], (
                f"n={n} button={button} seat={seat} pos={pos}: "
                f"got idx={idx}, want {expected[pos]}"
            )

    def test_6p_button0(self):
        self._check_all_seats(6, 0)

    def test_6p_button3(self):
        self._check_all_seats(6, 3)

    def test_6p_button5(self):
        self._check_all_seats(6, 5)

    def test_7p_button0(self):
        self._check_all_seats(7, 0)

    def test_7p_button6(self):
        self._check_all_seats(7, 6)

    def test_8p_button4(self):
        self._check_all_seats(8, 4)

    def test_2p_btn_is_player0(self):
        # HU: BTN/SB is OpenSpiel player 0
        assert _player_idx(seat=0, button=0, n=2) == 0  # BTN
        assert _player_idx(seat=1, button=0, n=2) == 1  # BB

    def test_2p_button1(self):
        assert _player_idx(seat=1, button=1, n=2) == 0  # BTN
        assert _player_idx(seat=0, button=1, n=2) == 1  # BB

    def test_btn_always_n_minus_1_for_ngt2(self):
        for n in range(3, 10):
            for button in range(n):
                idx = _player_idx(seat=button, button=button, n=n)
                assert idx == n - 1, f"BTN should be player {n-1} in {n}p, got {idx}"

    def test_sb_always_player0_for_ngt2(self):
        for n in range(3, 10):
            for button in range(n):
                sb_seat = (button + 1) % n
                idx = _player_idx(seat=sb_seat, button=button, n=n)
                assert idx == 0, f"SB should be player 0 in {n}p, got {idx}"

    def test_bb_always_player1_for_ngt2(self):
        for n in range(3, 10):
            for button in range(n):
                bb_seat = (button + 2) % n
                idx = _player_idx(seat=bb_seat, button=button, n=n)
                assert idx == 1, f"BB should be player 1 in {n}p, got {idx}"

    def test_all_indices_unique_per_hand(self):
        """Every seat maps to a distinct OpenSpiel player index."""
        for n in range(2, 10):
            for button in range(n):
                indices = [_player_idx(seat, button, n) for seat in range(n)]
                assert sorted(indices) == list(range(n)), (
                    f"Non-unique mapping for n={n} button={button}: {indices}"
                )


# ─── Strategy helpers ────────────────────────────────────────────────────────

class TestStrategyHelpers:
    """Pure helper functions — no pkl required."""

    def test_hand_category_pair(self):
        from strategy.preflop_charts import hand_category
        assert hand_category("Ah", "Ac") == "AA"

    def test_hand_category_suited(self):
        from strategy.preflop_charts import hand_category
        assert hand_category("Ah", "Kh") == "AKs"

    def test_hand_category_offsuit(self):
        from strategy.preflop_charts import hand_category
        assert hand_category("Ah", "Kc") == "AKo"

    def test_hand_category_order_independent(self):
        from strategy.preflop_charts import hand_category
        assert hand_category("Kc", "Ah") == "AKo"
        assert hand_category("Kh", "Ah") == "AKs"

    def test_open_raise_size_btn(self):
        from strategy.preflop_charts import open_raise_size
        assert open_raise_size(10000, 100, "btn") == 300

    def test_open_raise_size_utg(self):
        from strategy.preflop_charts import open_raise_size
        assert open_raise_size(10000, 100, "utg") == 250

    def test_effective_stack_hu(self):
        from strategy.preflop_charts import effective_stack_bb
        assert effective_stack_bb(10000, [10000, 8000], 0, 100, 2) == 80.0
