"""
Unit tests for OpponentTracker and PlayerStats.

Run with:  python3 -m pytest tests/test_tracker.py -v
"""
import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from strategy.tracker import PlayerStats, OpponentTracker


# ─── PlayerStats defaults ─────────────────────────────────────────────────────

class TestPlayerStatsDefaults:
    def test_all_counts_zero(self):
        s = PlayerStats()
        assert s.hands_dealt == 0
        assert s.vpip_count == 0
        assert s.limp_count == 0
        assert s.hands_ep == 0

    def test_vpip_default_no_hands(self):
        s = PlayerStats()
        assert s.vpip == 0.25   # sensible GTO prior

    def test_pfr_default_no_hands(self):
        s = PlayerStats()
        assert s.pfr == 0.15

    def test_limp_pct_default(self):
        assert PlayerStats().limp_pct == 0.05

    def test_call_3bet_pct_default(self):
        assert PlayerStats().call_3bet_pct == 0.30

    def test_positional_vpip_defaults(self):
        s = PlayerStats()
        assert s.vpip_ep_pct == 0.15
        assert s.vpip_mp_pct == 0.20
        assert s.vpip_lp_pct == 0.35
        assert s.vpip_blinds_pct == 0.40

    def test_turn_probe_default(self):
        assert PlayerStats().turn_probe_pct == 0.25

    def test_positional_awareness_default(self):
        s = PlayerStats()
        # hands_ep=0, so guard returns 2.0 regardless of default vpip priors
        assert s.positional_awareness == 2.0


# ─── Computed properties ──────────────────────────────────────────────────────

class TestComputedProperties:
    def test_vpip_computed(self):
        s = PlayerStats(hands_dealt=10, vpip_count=4)
        assert s.vpip == pytest.approx(0.4)

    def test_pfr_computed(self):
        s = PlayerStats(hands_dealt=10, pfr_count=2)
        assert s.pfr == pytest.approx(0.2)

    def test_limp_pct_computed(self):
        s = PlayerStats(limp_opps=20, limp_count=8)
        assert s.limp_pct == pytest.approx(0.4)

    def test_call_3bet_pct_computed(self):
        s = PlayerStats(three_bet_call_opps=10, call_3bet_count=4)
        assert s.call_3bet_pct == pytest.approx(0.4)

    def test_vpip_ep_pct(self):
        s = PlayerStats(hands_ep=8, vpip_ep=3)
        assert s.vpip_ep_pct == pytest.approx(0.375)

    def test_vpip_lp_pct(self):
        s = PlayerStats(hands_lp=10, vpip_lp=7)
        assert s.vpip_lp_pct == pytest.approx(0.7)

    def test_positional_awareness_ratio(self):
        s = PlayerStats(hands_ep=10, vpip_ep=2, hands_lp=10, vpip_lp=6)
        assert s.positional_awareness == pytest.approx(3.0)

    def test_positional_awareness_no_ep_hands(self):
        # hands_ep=0 → guard triggers → returns 2.0
        s = PlayerStats(hands_ep=0, hands_lp=10, vpip_lp=5)
        assert s.positional_awareness == 2.0

    def test_turn_probe_pct(self):
        s = PlayerStats(turn_probe_opps=8, turn_probe_count=3)
        assert s.turn_probe_pct == pytest.approx(0.375)

    def test_aggression_factor(self):
        s = PlayerStats(total_bets_raises=6, total_calls=4)
        assert s.aggression_factor == pytest.approx(0.6)

    def test_fold_to_cbet_combined(self):
        s = PlayerStats(cbet_faced_dry=4, cbet_fold_dry=2, cbet_faced_wet=4, cbet_fold_wet=1)
        assert s.fold_to_cbet_pct == pytest.approx(3/8)


# ─── merge_from ───────────────────────────────────────────────────────────────

class TestMergeFrom:
    def test_basic_merge(self):
        a = PlayerStats(hands_dealt=5, vpip_count=2)
        b = PlayerStats(hands_dealt=3, vpip_count=1)
        a.merge_from(b)
        assert a.hands_dealt == 8
        assert a.vpip_count == 3

    def test_merge_new_fields(self):
        a = PlayerStats(limp_opps=10, limp_count=3)
        b = PlayerStats(limp_opps=5, limp_count=2)
        a.merge_from(b)
        assert a.limp_opps == 15
        assert a.limp_count == 5

    def test_merge_positional(self):
        a = PlayerStats(hands_ep=4, vpip_ep=1)
        b = PlayerStats(hands_ep=6, vpip_ep=2)
        a.merge_from(b)
        assert a.hands_ep == 10
        assert a.vpip_ep == 3

    def test_merge_does_not_affect_b(self):
        a = PlayerStats(hands_dealt=5)
        b = PlayerStats(hands_dealt=3)
        a.merge_from(b)
        assert b.hands_dealt == 3  # b unchanged


# ─── Serialization ───────────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_roundtrip(self):
        s = PlayerStats(hands_dealt=10, vpip_count=4, limp_count=2,
                        hands_ep=3, vpip_ep=1, turn_probe_count=2)
        d = s.to_dict()
        s2 = PlayerStats.from_dict(d)
        assert s2.hands_dealt == 10
        assert s2.limp_count == 2
        assert s2.hands_ep == 3
        assert s2.turn_probe_count == 2

    def test_from_dict_ignores_unknown_keys(self):
        s = PlayerStats.from_dict({"hands_dealt": 5, "unknown_future_field": 99})
        assert s.hands_dealt == 5  # no error


# ─── OpponentTracker recording ───────────────────────────────────────────────

class TestOpponentTracker:
    def setup_method(self):
        self.t = OpponentTracker()

    def test_get_creates_default(self):
        s = self.t.get(0)
        assert isinstance(s, PlayerStats)
        assert s.hands_dealt == 0

    def test_record_hand_dealt(self):
        self.t.record_hand_dealt(0)
        assert self.t.get(0).hands_dealt == 1

    def test_record_hand_dealt_pos_ep(self):
        self.t.record_hand_dealt_pos(0, "ep")
        assert self.t.get(0).hands_ep == 1

    def test_record_hand_dealt_pos_lp(self):
        self.t.record_hand_dealt_pos(0, "lp")
        assert self.t.get(0).hands_lp == 1

    def test_record_hand_dealt_pos_blinds(self):
        self.t.record_hand_dealt_pos(0, "blinds")
        assert self.t.get(0).hands_blinds == 1

    def test_record_vpip_pos(self):
        self.t.record_vpip_pos(0, "mp")
        assert self.t.get(0).vpip_mp == 1

    def test_record_limp(self):
        self.t.record_limp_opp(0)
        self.t.record_limp(0)
        s = self.t.get(0)
        assert s.limp_opps == 1
        assert s.limp_count == 1
        assert s.limp_pct == 1.0

    def test_record_3bet_call(self):
        self.t.record_3bet_call_opp(0)
        self.t.record_call_3bet(0)
        s = self.t.get(0)
        assert s.three_bet_call_opps == 1
        assert s.call_3bet_count == 1

    def test_record_turn_probe(self):
        self.t.record_turn_probe_opp(0)
        self.t.record_turn_probe(0)
        s = self.t.get(0)
        assert s.turn_probe_opps == 1
        assert s.turn_probe_count == 1

    def test_record_action_types(self):
        self.t.record_action(0, "raise")
        self.t.record_action(0, "call")
        self.t.record_action(0, "fold")
        s = self.t.get(0)
        assert s.total_bets_raises == 1
        assert s.total_calls == 1
        assert s.total_folds == 1

    def test_stats_dict_keys(self):
        sd = self.t.stats_dict(0)
        for key in ("hands_dealt", "vpip", "pfr", "limp_pct", "call_3bet_pct",
                    "vpip_ep", "vpip_lp", "positional_awareness", "turn_probe_pct"):
            assert key in sd, f"missing key: {key}"

    def test_independent_seats(self):
        self.t.record_hand_dealt(0)
        self.t.record_hand_dealt(1)
        self.t.record_vpip(0)
        assert self.t.get(0).vpip_count == 1
        assert self.t.get(1).vpip_count == 0


# ─── Persistence ─────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, tmp_path, monkeypatch):
        import strategy.tracker as tracker_mod
        monkeypatch.setattr(tracker_mod, "TRACKER_DIR", tmp_path)

        t = OpponentTracker()
        t.set_player_name(0, "TestHero")
        t.record_hand_dealt(0)
        t.record_vpip(0)
        t.record_limp(0)
        t.record_hand_dealt_pos(0, "lp")
        t.save(0)

        # Verify file was written
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["hands_dealt"] == 1
        assert data["vpip_count"] == 1
        assert data["limp_count"] == 1
        assert data["hands_lp"] == 1

    def test_load_merges_prior_session(self, tmp_path, monkeypatch):
        import strategy.tracker as tracker_mod
        monkeypatch.setattr(tracker_mod, "TRACKER_DIR", tmp_path)

        # Write a prior session file manually
        prior = PlayerStats(hands_dealt=50, vpip_count=20, hands_lp=15, vpip_lp=9)
        path = tmp_path / "testhero.json"
        path.write_text(json.dumps(prior.to_dict()))

        t = OpponentTracker()
        t.set_player_name(0, "TestHero")  # triggers _load_from_disk
        t.record_hand_dealt(0)            # one new hand this session

        s = t.get(0)
        assert s.hands_dealt == 51
        assert s.vpip_count == 20
        assert s.hands_lp == 15
