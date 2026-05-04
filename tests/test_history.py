"""
Unit tests for HandHistoryWriter.

Run with:  python3 -m pytest tests/test_history.py -v
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch
from pathlib import Path


class TestHandHistoryWriter:
    def _make_writer(self, tmp_path):
        import config
        with patch.object(config, "HISTORY_DIR", tmp_path):
            from importlib import reload
            import engine.history as hmod
            reload(hmod)
            return hmod.HandHistoryWriter()

    def test_file_created(self, tmp_path):
        w = self._make_writer(tmp_path)
        w.begin_hand(1, 0, {"Hero": 10000, "Bot": 10000})
        w.flush_hand()
        files = list(tmp_path.glob("session_*.txt"))
        assert len(files) == 1

    def test_begin_hand_writes_header(self, tmp_path):
        w = self._make_writer(tmp_path)
        w.begin_hand(1, 0, {"Hero": 10000, "Bot": 9800})
        w.flush_hand()
        text = w.path.read_text()
        assert "PokerStars Hand #1" in text
        assert "Hero" in text
        assert "Bot" in text

    def test_action_without_note(self, tmp_path):
        w = self._make_writer(tmp_path)
        w.begin_hand(1, 0, {"Hero": 10000})
        w.action("Hero", "raises to", 300)
        w.flush_hand()
        text = w.path.read_text()
        assert "Hero: raises to $300" in text
        assert "[" not in text  # no note brackets

    def test_action_with_strategy_note(self, tmp_path):
        w = self._make_writer(tmp_path)
        w.begin_hand(1, 0, {"Hero": 10000})
        w.action("Bot", "bets", 200, strategy_note="MCCFR solver")
        w.flush_hand()
        text = w.path.read_text()
        assert "[MCCFR solver]" in text

    def test_fold_no_amount(self, tmp_path):
        w = self._make_writer(tmp_path)
        w.begin_hand(1, 0, {"Hero": 10000})
        w.action("Bot", "folds", strategy_note="nit/fold")
        w.flush_hand()
        text = w.path.read_text()
        assert "Bot: folds  [nit/fold]" in text

    def test_multiple_hands_same_file(self, tmp_path):
        w = self._make_writer(tmp_path)
        for i in range(1, 4):
            w.begin_hand(i, 0, {"Hero": 10000})
            w.action("Hero", "folds")
            w.flush_hand()
        text = w.path.read_text()
        assert "Hand #1" in text
        assert "Hand #2" in text
        assert "Hand #3" in text

    def test_summary_written(self, tmp_path):
        w = self._make_writer(tmp_path)
        w.begin_hand(1, 0, {"Hero": 10000})
        w.summary(300, ["Ah", "Kh", "Qh"], [("Hero", 300)])
        w.flush_hand()
        text = w.path.read_text()
        assert "SUMMARY" in text
        assert "Total pot $300" in text
