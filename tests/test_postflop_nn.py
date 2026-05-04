"""
Tests for postflop NN feature encoding and architecture.

These tests target:
  1. encode_features produces correct shape and value ranges
  2. PostflopNN query interface matches what server.py expects
  3. Postflop NN checkpoints are filtered (no optimizer files)

Run:  python3 -m pytest tests/test_postflop_nn.py -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np


# ─── Feature encoding ────────────────────────────────────────────────────────

from strategy.postflop_nn import (
    encode_features, N_ACTIONS, FEATURE_DIM, ACTION_NAMES,
    PostflopNN, nn_status,
)


class TestEncodeFeatures:
    """encode_features must produce a fixed-size vector with valid values."""

    def test_output_shape(self):
        feat = encode_features()
        assert feat.shape == (FEATURE_DIM,), f"Expected ({FEATURE_DIM},), got {feat.shape}"

    def test_output_dtype(self):
        feat = encode_features()
        assert feat.dtype == np.float32, f"Expected float32, got {feat.dtype}"

    def test_no_nan_or_inf(self):
        feat = encode_features(hand_cat=5, position=2, street=1, texture_id=42,
                               n_players=3, pot_size=10.0, stack_ratio=0.5,
                               facing_size=0.3, agg_actions=2)
        assert np.all(np.isfinite(feat)), "Features contain nan or inf"

    def test_hand_cat_in_range(self):
        """hand_cat should be one-hot encoded or in valid range."""
        feat0 = encode_features(hand_cat=0)
        feat20 = encode_features(hand_cat=20)
        # Should be different
        assert not np.array_equal(feat0, feat20), "hand_cat=0 and hand_cat=20 produce same features"

    def test_position_varies(self):
        """Different positions should produce different features."""
        feat0 = encode_features(position=0)
        feat5 = encode_features(position=5)
        assert not np.array_equal(feat0, feat5)

    def test_street_varies(self):
        """Different streets should produce different features."""
        feat_flop = encode_features(street=0)
        feat_turn = encode_features(street=1)
        feat_river = encode_features(street=2)
        assert not np.array_equal(feat_flop, feat_turn)
        assert not np.array_equal(feat_turn, feat_river)

    def test_n_players_varies(self):
        """2p vs 6p should produce different features."""
        feat2 = encode_features(n_players=2)
        feat6 = encode_features(n_players=6)
        assert not np.array_equal(feat2, feat6)

    def test_pot_size_varies(self):
        """Pot size should affect features."""
        feat_small = encode_features(pot_size=5.0)
        feat_big = encode_features(pot_size=100.0)
        assert not np.array_equal(feat_small, feat_big)

    def test_facing_size_varies(self):
        """Facing a bet vs not should produce different features."""
        feat_check = encode_features(facing_size=0.0)
        feat_bet = encode_features(facing_size=0.75)
        assert not np.array_equal(feat_check, feat_bet)


class TestConstants:
    """Verify NN constants are consistent."""

    def test_n_actions_matches_names(self):
        assert len(ACTION_NAMES) == N_ACTIONS, \
            f"N_ACTIONS={N_ACTIONS} but ACTION_NAMES has {len(ACTION_NAMES)} entries"

    def test_feature_dim_positive(self):
        assert FEATURE_DIM > 0

    def test_action_names_are_strings(self):
        for name in ACTION_NAMES:
            assert isinstance(name, str)


class TestNNStatus:
    """nn_status() must return structured data without optimizer files."""

    def test_returns_dict(self):
        status = nn_status()
        assert isinstance(status, dict)

    def test_no_optimizer_in_checkpoints(self):
        """Optimizer checkpoints must never appear in status."""
        status = nn_status()
        ckpts = status.get("checkpoints", [])
        for c in ckpts:
            assert "optimizer" not in str(c), f"Optimizer checkpoint leaked: {c}"

    def test_has_expected_keys(self):
        status = nn_status()
        expected = {"params", "step", "latest_checkpoint"}
        for key in expected:
            assert key in status, f"Missing key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
