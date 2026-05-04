"""Tests for postflop NN strategy and training.

Covers:
  - Feature encoding correctness
  - Network forward pass
  - PostflopNN query API
  - Action mapping (tabular ↔ NN)
  - Distillation data extraction
  - Loss computation with mask
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.postflop_nn import (
    PostflopNet, PostflopNN, encode_features, nn_status,
    INPUT_DIM, OUTPUT_DIM, N_ACTIONS, ACTION_NAMES,
    _hand_index, N_HAND_CATS, N_TEXTURES, N_STREETS,
    MAX_HIST_LEN, N_PLAYER_SLOTS, N_POSITIONS,
)
from strategy.board_abstraction import texture_id


# ─── Feature encoding ─────────────────────────────────────────────────────────

class TestFeatureEncoding:
    def test_output_shape(self):
        feat = encode_features(
            n_players=2, hole=(48, 49), board=[8, 12, 20],
            street=0, facing_bet=False, agg_count=0,
            history=[], position=0, pot_bb=5.0, spr=10.0,
        )
        assert feat.shape == (INPUT_DIM,)

    def test_hand_one_hot(self):
        feat = encode_features(
            n_players=2, hole=(48, 49), board=[0, 4, 8],
            street=0, facing_bet=False, agg_count=0,
            history=[], position=0, pot_bb=5.0, spr=10.0,
        )
        # First 169 dims = hand one-hot
        hand_slice = feat[:169]
        assert hand_slice.sum().item() == 1.0  # exactly one active

    def test_board_texture_one_hot(self):
        feat = encode_features(
            n_players=2, hole=(48, 49), board=[0, 4, 8],
            street=0, facing_bet=False, agg_count=0,
            history=[], position=0, pot_bb=5.0, spr=10.0,
        )
        tex_start = N_HAND_CATS
        tex_slice = feat[tex_start:tex_start+N_TEXTURES]
        assert tex_slice.sum().item() == 1.0

    def test_street_encoding(self):
        for street in [0, 1, 2]:
            feat = encode_features(
                n_players=2, hole=(48, 49), board=[0, 4, 8],
                street=street, facing_bet=False, agg_count=0,
                history=[], position=0, pot_bb=5.0, spr=10.0,
            )
            street_slice = feat[N_HAND_CATS+N_TEXTURES:N_HAND_CATS+N_TEXTURES+N_STREETS]
            assert street_slice.sum().item() == 1.0
            assert street_slice[street].item() == 1.0

    def test_facing_bet(self):
        feat_no = encode_features(
            n_players=2, hole=(48, 49), board=[0, 4, 8],
            street=0, facing_bet=False, agg_count=0,
            history=[], position=0, pot_bb=5.0, spr=10.0,
        )
        feat_yes = encode_features(
            n_players=2, hole=(48, 49), board=[0, 4, 8],
            street=0, facing_bet=True, agg_count=0,
            history=[], position=0, pot_bb=5.0, spr=10.0,
        )
        assert feat_no[N_HAND_CATS+N_TEXTURES+N_STREETS].item() == 0.0
        assert feat_yes[N_HAND_CATS+N_TEXTURES+N_STREETS].item() == 1.0

    def test_agg_count_normalized(self):
        for agg in [0, 1, 2, 3]:
            feat = encode_features(
                n_players=2, hole=(48, 49), board=[0, 4, 8],
                street=0, facing_bet=False, agg_count=agg,
                history=[], position=0, pot_bb=5.0, spr=10.0,
            )
            assert abs(feat[N_HAND_CATS+N_TEXTURES+N_STREETS+1].item() - agg / 3.0) < 1e-5

    def test_history_encoding(self):
        feat = encode_features(
            n_players=2, hole=(48, 49), board=[0, 4, 8],
            street=0, facing_bet=False, agg_count=0,
            history=[2, 1, 0], position=0, pot_bb=5.0, spr=10.0,
        )
        hist_slice = feat[N_HAND_CATS+N_TEXTURES+N_STREETS+2:N_HAND_CATS+N_TEXTURES+N_STREETS+2+10]
        assert abs(hist_slice[0].item() - 2 / 4.0) < 1e-5  # action 2
        assert abs(hist_slice[1].item() - 1 / 4.0) < 1e-5  # action 1
        assert abs(hist_slice[2].item() - 0 / 4.0) < 1e-5  # action 0
        assert hist_slice[3].item() == 0.0  # padded

    def test_n_players_one_hot(self):
        for n in [2, 3, 4, 5, 6]:
            feat = encode_features(
                n_players=n, hole=(48, 49), board=[0, 4, 8],
                street=0, facing_bet=False, agg_count=0,
                history=[], position=0, pot_bb=5.0, spr=10.0,
            )
            np_slice = feat[N_HAND_CATS+N_TEXTURES+N_STREETS+2+10:N_HAND_CATS+N_TEXTURES+N_STREETS+2+10+N_PLAYER_SLOTS]
            assert np_slice.sum().item() == 1.0

    def test_position_one_hot(self):
        for pos in [0, 1, 2]:
            feat = encode_features(
                n_players=2, hole=(48, 49), board=[0, 4, 8],
                street=0, facing_bet=False, agg_count=0,
                history=[], position=pos, pot_bb=5.0, spr=10.0,
            )
            pos_slice = feat[N_HAND_CATS+N_TEXTURES+N_STREETS+2+10+N_PLAYER_SLOTS:N_HAND_CATS+N_TEXTURES+N_STREETS+2+10+N_PLAYER_SLOTS+N_POSITIONS]
            assert pos_slice.sum().item() == 1.0
            assert pos_slice[pos].item() == 1.0


# ─── Network ──────────────────────────────────────────────────────────────────

class TestNetwork:
    def test_forward_shape(self):
        net = PostflopNet()
        x = torch.randn(8, INPUT_DIM)
        logits, value = net(x)
        assert logits.shape == (8, OUTPUT_DIM)
        assert value.shape == (8, 1)

    def test_predict_with_mask(self):
        net = PostflopNet()
        x = torch.randn(4, INPUT_DIM)
        mask = torch.ones(4, OUTPUT_DIM)
        mask[:, 0] = 0  # fold is illegal
        probs = net.predict(x, mask)
        assert probs.shape == (4, OUTPUT_DIM)
        assert (probs[:, 0] < 1e-6).all()  # fold probability ≈ 0
        assert abs(probs.sum(dim=-1) - 1.0).max() < 1e-4  # sums to 1

    def test_param_count(self):
        net = PostflopNet()
        n = sum(p.numel() for p in net.parameters())
        assert 800_000 < n < 1_200_000  # ~924K


# ─── PostflopNN wrapper ───────────────────────────────────────────────────────

class TestPostflopNNWrapper:
    def test_query_returns_all_legal_actions(self):
        nn = PostflopNN()
        if not nn.available:
            pytest.skip("No postflop NN checkpoint")
        result = nn.query(
            n_players=2, hole=(48, 49), board=[0, 4, 8],
            street=0, facing_bet=False, agg_count=0,
            history=[], position=0, pot_bb=5.0, spr=10.0,
            legal_actions=[0, 1, 2, 3, 4],
        )
        assert isinstance(result, dict)
        for name in ACTION_NAMES:
            assert name in result
        total = sum(result.values())
        assert abs(total - 1.0) < 0.05  # approximately normalized

    def test_query_with_limited_legal(self):
        nn = PostflopNN()
        if not nn.available:
            pytest.skip("No postflop NN checkpoint")
        result = nn.query(
            n_players=2, hole=(48, 49), board=[0, 4, 8],
            street=0, facing_bet=True, agg_count=2,
            history=[2, 1], position=0, pot_bb=10.0, spr=5.0,
            legal_actions=[0, 1, 4],  # fold, call, allin
        )
        assert "fold" in result
        assert "call" in result
        assert "allin" in result
        # bet33 and bet75 should not be present
        assert "bet33" not in result or result.get("bet33", 0) == 0


# ─── Action mapping ───────────────────────────────────────────────────────────

class TestActionMapping:
    def test_tabular_to_nn_opening(self):
        from solver_training.train_postflop_nn import _tabular_to_nn_legal
        # Opening: tabular [1,2,3] → NN [1,2,3] (check, bet33, bet75)
        legal = _tabular_to_nn_legal([1, 2, 3], facing_bet=False, spr=10.0)
        assert 1 in legal  # check/call
        assert 2 in legal  # bet33
        assert 3 in legal  # bet75

    def test_tabular_to_nn_facing_bet(self):
        from solver_training.train_postflop_nn import _tabular_to_nn_legal
        # Facing bet: tabular [0,1,2,3] → NN [0,1,3,4] (fold, call, raise/bet75, allin)
        legal = _tabular_to_nn_legal([0, 1, 2, 3], facing_bet=True, spr=10.0)
        assert 0 in legal  # fold
        assert 1 in legal  # call
        assert 3 in legal  # raise → bet75
        assert 4 in legal  # allin

    def test_tabular_to_nn_shallow(self):
        from solver_training.train_postflop_nn import _tabular_to_nn_legal
        # Shallow SPR: tabular action 3 → allin
        legal = _tabular_to_nn_legal([1, 3], facing_bet=False, spr=2.0)
        assert 4 in legal  # allin


# ─── Loss mask test ───────────────────────────────────────────────────────────

class TestLossMask:
    def test_masked_actions_dont_contribute_to_loss(self):
        """Verify that target mass on masked (illegal) actions doesn't inflate loss."""
        import torch.nn.functional as F

        net = PostflopNet()
        x = torch.randn(4, INPUT_DIM)
        y = torch.zeros(4, OUTPUT_DIM)
        y[:, 0] = 0.998
        y[:, 1] = 0.002
        y[:, 4] = 0.00003  # tiny mass on illegal action

        m = (y > 0.001).float()
        # action 4 is masked (0.00003 < 0.001)

        logits, _ = net(x)
        log_probs = F.log_softmax(logits.masked_fill(m < 0.5, -1e9), dim=-1)

        # Compute loss with mask
        per_action = -(y * m * log_probs)
        loss = per_action.sum(dim=-1).mean()

        # Loss should be reasonable (< 5), not 28K
        assert loss.item() < 5.0, f"Loss too high: {loss.item()}"


# ─── Hand index ───────────────────────────────────────────────────────────────

class TestHandIndex:
    def test_unique_indices(self):
        """All 169 canonical hands should get unique indices in 0-168."""
        indices = set()
        for r1 in range(13):
            # Pairs
            c1, c2 = r1 * 4, r1 * 4 + 1
            indices.add(_hand_index(c1, c2))
            for r2 in range(r1 + 1, 13):
                # Suited
                c1, c2 = r1 * 4, r2 * 4
                indices.add(_hand_index(c1, c2))
                # Offsuit
                c1, c2 = r1 * 4, r2 * 4 + 1
                indices.add(_hand_index(c1, c2))

        assert len(indices) == 169
        assert min(indices) == 0
        assert max(indices) == 168

    def test_aa_is_168(self):
        # A♣ A♦ → cards 48, 49
        assert _hand_index(48, 49) == 168

    def test_22_is_0(self):
        # 2♣ 2♦ → cards 0, 1
        assert _hand_index(0, 1) == 0


# ─── Board texture integration ────────────────────────────────────────────────

class TestBoardTextureIntegration:
    def test_texture_id_in_features(self):
        """Verify that board texture ID is correctly one-hot encoded."""
        board = [48, 36, 24]  # Ah, 9h, 8h → monotone hearts
        board_strs = [RANKS[c // 4] + SUITS[c % 4] for c in board]
        tid = texture_id(board_strs)
        assert 0 <= tid < 64

        feat = encode_features(
            n_players=2, hole=(0, 4), board=board,
            street=0, facing_bet=False, agg_count=0,
            history=[], position=0, pot_bb=5.0, spr=10.0,
        )
        tex_slice = feat[169:233]
        assert tex_slice[tid].item() == 1.0
        assert tex_slice.sum().item() == 1.0


RANKS = "23456789TJQKA"
SUITS = "cdhs"


# ─── nn_status ────────────────────────────────────────────────────────────────

class TestNNStatus:
    def test_status_structure(self):
        status = nn_status()
        assert "available" in status
        assert "input_dim" in status
        assert "output_dim" in status
        assert status["input_dim"] == INPUT_DIM
        assert status["output_dim"] == OUTPUT_DIM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
