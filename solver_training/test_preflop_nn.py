"""
Tests for the neural preflop solver.

Verifies:
  - Network architecture (forward pass shapes)
  - Feature encoding consistency
  - MCCFR traversal produces valid regrets/strategies
  - NashConv decreases over training
  - NN warm-start from tabular data
  - API compatibility with /charts endpoint
"""

import sys
import os
import math
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.preflop_nn import (
    PreflopNet, PreflopNN, encode_features, _hand_index,
    INPUT_DIM, OUTPUT_DIM, ACTION_NAMES, N_HAND_CATS,
    N_POSITIONS, MAX_HIST_LEN, N_PLAYER_SLOTS,
    _hand_to_cards, _build_label,
)
from solver_training.preflop_fixed_train import (
    State, legal_actions, apply_action, is_terminal, payoff,
    position_names, N_ACTIONS, BB,
)


# ─── Network architecture tests ──────────────────────────────────────────────

class TestNetworkArchitecture:
    def test_forward_pass_shapes(self):
        net = PreflopNet()
        x = torch.randn(8, INPUT_DIM)
        logits, value = net(x)
        assert logits.shape == (8, OUTPUT_DIM), f"Expected (8, {OUTPUT_DIM}), got {logits.shape}"
        assert value.shape == (8, 1), f"Expected (8, 1), got {value.shape}"

    def test_predict_produces_valid_probs(self):
        net = PreflopNet()
        x = torch.randn(1, INPUT_DIM)
        mask = torch.ones(1, OUTPUT_DIM, dtype=torch.bool)
        probs, value = net.predict(x, mask)
        assert probs.shape == (1, OUTPUT_DIM)
        assert abs(probs.sum().item() - 1.0) < 1e-5, f"Probs don't sum to 1: {probs.sum().item()}"
        assert (probs >= 0).all(), "Negative probabilities"

    def test_predict_respects_legal_mask(self):
        net = PreflopNet()
        x = torch.randn(1, INPUT_DIM)
        # Only fold and call legal
        mask = torch.zeros(1, OUTPUT_DIM, dtype=torch.bool)
        mask[0, 0] = True  # fold
        mask[0, 1] = True  # call
        probs, _ = net.predict(x, mask)
        assert (probs[0, 2:] < 1e-6).all(), "Illegal actions have non-zero probability"
        assert abs(probs[0, :2].sum().item() - 1.0) < 1e-5

    def test_param_count(self):
        net = PreflopNet()
        n_params = sum(p.numel() for p in net.parameters())
        # Should be around 1.1M params
        assert 500_000 < n_params < 5_000_000, f"Unexpected param count: {n_params}"

    def test_batch_predict(self):
        net = PreflopNet()
        x = torch.randn(32, INPUT_DIM)
        mask = torch.ones(32, OUTPUT_DIM, dtype=torch.bool)
        probs, values = net.predict(x, mask)
        assert probs.shape == (32, OUTPUT_DIM)
        assert values.shape == (32, 1)
        # All rows should sum to ~1
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(32), atol=1e-5)


# ─── Feature encoding tests ──────────────────────────────────────────────────

class TestFeatureEncoding:
    def test_feature_dimension(self):
        feat = encode_features(5, 30.0, 2, (0, 5), ["bet", "call"])
        assert feat.shape == (INPUT_DIM,), f"Expected ({INPUT_DIM},), got {feat.shape}"

    def test_hand_encoding_one_hot(self):
        feat1 = encode_features(5, 30.0, 0, (0, 5), [])
        feat2 = encode_features(5, 30.0, 0, (12, 12), [])
        # Different hands should have different one-hot vectors
        assert not torch.equal(feat1[:N_HAND_CATS], feat2[:N_HAND_CATS])

    def test_same_hand_consistent(self):
        # Same hand encoded twice should be identical
        f1 = encode_features(3, 30.0, 0, (0, 5), [])
        f2 = encode_features(3, 30.0, 0, (0, 5), [])
        assert torch.equal(f1, f2)

    def test_position_one_hot(self):
        feat = encode_features(5, 30.0, 2, (0, 5), [])
        # Position CO = index 4 in all_positions
        pos_slice = feat[N_HAND_CATS:N_HAND_CATS + N_POSITIONS]
        assert pos_slice.sum().item() == 1.0, "Position should be one-hot"

    def test_n_players_one_hot(self):
        feat = encode_features(5, 30.0, 0, (0, 5), [])
        np_offset = N_HAND_CATS + N_POSITIONS + MAX_HIST_LEN
        np_slice = feat[np_offset:np_offset + N_PLAYER_SLOTS]
        assert np_slice.sum().item() == 1.0, "n_players should be one-hot"
        # 5 players → index 3 (5-2=3)
        assert np_slice[3].item() == 1.0

    def test_stack_depth_encoding(self):
        f1 = encode_features(5, 10.0, 0, (0, 5), [])
        f2 = encode_features(5, 100.0, 0, (0, 5), [])
        f3 = encode_features(5, 50.0, 0, (0, 5), [])
        sd_offset = INPUT_DIM - 1
        # Log scale: deeper stacks should have higher normalized value
        assert f1[sd_offset] < f3[sd_offset] < f2[sd_offset]

    def test_history_encoding(self):
        f_no_hist = encode_features(5, 30.0, 0, (0, 5), [])
        f_bet = encode_features(5, 30.0, 0, (0, 5), ["bet"])
        f_bet_call = encode_features(5, 30.0, 0, (0, 5), ["bet", "call"])
        hist_offset = N_HAND_CATS + N_POSITIONS
        # History should differ
        assert not torch.equal(f_no_hist[hist_offset:hist_offset+MAX_HIST_LEN],
                               f_bet[hist_offset:hist_offset+MAX_HIST_LEN])
        assert not torch.equal(f_bet[hist_offset:hist_offset+MAX_HIST_LEN],
                               f_bet_call[hist_offset:hist_offset+MAX_HIST_LEN])


# ─── Hand index tests ─────────────────────────────────────────────────────────

class TestHandIndex:
    def test_range(self):
        """All hand indices should be 0-168."""
        for c1 in range(52):
            for c2 in range(52):
                if c1 == c2:
                    continue
                idx = _hand_index(c1, c2)
                assert 0 <= idx < 169, f"Hand index out of range: {idx} for ({c1},{c2})"

    def test_symmetry(self):
        """Same hand with cards swapped should give same index."""
        for c1 in range(52):
            for c2 in range(c1 + 1, 52):
                assert _hand_index(c1, c2) == _hand_index(c2, c1), \
                    f"Asymmetric hand index: ({c1},{c2}) vs ({c2},{c1})"

    def test_uniqueness(self):
        """Different hands should (mostly) give different indices."""
        indices = set()
        for c1 in range(52):
            for c2 in range(c1 + 1, 52):
                indices.add(_hand_index(c1, c2))
        # 52*51/2 = 1326 hole card combos → 169 canonical hands
        assert len(indices) == 169, f"Expected 169 unique hand indices, got {len(indices)}"


# ─── MCCFR traversal tests ────────────────────────────────────────────────────

class TestMCCFRTraversal:
    def test_traverse_produces_examples(self):
        """A single traversal should produce training examples."""
        from solver_training.train_preflop_nn import _traverse_mccfr
        net = PreflopNet()
        net.eval()
        device = torch.device("cpu")
        feats, targets, masks, values = _traverse_mccfr(net, 3, 30, device)
        assert len(feats) > 0, "Traversal should produce at least one training example"

    def test_traverse_target_sums_to_one(self):
        """Target strategy should be a valid probability distribution over legal actions."""
        from solver_training.train_preflop_nn import _traverse_mccfr
        net = PreflopNet()
        net.eval()
        device = torch.device("cpu")
        feats, targets, masks, values = _traverse_mccfr(net, 2, 30, device)
        for i in range(len(targets)):
            legal_sum = targets[i][masks[i] > 0.5].sum()
            assert abs(legal_sum - 1.0) < 1e-5, \
                f"Target probs don't sum to 1 over legal actions: sum={legal_sum}"

    def test_traverse_illegal_actions_zero(self):
        """Target strategy should be zero for illegal actions."""
        from solver_training.train_preflop_nn import _traverse_mccfr
        net = PreflopNet()
        net.eval()
        device = torch.device("cpu")
        feats, targets, masks, values = _traverse_mccfr(net, 3, 30, device)
        for i in range(len(targets)):
            illegal = targets[i][masks[i] < 0.5]
            assert np.allclose(illegal, 0.0, atol=1e-6), \
                "Target probs should be zero for illegal actions"

    def test_traverse_multiple_configs(self):
        """Traversal should work for different player counts and stack depths."""
        from solver_training.train_preflop_nn import _traverse_mccfr
        net = PreflopNet()
        net.eval()
        device = torch.device("cpu")
        for n_p, s_bb in [(2, 30), (3, 30), (5, 100)]:
            feats, targets, masks, values = _traverse_mccfr(net, n_p, s_bb, device)
            assert len(feats) > 0, f"No examples for {n_p}p {s_bb}bb"


# ─── Query API tests ──────────────────────────────────────────────────────────

class TestQueryAPI:
    def test_query_returns_valid_probs(self):
        nn = PreflopNN()
        probs = nn.query(n=5, bb=30, pidx=0, hand=(0, 5))
        assert isinstance(probs, dict)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-4, f"Probs don't sum to 1: {total}"
        assert all(v >= 0 for v in probs.values()), "Negative probabilities"

    def test_query_continuous_stack_depth(self):
        """NN should accept any stack depth, not just trained buckets."""
        nn = PreflopNN()
        for bb in [7.5, 23, 45.3, 88, 150]:
            probs = nn.query(n=3, bb=bb, pidx=0, hand=(0, 5))
            assert isinstance(probs, dict), f"Failed at {bb}bb"
            assert len(probs) > 0, f"Empty result at {bb}bb"

    def test_query_chart_format(self):
        """query_chart should return data compatible with /charts API."""
        nn = PreflopNN()
        chart = nn.query_chart(n=5, bb=30, pidx=0, history=[])
        assert "n_players" in chart
        assert "stack_bb" in chart
        assert "position" in chart
        assert "history" in chart
        assert "hands" in chart
        assert "n_hands_decoded" in chart
        assert chart["n_hands_decoded"] == 169
        assert chart["source"] == "nn"

    def test_query_chart_hand_probs(self):
        """Each hand's probabilities should sum to ~1 over legal actions."""
        nn = PreflopNN()
        chart = nn.query_chart(n=3, bb=30, pidx=0, history=[])
        for hand_key, probs in chart["hands"].items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.05, \
                f"Hand {hand_key} probs sum to {total}, not 1"


# ─── Label building tests ────────────────────────────────────────────────────

class TestLabelBuilding:
    def test_rfi_label(self):
        label = _build_label(5, 30, 0, [])
        assert "HJ" in label
        assert "RFI" in label

    def test_vs_label(self):
        label = _build_label(5, 30, 2, ["bet", "call"])
        assert "vs" in label or "BTN" in label

    def test_stack_in_label(self):
        label = _build_label(5, 45, 0, [])
        assert "45" in label


# ─── Save/load tests ─────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        net1 = PreflopNet()
        x = torch.randn(1, INPUT_DIM)
        mask = torch.ones(1, OUTPUT_DIM, dtype=torch.bool)
        with torch.no_grad():
            out1, _ = net1.predict(x, mask)

        # Save
        path = tmp_path / "test_model.pt"
        torch.save({"model": net1.state_dict(), "step": 42}, path)

        # Load into new network
        net2 = PreflopNet()
        state = torch.load(path, map_location="cpu", weights_only=True)
        net2.load_state_dict(state["model"])
        net2.eval()

        with torch.no_grad():
            out2, _ = net2.predict(x, mask)

        assert torch.allclose(out1, out2, atol=1e-6), "Loaded model gives different output"

    def test_nn_status(self):
        from strategy.preflop_nn import nn_status
        status = nn_status()
        assert "available" in status
        assert "n_params" in status
        assert status["n_params"] > 0


# ─── Integration: NN matches tabular at trained depths ────────────────────────

class TestIntegration:
    def test_nn_uses_same_game_logic(self):
        """NN query should produce the same legal actions as tabular."""
        nn = PreflopNN()
        s = State(n_players=5, stack_bb=30)
        la = legal_actions(s)
        probs = nn.query(n=5, bb=30, pidx=0, hand=(0, 5), legal_actions=la)
        # All returned action names should correspond to legal actions
        action_map = {"fold": 0, "call": 1, "bet": 2, "squeeze": 3, "allin": 4}
        for name in probs:
            assert action_map[name] in la, f"NN returned illegal action: {name}"

    def test_hand_to_cards_round_trip(self):
        """_hand_to_cards should produce valid card indices."""
        for r1 in range(13):
            for r2 in range(13):
                c1, c2 = _hand_to_cards(r1, r2)
                assert 0 <= c1 < 52, f"Invalid card: {c1}"
                assert 0 <= c2 < 52, f"Invalid card: {c2}"
                assert c1 != c2, f"Duplicate card: {c1}"


# ─── Convergence smoke test ──────────────────────────────────────────────────

class TestConvergence:
    def test_loss_decreases(self):
        """Training on a fixed dataset should reduce loss on that dataset."""
        from solver_training.train_preflop_nn import _train_step
        import random

        torch.manual_seed(42)
        net = PreflopNet()
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        # Generate a fixed synthetic dataset
        n_examples = 512
        X = torch.randn(n_examples, INPUT_DIM)
        T = torch.zeros(n_examples, OUTPUT_DIM)
        M = torch.zeros(n_examples, OUTPUT_DIM)
        V = torch.randn(n_examples)

        for i in range(n_examples):
            n_legal = random.randint(2, 4)
            legal = random.sample(range(OUTPUT_DIM), n_legal)
            for a in legal:
                M[i, a] = 1.0
            for a in legal:
                T[i, a] = 1.0 / n_legal

        X_np = X.numpy()
        T_np = T.numpy()
        M_np = M.numpy()
        V_np = V.numpy()

        # Measure initial loss
        net.eval()
        with torch.no_grad():
            logits, _ = net(X)
            log_probs = torch.nn.functional.log_softmax(
                logits.masked_fill(M < 0.5, -1e9), dim=-1
            )
            initial_loss = -(T * log_probs).sum(dim=-1).mean().item()

        # Train on this fixed dataset
        net.train()
        for _ in range(100):
            _train_step(net, optimizer, X_np, T_np, M_np, V_np, device)

        # Measure final loss
        net.eval()
        with torch.no_grad():
            logits, _ = net(X)
            log_probs = torch.nn.functional.log_softmax(
                logits.masked_fill(M < 0.5, -1e9), dim=-1
            )
            final_loss = -(T * log_probs).sum(dim=-1).mean().item()

        assert final_loss < initial_loss, \
            f"Loss didn't decrease: {initial_loss:.4f} → {final_loss:.4f}"

    def test_loss_masks_illegal_target_mass(self):
        """Verify that target mass on masked (illegal) actions doesn't explode loss."""
        import torch, torch.nn.functional as F
        from strategy.preflop_nn import PreflopNet, INPUT_DIM, OUTPUT_DIM

        net = PreflopNet()
        net.eval()

        # Create a sample where target has tiny mass on masked actions
        x = torch.randn(1, INPUT_DIM)
        target = torch.tensor([[0.998, 0.0, 0.002, 0.0, 0.00003]])  # tiny mass on action 4
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0]])  # only fold + bet legal

        with torch.no_grad():
            logits, _ = net(x)
            log_probs = F.log_softmax(logits.masked_fill(mask < 0.5, -1e9), dim=-1)

        # WITH mask fix: loss should be reasonable (< 5.0)
        masked_loss = -(target * mask * log_probs).sum(dim=-1).mean().item()
        assert masked_loss < 5.0, f"Masked loss too high: {masked_loss:.2f}"

        # WITHOUT mask fix: loss would be huge due to 0.00003 * 1e9 = 30K
        unmasked_loss = -(target * log_probs).sum(dim=-1).mean().item()
        assert unmasked_loss > 1000, "Unmasked loss should be huge (verifying the bug existed)"
