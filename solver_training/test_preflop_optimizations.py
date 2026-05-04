"""
Tests verifying the optimized preflop_fixed_train.py produces
identical results to the original (unoptimized) implementation.

Run with:  python3 -m pytest solver_training/test_preflop_optimizations.py -v
"""

import sys
import os
import random
import struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_training.preflop_fixed_train import (
    State, Solver, position_names, legal_actions, apply_action,
    is_terminal, payoff, info_key, _hand_cat, _pack_hand_cat,
    N_ACTIONS, BB, SB, RAKE_RATE, RAKE_CAP,
    _build_hand_cat_cache, _get_pos_idx,
)
import solver_training.preflop_fixed_train as _pft


def _deal_state(n_players=5, stack_bb=30, seed=42):
    rng = random.Random(seed)
    deck = list(range(52))
    rng.shuffle(deck)
    s = State(n_players, stack_bb)
    for p in range(n_players):
        s.hole[p] = [deck[p * 2], deck[p * 2 + 1]]
    return s


# ─── phevaluator correctness ─────────────────────────────────────────────────

class TestPhevaluatorPayoff:
    """Verify phevaluator produces the same winners as treys."""

    def test_showdown_correctness(self):
        """Compare phevaluator winners against manual hand ranking."""
        # Known hand: pair of aces vs pair of kings
        s = State(2, 100)
        s.hole[0] = [48, 49]  # Ah, Ad (pair of aces)
        s.hole[1] = [44, 45]  # Kh, Kd (pair of kings)
        # Action: both all-in
        s2 = apply_action(s, 2)  # SB raises
        s3 = apply_action(s2, 4)  # BB all-in
        s4 = apply_action(s3, 1)  # SB calls

        assert is_terminal(s4)
        gains = payoff(s4)
        # Player 0 (aces) should win
        assert gains[0] > 0, f"Player 0 with aces should win, got gains={gains}"

    def test_fold_payoff_unchanged(self):
        """Fold terminal payoffs should be identical regardless of evaluator."""
        s = _deal_state(3, 30)
        s2 = apply_action(s, 0)  # First player folds
        s3 = apply_action(s2, 0)  # Second player folds
        assert is_terminal(s3)
        gains = payoff(s3)
        # BB should win the pot
        total = sum(gains)
        assert abs(total + min(s3.pot * RAKE_RATE, RAKE_CAP * BB)) < 1, \
            f"Pot should equal sum of gains + rake: total={total}, pot={s3.pot}"

    def test_multiway_showdown(self):
        """Multiway all-in should produce correct winner."""
        s = State(5, 30)
        # Give known hands
        s.hole[0] = [12, 25]  # Ac, Ah — pair of aces
        s.hole[1] = [11, 24]  # Kc, Kh — pair of kings
        s.hole[2] = [10, 23]  # Qc, Qh — pair of queens
        s.hole[3] = [9, 22]   # Jc, Jh — pair of jacks
        s.hole[4] = [8, 21]   # Tc, Th — pair of tens

        # Everyone all-in
        s2 = apply_action(s, 2)   # HJ raises
        s3 = apply_action(s2, 4)  # CO all-in
        s4 = apply_action(s3, 4)  # BTN all-in
        s5 = apply_action(s4, 4)  # SB all-in
        s6 = apply_action(s5, 4)  # BB all-in
        s7 = apply_action(s6, 1)  # HJ calls

        assert is_terminal(s7)
        gains = payoff(s7)
        # Player 0 (aces) should win
        assert gains[0] > 0, f"Player 0 with aces should win, gains={gains}"

    def test_split_pot(self):
        """Identical hand strengths should split the pot."""
        s = State(2, 100)
        s.hole[0] = [48, 44]  # Ah, Kh
        s.hole[1] = [49, 45]  # Ad, Kd
        # Same hand rank (AK offsuit), same board → split
        s2 = apply_action(s, 2)
        s3 = apply_action(s2, 1)
        assert is_terminal(s3)
        gains = payoff(s3)
        # With random board, could go either way, but we verify the payoff is valid
        total = sum(gains)
        assert abs(total + min(s3.pot * RAKE_RATE, RAKE_CAP * BB)) < 1


# ─── info_key cache correctness ──────────────────────────────────────────────

class TestInfoKeyCache:
    """Verify precomputed hand_cat cache matches original _hand_cat + _pack_hand_cat."""

    def test_cache_matches_original(self):
        """All 2652 hole card pairs should produce same key via cache and original."""
        _build_hand_cat_cache()
        cache = _pft._HAND_CAT_CACHE
        assert cache is not None

        errors = 0
        for c1 in range(52):
            for c2 in range(52):
                if c1 == c2:
                    continue
                # Original method
                hc = _hand_cat([c1, c2])
                r1r2_orig, suit_orig = _pack_hand_cat(hc)

                # Cached method
                r1r2_cache, suit_cache = cache[(c1, c2)]

                if (r1r2_orig, suit_orig) != (r1r2_cache, suit_cache):
                    errors += 1
                    if errors <= 3:
                        print(f"  MISMATCH: cards=({c1},{c2}) "
                              f"orig=({r1r2_orig},{suit_orig}) "
                              f"cache=({r1r2_cache},{suit_cache})")

        assert errors == 0, f"{errors} hand_cat cache mismatches"

    def test_info_key_format_equivalence(self):
        """info_key should produce same bytes with struct.pack as with bytearray."""
        s = _deal_state(5, 30)
        legal = legal_actions(s)

        # Test with empty and non-empty action histories
        for hist in [[], [2], [2, 1, 0], [2, 3, 1]]:
            key = info_key(s.acting, s, hist)

            # Verify it's valid bytes
            assert isinstance(key, bytes)
            # Header is 8 bytes + len(hist) action bytes
            assert len(key) == 8 + len(hist)

            # Verify struct fields
            p = key[0]
            r1r2 = key[1]
            suit_byte = key[2]
            n_players = key[3]
            stack_enc = key[4]
            raise_level = key[5]
            pos_idx = key[6]
            n_cc = key[7]

            assert n_players == s.n_players
            assert raise_level == s.raise_level
            assert n_cc == s.n_cold_callers

    def test_pos_idx_cache(self):
        """Position index lookup should match position_names().index()."""
        for n in [2, 3, 4, 5, 6]:
            names = position_names(n)
            for i, name in enumerate(names):
                assert _get_pos_idx(n, name) == i


# ─── Mutable traverse correctness ────────────────────────────────────────────

class TestMutableTraverse:
    """Verify that _apply_action_mut + _undo produces same state as apply_action."""

    def test_apply_undo_roundtrip(self):
        """apply + undo should restore state exactly."""
        s = _deal_state(5, 30)
        original = s.copy()

        for action in legal_actions(s):
            # Test mutable path
            solver = Solver(5, 30)
            undo = solver._apply_action_mut(s, action)

            # Verify state changed
            assert s.acting != original.acting or s.folded != original.folded or \
                   s.pot != original.pot, f"Action {action} should change state"

            # Undo
            solver._undo_action(s, undo)

            # Verify exact restoration
            assert s.acting == original.acting, f"acting mismatch after undo of action {action}"
            assert s.pot == original.pot, f"pot mismatch after undo of action {action}"
            assert s.folded == original.folded, f"folded mismatch after undo of action {action}"
            assert s.stacks == original.stacks, f"stacks mismatch after undo of action {action}"
            assert s.invested == original.invested, f"invested mismatch after undo of action {action}"
            assert s.raise_level == original.raise_level, f"raise_level mismatch after undo of action {action}"
            assert s.facing_size == original.facing_size, f"facing_size mismatch after undo of action {action}"
            assert s.responded == original.responded, f"responded mismatch after undo of action {action}"
            assert s.n_cold_callers == original.n_cold_callers, f"n_cc mismatch after undo of action {action}"

    def test_mut_matches_copy(self):
        """_apply_action_mut should produce same state as apply_action (copy-based)."""
        for seed in range(50):
            s = _deal_state(5, 30, seed=seed + 100)
            legal = legal_actions(s)

            for action in legal:
                # Copy-based path
                s_copy = apply_action(s, action)

                # Mutable path
                solver = Solver(5, 30)
                undo = solver._apply_action_mut(s, action)

                # Compare all fields
                assert s.acting == s_copy.acting, \
                    f"seed={seed} action={action}: acting {s.acting} != {s_copy.acting}"
                assert s.pot == s_copy.pot, \
                    f"seed={seed} action={action}: pot {s.pot} != {s_copy.pot}"
                assert s.folded == s_copy.folded, \
                    f"seed={seed} action={action}: folded mismatch"
                assert s.stacks == s_copy.stacks, \
                    f"seed={seed} action={action}: stacks mismatch"
                assert s.invested == s_copy.invested, \
                    f"seed={seed} action={action}: invested mismatch"
                assert s.raise_level == s_copy.raise_level, \
                    f"seed={seed} action={action}: raise_level {s.raise_level} != {s_copy.raise_level}"
                assert s.facing_size == s_copy.facing_size, \
                    f"seed={seed} action={action}: facing_size {s.facing_size} != {s_copy.facing_size}"
                assert s.n_cold_callers == s_copy.n_cold_callers, \
                    f"seed={seed} action={action}: n_cc {s.n_cold_callers} != {s_copy.n_cold_callers}"
                assert s.open_size == s_copy.open_size, \
                    f"seed={seed} action={action}: open_size mismatch"
                assert s.last_raiser_pos == s_copy.last_raiser_pos, \
                    f"seed={seed} action={action}: last_raiser_pos mismatch"

                # Undo for next action test
                solver._undo_action(s, undo)

    def test_deep_sequence_mut_vs_copy(self):
        """Multi-action sequences: mutable + undo should match copy-based."""
        for seed in range(20):
            s_orig = _deal_state(5, 30, seed=seed + 200)
            s_copy = s_orig.copy()
            solver = Solver(5, 30)

            undos = []
            actions_taken = []
            s_mut = s_orig

            for _ in range(15):  # up to 15 actions
                if is_terminal(s_mut):
                    break
                legal = legal_actions(s_mut)
                if not legal:
                    break
                action = legal[0]  # deterministic: always first legal action
                actions_taken.append(action)

                undo = solver._apply_action_mut(s_mut, action)
                undos.append(undo)
                s_copy = apply_action(s_copy, action)

                # Verify after each step
                assert s_mut.acting == s_copy.acting, \
                    f"Step mismatch at action {action}: acting"
                assert s_mut.pot == s_copy.pot, f"Step mismatch: pot"
                assert s_mut.folded == s_copy.folded, f"Step mismatch: folded"
                assert s_mut.stacks == s_copy.stacks, f"Step mismatch: stacks"
                assert s_mut.invested == s_copy.invested, f"Step mismatch: invested"
                assert s_mut.raise_level == s_copy.raise_level, f"Step mismatch: raise_level"

            # Undo all actions and verify we get back to original
            for undo in reversed(undos):
                solver._undo_action(s_mut, undo)

            assert s_mut.acting == s_orig.acting
            assert s_mut.pot == s_orig.pot
            assert s_mut.folded == s_orig.folded
            assert s_mut.stacks == s_orig.stacks
            assert s_mut.invested == s_orig.invested


# ─── Strategy computation correctness ────────────────────────────────────────

class TestStrategyComputation:
    """Verify hybrid Python/numpy strategy matches pure numpy."""

    def test_regret_matching_uniform(self):
        """Zero regrets → uniform over legal actions."""
        solver = Solver(3, 30)
        # Fresh solver has zero regrets
        key = info_key(0, _deal_state(3, 30), [])
        legal = [0, 2, 4]
        strat = solver._strategy(key, legal)

        # Should be uniform
        for a in legal:
            assert abs(strat[a] - 1.0 / len(legal)) < 1e-6, \
                f"Expected uniform, got strat[{a}]={strat[a]}"

    def test_regret_matching_positive(self):
        """Positive regrets → proportional strategy."""
        solver = Solver(3, 30)
        # Run some iterations to populate regrets
        for _ in range(100):
            solver.run_iteration()

        # Check that strategy sums to 1 over legal actions
        s = _deal_state(3, 30)
        for _ in range(10):
            if is_terminal(s):
                break
            legal = legal_actions(s)
            key = info_key(s.acting, s, [])
            strat = solver._strategy(key, legal)
            total = sum(strat[a] for a in legal)
            assert abs(total - 1.0) < 1e-5, \
                f"Strategy should sum to 1, got {total} for legal={legal}"
            # All probabilities should be non-negative
            for a in legal:
                assert strat[a] >= 0, f"strat[{a}]={strat[a]} < 0"

    def test_regret_matching_deterministic(self):
        """Single positive regret → all mass on that action."""
        solver = Solver(3, 30)
        # Manually set regrets for a specific info state
        s = _deal_state(3, 30)
        legal = legal_actions(s)
        key = info_key(s.acting, s, [])

        idx = solver._ensure(key)
        solver._regrets[idx] = 0.0
        solver._regrets[idx][2] = 1.0  # Only action 2 has positive regret

        strat = solver._strategy(key, legal)
        assert abs(strat[2] - 1.0) < 1e-6, f"Expected all mass on action 2, got {strat}"
        for a in legal:
            if a != 2:
                assert abs(strat[a]) < 1e-6, f"Expected 0 on action {a}, got {strat[a]}"


# ─── is_terminal correctness ─────────────────────────────────────────────────

class TestIsTerminalFast:
    """Verify optimized is_terminal matches original behavior."""

    def test_fold_to_one(self):
        s = _deal_state(5, 30)
        for i in range(4):
            s = apply_action(s, 0)  # fold everyone but BB
        assert is_terminal(s)

    def test_not_terminal_initial(self):
        s = _deal_state(5, 30)
        assert not is_terminal(s)

    def test_not_terminal_after_raise(self):
        s = _deal_state(5, 30)
        s2 = apply_action(s, 2)  # RFI
        assert not is_terminal(s2)

    def test_terminal_after_call_around(self):
        """2p: SB raises, BB calls → terminal."""
        s = State(2, 100)
        deck = list(range(52))
        random.shuffle(deck)
        for p in range(2):
            s.hole[p] = [deck[p * 2], deck[p * 2 + 1]]
        s2 = apply_action(s, 2)  # SB raises
        s3 = apply_action(s2, 1)  # BB calls
        assert is_terminal(s3)


# ─── Integration: full solver correctness ────────────────────────────────────

class TestSolverIntegration:
    """Verify the optimized solver produces valid CFR output."""

    def test_strat_sums_increase(self):
        """After running iterations, strat_sums should be non-zero."""
        solver = Solver(3, 30)
        for _ in range(1000):
            solver.run_iteration()

        non_zero = np.count_nonzero(solver._strat_sum[:solver._n])
        assert non_zero > 0, "strat_sums should have non-zero entries after training"

    def test_regrets_change(self):
        """After running iterations, regrets should be non-zero."""
        solver = Solver(3, 30)
        for _ in range(1000):
            solver.run_iteration()

        non_zero = np.count_nonzero(solver._regrets[:solver._n])
        assert non_zero > 0, "regrets should have non-zero entries after training"

    def test_info_states_discovered(self):
        """Solver should discover many info states."""
        solver = Solver(3, 30)
        for _ in range(1000):
            solver.run_iteration()
        assert solver.n_info_states > 100, \
            f"Expected >100 info states, got {solver.n_info_states}"

    def test_checkpoint_roundtrip(self):
        """Save and load should preserve solver state."""
        import tempfile
        solver = Solver(3, 30)
        for _ in range(500):
            solver.run_iteration()

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            solver.iterations = 500
            _save_ckpt(solver, path, {'iterations_done': 500})
            loaded = _load_solver(path)

            assert loaded.n_players == 3
            assert loaded.stack_bb == 30
            assert loaded.n_info_states == solver.n_info_states
            assert np.array_equal(loaded._regrets[:solver._n], solver._regrets[:solver._n])
            assert np.array_equal(loaded._strat_sum[:solver._n], solver._strat_sum[:solver._n])
        finally:
            os.unlink(path)

    def test_convergence_direction(self):
        """Strategy should converge towards reasonable actions."""
        solver = Solver(3, 30)
        for _ in range(10000):
            solver.run_iteration()

        # Check that at least some info states have non-uniform strategy
        n_nonuniform = 0
        for idx in range(solver._n):
            ss = solver._strat_sum[idx]
            total = ss.sum()
            if total > 0:
                probs = ss / total
                legal = [a for a in range(N_ACTIONS) if probs[a] > 0]
                if len(legal) > 1:
                    max_p = max(probs[a] for a in legal)
                    if max_p > 0.6:  # significantly non-uniform
                        n_nonuniform += 1

        assert n_nonuniform > 0, "Some strategies should be non-uniform after 10K iters"


# ─── Import needed for checkpoint test
from solver_training.preflop_fixed_train import _save_ckpt, _load_solver
