"""
Comprehensive tests for preflop_fixed_train.py — verifies every rule from the spec.

Run with:  python3 -m pytest solver_training/test_preflop_fixed.py -v
"""

import sys
import os
import random
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_training.preflop_fixed_train import (
    State, Solver, position_names, legal_actions, apply_action,
    is_terminal, payoff, info_key, _hand_cat, _pack_hand_cat,
    _rfi_size, _three_bet_size, _squeeze_size, _four_bet_size,
    _is_ip_vs, _betting_complete, _save_ckpt, _load_solver,
    N_ACTIONS, BB, SB, RAKE_RATE, RAKE_CAP,
)


def _deal_state(n_players=6, stack_bb=100):
    """Deal a random state with proper hole cards."""
    deck = list(range(52))
    random.shuffle(deck)
    s = State(n_players, stack_bb)
    for p in range(n_players):
        s.hole[p] = [deck[p * 2], deck[p * 2 + 1]]
    return s


# ─── Position labels ──────────────────────────────────────────────────────────

class TestPositions:
    def test_2p(self):
        assert position_names(2) == ["SB", "BB"]

    def test_3p(self):
        assert position_names(3) == ["BTN", "SB", "BB"]

    def test_4p(self):
        assert position_names(4) == ["CO", "BTN", "SB", "BB"]

    def test_5p(self):
        assert position_names(5) == ["HJ", "CO", "BTN", "SB", "BB"]

    def test_6p(self):
        assert position_names(6) == ["LJ", "HJ", "CO", "BTN", "SB", "BB"]

    def test_7p(self):
        assert position_names(7) == ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"]

    def test_8p(self):
        assert position_names(8) == ["UTG1", "UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"]


# ─── RFI sizing ───────────────────────────────────────────────────────────────

class TestRFISizing:
    """RFI: BTN/SB open 3bb, all others open 2.5bb."""

    def test_btn_rfi_3bb(self):
        assert _rfi_size("BTN", BB) == 3 * BB

    def test_sb_rfi_3bb(self):
        assert _rfi_size("SB", BB) == 3 * BB

    def test_lj_rfi_2_5bb(self):
        assert _rfi_size("LJ", BB) == int(2.5 * BB)

    def test_hj_rfi_2_5bb(self):
        assert _rfi_size("HJ", BB) == int(2.5 * BB)

    def test_co_rfi_2_5bb(self):
        assert _rfi_size("CO", BB) == int(2.5 * BB)

    def test_utg_rfi_2_5bb(self):
        assert _rfi_size("UTG", BB) == int(2.5 * BB)

    def test_utg1_rfi_2_5bb(self):
        assert _rfi_size("UTG1", BB) == int(2.5 * BB)

    def test_6p_lj_opens_250(self):
        """In 6p, LJ (first to act) RFI should invest 250 total."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)  # LJ RFI
        assert s2.invested[0] == 250  # LJ invested 250 = 2.5bb

    def test_2p_sb_opens_300(self):
        """In 2p, SB/BTN RFI should invest 300 total."""
        s = _deal_state(2)
        s2 = apply_action(s, 2)  # SB RFI
        assert s2.invested[0] == 300  # SB invested 300 = 3bb


# ─── No open-limping ──────────────────────────────────────────────────────────

class TestNoLimping:
    """At raise_level=0, no position can limp (call action = 1 is not available)."""

    def test_no_limp_6p_first_to_act(self):
        s = _deal_state(6)
        acts = legal_actions(s)
        assert 1 not in acts, f"LJ should not be able to limp, got actions: {acts}"

    def test_no_limp_2p_sb(self):
        s = _deal_state(2)
        acts = legal_actions(s)
        assert 1 not in acts, f"SB/BTN should not be able to limp, got actions: {acts}"

    def test_no_limp_3p_btn(self):
        s = _deal_state(3)
        acts = legal_actions(s)
        assert 1 not in acts, f"BTN should not be able to limp, got actions: {acts}"

    def test_all_positions_no_limp(self):
        """Walk through every position at raise_level=0 and verify no limp."""
        for n in range(2, 9):
            s = _deal_state(n)
            acts = legal_actions(s)
            assert 1 not in acts, (
                f"{n}p: {s.pos(s.acting)} should not limp at raise_level=0, "
                f"got actions: {acts}"
            )


# ─── 3-bet sizing ─────────────────────────────────────────────────────────────

class TestThreeBetSizing:
    """3-bet: 9bb IP, 12bb OOP."""

    def test_3bet_ip_9bb(self):
        assert _three_bet_size(True, BB) == 9 * BB

    def test_3bet_oop_12bb(self):
        assert _three_bet_size(False, BB) == 12 * BB

    def test_6p_co_3bet_ip_vs_lj_open(self):
        """CO is IP vs LJ's open. CO's 3-bet should be 9bb = 900."""
        s = _deal_state(6)
        # LJ opens
        s2 = apply_action(s, 2)  # LJ RFI
        # HJ folds
        s3 = apply_action(s2, 0)  # HJ fold
        # CO faces the open — CO is IP vs LJ
        co_pos = s3.pos(s3.acting)
        assert co_pos == "CO"
        # CO 3-bets (action 2)
        s4 = apply_action(s3, 2)  # CO 3-bet
        assert s4.invested[2] == 900, f"CO 3bet should be 900, got {s4.invested[2]}"

    def test_6p_sb_3bet_oop_vs_lj_open(self):
        """SB is OOP vs LJ's open. SB's 3-bet should be 12bb = 1200."""
        s = _deal_state(6)
        # LJ opens, everyone folds to SB
        s2 = apply_action(s, 2)   # LJ RFI
        s3 = apply_action(s2, 0)  # HJ fold
        s4 = apply_action(s3, 0)  # CO fold
        s5 = apply_action(s4, 0)  # BTN fold
        # SB faces the open — SB is OOP
        sb_pos = s5.pos(s5.acting)
        assert sb_pos == "SB"
        # SB 3-bets (action 2)
        s6 = apply_action(s5, 2)  # SB 3-bet
        assert s6.invested[4] == 1200, f"SB 3bet should be 1200, got {s6.invested[4]}"


# ─── Squeeze sizing ────────────────────────────────────────────────────────────

class TestSqueezeSizing:
    """Squeeze: IP (3+n_callers)×open, OOP (4+n_callers)×open."""

    def test_squeeze_ip_1_caller(self):
        """IP squeeze vs 2.5bb open with 1 caller: (3+1)*250 = 1000."""
        assert _squeeze_size(True, 1, 250) == 1000

    def test_squeeze_oop_1_caller(self):
        """OOP squeeze vs 2.5bb open with 1 caller: (4+1)*250 = 1250."""
        assert _squeeze_size(False, 1, 250) == 1250

    def test_squeeze_ip_2_callers(self):
        """IP squeeze vs 2.5bb open with 2 callers: (3+2)*250 = 1250."""
        assert _squeeze_size(True, 2, 250) == 1250

    def test_squeeze_oop_2_callers(self):
        """OOP squeeze vs 2.5bb open with 2 callers: (4+2)*250 = 1500."""
        assert _squeeze_size(False, 2, 250) == 1500

    def test_squeeze_not_available_without_callers(self):
        """At raise_level=1 with 0 cold callers, squeeze (action 3) should not be available."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI
        # HJ faces the open — n_cold_callers is still 0
        assert s2.n_cold_callers == 0
        acts = legal_actions(s2)
        assert 3 not in acts, f"Squeeze should not be available with 0 callers, got: {acts}"

    def test_squeeze_available_with_callers(self):
        """At raise_level=1 with 1+ cold callers, squeeze (action 3) should be available."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI
        s3 = apply_action(s2, 1)  # HJ calls
        # CO faces the open with 1 cold caller
        assert s3.n_cold_callers == 1
        acts = legal_actions(s3)
        assert 3 in acts, f"Squeeze should be available with 1 caller, got: {acts}"

    def test_co_squeeze_ip_1_caller(self):
        """CO squeeze vs LJ open + HJ call: IP (3+1)*250 = 1000."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI (250)
        s3 = apply_action(s2, 1)  # HJ call
        s4 = apply_action(s3, 3)  # CO squeeze
        assert s4.invested[2] == 1000, f"CO squeeze should be 1000, got {s4.invested[2]}"

    def test_sb_squeeze_oop_1_caller(self):
        """SB squeeze vs LJ open + HJ call + CO/BTN folds: OOP (4+1)*250 = 1250."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI (250)
        s3 = apply_action(s2, 1)  # HJ call
        s4 = apply_action(s3, 0)  # CO fold
        s5 = apply_action(s4, 0)  # BTN fold
        # SB faces the open with 1 cold caller
        s6 = apply_action(s5, 3)  # SB squeeze
        assert s6.invested[4] == 1250, f"SB squeeze should be 1250, got {s6.invested[4]}"


# ─── 4-bet sizing ──────────────────────────────────────────────────────────────

class TestFourBetSizing:
    """4-bet: 2.3× IP, 2.8× OOP (of the facing 3-bet/squeeze size)."""

    def test_4bet_ip(self):
        assert _four_bet_size(True, 900) == int(2.3 * 900)

    def test_4bet_oop(self):
        assert _four_bet_size(False, 900) == int(2.8 * 900)

    def test_4bet_ip_vs_3bet_900(self):
        """IP 4-bet vs 9bb 3-bet: 2.3 * 900 = 2070."""
        assert _four_bet_size(True, 900) == 2070

    def test_4bet_oop_vs_3bet_900(self):
        """OOP 4-bet vs 9bb 3-bet: 2.8 * 900 = 2520."""
        assert _four_bet_size(False, 900) == 2520

    def test_lj_4bet_oop_vs_co_3bet(self):
        """LJ is OOP vs CO's 3-bet. LJ 4-bet = 2.8 * 900 = 2520."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI (250)
        s3 = apply_action(s2, 0)  # HJ fold
        s4 = apply_action(s3, 2)  # CO 3-bet IP (900)
        s5 = apply_action(s4, 0)  # BTN fold
        s6 = apply_action(s5, 0)  # SB fold
        s7 = apply_action(s6, 0)  # BB fold
        # Action back to LJ — LJ is OOP vs CO
        assert s7.pos(s7.acting) == "LJ"
        s8 = apply_action(s7, 2)  # LJ 4-bet OOP
        assert s8.invested[0] == 2520, f"LJ 4bet should be 2520, got {s8.invested[0]}"

    def test_btn_4bet_ip_vs_co_3bet(self):
        """BTN is IP vs CO's 3-bet. BTN 4-bet = 2.3 * 900 = 2070."""
        # CO opens, BTN 3-bets, CO 4-bets — wait, we need BTN to 4-bet.
        # Better: LJ opens, HJ folds, CO calls, BTN 3-bets IP, LJ folds, CO 4-bets
        # Actually simplest: build a state where BTN faces a 3-bet and is IP
        # LJ opens, CO 3-bets, action to BTN who 4-bets IP
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI (250)
        s3 = apply_action(s2, 0)  # HJ fold
        s4 = apply_action(s3, 2)  # CO 3-bet IP (900)
        # BTN faces the 3-bet — BTN is IP vs CO
        assert s4.pos(s4.acting) == "BTN"
        # BTN 4-bets (action 2)
        s5 = apply_action(s4, 2)  # BTN 4-bet IP
        assert s5.invested[3] == 2070, f"BTN 4bet should be 2070, got {s5.invested[3]}"


# ─── 5-bet = all-in only ──────────────────────────────────────────────────────

class TestFiveBetAllIn:
    """At raise_level>=3, no sizing choices — only fold, call, all-in."""

    def test_no_raise_at_level_3(self):
        """Facing a 4-bet, no raise action should be available."""
        # Build a state at raise_level=3
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI
        s3 = apply_action(s2, 0)  # HJ fold
        s4 = apply_action(s3, 2)  # CO 3-bet
        s5 = apply_action(s4, 0)  # BTN fold
        s6 = apply_action(s5, 0)  # SB fold
        s7 = apply_action(s6, 0)  # BB fold
        s8 = apply_action(s7, 2)  # LJ 4-bet
        # CO faces the 4-bet
        assert s8.raise_level == 3
        acts = legal_actions(s8)
        # Should only have fold(0), call(1), allin(4)
        assert 2 not in acts, f"No 4-bet sizing at raise_level=3, got: {acts}"
        assert 3 not in acts, f"No squeeze at raise_level=3, got: {acts}"
        assert 0 in acts, "fold should be available"
        assert 1 in acts, "call should be available"
        assert 4 in acts, "all-in should be available"

    def test_5bet_is_allin_only(self):
        """5-bet action is always all-in (action 4)."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI
        s3 = apply_action(s2, 0)  # HJ fold
        s4 = apply_action(s3, 2)  # CO 3-bet
        s5 = apply_action(s4, 0)  # BTN fold
        s6 = apply_action(s5, 0)  # SB fold
        s7 = apply_action(s6, 0)  # BB fold
        s8 = apply_action(s7, 2)  # LJ 4-bet
        # CO 5-bets (all-in)
        s9 = apply_action(s8, 4)  # CO all-in
        # After CO all-in (which is a raise), LJ faces a 5-bet
        # LJ should only have fold, call, or all-in
        if not is_terminal(s9):
            acts = legal_actions(s9)
            assert 2 not in acts, f"No raise sizing at raise_level>=4, got: {acts}"
            assert 3 not in acts, f"No squeeze at raise_level>=4, got: {acts}"


# ─── Rake ──────────────────────────────────────────────────────────────────────

class TestRake:
    """Rake: 3% of pot, capped at 2×BB."""

    def test_rake_small_pot(self):
        """Small pot: 3% rake applies."""
        pot = 300  # 3bb pot
        expected_rake = min(pot * RAKE_RATE, RAKE_CAP * BB)
        assert expected_rake == 9, f"Rake on 300 pot should be 9, got {expected_rake}"

    def test_rake_large_pot_capped(self):
        """Large pot: rake should be capped at 2×BB = 200."""
        pot = 10000  # 100bb pot
        expected_rake = min(pot * RAKE_RATE, RAKE_CAP * BB)
        assert expected_rake == 200, f"Rake should be capped at 200, got {expected_rake}"

    def test_rake_applied_at_fold_terminal(self):
        """Rake should be applied even when a player folds preflop."""
        # 2p: SB opens, BB folds — pot is 350 (SB 50 + BB 100 + SB's 200 raise)
        s = _deal_state(2)
        s2 = apply_action(s, 2)   # SB RFI (300 total, pot = 150+250=400)
        s3 = apply_action(s2, 0)  # BB fold
        assert is_terminal(s3)
        gains = payoff(s3)
        # SB wins pot minus rake
        pot = s3.pot
        rake = min(pot * RAKE_RATE, RAKE_CAP * BB)
        assert gains[0] == pot - rake - s3.invested[0], (
            f"SB should win pot-rake minus invested, got gains[0]={gains[0]}"
        )


# ─── 169-hand abstraction ─────────────────────────────────────────────────────

class TestHandAbstraction:
    """169-hand categories cover all 1326 combos."""

    def test_169_categories(self):
        """There should be exactly 169 canonical hand categories."""
        cats = set()
        for c1 in range(52):
            for c2 in range(c1 + 1, 52):
                cats.add(_hand_cat([c1, c2]))
        assert len(cats) == 169, f"Expected 169 hand categories, got {len(cats)}"

    def test_pair_category(self):
        assert _hand_cat([0, 1]) == "22"   # 2c 2d (rank_idx 0,0 same = pair)

    def test_suited_category(self):
        # Ac = rank 12 * 4 + suit 0 = 48; Kc = rank 11 * 4 + suit 0 = 44
        assert _hand_cat([48, 44]) == "AKs"  # Ac Kc (same suit)

    def test_offsuit_category(self):
        # Ac = 48; Kd = 45 (rank 11, suit 1)
        assert _hand_cat([48, 45]) == "AKo"  # Ac Kd (different suits)

    def test_pack_unpack_roundtrip(self):
        """Pack and hand_cat strings should be consistent."""
        from solver_training.preflop_fixed_train import RANKS
        for c1 in range(52):
            for c2 in range(c1 + 1, 52):
                cat = _hand_cat([c1, c2])
                r1r2, suit = _pack_hand_cat(cat)
                r1 = (r1r2 >> 4) & 0xF
                r2 = r1r2 & 0xF
                reconstructed = RANKS[r1] + RANKS[r2]
                if suit == 2:
                    assert reconstructed == cat, f"Pair mismatch: {cat} vs {reconstructed}"
                elif suit == 1:
                    assert reconstructed + "s" == cat, f"Suited mismatch: {cat} vs {reconstructed}s"
                else:
                    assert reconstructed + "o" == cat, f"Offsuit mismatch: {cat} vs {reconstructed}o"


# ─── Legal actions at every decision point ─────────────────────────────────────

class TestLegalActions:
    """Verify legal actions are correct at every game state."""

    def test_unopened_has_no_call(self):
        """At raise_level=0, no position can call (limp)."""
        for n in range(2, 9):
            s = _deal_state(n)
            acts = legal_actions(s)
            assert 1 not in acts, f"{n}p: call should not be available at raise_level=0"

    def test_unopened_has_fold_rfi(self):
        """At raise_level=0 with deep stacks, actions are fold(0), RFI(2) only.
        All-in only appears when stacks are shallow (stack < 3x RFI size)."""
        for n in range(2, 9):
            s = _deal_state(n, stack_bb=100)
            acts = legal_actions(s)
            assert 0 in acts, "fold should be available"
            assert 2 in acts, "RFI should be available"
            assert 4 not in acts, "all-in should NOT be available at 100bb RFI"

    def test_unopened_shallow_has_allin(self):
        """At raise_level=0 with shallow stacks, all-in is available."""
        for n in range(2, 9):
            s = _deal_state(n, stack_bb=5)  # 5bb: RFI 2.5-3bb = 50-60% of stack
            acts = legal_actions(s)
            assert 4 in acts, "all-in should be available at 5bb RFI"

    def test_facing_open_has_fold_call(self):
        """At raise_level=1, fold and call are always available."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI
        acts = legal_actions(s2)
        assert 0 in acts, "fold should be available"
        assert 1 in acts, "call should be available"

    def test_facing_3bet_has_no_squeeze(self):
        """At raise_level=2, no squeeze action available."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI
        s3 = apply_action(s2, 0)  # HJ fold
        s4 = apply_action(s3, 2)  # CO 3-bet
        # Now LJ faces the 3-bet
        s5 = apply_action(s4, 0)  # BTN fold
        s6 = apply_action(s5, 0)  # SB fold
        s7 = apply_action(s6, 0)  # BB fold
        assert s7.raise_level == 2
        acts = legal_actions(s7)
        assert 3 not in acts, "squeeze should not be available at raise_level=2"

    def test_facing_4bet_only_fold_call_allin(self):
        """At raise_level=3, only fold/call/all-in available."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI
        s3 = apply_action(s2, 0)  # HJ fold
        s4 = apply_action(s3, 2)  # CO 3-bet
        s5 = apply_action(s4, 0)  # BTN fold
        s6 = apply_action(s5, 0)  # SB fold
        s7 = apply_action(s6, 0)  # BB fold
        s8 = apply_action(s7, 2)  # LJ 4-bet
        assert s8.raise_level == 3
        acts = legal_actions(s8)
        for a in acts:
            assert a in (0, 1, 4), f"Only fold/call/allin at level 3, got action {a}"


# ─── Game tree walk ────────────────────────────────────────────────────────────

class TestGameTreeWalk:
    """Walk the game tree to verify it terminates correctly."""

    def _walk(self, s, depth=0, max_depth=20):
        """DFS through the game tree. Returns (total_nodes, max_depth_reached)."""
        if is_terminal(s) or depth > max_depth:
            return (1, depth)

        total = 1
        max_d = depth
        acts = legal_actions(s)
        assert len(acts) > 0, f"No legal actions at depth {depth}"

        for a in acts:
            ns = apply_action(s, a)
            t, d = self._walk(ns, depth + 1, max_depth)
            total += t
            max_d = max(max_d, d)
        return (total, max_d)

    def test_2p_tree_walks(self):
        """2-player game tree should terminate."""
        s = _deal_state(2)
        total, max_d = self._walk(s)
        assert total > 0
        assert max_d < 30, f"2p tree too deep: {max_d}"

    def test_6p_tree_walks(self):
        """6-player game tree should terminate."""
        s = _deal_state(6)
        total, max_d = self._walk(s)
        assert total > 0
        assert max_d < 30, f"6p tree too deep: {max_d}"

    def test_8p_tree_walks(self):
        """8-player game tree should terminate."""
        s = _deal_state(8)
        total, max_d = self._walk(s)
        assert total > 0
        assert max_d < 30, f"8p tree too deep: {max_d}"


# ─── Chip accounting ──────────────────────────────────────────────────────────

class TestChipAccounting:
    """Total chips in the system must be conserved."""

    def test_chips_conserved_after_actions(self):
        """pot + sum(stacks) + sum(folded_chips) = total_buy_in * n_players."""
        s = _deal_state(6)
        total = 6 * s.stack_bb * BB
        for a in legal_actions(s):
            ns = apply_action(s, a)
            pot_stacks = ns.pot + sum(ns.stacks)
            # Some chips are in invested amounts that are already in the pot
            # pot = sum(invested), so pot + sum(stacks) should equal total
            assert pot_stacks == total, (
                f"Chips not conserved after action {a}: "
                f"pot+stacks={pot_stacks}, expected={total}"
            )

    def test_chips_conserved_deep_tree(self):
        """Walk several actions deep and verify chip conservation."""
        s = _deal_state(6)
        total = 6 * s.stack_bb * BB

        actions_seq = [2, 1, 0, 2, 0, 0, 0]  # RFI, call, fold, 3bet, fold, fold, fold
        ns = s
        for a in actions_seq:
            if is_terminal(ns):
                break
            acts = legal_actions(ns)
            if a not in acts:
                a = acts[0]  # fallback
            ns = apply_action(ns, a)
            assert ns.pot + sum(ns.stacks) == total

    def test_payoff_sums_to_pot_minus_rake(self):
        """Sum of all payoffs should equal pot minus rake (zero-sum with rake)."""
        # Build a terminal state
        s = _deal_state(2)
        s2 = apply_action(s, 2)   # SB RFI
        s3 = apply_action(s2, 0)  # BB fold
        assert is_terminal(s3)
        gains = payoff(s3)
        pot = s3.pot
        rake = min(pot * RAKE_RATE, RAKE_CAP * BB)
        assert sum(gains) == -(rake), (
            f"Payoff sum should be -rake={-rake}, got {sum(gains)}"
        )


# ─── Betting completion ───────────────────────────────────────────────────────

class TestBettingCompletion:
    """Betting should complete correctly."""

    def test_all_fold_to_rfi_is_terminal(self):
        """If everyone folds to an RFI, it's terminal."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI
        s3 = apply_action(s2, 0)  # HJ fold
        s4 = apply_action(s3, 0)  # CO fold
        s5 = apply_action(s4, 0)  # BTN fold
        s6 = apply_action(s5, 0)  # SB fold
        s7 = apply_action(s6, 0)  # BB fold
        assert is_terminal(s7)

    def test_call_then_check_is_not_terminal_unless_bb(self):
        """After RFI and one call, betting continues."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI
        s3 = apply_action(s2, 1)  # HJ call
        # Not terminal — other players still to act
        assert not is_terminal(s3)

    def test_all_call_rfi_is_terminal(self):
        """If everyone calls the RFI, betting is complete."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)   # LJ RFI
        s3 = apply_action(s2, 1)  # HJ call
        s4 = apply_action(s3, 1)  # CO call
        s5 = apply_action(s4, 1)  # BTN call
        s6 = apply_action(s5, 1)  # SB call
        s7 = apply_action(s6, 1)  # BB call
        assert is_terminal(s7), "All players called RFI — should be terminal"


# ─── Checkpoint save/load ─────────────────────────────────────────────────────

class TestCheckpoint:
    """Checkpoint save/load round-trip."""

    def test_save_load_roundtrip(self):
        """Save a solver, load it back, verify data matches."""
        solver = Solver(6, 100)
        # Run a few iterations
        for _ in range(10):
            solver.run_iteration()

        n_before = solver.n_info_states
        iters_before = solver.iterations

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            progress = {"iterations_done": 10, "n_players": 6, "stack_bb": 100}
            _save_ckpt(solver, path, progress)

            loaded = _load_solver(path)
            assert loaded.n_players == 6
            assert loaded.stack_bb == 100
            assert loaded.n_info_states == n_before
            assert loaded.iterations == iters_before
            assert np.array_equal(loaded._regrets[:n_before], solver._regrets[:n_before])
            assert np.array_equal(loaded._strat_sum[:n_before], solver._strat_sum[:n_before])
        finally:
            os.unlink(path)
            pp = path.replace(".pkl", ".progress.json")
            if os.path.exists(pp):
                os.unlink(pp)


# ─── Solver runs iterations ───────────────────────────────────────────────────

class TestSolverRuns:
    """Solver should run iterations without crash."""

    def test_2p_100_iterations(self):
        solver = Solver(2, 100)
        for _ in range(100):
            solver.run_iteration()
        assert solver.n_info_states > 0

    def test_6p_50_iterations(self):
        solver = Solver(6, 100)
        for _ in range(50):
            solver.run_iteration()
        assert solver.n_info_states > 0

    def test_8p_50_iterations(self):
        solver = Solver(8, 100)
        for _ in range(50):
            solver.run_iteration()
        assert solver.n_info_states > 0

    def test_short_stack_30bb(self):
        solver = Solver(2, 30)
        for _ in range(100):
            solver.run_iteration()
        assert solver.n_info_states > 0

    def test_deep_stack_200bb(self):
        solver = Solver(2, 200)
        for _ in range(100):
            solver.run_iteration()
        assert solver.n_info_states > 0


# ─── IP/OOP determination ─────────────────────────────────────────────────────

class TestIPOOP:
    """Test in-position vs out-of-position determination."""

    def test_co_ip_vs_lj(self):
        assert _is_ip_vs("CO", "LJ", 6) is True

    def test_co_ip_vs_hj(self):
        assert _is_ip_vs("CO", "HJ", 6) is True

    def test_btn_ip_vs_co(self):
        assert _is_ip_vs("BTN", "CO", 6) is True

    def test_lj_oop_vs_co(self):
        assert _is_ip_vs("LJ", "CO", 6) is False

    def test_sb_oop_vs_btn(self):
        assert _is_ip_vs("SB", "BTN", 6) is False

    def test_bb_oop_vs_co(self):
        assert _is_ip_vs("BB", "CO", 6) is False

    def test_2p_sb_oop_vs_bb(self):
        """In 2p, SB acts first postflop (OOP) vs BB."""
        assert _is_ip_vs("SB", "BB", 2) is False

    def test_2p_bb_ip_vs_sb(self):
        """In 2p, BB acts last postflop (IP) vs SB."""
        assert _is_ip_vs("BB", "SB", 2) is True


# ─── Info key properties ──────────────────────────────────────────────────────

class TestInfoKey:
    """Info keys should distinguish different game states."""

    def test_different_positions_different_keys(self):
        """Same hand, different position → different key."""
        s1 = _deal_state(6)
        s2 = _deal_state(6)
        # Copy hole cards to be the same
        s2.hole[0] = s1.hole[0][:]
        k1 = info_key(0, s1, [])
        k2 = info_key(1, s2, [])  # player 1 = HJ
        assert k1 != k2

    def test_different_raise_levels_different_keys(self):
        """Same hand, different raise_level → different key."""
        s = _deal_state(6)
        s2 = apply_action(s, 2)  # LJ RFI
        k0 = info_key(0, s, [])    # LJ at level 0
        k1 = info_key(0, s2, [2])  # LJ at level 1 (after their own RFI)
        # These should be different because raise_level differs
        # (k0 is LJ before acting, k1 is LJ after RFI — different contexts)
        # Actually info_key for the acting player includes raise_level
        # But k0 has raise_level=0 and k1 has raise_level=1
        assert k0 != k1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
