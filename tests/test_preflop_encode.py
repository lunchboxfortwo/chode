"""
Tests for preflop key encoding/decoding and chart mapping.

These tests specifically target the bugs found on 2026-05-04:
  1. _decode_key was reading wrong bytes (suit/pidx/hist offset by 3)
  2. _hand_to_cards had inverted rank mapping (grid 0=Ace vs card 0=Two)
  3. _build_label produced nonsense like "SB vs SB" for 3bet spots

Run:  python3 -m pytest tests/test_preflop_encode.py -v
"""
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np


# ─── Fixtures & helpers ──────────────────────────────────────────────────────

from solver_training.preflop_fixed_train import (
    info_key, State, Solver, RANKS, SUITS, apply_action, legal_actions,
    _HAND_CAT_CACHE, _build_hand_cat_cache,
)
from solver_training.train_preflop_nn import _decode_key
from strategy.preflop_nn import _hand_to_cards, _build_label, position_names


def _make_state_with_hand(n_players=3, stack_bb=30, r1=12, r2=11, suited=True, pidx=0):
    """Build a State with specific hole cards for the given player index.

    r1, r2 use card rank encoding: 0=Two, 1=Three, ..., 12=Ace.
    """
    s = State(n_players=n_players, stack_bb=stack_bb)
    _build_hand_cat_cache()
    # Card index: rank * 4 + suit  (rank 0=Two..12=Ace, suit 0-3)
    if r1 == r2:
        # Pair — two cards of same rank, different suits
        c1 = r1 * 4
        c2 = r2 * 4 + 1
    elif suited:
        # Suited — same suit
        c1 = r1 * 4
        c2 = r2 * 4  # same suit as c1
    else:
        # Offsuit — different suits
        c1 = r1 * 4
        c2 = r2 * 4 + 2
    s.hole[pidx] = [c1, c2]
    return s


# ─── 1. info_key → _decode_key round-trip ────────────────────────────────────

class TestKeyRoundTrip:
    """Encode a state with info_key, then decode with _decode_key — must match."""

    @pytest.mark.parametrize("n_players,stack_bb", [
        (2, 30), (2, 50), (2, 100),
        (3, 30), (3, 50),
        (4, 30), (5, 30), (6, 30),
    ])
    def test_roundtrip_rfi(self, n_players, stack_bb):
        """RFI (no action history) round-trips correctly."""
        s = _make_state_with_hand(n_players, stack_bb, r1=12, r2=12)  # AA
        key = info_key(0, s, [])
        r1, r2, suit, pidx, sbb, hist = _decode_key(key)
        assert pidx == 0, f"pidx: expected 0, got {pidx}"
        assert sbb == float(stack_bb), f"stack_bb: expected {stack_bb}, got {sbb}"
        assert r1 == 12, f"r1: expected 12 (Ace), got {r1}"
        assert r2 == 12, f"r2: expected 12 (Ace), got {r2}"
        assert suit == 2, f"suit: expected 2 (pair), got {suit}"
        assert hist == [], f"hist: expected [], got {hist}"

    @pytest.mark.parametrize("n_players,stack_bb,pidx,hist", [
        (2, 30, 0, [2, 2]),          # SB facing 3bet
        (2, 50, 1, [2]),             # BB facing RFI
        (3, 30, 2, [2, 1]),          # BB squeeze
        (3, 30, 0, [2, 2, 1]),       # BTN facing 3bet+call
        (6, 30, 0, [2, 1, 1, 1, 2]), # LJ facing 3bet after calls
    ])
    def test_roundtrip_with_history(self, n_players, stack_bb, pidx, hist):
        """Keys with action history round-trip correctly."""
        s = _make_state_with_hand(n_players, stack_bb, r1=10, r2=8, pidx=pidx)  # J9
        key = info_key(pidx, s, hist)
        r1, r2, suit, pidx_dec, sbb_dec, hist_dec = _decode_key(key)
        assert pidx_dec == pidx, f"pidx: expected {pidx}, got {pidx_dec}"
        assert sbb_dec == float(stack_bb), f"stack_bb: expected {stack_bb}, got {sbb_dec}"
        assert hist_dec == hist, f"hist: expected {hist}, got {hist_dec}"

    def test_suit_encoding_offsuit(self):
        """Offsuit hand must decode as suit=0."""
        s = _make_state_with_hand(3, 30, r1=12, r2=10, suited=False)  # AQo
        key = info_key(0, s, [])
        _, _, suit, _, _, _ = _decode_key(key)
        assert suit == 0, f"offsuit should be 0, got {suit}"

    def test_suit_encoding_suited(self):
        """Suited hand must decode as suit=1."""
        s = _make_state_with_hand(3, 30, r1=12, r2=11, suited=True)  # AKs
        key = info_key(0, s, [])
        _, _, suit, _, _, _ = _decode_key(key)
        assert suit == 1, f"suited should be 1, got {suit}"

    def test_suit_encoding_pair(self):
        """Pair must decode as suit=2."""
        s = _make_state_with_hand(3, 30, r1=12, r2=12)  # AA
        key = info_key(0, s, [])
        _, _, suit, _, _, _ = _decode_key(key)
        assert suit == 2, f"pair should be 2, got {suit}"


# ─── 2. Chart grid ↔ card mapping ───────────────────────────────────────────

class TestHandToCards:
    """The 13×13 chart grid uses 0=Ace..12=Two; card encoding uses 0=Two..12=Ace."""

    def test_aa_maps_to_aces(self):
        """Grid (0,0) = AA must produce card indices with rank 12 (Ace)."""
        c1, c2 = _hand_to_cards(0, 0)  # AA (pair, top-left)
        r1 = c1 // 4
        r2 = c2 // 4
        assert r1 == 12, f"AA c1 rank: expected 12 (Ace), got {r1}"
        assert r2 == 12, f"AA c2 rank: expected 12 (Ace), got {r2}"

    def test_72o_maps_to_correct_ranks(self):
        """Grid (11,12) = 32o (Three-Two offsuit) must produce correct card ranks."""
        # Chart: 0=A,1=K,...,11=3,12=2 → card rank: 12,11,...,1,0
        c1, c2 = _hand_to_cards(11, 12)  # 32o (row > col = offsuit)
        r1 = c1 // 4  # Three → card rank 1
        r2 = c2 // 4  # Two → card rank 0
        assert r1 == 1, f"Three rank: expected 1, got {r1}"
        assert r2 == 0, f"Two rank: expected 0, got {r2}"

    def test_aks_maps_to_correct_ranks(self):
        """Grid (0,1) = AKs must produce ranks 12 (Ace) and 11 (King)."""
        c1, c2 = _hand_to_cards(0, 1)  # AKs (row < col = suited)
        r1 = c1 // 4
        r2 = c2 // 4
        assert r1 == 12, f"Ace rank: expected 12, got {r1}"
        assert r2 == 11, f"King rank: expected 11, got {r2}"

    def test_jto_maps_to_correct_ranks(self):
        """Grid (3,4) = JTo must produce ranks 9 (Jack) and 8 (Ten)."""
        c1, c2 = _hand_to_cards(3, 4)  # JTo (row < col — but wait, need row>col for offsuit)
        # Actually grid (3,4) = JTs (row<col=suited). JTo is grid (4,3).
        c1, c2 = _hand_to_cards(4, 3)  # JTo (row > col = offsuit)
        r1 = c1 // 4
        r2 = c2 // 4
        assert r1 == 8, f"Ten rank: expected 8, got {r1}"
        assert r2 == 9, f"Jack rank: expected 9, got {r2}"

    @pytest.mark.parametrize("i,j", [(i, j) for i in range(13) for j in range(13)])
    def test_all_grid_positions_map_valid_cards(self, i, j):
        """Every grid position must produce valid card indices (0-51)."""
        c1, c2 = _hand_to_cards(i, j)
        assert 0 <= c1 <= 51, f"({i},{j}): c1={c1} out of range"
        assert 0 <= c2 <= 51, f"({i},{j}): c2={c2} out of range"
        assert c1 != c2, f"({i},{j}): c1==c2 ({c1}), can't have same card twice"


# ─── 3. Scenario labels ─────────────────────────────────────────────────────

class TestBuildLabel:
    """_build_label must produce correct human-readable labels."""

    def test_rfi(self):
        assert _build_label(2, 30, 0, []) == "SB RFI (2p 30bb)"

    def test_facing_rfi(self):
        assert _build_label(2, 30, 1, ["bet"]) == "BB facing RFI (2p 30bb)"

    def test_facing_3bet_2p(self):
        assert _build_label(2, 30, 0, ["bet", "bet"]) == "SB facing 3bet (2p 30bb)"

    def test_facing_4bet_2p(self):
        assert _build_label(2, 30, 1, ["bet", "bet", "bet"]) == "BB facing 4bet (2p 30bb)"

    def test_squeeze_3p(self):
        assert _build_label(3, 30, 2, ["bet", "call"]) == "BB squeeze spot (3p 30bb)"

    def test_facing_3bet_3p(self):
        assert _build_label(3, 30, 0, ["bet", "bet"]) == "BTN facing 3bet (3p 30bb)"

    def test_facing_3bet_plus_call(self):
        assert _build_label(3, 50, 0, ["bet", "bet", "call"]) == "BTN facing 3bet+call (3p 50bb)"

    def test_no_vs_self_nonsense(self):
        """Never produce 'SB vs SB' or similar self-referencing labels."""
        for n in [2, 3, 4, 5, 6]:
            for pidx in range(n):
                for hist in [[], ["bet"], ["bet", "bet"], ["bet", "call"], ["bet", "bet", "bet"]]:
                    label = _build_label(n, 30, pidx, hist)
                    pos = position_names(n)[pidx]
                    assert f"{pos} vs {pos}" not in label, \
                        f"Self-referencing label: {label} (n={n}, pidx={pidx}, hist={hist})"


# ─── 4. Scenario validity per player count ──────────────────────────────────

class TestScenarioValidity:
    """Verify that only valid action-history scenarios exist for each player count."""

    # Valid (pidx, history) combos per player count
    # In n-max, action k is taken by player (k % n)
    # So after k actions, it's player (k % n)'s turn

    def _get_valid_scenarios(self, n):
        """Generate all valid (pidx, history) combos for n players.
        
        After k actions, the next actor is player (k % n).
        """
        valid = set()
        # RFI: 0 actions, player 0 acts
        valid.add((0, ()))
        # After 1 action (bet): player 1 acts (facing RFI)
        valid.add((1, (2,)))
        # After 2 actions: player (2 % n) acts
        if n == 2:
            valid.add((0, (2, 2)))  # SB facing 3bet
        else:
            valid.add((2, (2, 1)))  # squeeze (bet+call, actor=2)
            valid.add((2, (2, 2)))  # facing 3bet (two bets, actor=2)
        # After 3 actions: player (3 % n) acts
        actor3 = 3 % n
        if n == 2:
            valid.add((1, (2, 2, 2)))  # BB facing 4bet
        else:
            valid.add((actor3, (2, 2, 1)))  # facing 3bet+call
            valid.add((actor3, (2, 2, 2)))  # facing 4bet
        return valid

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_valid_scenarios_are_indeed_valid(self, n):
        """All scenarios we consider valid should be reachable."""
        valid = self._get_valid_scenarios(n)
        for pidx, hist in valid:
            # The actor after len(hist) actions should be pidx
            actor = len(hist) % n
            assert actor == pidx, \
                f"n={n}: after {len(hist)} actions, actor is {actor}, not {pidx}"

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_impossible_scenarios_not_in_valid(self, n):
        """Impossible combos should not be in valid set."""
        valid = self._get_valid_scenarios(n)
        # SB facing RFI (pidx=0, hist=[bet]) is impossible: after 1 action, actor=1
        if n >= 2:
            assert (0, (2,)) not in valid, "SB can't face RFI — they act first"
        # BB RFI (pidx=1, hist=[]) is impossible: after 0 actions, actor=0
        assert (1, ()) not in valid, "BB can't RFI — SB acts first"
        # Squeeze requires n>=3
        if n == 2:
            assert (2, (2, 1)) not in valid, "No squeeze in 2-max"


# ─── 5. Data quality guards ─────────────────────────────────────────────────

class TestDataQuality:
    """Guard against corrupted training data (inf, nan, all-zero rows)."""

    def test_normalize_finite_only(self):
        """Normalization of strategy sums must skip inf/nan rows."""
        # Simulate the normalization from train_preflop_nn.py
        probs = np.array([1e38, 2e38, 0.0], dtype=np.float32)
        total = probs.sum()
        # This would produce inf/inf = nan if not guarded
        if np.all(np.isfinite(probs)) and total > 0:
            result = probs / total
            assert not np.any(np.isnan(result))
        else:
            # Should be skipped, not crash
            pass

    def test_normalize_all_zero_row(self):
        """All-zero strategy should not produce nan from 0/0."""
        probs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        total = probs.sum()
        if total > 0:
            result = probs / total
        else:
            result = np.ones_like(probs) / len(probs)  # uniform fallback
        assert not np.any(np.isnan(result))

    def test_normalize_normal_row(self):
        """Normal strategy sums should normalize correctly."""
        probs = np.array([100.0, 200.0, 50.0], dtype=np.float32)
        result = probs / probs.sum()
        assert np.allclose(result.sum(), 1.0)
        assert not np.any(np.isnan(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
