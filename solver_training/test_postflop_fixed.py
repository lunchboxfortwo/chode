"""
Comprehensive tests for postflop_fixed_train game logic.

Covers:
  1. Chip conservation invariant (pot + stacks == const after every action)
  2. Equal investment after call (invested[caller] == facing_size)
  3. Facing bet resolution in multi-player (no free rides)
  4. Game tree termination (2p and 3p)
  5. Allin edge cases (short stack, over-shove, multi-way allin)
  6. Betting round completion (check-around, bet-call, bet-raise-call)
  7. Key encoding round-trip (trainer info_key matches runtime _fixed_key_bytes)
  8. Street advancement
  9. Aggression cap enforcement

Run:  python3 -m pytest solver_training/test_postflop_fixed.py -v
"""
import random
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_training.postflop_fixed_train import (
    State, apply_action, legal_actions, is_terminal, needs_card,
    payoff, info_key, _put_in, _end_street, _call_amt, _raise_total,
    _bet_small, _bet_large, _hand_cat, _pack_hand_cat, _board_norm_raw,
    BB, SB, BUY_IN, N_ACTIONS, AGG_CAP, MIN_BET, RAKE_RATE, RAKE_CAP,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_state_2p(stacks=None, invested=None, street=0, acting=0,
                  facing_size=0, agg_count=0):
    """Create a 2p postflop state with reasonable defaults.
    
    In 2p postflop: SB=50, BB=100 blinds.
    Pot = sum(invested) = 150, stacks = BUY_IN - invested.
    """
    deck = list(range(52))
    random.shuffle(deck)
    if invested is None:
        invested = [SB, BB]
    if stacks is None:
        stacks = [BUY_IN - invested[0], BUY_IN - invested[1]]
    pot = sum(invested)
    s = State(
        hole=[[deck[0], deck[1]], [deck[2], deck[3]]],
        board=[deck[4], deck[5], deck[6]],
        pot=pot, stacks=stacks, invested=invested,
        n_players=2, street=street, acting=acting,
    )
    s.facing_size = facing_size
    s.agg_count = agg_count
    return s


def make_state_3p(stacks=None, invested=None, street=0, acting=0,
                  facing_size=0, agg_count=0):
    """Create a 3p postflop state with reasonable defaults.
    
    In 3p postflop: SB=50, BB=100, BTN=0 blinds.
    Pot = sum(invested) = 150, stacks = BUY_IN - invested.
    """
    deck = list(range(52))
    random.shuffle(deck)
    if invested is None:
        invested = [SB, BB, 0]
    if stacks is None:
        stacks = [BUY_IN - invested[0], BUY_IN - invested[1], BUY_IN - invested[2]]
    pot = sum(invested)
    s = State(
        hole=[[deck[0], deck[1]], [deck[2], deck[3]], [deck[4], deck[5]]],
        board=[deck[6], deck[7], deck[8]],
        pot=pot, stacks=stacks, invested=invested,
        n_players=3, street=street, acting=acting,
    )
    s.facing_size = facing_size
    s.agg_count = agg_count
    return s


def total_chips(s):
    """Total chips in the system (pot + all stacks)."""
    return s.pot + sum(s.stacks)


# ─── 1. Chip Conservation ────────────────────────────────────────────────────

class TestChipConservation:
    """pot + stacks must remain constant after every action."""

    def test_2p_check_check(self):
        s = make_state_2p()
        expected = total_chips(s)
        s1 = apply_action(s, 1)  # P0 check
        assert total_chips(s1) == expected
        s2 = apply_action(s1, 1)  # P1 check
        assert total_chips(s2) == expected

    def test_2p_bet_call(self):
        s = make_state_2p()
        expected = total_chips(s)
        s1 = apply_action(s, 2)  # P0 bet
        assert total_chips(s1) == expected
        s2 = apply_action(s1, 1)  # P1 call
        assert total_chips(s2) == expected

    def test_2p_bet_raise_call(self):
        s = make_state_2p()
        expected = total_chips(s)
        s1 = apply_action(s, 2)  # P0 bet
        s2 = apply_action(s1, 2)  # P1 raise
        s3 = apply_action(s2, 1)  # P0 call
        assert total_chips(s1) == expected
        assert total_chips(s2) == expected
        assert total_chips(s3) == expected

    def test_2p_bet_fold(self):
        s = make_state_2p()
        expected = total_chips(s)
        s1 = apply_action(s, 3)  # P0 bet large
        s2 = apply_action(s1, 0)  # P1 fold
        assert total_chips(s1) == expected
        assert total_chips(s2) == expected

    def test_3p_bet_call_call(self):
        s = make_state_3p()
        expected = total_chips(s)
        s1 = apply_action(s, 2)  # P0 bet
        s2 = apply_action(s1, 1)  # P1 call
        s3 = apply_action(s2, 1)  # P2 call
        assert total_chips(s1) == expected
        assert total_chips(s2) == expected
        assert total_chips(s3) == expected

    def test_3p_bet_call_fold(self):
        s = make_state_3p()
        expected = total_chips(s)
        s1 = apply_action(s, 2)   # P0 bet
        s2 = apply_action(s1, 1)  # P1 call
        s3 = apply_action(s2, 0)  # P2 fold
        assert total_chips(s3) == expected

    def test_3p_bet_raise_call_call(self):
        s = make_state_3p()
        expected = total_chips(s)
        s1 = apply_action(s, 2)   # P0 bet
        s2 = apply_action(s1, 2)  # P1 raise
        s3 = apply_action(s2, 1)  # P0 call
        s4 = apply_action(s3, 1)  # P2 call
        for st in [s1, s2, s3, s4]:
            assert total_chips(st) == expected

    def test_allin_call(self):
        """Allin + call preserves chips."""
        s = make_state_2p()
        expected = total_chips(s)
        s1 = apply_action(s, 3)  # P0 allin/large
        s2 = apply_action(s1, 3)  # P1 allin
        assert total_chips(s2) == expected

    def test_full_hand_to_showdown(self):
        """Full 2p hand: bet-call, check-check, check-check preserves chips."""
        s = make_state_2p()
        expected = total_chips(s)
        # Flop: bet-call
        s1 = apply_action(s, 2)   # P0 bet
        s2 = apply_action(s1, 1)  # P1 call → street ends
        # Deal turn
        assert needs_card(s2)
        s2.board.append(40)  # dummy card
        # Turn: check-check
        s3 = apply_action(s2, 1)  # P0 check
        s4 = apply_action(s3, 1)  # P1 check → street ends
        # Deal river
        assert needs_card(s4)
        s4.board.append(41)
        # River: check-check
        s5 = apply_action(s4, 1)  # P0 check
        s6 = apply_action(s5, 1)  # P1 check → terminal
        assert is_terminal(s6)
        assert total_chips(s6) == expected


# ─── 2. Equal Investment After Call ──────────────────────────────────────────

class TestCallInvestment:
    """After a call, the caller's invested must equal facing_size."""

    def test_2p_bet_call_equal_invested(self):
        s = make_state_2p()
        s1 = apply_action(s, 2)   # P0 bet
        s2 = apply_action(s1, 1)  # P1 call
        # Both players should have equal invested after the call
        assert s2.invested[0] == s2.invested[1], \
            f"After call: P0 invested {s2.invested[0]} != P1 invested {s2.invested[1]}"

    def test_2p_bet_large_call_equal_invested(self):
        s = make_state_2p()
        s1 = apply_action(s, 3)   # P0 bet large
        s2 = apply_action(s1, 1)  # P1 call
        assert s2.invested[0] == s2.invested[1], \
            f"After call: P0 invested {s2.invested[0]} != P1 invested {s2.invested[1]}"

    def test_2p_bet_raise_call_equal_invested(self):
        s = make_state_2p()
        s1 = apply_action(s, 2)   # P0 bet
        s2 = apply_action(s1, 2)  # P1 raise
        s3 = apply_action(s2, 1)  # P0 call
        assert s3.invested[0] == s3.invested[1], \
            f"After call: P0 invested {s3.invested[0]} != P1 invested {s3.invested[1]}"

    def test_3p_bet_call_call_equal_invested(self):
        """All 3 players should match the bettor's investment after calling."""
        s = make_state_3p()
        s1 = apply_action(s, 2)   # P0 bet
        s2 = apply_action(s1, 1)  # P1 call
        s3 = apply_action(s2, 1)  # P2 call
        assert s3.invested[0] == s3.invested[1] == s3.invested[2], \
            f"After all call: invested={[s3.invested[i] for i in range(3)]}"

    def test_3p_bet_raise_call_call_equal_invested(self):
        """All players match the raiser's investment."""
        s = make_state_3p()
        s1 = apply_action(s, 2)   # P0 bet
        s2 = apply_action(s1, 2)  # P1 raise
        s3 = apply_action(s2, 1)  # P0 call
        s4 = apply_action(s3, 1)  # P2 call
        assert s4.invested[0] == s4.invested[1] == s4.invested[2], \
            f"After all call: invested={[s4.invested[i] for i in range(3)]}"


# ─── 3. Multi-player Facing Bet ──────────────────────────────────────────────

class TestMultiPlayerFacingBet:
    """Players after the first caller must still face the bet."""

    def test_3p_p0_bet_p1_call_p2_still_faces_bet(self):
        """After P0 bets and P1 calls, P2 must call — not check for free."""
        s = make_state_3p()
        s1 = apply_action(s, 2)   # P0 bet → facing_size = invested[0]
        bettor_inv = s1.invested[0]
        s2 = apply_action(s1, 1)  # P1 call
        # P2 should still need to match the bet
        assert s2.facing_size == bettor_inv, \
            f"After P1 call: facing_size={s2.facing_size} but bettor invested={bettor_inv}"
        # P2's call amount should be facing_size - invested[2]
        p2_call = _call_amt(s2, 2)
        assert p2_call > 0, f"P2 has 0 call amount — gets a free ride!"
        # P2 cannot check (action 1 when facing_size > 0 means call, not check)
        legal = legal_actions(s2)
        assert 0 in legal, "P2 should be able to fold"
        assert 1 in legal, "P2 should be able to call"

    def test_3p_p0_bet_p2_must_match_after_p1_folds(self):
        """If P1 folds after P0 bets, P2 still faces the bet."""
        s = make_state_3p()
        s1 = apply_action(s, 2)   # P0 bet
        bettor_inv = s1.invested[0]
        s2 = apply_action(s1, 0)  # P1 fold
        # P2 still faces the bet
        assert s2.facing_size == bettor_inv
        assert _call_amt(s2, 2) > 0

    def test_3p_p0_bet_all_call_investments_match(self):
        """All callers match the bettor's investment exactly."""
        s = make_state_3p()
        s1 = apply_action(s, 2)   # P0 bet
        s2 = apply_action(s1, 1)  # P1 call
        s3 = apply_action(s2, 1)  # P2 call
        target = s1.invested[0]   # bettor's investment
        for p in range(3):
            assert s3.invested[p] == target, \
                f"Player {p} invested {s3.invested[p]} != bettor {target}"


# ─── 4. Game Tree Termination ────────────────────────────────────────────────

class TestTermination:
    """Verify is_terminal works correctly."""

    def test_2p_fold_is_terminal(self):
        s = make_state_2p()
        s1 = apply_action(s, 2)   # P0 bet
        s2 = apply_action(s1, 0)  # P1 fold
        assert is_terminal(s2)

    def test_3p_one_left_is_terminal(self):
        s = make_state_3p()
        s1 = apply_action(s, 2)   # P0 bet
        s2 = apply_action(s1, 0)  # P1 fold
        s3 = apply_action(s2, 0)  # P2 fold
        assert is_terminal(s3)

    def test_2p_check_check_not_terminal_on_flop(self):
        s = make_state_2p()
        s1 = apply_action(s, 1)   # P0 check
        s2 = apply_action(s1, 1)  # P1 check → end flop
        # Not terminal — there are more streets
        assert not is_terminal(s2)

    def test_river_check_check_is_terminal(self):
        s = make_state_2p(street=2)
        # Need 5 board cards for river
        deck = list(range(52))
        random.shuffle(deck)
        s.board = deck[:5]
        s1 = apply_action(s, 1)  # P0 check
        s2 = apply_action(s1, 1)  # P1 check
        assert is_terminal(s2)

    def test_2p_both_allin_is_terminal_or_runout(self):
        """When both players are allin, the board should be run out."""
        # Use stacks small enough that action 3 = allin
        s = make_state_2p(invested=[500, 500], stacks=[500, 500])
        s.pot = 1000
        s1 = apply_action(s, 3)  # P0 allin (500 chips)
        s2 = apply_action(s1, 3)  # P1 allin (500 chips)
        # Both stacks should be 0, board run out
        assert s2.stacks[0] == 0 and s2.stacks[1] == 0
        assert is_terminal(s2) or s2.street == 3


# ─── 5. Allin Edge Cases ────────────────────────────────────────────────────

class TestAllin:
    """Allin scenarios."""

    def test_allin_sets_facing_size(self):
        s = make_state_2p()
        s1 = apply_action(s, 3)  # P0 large/allin
        assert s1.facing_size == s1.invested[0]

    def test_allin_call_matches(self):
        s = make_state_2p()
        s1 = apply_action(s, 3)  # P0 allin
        target = s1.invested[0]
        s2 = apply_action(s1, 1)  # P1 call
        # P1 should match P0's investment (or go allin for less)
        assert s2.invested[1] >= target or s2.stacks[1] == 0

    def test_short_stack_allin_caller_matches(self):
        """Player with fewer chips goes allin; caller puts in more."""
        invested = [SB, BB]
        stacks = [500, BUY_IN - BB]
        s = make_state_2p(stacks=stacks, invested=invested)
        # P0 allin
        s1 = apply_action(s, 3)
        # P1 calls — should put in enough to match P0's allin
        s2 = apply_action(s1, 1)
        assert s2.invested[0] == s2.invested[1], \
            f"Short stack allin: P0 invested {s2.invested[0]} != P1 invested {s2.invested[1]}"

    def test_allin_below_facing_size(self):
        """Player goes allin for less than the bet — doesn't change facing_size."""
        # P0 bets, P1 shoves for more (raise), P0 calls
        s = make_state_2p()
        s1 = apply_action(s, 2)   # P0 bet small
        s2 = apply_action(s1, 3)  # P1 allin/raise
        assert s2.facing_size == s2.invested[1]  # facing the allin amount
        s3 = apply_action(s2, 1)  # P0 calls the allin
        assert s3.invested[0] == s3.invested[1]


# ─── 6. Betting Round Completion ────────────────────────────────────────────

class TestBettingRound:
    """Verify street ends at the right time."""

    def test_check_around_ends_street(self):
        s = make_state_2p()
        s1 = apply_action(s, 1)  # P0 check
        s2 = apply_action(s1, 1)  # P1 check
        assert s2.street == 1  # advanced from flop to turn

    def test_bet_call_ends_street(self):
        s = make_state_2p()
        s1 = apply_action(s, 2)  # P0 bet
        s2 = apply_action(s1, 1)  # P1 call
        assert s2.street == 1

    def test_bet_raise_call_ends_street(self):
        s = make_state_2p()
        s1 = apply_action(s, 2)  # P0 bet
        s2 = apply_action(s1, 2)  # P1 raise
        s3 = apply_action(s2, 1)  # P0 call
        assert s3.street == 1

    def test_3p_bet_call_call_ends_street(self):
        s = make_state_3p()
        s1 = apply_action(s, 2)  # P0 bet
        s2 = apply_action(s1, 1)  # P1 call
        s3 = apply_action(s2, 1)  # P2 call
        assert s3.street == 1

    def test_3p_bet_fold_call_does_not_end_street_prematurely(self):
        """Bet → fold → call should still end the street (2 players left)."""
        s = make_state_3p()
        s1 = apply_action(s, 2)  # P0 bet
        s2 = apply_action(s1, 0)  # P1 fold
        s3 = apply_action(s2, 1)  # P2 call
        assert s3.street == 1  # street ends after call (2 players matched)

    def test_3p_bet_call_fold_ends_street(self):
        """Bet → call → fold: all remaining active players have acted, street ends."""
        s = make_state_3p()
        s1 = apply_action(s, 2)  # P0 bet
        s2 = apply_action(s1, 1)  # P1 call
        s3 = apply_action(s2, 0)  # P2 fold → P0 and P1 both acted → street ends
        assert s3.street == 1
        assert s3.facing_size == 0

    def test_3p_bet_fold_cannot_end_street_if_caller_remaining(self):
        """Bet → fold: one player still faces the bet, street should NOT end."""
        s = make_state_3p()
        s1 = apply_action(s, 2)  # P0 bet
        s2 = apply_action(s1, 0)  # P1 fold
        assert s2.street == 0  # P2 still needs to act

    def test_3p_raise_fold_call_ends_street(self):
        """Bet → raise → fold → call: P0 and P1 acted, street ends after P0 calls."""
        s = make_state_3p()
        s1 = apply_action(s, 2)  # P0 bet
        s2 = apply_action(s1, 2)  # P1 raise
        s3 = apply_action(s2, 0)  # P2 fold
        assert s3.street == 0  # P0 still needs to respond to raise
        s4 = apply_action(s3, 1)  # P0 call → street ends
        assert s4.street == 1

    def test_facing_size_cleared_on_new_street(self):
        s = make_state_2p()
        s1 = apply_action(s, 2)  # P0 bet
        s2 = apply_action(s1, 1)  # P1 call → street ends
        assert s2.facing_size == 0

    def test_acted_cleared_on_new_street(self):
        s = make_state_2p()
        s1 = apply_action(s, 2)  # P0 bet
        s2 = apply_action(s1, 1)  # P1 call → street ends
        assert len(s2.acted) == 0

    def test_agg_count_cleared_on_new_street(self):
        s = make_state_2p()
        s1 = apply_action(s, 2)  # P0 bet
        s2 = apply_action(s1, 2)  # P1 raise
        s3 = apply_action(s2, 1)  # P0 call → street ends
        assert s3.agg_count == 0


# ─── 7. Key Encoding Round-Trip ──────────────────────────────────────────────

class TestKeyEncoding:
    """Verify trainer info_key matches runtime _fixed_key_bytes."""

    def test_info_key_is_deterministic(self):
        """Same state + history always produces the same key."""
        random.seed(42)
        s = make_state_2p()
        k1 = info_key(0, s, [2, 1])
        k2 = info_key(0, s, [2, 1])
        assert k1 == k2

    def test_info_key_differs_by_player(self):
        random.seed(42)
        s = make_state_2p()
        k0 = info_key(0, s, [2, 1])
        k1 = info_key(1, s, [2, 1])
        assert k0 != k1

    def test_info_key_differs_by_facing(self):
        random.seed(42)
        s1 = make_state_2p(facing_size=0)
        s2 = make_state_2p(facing_size=100)
        k1 = info_key(0, s1, [])
        k2 = info_key(0, s2, [])
        assert k1 != k2

    def test_info_key_differs_by_agg_count(self):
        random.seed(42)
        s1 = make_state_2p(agg_count=0)
        s2 = make_state_2p(agg_count=2)
        k1 = info_key(0, s1, [])
        k2 = info_key(0, s2, [])
        assert k1 != k2

    def test_info_key_differs_by_action_history(self):
        random.seed(42)
        s = make_state_2p()
        k1 = info_key(0, s, [2, 1])
        k2 = info_key(0, s, [3, 1])
        assert k1 != k2

    def test_info_key_differs_by_street(self):
        random.seed(42)
        s1 = make_state_2p(street=0)
        s2 = make_state_2p(street=1)
        k1 = info_key(0, s1, [])
        k2 = info_key(0, s2, [])
        assert k1 != k2

    def test_runtime_key_matches_trainer(self):
        """Runtime _fixed_key_bytes produces the same key as trainer info_key."""
        from strategy.postflop_solver import _fixed_key_bytes

        RANKS = "23456789TJQKA"
        SUITS = "cdhs"

        def card_str(c):
            return RANKS[c // 4] + SUITS[c % 4]

        random.seed(42)
        s = make_state_2p()
        # Simulate a bet-call history
        s1 = apply_action(s, 2)  # P0 bet
        hist = [2]

        trainer_key = info_key(0, s1, hist)

        # Build runtime key
        hole_strs = [card_str(s1.hole[0][0]), card_str(s1.hole[0][1])]
        board_strs = [card_str(c) for c in s1.board]
        action_hist = ['bet' if a == 2 else 'call' if a == 1 else 'fold' if a == 0 else 'allin' for a in hist]
        prev = [[]]  # no previous street actions

        runtime_key = _fixed_key_bytes(hole_strs, board_strs, 0, action_hist, prev)

        assert trainer_key == runtime_key, \
            f"Key mismatch:\n  trainer: {list(trainer_key)}\n  runtime: {list(runtime_key)}"


# ─── 8. Street Advancement ───────────────────────────────────────────────────

class TestStreetAdvancement:
    """Verify correct street advancement and card dealing."""

    def test_flop_is_street_0(self):
        s = make_state_2p()
        assert s.street == 0

    def test_after_flop_check_check_street_1(self):
        s = make_state_2p()
        s1 = apply_action(s, 1)
        s2 = apply_action(s1, 1)
        assert s2.street == 1

    def test_needs_card_after_flop(self):
        s = make_state_2p()
        s1 = apply_action(s, 1)
        s2 = apply_action(s1, 1)
        assert needs_card(s2)

    def test_no_needs_card_on_flop(self):
        s = make_state_2p()
        assert not needs_card(s)


# ─── 9. Aggression Cap ───────────────────────────────────────────────────────

class TestAggCap:
    """After AGG_CAP aggressive actions, only fold/call/allin allowed."""

    def test_cannot_raise_after_cap(self):
        s = make_state_2p()
        # P0 bet (agg=1), P1 raise (agg=2), P0 re-raise (agg=3)
        s1 = apply_action(s, 2)   # P0 bet → agg=1
        s2 = apply_action(s1, 2)  # P1 raise → agg=2
        s3 = apply_action(s2, 2)  # P0 re-raise → agg=3
        # P1 should not be able to raise (agg=3 == AGG_CAP)
        legal = legal_actions(s3)
        assert 2 not in legal, f"Raise should not be available after agg cap"

    def test_can_call_after_cap(self):
        s = make_state_2p()
        s1 = apply_action(s, 2)
        s2 = apply_action(s1, 2)
        s3 = apply_action(s2, 2)
        legal = legal_actions(s3)
        assert 1 in legal  # call should always be available

    def test_can_allin_after_cap(self):
        s = make_state_2p()
        s1 = apply_action(s, 2)
        s2 = apply_action(s1, 2)
        s3 = apply_action(s2, 2)
        legal = legal_actions(s3)
        assert 3 in legal  # allin should be available


# ─── 10. Payoff Correctness ─────────────────────────────────────────────────

class TestPayoff:
    """Verify payoff sums and rake."""

    def test_fold_payoff_sum_equals_negative_rake(self):
        s = make_state_2p()
        s1 = apply_action(s, 2)   # P0 bet
        s2 = apply_action(s1, 0)  # P1 fold
        g = payoff(s2)
        rake = min(s2.pot * RAKE_RATE, RAKE_CAP * BB)
        assert abs(sum(g) + rake) < 1.0, f"Payoff sum {sum(g)} should be -rake {-rake}"

    def test_showdown_payoff_sum_equals_negative_rake(self):
        """Full hand to showdown, verify payoff sum = -rake."""
        s = make_state_2p()
        expected = total_chips(s)
        # Flop: check-check
        s1 = apply_action(s, 1)
        s2 = apply_action(s1, 1)
        # Deal turn
        s2.board.append(40)
        # Turn: check-check
        s3 = apply_action(s2, 1)
        s4 = apply_action(s3, 1)
        # Deal river
        s4.board.append(41)
        # River: check-check
        s5 = apply_action(s4, 1)
        s6 = apply_action(s5, 1)
        assert is_terminal(s6)
        g = payoff(s6)
        rake = min(s6.pot * RAKE_RATE, RAKE_CAP * BB)
        assert abs(sum(g) + rake) < 1.0

    def test_payoff_zero_sum_minus_rake(self):
        """Win + loss = -rake for 2p."""
        s = make_state_2p()
        s1 = apply_action(s, 2)  # P0 bet
        s2 = apply_action(s1, 0) # P1 fold
        g = payoff(s2)
        # Winner gains (pot - rake - invested), loser loses invested
        # Sum should be -rake
        rake = min(s2.pot * RAKE_RATE, RAKE_CAP * BB)
        assert abs(g[0] + g[1] + rake) < 0.01


# ─── 11. Property-Based: Random Walk ────────────────────────────────────────

class TestRandomWalk:
    """Walk random game paths and verify invariants at every step."""

    @pytest.mark.parametrize("n_players", [2, 3])
    def test_invariants_on_random_walk(self, n_players):
        """Chip conservation + correct investment on random action sequences."""
        random.seed(12345)
        for trial in range(200):
            if n_players == 2:
                s = make_state_2p()
            else:
                s = make_state_3p()
            expected_total = total_chips(s)

            for step in range(30):
                if is_terminal(s):
                    break
                if needs_card(s):
                    # Deal a card
                    exclude = set()
                    for h in s.hole:
                        exclude.update(h)
                    exclude.update(s.board)
                    card = random.choice([c for c in range(52) if c not in exclude])
                    s.board.append(card)
                    continue
                legal = legal_actions(s)
                if not legal:
                    break
                action = random.choice(legal)
                s = apply_action(s, action)
                # INVARIANT 1: chip conservation
                assert total_chips(s) == expected_total, \
                    f"Chip leak at step {step}: {total_chips(s)} != {expected_total}"

            # If we reached a terminal state, check payoff
            if is_terminal(s):
                g = payoff(s)
                rake = min(s.pot * RAKE_RATE, RAKE_CAP * BB)
                assert abs(sum(g) + rake) < 1.0, \
                    f"Payoff sum {sum(g)} != -rake {-rake}"


# ─── 12. Call Amount Calculation ────────────────────────────────────────────

class TestCallAmt:
    """Test _call_amt helper."""

    def test_call_amt_no_bet(self):
        s = make_state_2p(facing_size=0)
        assert _call_amt(s, 0) == 0

    def test_call_amt_with_bet(self):
        s = make_state_2p(facing_size=200, invested=[200, 100])
        assert _call_amt(s, 1) == 100  # P1 needs 100 more

    def test_call_amt_already_matched(self):
        s = make_state_2p(facing_size=200, invested=[200, 200])
        assert _call_amt(s, 1) == 0  # P1 already matched

    def test_call_amt_short_stack(self):
        s = make_state_2p(facing_size=5000, invested=[5000, 100], stacks=[5000, 50])
        # P1 only has 50 chips but needs 4900 to match
        assert _call_amt(s, 1) == 4900  # theoretical call amount
        # But actual call is min(4900, 50) = 50 (allin for less)
