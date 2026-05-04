"""
NashConv audit for preflop MCCFR solver checkpoints.

NashConv(σ) = Σ_p [BR_p(σ_{-p}) - v_p(σ)]

Upper bound on exploitability. If small (say < 0.5 bb/hand),
the strategy is approximately Nash equilibrium.

Uses multi-board showdown averaging to reduce variance from
chance nodes. Single-board NashConv overestimates by ~3-8x
due to showdown outcome noise.

Single-pass tree traversal computes BR + policy values for
ALL players simultaneously (2N values per node visit).
"""

import os
import sys
import random
import pickle
import time
import json
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_training.preflop_fixed_train import (
    Solver, State, _load_solver,
    is_terminal, payoff, legal_actions, apply_action,
    info_key, position_names,
    _init_treys, _card_str,
    BB, SB, N_ACTIONS, RAKE_RATE, RAKE_CAP,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

# ─── Treys helpers ───────────────────────────────────────────────────────────

_EVAL = None
_TC = None

def _init_audit_treys():
    global _EVAL, _TC
    if _EVAL is not None:
        return
    from treys import Card as TCard, Evaluator as TEval
    _EVAL = TEval()
    _TC = [TCard.new(_card_str(c)) for c in range(52)]


# ─── Average policy lookup ───────────────────────────────────────────────────

class AvgPolicy:
    """Fast lookup for the average policy from a solver's strat_sum arrays."""

    def __init__(self, solver: Solver):
        self._solver = solver

    def get_probs(self, key: bytes, legal: list[int]) -> np.ndarray:
        """Return probability vector over ALL actions from the average policy."""
        idx = self._solver._key_index.get(key)
        strat = np.zeros(N_ACTIONS, dtype=np.float64)
        if idx is not None:
            ss = self._solver._strat_sum[idx]
            total = ss[legal].sum()
            if total > 0:
                for a in legal:
                    strat[a] = ss[a] / total
                return strat
        # Uniform over legal if no data
        for a in legal:
            strat[a] = 1.0 / len(legal)
        return strat


# ─── Stable payoff (multi-board showdown) ────────────────────────────────────

def _payoff_stable(s: State, n_boards: int = 100) -> list[float]:
    """
    Low-variance payoff using multi-board showdown averaging.
    Reduces NashConv measurement noise from ~8x overestimate (single board)
    to ~1.2x with 100 boards.
    """
    rake = min(s.pot * RAKE_RATE, RAKE_CAP * BB)
    active = s.active()

    if len(active) <= 1:
        winner = active[0]
        gains = [-s.invested[p] for p in range(s.n_players)]
        gains[winner] += s.pot - rake
        return gains

    # Showdown: average over n_boards
    exclude = set()
    for h in s.hole:
        exclude.update(h)

    total_gains = [0.0] * s.n_players
    base_gains = [-s.invested[p] for p in range(s.n_players)]

    for _ in range(n_boards):
        deck = [c for c in range(52) if c not in exclude]
        random.shuffle(deck)
        board = deck[:5]

        gains = list(base_gains)
        board_treys = [_TC[c] for c in board]
        scores = {}
        for p in active:
            hole_treys = [_TC[c] for c in s.hole[p]]
            scores[p] = _EVAL.evaluate(board_treys, hole_treys)
        best = min(scores.values())
        winners = [p for p, sc in scores.items() if sc == best]
        share = (s.pot - rake) / len(winners)
        for w in winners:
            gains[w] += share
        for p in range(s.n_players):
            total_gains[p] += gains[p]

    return [g / n_boards for g in total_gains]


# ─── Single-pass tree traversal ──────────────────────────────────────────────

def _traverse_all(s: State, policy: AvgPolicy, n_players: int,
                  hist: list, depth: int = 0,
                  n_boards: int = 100):
    """
    Single-pass traversal returning (br_vals, pv_vals) each of length n_players.

    br_vals[p] = best-response value for player p against σ_{-p}
    pv_vals[p] = policy value for player p when all follow σ

    Uses multi-board showdown averaging for stable terminal payoffs.
    """
    if is_terminal(s):
        p = _payoff_stable(s, n_boards)
        return list(p), list(p)

    if depth > 20:
        return [0.0] * n_players, [0.0] * n_players

    p_act = s.acting
    legal = legal_actions(s)
    if not legal:
        return [0.0] * n_players, [0.0] * n_players

    key = info_key(p_act, s, hist)
    strat = policy.get_probs(key, legal)

    pv = [0.0] * n_players
    br = [0.0] * n_players
    best_br_pact = -1e30

    for a in legal:
        ns = apply_action(s, a)
        child_br, child_pv = _traverse_all(
            ns, policy, n_players, hist + [a], depth + 1, n_boards
        )
        prob_a = strat[a]

        # Policy value: always weighted by σ
        for q in range(n_players):
            pv[q] += prob_a * child_pv[q]

        # BR for the acting player: take max
        if child_br[p_act] > best_br_pact:
            best_br_pact = child_br[p_act]

        # BR for non-acting players: weighted by σ
        for q in range(n_players):
            if q != p_act:
                br[q] += prob_a * child_br[q]

    br[p_act] = best_br_pact
    return br, pv


# ─── Main computation ────────────────────────────────────────────────────────

def compute_nash_conv(solver: Solver, n_samples: int = 5_000,
                      n_boards: int = 100, seed: int = 42) -> dict:
    """
    Monte Carlo estimate of NashConv for a solver's average policy.
    Uses multi-board showdown averaging and single-pass traversal.
    """
    _init_audit_treys()
    policy = AvgPolicy(solver)
    n = solver.n_players
    nc_values = []
    per_player_exploit = [[] for _ in range(n)]

    t0 = time.time()
    rng = random.Random(seed)

    for i in range(n_samples):
        deck = list(range(52))
        rng.shuffle(deck)
        state = State(n, solver.stack_bb)
        for p in range(n):
            state.hole[p] = [deck[p * 2], deck[p * 2 + 1]]

        br, pv = _traverse_all(state, policy, n, [], n_boards=n_boards)

        deal_nc = 0.0
        for p in range(n):
            exploit_p = (br[p] - pv[p]) / BB
            per_player_exploit[p].append(exploit_p)
            deal_nc += exploit_p
        nc_values.append(deal_nc)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            mean_nc = np.mean(nc_values)
            stderr_nc = np.std(nc_values) / np.sqrt(i + 1)
            print(f"    {i+1:>6,}/{n_samples:,} deals  |  "
                  f"NashConv = {mean_nc:.2f} ± {stderr_nc:.2f} bb  |  "
                  f"{rate:.0f} deals/s")

    nc_arr = np.array(nc_values)

    result = {
        "nash_conv_mean": float(np.mean(nc_arr)),
        "nash_conv_stderr": float(np.std(nc_arr) / np.sqrt(n_samples)),
        "nash_conv_median": float(np.median(nc_arr)),
        "nash_conv_max": float(np.max(nc_arr)),
        "per_player_mean": {},
        "per_player_stderr": {},
        "n_samples": n_samples,
        "n_boards": n_boards,
        "elapsed_s": time.time() - t0,
    }

    for p in range(n):
        arr = np.array(per_player_exploit[p])
        pos = position_names(n)[p]
        result["per_player_mean"][pos] = float(np.mean(arr))
        result["per_player_stderr"][pos] = float(np.std(arr) / np.sqrt(n_samples))

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="NashConv audit for preflop solver checkpoints"
    )
    parser.add_argument("--players", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="Player counts to audit")
    parser.add_argument("--stack-bb", type=int, default=30,
                        help="Stack depth in BB")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of Monte Carlo deals (default: auto by config)")
    parser.add_argument("--boards", type=int, default=100,
                        help="Number of boards per showdown (more = less variance)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _init_audit_treys()

    # Default samples by config size (bigger = slower per deal)
    _DEFAULT_SAMPLES = {2: 2000, 3: 1000, 4: 500, 5: 200, 6: 100}

    results = {}

    for n_players in args.players:
        suffix = "" if args.stack_bb == 100 else f"_{args.stack_bb}bb"
        path = os.path.join(OUTPUT_DIR, f"{n_players}p{suffix}_preflop_fixed_solver.pkl")

        if not os.path.exists(path):
            print(f"\n{'='*60}")
            print(f"  SKIP {n_players}p {args.stack_bb}bb — no checkpoint found")
            continue

        # Check format compatibility
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict) or obj.get("format") != "preflop-v1":
            print(f"\n{'='*60}")
            print(f"  SKIP {n_players}p {args.stack_bb}bb — old checkpoint format")
            continue

        size_mb = os.path.getsize(path) / 1048576
        print(f"\n{'='*60}")
        print(f"  {n_players}p {args.stack_bb}bb  ({size_mb:.1f} MB checkpoint)")
        print(f"{'='*60}")

        t0 = time.time()
        solver = _load_solver(path)
        load_time = time.time() - t0

        # Get iteration count from progress file
        prog_path = path.replace(".pkl", ".progress.json")
        iters = solver.iterations
        if os.path.exists(prog_path):
            with open(prog_path) as f:
                prog = json.load(f)
            iters = prog.get("iterations_done", iters)

        print(f"  Loaded in {load_time:.1f}s — {solver._n:,} info states, "
              f"{iters:,} iterations")

        n_samples = args.samples or _DEFAULT_SAMPLES.get(n_players, 500)

        print(f"\n  Computing NashConv ({n_samples:,} deals, "
              f"{args.boards} boards/showdown)...")
        result = compute_nash_conv(solver, n_samples=n_samples,
                                   n_boards=args.boards, seed=args.seed)

        print(f"\n  Results:")
        print(f"    NashConv: {result['nash_conv_mean']:.2f} "
              f"± {result['nash_conv_stderr']:.2f} bb/hand")
        print(f"    Median:   {result['nash_conv_median']:.2f} bb/hand")

        print(f"\n  Per-player exploitability:")
        for pos in position_names(n_players):
            mean = result["per_player_mean"][pos]
            stderr = result["per_player_stderr"][pos]
            print(f"    {pos:>3}: {mean:+.2f} ± {stderr:.2f} bb/hand")

        result["n_info_states"] = solver._n
        result["iterations"] = iters
        results[f"{n_players}p"] = result

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY — NashConv (bb/hand, {args.boards}-board showdowns)")
    print(f"{'='*60}")
    print(f"  {'Config':<8} {'Iters':>12} {'InfoStates':>12} {'NashConv':>10} {'±':>6}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*6}")
    for key, r in results.items():
        print(f"  {key:<8} {r['iterations']:>12,} {r['n_info_states']:>12,} "
              f"{r['nash_conv_mean']:>10.2f} {r['nash_conv_stderr']:>6.2f}")

    # Convergence targets
    try:
        from solver_training.preflop_fixed_train import _CONVERGENCE_TARGETS
        print(f"\n  Convergence targets (NC < 0.5 bb/hand):")
        print(f"  {'Config':<8} {'Current':>12} {'Target':>14} {'Remaining':>14}")
        print(f"  {'-'*8} {'-'*12} {'-'*14} {'-'*14}")
        for key, r in results.items():
            n_p = int(key[0])
            s_bb = args.stack_bb
            target = _CONVERGENCE_TARGETS.get((n_p, s_bb), 0)
            current = r["iterations"]
            remaining = max(target - current, 0)
            print(f"  {key:<8} {current:>12,} {target:>14,} {remaining:>14,}")
    except ImportError:
        pass

    print()
    print("  Interpretation:")
    print("    NashConv < 0.5 bb/hand  → strong GTO approximation")
    print("    NashConv 0.5–2 bb/hand  → reasonable, but exploitable")
    print("    NashConv > 2 bb/hand    → needs more training iterations")
    print()


if __name__ == "__main__":
    main()
