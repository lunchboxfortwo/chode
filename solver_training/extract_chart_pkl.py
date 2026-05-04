#!/usr/bin/env python3
"""Re-extract chart-friendly policy PKLs from solver-internal PKLs.

The trainer saves two formats:
  - {N}p{bb}_preflop_fixed_solver.pkl  — bytes keys, int actions (for resuming)
  - {N}p{bb}_preflop_policy.pkl         — string keys, string actions (for chart UI)

If a chart PKL is missing but the solver PKL exists, this script generates it.
Run after training completes or to backfill missing chart PKLs.

Usage:
    python3 solver_training/extract_chart_pkl.py
    python3 solver_training/extract_chart_pkl.py --force   # overwrite existing
"""
import os, sys, pickle, glob, argparse

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_training.preflop_fixed_train import (
    _hand_cat, _pack_hand_cat, RANKS, OUTPUT_DIR,
)


def _build_hand_map() -> dict[tuple[int, int], str]:
    m: dict[tuple[int, int], str] = {}
    for r1 in range(13):
        for r2 in range(r1 + 1, 13):
            hc = _hand_cat([r1 * 4 + 0, r2 * 4 + 1])
            r1r2, sb = _pack_hand_cat(hc)
            m[(r1r2, sb)] = f"{RANKS[r1]}{RANKS[r2]}s"
            hc = _hand_cat([r1 * 4 + 0, r2 * 4 + 2])
            r1r2, sb = _pack_hand_cat(hc)
            m[(r1r2, sb)] = f"{RANKS[r1]}{RANKS[r2]}o"
        hc = _hand_cat([r1 * 4 + 0, r1 * 4 + 1])
        r1r2, sb = _pack_hand_cat(hc)
        m[(r1r2, sb)] = f"{RANKS[r1]}{RANKS[r1]}"
    return m


_ACTION_MAP = {0: "fold", 1: "call", 2: "bet", 3: "bet", 4: "allin"}


def convert(solver_policy: dict) -> dict:
    hand_map = _build_hand_map()
    chart: dict[tuple[str, int, tuple[str, ...]], dict[str, float]] = {}
    for k, v in solver_policy.items():
        if not isinstance(k, (bytes, bytearray)) or len(k) < 8:
            continue
        player_idx = k[0]
        hand_str = hand_map.get((k[1], k[2]))
        if hand_str is None:
            continue
        hist_ints = list(k[8:])
        hist_strs = tuple(_ACTION_MAP.get(a, f"?{a}") for a in hist_ints)
        chart_val: dict[str, float] = {}
        for a_int, prob in v.items():
            a_name = _ACTION_MAP.get(a_int, f"?{a_int}")
            chart_val[a_name] = chart_val.get(a_name, 0.0) + prob
        chart[(hand_str, player_idx, hist_strs)] = chart_val
    return chart


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing chart PKLs")
    parser.add_argument("--dir", default=OUTPUT_DIR, help="Directory to scan")
    args = parser.parse_args()

    solver_pkls = sorted(glob.glob(os.path.join(args.dir, "*_preflop_fixed_solver.pkl")))
    if not solver_pkls:
        print("No solver PKLs found.")
        return

    from solver_training.preflop_fixed_train import _load_solver, extract_and_save

    for sp in solver_pkls:
        base = os.path.basename(sp).replace("_preflop_fixed_solver.pkl", "")
        chart_path = os.path.join(args.dir, f"{base}_preflop_policy.pkl")

        if os.path.exists(chart_path) and not args.force:
            print(f"  SKIP {base} — chart PKL already exists (use --force to overwrite)")
            continue

        print(f"  Loading {os.path.basename(sp)} ...")
        try:
            solver = _load_solver(sp)
        except Exception as e:
            print(f"  ERROR loading {sp}: {e}")
            continue

        # Extract both solver and chart PKLs (extract_and_save does both)
        extract_and_save(solver, solver.n_players, solver.stack_bb)

        size_kb = os.path.getsize(chart_path) // 1024 if os.path.exists(chart_path) else 0
        print(f"  → {os.path.basename(chart_path)}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
