#!/usr/bin/env python3
"""
Standalone policy extraction script — runs extraction on an existing solver
checkpoint WITHOUT the OOM-causing average_policy() dict.

Usage:
    python3 solver_training/extract_policy.py --solver data/postflop_tables/2p_postflop_fixed_solver.pkl
    python3 solver_training/extract_policy.py --solver data/preflop_tables/3p_30bb_preflop_fixed_solver.pkl --preflop
"""
import os
import sys
import argparse
import time
import hashlib
import pickle
import tempfile
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RANKS = "23456789TJQKA"


def _hash_key(key) -> np.uint64:
    if isinstance(key, bytes):
        return np.uint64(int.from_bytes(
            hashlib.md5(key).digest()[:8], "little", signed=False
        ))
    return np.uint64(int.from_bytes(
        hashlib.md5(str(key).encode()).digest()[:8], "little", signed=False
    ))


def _load_postflop_solver(path: str):
    """Load a postflop solver checkpoint (flat-v1/v2/v3 format)."""
    with open(path, "rb") as f:
        obj = pickle.load(f)

    N_ACTIONS = 4

    if isinstance(obj, dict) and obj.get("format") in ("flat-v1", "flat-v2", "flat-v3"):
        keys = obj["keys"]
        regrets = obj["regrets"]
        strats = obj["strat_sums"]
        n = len(keys)
        return {
            "n_players": obj["n_players"],
            "iterations": obj.get("iterations", 0),
            "n_info_states": n,
            "N_ACTIONS": N_ACTIONS,
            "keys": keys,
            "regrets": regrets,
            "strat_sums": strats,
        }

    raise ValueError(f"Unknown checkpoint format: {type(obj)}")


def _load_preflop_solver(path: str):
    """Load a preflop solver checkpoint (preflop-v1 format)."""
    with open(path, "rb") as f:
        obj = pickle.load(f)

    N_ACTIONS = 5

    if isinstance(obj, dict) and obj.get("format") == "preflop-v1":
        keys = obj["keys"]
        regrets = obj["regrets"]
        strats = obj["strat_sums"]
        n = len(keys)
        return {
            "n_players": obj["n_players"],
            "stack_bb": obj.get("stack_bb", 100),
            "iterations": obj.get("iterations", 0),
            "n_info_states": n,
            "N_ACTIONS": N_ACTIONS,
            "keys": keys,
            "regrets": regrets,
            "strat_sums": strats,
        }

    raise ValueError(f"Unknown checkpoint format: {type(obj)}")


def extract_postflop(solver_data: dict, output_dir: str):
    """Stream-extract postflop policy to NPZ."""
    n = solver_data["n_info_states"]
    n_players = solver_data["n_players"]
    N_ACTIONS = solver_data["N_ACTIONS"]
    regrets = solver_data["regrets"]
    strats = solver_data["strat_sums"]
    keys_list = solver_data["keys"]

    logger.info(f"Extracting {n_players}p postflop policy: {n:,} info states, {N_ACTIONS} actions")

    # DOMINANCE simplification (same as trainer)
    DOMINANCE_HARD = 0.95
    DOMINANCE_FREQ = 0.80
    DOMINANCE_EV_THRESHOLD = 0.005

    keys_out = np.empty(n, dtype=np.uint64)
    acts_out = np.full((n, 4), -1, dtype=np.int16)
    probs_out = np.zeros((n, 4), dtype=np.float16)
    n_valid = 0

    t0 = time.time()
    for i in range(n):
        ss = strats[i]
        total = ss.sum()
        if total <= 0:
            continue

        r = regrets[i]
        pos = np.maximum(r, 0.0)
        pos_total = pos.sum()
        if pos_total > 0:
            strat = pos / pos_total
        else:
            nz = (ss > 0).sum()
            if nz == 0:
                continue
            strat = np.where(ss > 0, 1.0 / nz, 0.0)

        raw_probs = {a: float(strat[a]) for a in range(N_ACTIONS) if ss[a] > 0}

        # Simplification
        items = list(raw_probs.items())
        best_a, best_p = max(items, key=lambda x: x[1])
        probs = raw_probs
        if best_p >= DOMINANCE_HARD:
            probs = {best_a: 1.0}
        elif best_p >= DOMINANCE_FREQ:
            ev_loss_bound = 1.0 - best_p
            if ev_loss_bound <= DOMINANCE_EV_THRESHOLD:
                probs = {best_a: 1.0}

        ptotal = sum(probs.values())
        if ptotal <= 0:
            continue

        keys_out[n_valid] = _hash_key(keys_list[i])
        for j, (a, p) in enumerate(sorted(probs.items())):
            if j >= 4:
                break
            acts_out[n_valid, j] = a
            probs_out[n_valid, j] = p / ptotal
        n_valid += 1

        if n_valid % 5_000_000 == 0:
            elapsed = time.time() - t0
            logger.info(f"  ... {n_valid:,} entries ({i:,}/{n:,}) — {elapsed:.1f}s")

    if n_valid == 0:
        logger.warning("No valid entries — nothing to save")
        return

    keys_out = keys_out[:n_valid]
    acts_out = acts_out[:n_valid]
    probs_out = probs_out[:n_valid]

    order = np.argsort(keys_out)
    keys_out = keys_out[order]
    acts_out = acts_out[order]
    probs_out = probs_out[order]

    npz_path = os.path.join(output_dir, f"{n_players}p_postflop_fixed_policy.npz")
    dir_ = os.path.dirname(npz_path)
    fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".tmp_", suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, keys=keys_out, actions=acts_out, probs=probs_out,
                                n_players=np.array([n_players]), max_actions=np.array([4]))
        os.replace(tmp, npz_path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise

    size_mb = os.path.getsize(npz_path) // (1024 ** 2)
    logger.info(f"NPZ saved: {npz_path}  ({size_mb} MB, {n_valid:,} entries, {time.time()-t0:.1f}s)")


def extract_preflop(solver_data: dict, output_dir: str):
    """Stream-extract preflop policy to NPZ + chart PKL."""
    n = solver_data["n_info_states"]
    n_players = solver_data["n_players"]
    stack_bb = solver_data.get("stack_bb", 100)
    N_ACTIONS = solver_data["N_ACTIONS"]
    regrets = solver_data["regrets"]
    strats = solver_data["strat_sums"]
    keys_list = solver_data["keys"]

    suffix = "" if stack_bb == 100 else f"_{stack_bb}bb"
    logger.info(f"Extracting {n_players}p{suffix} preflop policy: {n:,} info states, {N_ACTIONS} actions")

    # ── NPZ ────────────────────────────────────────────────────────────────
    keys_out = np.empty(n, dtype=np.uint64)
    acts_out = np.full((n, 5), -1, dtype=np.int16)
    probs_out = np.zeros((n, 5), dtype=np.float16)
    n_valid = 0

    t0 = time.time()
    for i in range(n):
        ss = strats[i]
        total = ss.sum()
        if total <= 0:
            continue

        r = regrets[i]
        pos = np.maximum(r, 0.0)
        pos_total = pos.sum()
        if pos_total > 0:
            strat = pos / pos_total
        else:
            nz = (ss > 0).sum()
            if nz == 0:
                continue
            strat = np.where(ss > 0, 1.0 / nz, 0.0)

        raw_probs = {a: float(strat[a]) for a in range(N_ACTIONS) if ss[a] > 0}
        ptotal = sum(raw_probs.values())
        if ptotal <= 0:
            continue

        keys_out[n_valid] = _hash_key(keys_list[i])
        for j, (a, p) in enumerate(sorted(raw_probs.items())):
            if j >= 5:
                break
            acts_out[n_valid, j] = a
            probs_out[n_valid, j] = p / ptotal
        n_valid += 1

        if n_valid % 5_000_000 == 0:
            elapsed = time.time() - t0
            logger.info(f"  ... {n_valid:,} entries ({i:,}/{n:,}) — {elapsed:.1f}s")

    if n_valid > 0:
        keys_out = keys_out[:n_valid]
        acts_out = acts_out[:n_valid]
        probs_out = probs_out[:n_valid]

        order = np.argsort(keys_out)
        keys_out = keys_out[order]
        acts_out = acts_out[order]
        probs_out = probs_out[order]

        npz_path = os.path.join(output_dir, f"{n_players}p{suffix}_preflop_fixed_policy.npz")
        dir_ = os.path.dirname(npz_path)
        fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".tmp_", suffix=".npz")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, keys=keys_out, actions=acts_out, probs=probs_out,
                                    n_players=np.array([n_players]), stack_bb=np.array([stack_bb]),
                                    max_actions=np.array([5]))
            os.replace(tmp, npz_path)
        except Exception:
            try: os.unlink(tmp)
            except OSError: pass
            raise

        size_mb = os.path.getsize(npz_path) // (1024 ** 2)
        logger.info(f"NPZ saved: {npz_path}  ({size_mb} MB, {n_valid:,} entries, {time.time()-t0:.1f}s)")

    # ── Chart PKL (only for manageable sizes) ─────────────────────────────
    if n <= 10_000_000:
        logger.info(f"Building chart PKL ({n:,} info states)...")
        SUITS = "cdhs"
        _ACTION_MAP = {0: "fold", 1: "call", 2: "bet", 3: "bet", 4: "allin"}

        def _hand_cat_unpack(r1r2_byte, suit_byte):
            r1 = (r1r2_byte >> 4) & 0xF
            r2 = r1r2_byte & 0xF
            if r1 == r2:
                return RANKS[r1] * 2
            return RANKS[r1] + RANKS[r2] + ('s' if suit_byte == 1 else 'o')

        chart = {}
        for i in range(n):
            ss = strats[i]
            total = ss.sum()
            if total <= 0:
                continue
            r = regrets[i]
            pos = np.maximum(r, 0.0)
            pos_total = pos.sum()
            if pos_total > 0:
                strat = pos / pos_total
            else:
                nz = (ss > 0).sum()
                if nz == 0:
                    continue
                strat = np.where(ss > 0, 1.0 / nz, 0.0)

            key = keys_list[i]
            if not isinstance(key, bytes) or len(key) < 8:
                continue
            player_idx = key[0]
            hand_str = _hand_cat_unpack(key[1], key[2])
            hist_ints = list(key[8:])
            hist_strs = tuple(_ACTION_MAP.get(a, f"?{a}") for a in hist_ints)
            ptotal = sum(float(strat[a]) for a in range(N_ACTIONS) if ss[a] > 0)
            if ptotal <= 0:
                continue
            chart_val = {}
            for a in range(N_ACTIONS):
                if ss[a] > 0:
                    a_name = _ACTION_MAP.get(a, f"?{a}")
                    chart_val[a_name] = chart_val.get(a_name, 0.0) + float(strat[a]) / ptotal
            chart[(hand_str, player_idx, hist_strs)] = chart_val

        chart_path = os.path.join(output_dir, f"{n_players}p{suffix}_preflop_policy.pkl")
        with open(chart_path, "wb") as f:
            pickle.dump(chart, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Chart PKL: {chart_path}  ({os.path.getsize(chart_path)//1024} KB, {len(chart):,} entries)")
    else:
        logger.info(f"Skipping chart PKL ({n:,} > 10M — would OOM)")


def main():
    parser = argparse.ArgumentParser(description="Extract policy from solver checkpoint (OOM-safe)")
    parser.add_argument("--solver", required=True, help="Path to solver .pkl checkpoint")
    parser.add_argument("--preflop", action="store_true", help="Preflop solver (default: postflop)")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    if not os.path.exists(args.solver):
        logger.error(f"File not found: {args.solver}")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {args.solver} ({os.path.getsize(args.solver)//1024//1024} MB)")
    t0 = time.time()

    if args.preflop:
        solver_data = _load_preflop_solver(args.solver)
        output_dir = args.output_dir or os.path.dirname(args.solver)
        extract_preflop(solver_data, output_dir)
    else:
        solver_data = _load_postflop_solver(args.solver)
        output_dir = args.output_dir or os.path.dirname(args.solver)
        extract_postflop(solver_data, output_dir)

    # Free the large arrays
    del solver_data
    logger.info(f"Total extraction time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
