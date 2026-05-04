#!/usr/bin/env python3
"""
Parallel MCCFR training with periodic strategy merging.

External-sampling MCCFR's strat_sums are additive across independent
solver instances. This script runs N parallel solvers on separate cores,
each with a different random seed, and periodically merges their
strat_sums back into a shared checkpoint.

Speedup: ~N× (linear scaling, limited by RAM and merge overhead).

Usage:
    # Run 6 parallel workers for 5p 30bb (default concurrency = nproc - 2)
    python3 solver_training/parallel_train.py --players 5 --stack-bb 30 --resume

    # Explicit worker count
    python3 solver_training/parallel_train.py --players 5 --stack-bb 30 --resume --workers 4

    # Custom merge interval
    python3 solver_training/parallel_train.py --players 5 --stack-bb 30 --resume --merge-every 500000
"""

import os
import sys
import json
import time
import pickle
import shutil
import random
import logging
import argparse
import tempfile
import subprocess
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_training.preflop_fixed_train import (
    Solver, _load_solver, _save_ckpt, OUTPUT_DIR, BB
)

logger = logging.getLogger(__name__)


def _solver_path(n_players: int, stack_bb: int) -> str:
    suffix = "" if stack_bb == 100 else f"_{stack_bb}bb"
    return os.path.join(OUTPUT_DIR, f"{n_players}p{suffix}_preflop_fixed_solver.pkl")


def _progress_path(n_players: int, stack_bb: int) -> str:
    suffix = "" if stack_bb == 100 else f"_{stack_bb}bb"
    return os.path.join(OUTPUT_DIR, f"{n_players}p{suffix}_preflop_fixed_solver.progress.json")


def _worker_path(n_players: int, stack_bb: int, worker_id: int) -> str:
    suffix = "" if stack_bb == 100 else f"_{stack_bb}bb"
    return os.path.join(OUTPUT_DIR, f".worker_{worker_id}_{n_players}p{suffix}_preflop_fixed_solver.pkl")


def _merge_solvers(base_path: str, worker_paths: list[str]) -> int:
    """
    Merge strat_sums from multiple worker checkpoints into the base checkpoint.
    Returns the total additional iterations merged.
    """
    # Load base
    with open(base_path, 'rb') as f:
        base = pickle.load(f)
    
    base_strat = base['strat_sums']
    base_iters = base.get('iterations', 0)
    total_added = 0

    for wpath in worker_paths:
        if not os.path.exists(wpath):
            continue
        with open(wpath, 'rb') as f:
            worker = pickle.load(f)
        
        if worker.get('format') != 'preflop-v1':
            logger.warning(f"Skipping worker {wpath}: not preflop-v1 format")
            continue
        
        # Ensure keys match
        if worker['keys'] != base['keys']:
            logger.warning(f"Worker {wpath} has different keys — skipping")
            continue
        
        # Add strat_sums
        w_strat = worker['strat_sums']
        iters = worker.get('iterations', 0)
        base_strat += w_strat
        total_added += iters
        logger.info(f"  Merged {wpath}: +{iters:,} iterations")

    # Update base
    base['iterations'] = base_iters + total_added
    
    # Atomic write
    tmp = base_path + '.tmp_merge'
    with open(tmp, 'wb') as f:
        pickle.dump(base, f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.move(tmp, base_path)
    
    return total_added


def _run_worker(args: dict):
    """
    Worker process: run a solver instance for `iters` iterations,
    then save to a worker-specific checkpoint.
    """
    n_players = args['n_players']
    stack_bb = args['stack_bb']
    iters = args['iters']
    worker_id = args['worker_id']
    seed = args['seed']
    base_path = args['base_path']
    
    # Load base checkpoint as starting point
    solver = _load_solver(base_path)
    
    # Override random seed
    random.seed(seed)
    
    logger.info(f"Worker {worker_id}: starting {iters:,} iterations (seed={seed})")
    
    t0 = time.time()
    for i in range(1, iters + 1):
        solver.run_iteration()
        
        if i % 10000 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            logger.info(f"Worker {worker_id}: {i:,}/{iters:,} iters ({rate:.0f} it/s)")
    
    elapsed = time.time() - t0
    logger.info(f"Worker {worker_id}: done in {elapsed:.0f}s")
    
    # Save to worker-specific path
    wpath = _worker_path(n_players, stack_bb, worker_id)
    solver.iterations = iters  # iterations THIS worker did
    _save_ckpt(solver, wpath, {'iterations_done': iters})
    
    return worker_id, iters


def main():
    parser = argparse.ArgumentParser(
        description="Parallel MCCFR training with periodic merging"
    )
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--stack-bb", type=int, default=30)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: nproc-2)")
    parser.add_argument("--merge-every", type=int, default=1_000_000,
                        help="Iterations per worker between merges")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--iters", type=int, default=None,
                        help="Total iterations target (default: convergence target)")
    parser.add_argument("--checkpoint-every", type=int, default=10_000_000,
                        help="Save main checkpoint every N total iterations")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(process)d] %(message)s",
    )

    n_players = args.players
    stack_bb = args.stack_bb
    n_workers = args.workers or max(1, mp.cpu_count() - 2)
    base_path = _solver_path(n_players, stack_bb)
    prog_path = _progress_path(n_players, stack_bb)

    # Determine target iterations
    target = args.iters
    if target is None:
        from solver_training.preflop_fixed_train import _CONVERGENCE_TARGETS
        target = _CONVERGENCE_TARGETS.get((n_players, stack_bb), 50_000_000)
    
    # Get current progress
    current_iters = 0
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            prog = json.load(f)
        current_iters = prog.get('iterations_done', 0)
    
    remaining = max(target - current_iters, 0)
    
    logger.info(f"=== Parallel MCCFR: {n_players}p {stack_bb}bb ===")
    logger.info(f"  Target: {target:,} iters | Done: {current_iters:,} | Remaining: {remaining:,}")
    logger.info(f"  Workers: {n_workers} | Merge every: {args.merge_every:,} iters/worker")
    
    if remaining <= 0:
        logger.info("Already at target — nothing to do")
        return

    # Check checkpoint exists
    if not os.path.exists(base_path):
        logger.error(f"No checkpoint at {base_path} — run single-worker training first")
        return

    # Calculate iterations per worker per merge round
    iters_per_worker = args.merge_every
    total_rounds = (remaining + n_workers * iters_per_worker - 1) // (n_workers * iters_per_worker)
    
    logger.info(f"  Rounds: {total_rounds} | Iters/worker/round: {iters_per_worker:,}")
    logger.info(f"  Expected speedup: ~{n_workers}x")

    # Clean up any stale worker checkpoints
    for wid in range(n_workers):
        wpath = _worker_path(n_players, stack_bb, wid)
        if os.path.exists(wpath):
            os.remove(wpath)

    total_merged = 0
    t_start = time.time()

    for round_num in range(total_rounds):
        logger.info(f"\n--- Round {round_num + 1}/{total_rounds} ---")
        
        # Check if we need fewer iters in the last round
        still_remaining = target - current_iters - total_merged
        if still_remaining <= 0:
            break
        
        # Adjust iters per worker if last round
        round_iters = min(iters_per_worker, -(-still_remaining // n_workers))  # ceil div
        
        # Launch workers
        worker_args = []
        for wid in range(n_workers):
            worker_args.append({
                'n_players': n_players,
                'stack_bb': stack_bb,
                'iters': round_iters,
                'worker_id': wid,
                'seed': (round_num * n_workers + wid + 1) * 7919,  # unique seeds
                'base_path': base_path,
            })
        
        with mp.Pool(n_workers) as pool:
            results = pool.map(_run_worker, worker_args)
        
        # Merge
        worker_paths = [_worker_path(n_players, stack_bb, wid) for wid in range(n_workers)]
        merged = _merge_solvers(base_path, worker_paths)
        total_merged += merged
        
        # Update progress file
        new_total = current_iters + total_merged
        with open(prog_path, 'w') as f:
            json.dump({
                'iterations_done': new_total,
                'n_players': n_players,
                'stack_bb': stack_bb,
                'checkpoint_time': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'parallel_workers': n_workers,
            }, f)
        
        # Clean up worker files
        for wpath in worker_paths:
            if os.path.exists(wpath):
                os.remove(wpath)
        
        # Progress report
        elapsed = time.time() - t_start
        rate = total_merged / elapsed
        eta = (target - new_total) / rate / 86400 if rate > 0 else 0
        
        logger.info(f"  Round {round_num+1} merged: +{merged:,} iters")
        logger.info(f"  Total: {new_total:,}/{target:,} ({100*new_total/target:.1f}%)")
        logger.info(f"  Rate: {rate:.0f} it/s (wall-clock) | ETA: {eta:.1f} days")
        
        # Main checkpoint
        if new_total - current_iters >= args.checkpoint_every or new_total >= target:
            logger.info(f"  Checkpoint saved at {new_total:,} iterations")

    elapsed = time.time() - t_start
    final_total = current_iters + total_merged
    logger.info(f"\n=== DONE ===")
    logger.info(f"  Final: {final_total:,}/{target:,} iterations")
    logger.info(f"  Total time: {elapsed/3600:.1f} hours")
    logger.info(f"  Effective rate: {total_merged/elapsed:.0f} it/s (wall-clock)")
    
    # Re-extract policy
    logger.info(f"\nRe-extracting policy PKL...")
    from solver_training.preflop_fixed_train import extract_and_save
    solver = _load_solver(base_path)
    solver.iterations = final_total
    extract_and_save(solver, n_players, stack_bb)
    logger.info("Policy extraction complete")


if __name__ == "__main__":
    main()
