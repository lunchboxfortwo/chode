"""
Postflop MCCFR trainer — one solver for the full HU postflop game.

Trains on flop+turn+river using OpenSpiel universal_poker (no preflop betting).
At query time, states are reconstructed with real cards (same pattern as
the preflop solver) so no BFS extraction is needed.

Output: data/postflop_tables/postflop_solver.pkl

Usage:
    python3 solver_training/postflop_train.py --iters 2000000
"""
import sys
import os
import pickle
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspiel

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "postflop_tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUY_IN = 10_000
SB = 50
BB = 100


def build_postflop_game() -> object:
    """
    HU game starting from the flop: no preflop betting round.
    numBoardCards = 3 1 1  →  deal flop, then betting round, then turn, etc.
    firstPlayer = 2 1 1   →  in ACPC 1-indexed: player 2 = OOP (BB) acts first postflop.
    """
    gamedef = f"""\
GAMEDEF
nolimit
numPlayers = 2
numRounds = 3
blind = {SB} {BB}
firstPlayer = 2 1 1
numSuits = 4
numRanks = 13
numHoleCards = 2
numBoardCards = 3 1 1
stack = {BUY_IN} {BUY_IN}
END GAMEDEF
"""
    return pyspiel.universal_poker.load_universal_poker_from_acpc_gamedef(gamedef)


def train(n_iters: int, checkpoint_every: int = 200_000):
    output_path = os.path.join(OUTPUT_DIR, "postflop_solver.pkl")

    print(f"\n{'='*60}")
    print(f"Training HU postflop solver")
    print(f"Iterations: {n_iters:,}  |  Checkpoint every: {checkpoint_every:,}")
    print(f"{'='*60}")

    game = build_postflop_game()
    print(f"Game: {game.get_type().short_name}, info state size: {game.information_state_tensor_size()}")

    solver = pyspiel.OutcomeSamplingMCCFRSolver(game)

    start = time.time()
    for i in range(1, n_iters + 1):
        solver.run_iteration()

        if i % 50_000 == 0:
            elapsed = time.time() - start
            rate = i / elapsed
            rem = (n_iters - i) / rate
            print(f"  iter {i:>8,}/{n_iters:,}  |  {rate:,.0f} iter/s  |  ~{rem/60:.1f}min remaining",
                  flush=True)

        if i % checkpoint_every == 0 or i == n_iters:
            with open(output_path, "wb") as f:
                pickle.dump(solver, f)
            print(f"  Checkpoint saved: {output_path}")

    elapsed = time.time() - start
    print(f"\nDone. {n_iters:,} iterations in {elapsed/60:.1f} min")
    print(f"Saved: {output_path}")
    return solver, game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=2_000_000)
    args = parser.parse_args()
    train(args.iters)


if __name__ == "__main__":
    main()
