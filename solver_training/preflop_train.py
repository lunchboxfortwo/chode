"""
Preflop MCCFR trainer using OpenSpiel universal_poker.

Trains a separate strategy table for each player count (2-9).
Output: data/preflop_tables/{n}p_strategy.pkl

Usage:
    python3 solver_training/preflop_train.py --players 6 --iters 500000
    python3 solver_training/preflop_train.py --players all --iters 500000
"""
import sys
import os
import pickle
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspiel

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "preflop_tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUY_IN   = 10_000
SB       = 50
BB       = 100


# ─── ACPC game definition builders ───────────────────────────────────────────

def _blinds_str(n: int) -> str:
    """SB posts position 1, BB posts position 2, rest post 0."""
    blinds = [0] * n
    blinds[0] = SB
    blinds[1] = BB
    return " ".join(str(b) for b in blinds)


def _stacks_str(n: int) -> str:
    return " ".join(str(BUY_IN) for _ in range(n))


def _first_player_str(n: int) -> str:
    """UTG (position 3 in ACPC 1-indexed) acts first preflop."""
    if n == 2:
        return "1"   # SB acts first HU
    return str(min(3, n))


def build_preflop_game(n_players: int):
    """Build a preflop-only NLHE game for n_players."""
    gamedef = f"""\
GAMEDEF
nolimit
numPlayers = {n_players}
numRounds = 1
blind = {_blinds_str(n_players)}
firstPlayer = {_first_player_str(n_players)}
numSuits = 4
numRanks = 13
numHoleCards = 2
numBoardCards = 0
stack = {_stacks_str(n_players)}
END GAMEDEF
"""
    universal_poker = pyspiel.universal_poker
    return universal_poker.load_universal_poker_from_acpc_gamedef(gamedef)


# ─── Strategy extraction ──────────────────────────────────────────────────────

RANKS = "23456789TJQKA"
RANK_VAL = {r: i for i, r in enumerate(RANKS)}


def _card_str(card_int: int) -> str:
    """Convert OpenSpiel card int → 'Ah' style string."""
    rank_idx = card_int % 13
    suit_idx = card_int // 13
    suits = "cdhs"
    return RANKS[rank_idx] + suits[suit_idx]


def _hand_category(c1: int, c2: int) -> str:
    """Convert two card ints to canonical hand string like 'AKs', '99'."""
    r1 = RANKS[c1 % 13]
    s1 = c1 // 13
    r2 = RANKS[c2 % 13]
    s2 = c2 // 13
    if RANK_VAL[r1] < RANK_VAL[r2]:
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return f"{r1}{r2}"
    return f"{r1}{r2}s" if s1 == s2 else f"{r1}{r2}o"


def extract_strategy(game, policy, n_players: int) -> dict:
    """
    Walk the game tree and collect action probabilities keyed by
    (hand_category, position_name, action_history_str).

    Returns:
        {
          "AA": {
            "BTN": {
              "": {"fold": 0.0, "call": 0.05, "raise_2.5x": 0.95},
              "raise": {"fold": 0.0, "call": 0.3, "raise_3x": 0.7},
              ...
            },
            ...
          },
          ...
        }
    """
    positions = _position_names(n_players)
    strategy = {}

    # BFS over all information states
    seen = set()
    stack = [game.new_initial_state()]

    while stack:
        state = stack.pop()

        if state.is_terminal():
            continue

        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                stack.append(state.child(action))
            continue

        player = state.current_player()
        info_str = state.information_state_string(player)

        if info_str in seen:
            continue
        seen.add(info_str)

        # Parse cards from info state
        try:
            cards = _parse_cards_from_info(info_str, state, player)
        except Exception:
            for a in state.legal_actions():
                stack.append(state.child(a))
            continue

        if len(cards) < 2:
            for a in state.legal_actions():
                stack.append(state.child(a))
            continue

        hand = _hand_category(cards[0], cards[1])
        position = positions[player]
        action_hist = _parse_action_history(info_str)

        # Get action probabilities from policy
        probs = policy.action_probabilities(state, player)
        action_labels = _label_actions(state.legal_actions(), state, n_players)

        entry = {label: probs.get(a, 0.0)
                 for a, label in zip(state.legal_actions(), action_labels)}

        strategy.setdefault(hand, {}).setdefault(position, {})[action_hist] = entry

        for a in state.legal_actions():
            child = state.child(a)
            if not child.is_terminal():
                stack.append(child)

    return strategy


def _position_names(n: int) -> list[str]:
    """Map player index → position name."""
    if n == 2:
        return ["SB", "BB"]
    pos = ["SB", "BB"] + [f"P{i}" for i in range(3, n + 1)]
    # Rename by standard 6-max names if applicable
    names_6 = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
    names_9 = ["SB", "BB", "UTG", "UTG1", "UTG2", "HJ", "LJ", "CO", "BTN"]
    if n == 6:
        return names_6
    if n == 9:
        return names_9
    return pos[:n]


def _parse_cards_from_info(info_str: str, state, player: int) -> list[int]:
    """Extract hole card ints from state."""
    hand = state.get_game().new_initial_state()
    # Use state's private information directly
    try:
        private = state.private_observation_string(player)
        # OpenSpiel encodes cards as integers accessible via state
        return list(state.observer_history(player)) if hasattr(state, 'observer_history') else []
    except Exception:
        pass
    return []


def _parse_action_history(info_str: str) -> str:
    """Extract the betting action history string from info state."""
    # universal_poker info strings contain the action sequence
    lines = info_str.strip().splitlines()
    for line in lines:
        if line.startswith("["):
            return line.strip()
    return ""


def _label_actions(actions: list[int], state, n_players: int) -> list[str]:
    """Convert action integers to human-readable labels."""
    labels = []
    pot = sum(state.get_game().new_initial_state().returns()) if False else 0
    for a in actions:
        action_str = state.action_to_string(state.current_player(), a)
        labels.append(action_str)
    return labels


# ─── Training loop ────────────────────────────────────────────────────────────

def train(n_players: int, n_iters: int, checkpoint_every: int = 50_000):
    print(f"\n{'='*60}")
    print(f"Training {n_players}-player preflop strategy")
    print(f"Iterations: {n_iters:,}  |  Checkpoint every: {checkpoint_every:,}")
    print(f"{'='*60}")

    game = build_preflop_game(n_players)
    print(f"Game: {game.get_type().short_name}, {game.num_players()} players")
    print(f"Info state size: {game.information_state_tensor_size()}")

    solver = pyspiel.OutcomeSamplingMCCFRSolver(game)

    start = time.time()
    for i in range(1, n_iters + 1):
        solver.run_iteration()

        if i % 10_000 == 0:
            elapsed = time.time() - start
            rate = i / elapsed
            remaining = (n_iters - i) / rate
            print(f"  iter {i:>8,} / {n_iters:,}  |  "
                  f"{rate:,.0f} iter/s  |  "
                  f"~{remaining/60:.1f} min remaining", flush=True)

        if i % checkpoint_every == 0 or i == n_iters:
            policy = solver.average_policy()
            path = os.path.join(OUTPUT_DIR, f"{n_players}p_solver.pkl")
            with open(path, "wb") as f:
                pickle.dump(solver, f)
            print(f"  Checkpoint saved: {path}")

    policy = solver.average_policy()
    path = os.path.join(OUTPUT_DIR, f"{n_players}p_solver.pkl")
    with open(path, "wb") as f:
        pickle.dump(solver, f)

    elapsed = time.time() - start
    print(f"\nDone. {n_iters:,} iterations in {elapsed/60:.1f} min")
    print(f"Saved: {path}")
    return solver, game


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", default="6",
                        help="Player count or 'all' for 2-9")
    parser.add_argument("--iters", type=int, default=500_000,
                        help="MCCFR iterations (500k = ~10min, 5M = ~2hr)")
    args = parser.parse_args()

    if args.players == "all":
        counts = [2, 3, 4, 5, 6, 7, 8, 9]
    else:
        counts = [int(args.players)]

    for n in counts:
        train(n, args.iters)

    print("\nAll training complete.")
    print(f"Tables saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
