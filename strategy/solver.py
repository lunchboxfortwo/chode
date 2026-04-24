"""
Wrapper for TexasSolver-Console. Falls back to heuristic when solver is unavailable.
"""
import subprocess
import tempfile
import os
import random
import resource
from pathlib import Path
from config import SOLVER_PATH, SOLVER_THREADS, SOLVER_RAM_GB, SOLVER_ITERATIONS, SOLVER_TIME_LIMIT

RANK_VALUE = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
              "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def _solver_available() -> bool:
    return Path(SOLVER_PATH).exists()


def _hand_strength_heuristic(hole: list[str], board: list[str]) -> float:
    """Rough [0,1] hand strength estimate without a full evaluator."""
    all_cards = hole + board
    ranks = sorted([RANK_VALUE.get(c[0].upper(), 0) for c in all_cards], reverse=True)
    suits = [c[1].lower() for c in all_cards]

    rank_counts: dict[int, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    suit_counts: dict[str, int] = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1

    max_pair = max(rank_counts.values(), default=1)
    flush_draw = max(suit_counts.values(), default=0) >= 4
    flush = max(suit_counts.values(), default=0) >= 5

    hole_ranks = [RANK_VALUE.get(c[0].upper(), 0) for c in hole]
    top_hole = max(hole_ranks) if hole_ranks else 0

    if max_pair >= 4:
        return 0.97
    if max_pair == 3 and len([v for v in rank_counts.values() if v >= 2]) >= 2:
        return 0.94  # full house-ish
    if flush:
        return 0.92
    if max_pair == 3:
        return 0.80
    pairs = [v for v in rank_counts.values() if v == 2]
    if len(pairs) >= 2:
        return 0.70
    if len(pairs) == 1:
        # Top pair bonus
        paired_rank = [k for k, v in rank_counts.items() if v == 2][0]
        strength = 0.45 + (paired_rank / 14) * 0.15
        return strength
    if flush_draw:
        return 0.35
    return (top_hole / 14) * 0.30


def _range_for_position(position: str) -> str:
    """Simple GTO range string for TexasSolver."""
    ranges = {
        "btn": "AA:1,KK:1,QQ:1,JJ:1,TT:1,99:1,88:0.75,AKs:1,AQs:1,AJs:1,ATs:1,KQs:1,QJs:1,JTs:1",
        "bb":  "AA:1,KK:1,QQ:1,JJ:1,TT:1,99:1,88:1,77:1,66:1,AKs:1,AQs:1,AJs:1,ATs:1,A9s:1,KQs:1",
        "sb":  "AA:1,KK:1,QQ:1,JJ:1,TT:1,99:1,AKs:1,AQs:1,AJs:1,KQs:1,QJs:1",
        "default": "AA:1,KK:1,QQ:1,JJ:1,TT:1,AKs:1,AQs:1,AJs:1,AKo:1",
    }
    return ranges.get(position.lower(), ranges["default"])


def _set_solver_limits():
    """Called in subprocess before exec — limits RAM to SOLVER_RAM_GB."""
    ram_bytes = SOLVER_RAM_GB * 1024 ** 3
    try:
        resource.setrlimit(resource.RLIMIT_AS, (ram_bytes, ram_bytes))
    except Exception:
        pass


def solve_postflop(
    hole: list[str],
    board: list[str],
    pot: int,
    effective_stack: int,
    hero_position: str = "btn",
    villain_position: str = "bb",
) -> dict:
    """
    Returns a dict: {'action': 'bet'|'check'|'call'|'fold', 'amount': int, 'equity': float}
    """
    equity = _hand_strength_heuristic(hole, board)

    if not _solver_available():
        return _heuristic_decision(equity, pot, effective_stack)

    return _run_solver(hole, board, pot, effective_stack, hero_position, villain_position, equity)


def _run_solver(hole, board, pot, effective_stack, hero_pos, villain_pos, fallback_equity):
    board_str = " ".join(board[:3]) if len(board) >= 3 else " ".join(board)
    hero_range = _range_for_position(hero_pos)
    villain_range = _range_for_position(villain_pos)

    config_lines = [
        f"set_pot {pot}",
        f"set_effective_stack {effective_stack}",
        f"set_board {board_str}",
        f"set_range_ip {hero_range}",
        f"set_range_oop {villain_range}",
        f"set_bet_sizes 0.5,1",
        f"set_raise_sizes 2.5",
        f"set_allin_threshold 1.5",
        f"solve",
        f"show_result",
    ]

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(config_lines))
            cfg_path = f.name

        result = subprocess.run(
            [str(SOLVER_PATH), cfg_path],
            capture_output=True,
            text=True,
            timeout=SOLVER_TIME_LIMIT * 2,
            preexec_fn=_set_solver_limits,
        )
        os.unlink(cfg_path)

        return _parse_solver_output(result.stdout, fallback_equity, pot, effective_stack)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return _heuristic_decision(fallback_equity, pot, effective_stack)


def _parse_solver_output(output: str, equity: float, pot: int, stack: int) -> dict:
    """Parse TexasSolver output for recommended action/frequency."""
    lines = output.lower().splitlines()
    bet_freq = 0.0
    check_freq = 0.0

    for line in lines:
        if "bet" in line and "%" in line:
            try:
                bet_freq = float(line.split("%")[0].split()[-1]) / 100
            except ValueError:
                pass
        if "check" in line and "%" in line:
            try:
                check_freq = float(line.split("%")[0].split()[-1]) / 100
            except ValueError:
                pass

    if bet_freq > 0 and random.random() < bet_freq:
        amount = int(pot * 0.6)
        return {"action": "bet", "amount": min(amount, stack), "equity": equity}
    return {"action": "check", "amount": 0, "equity": equity}


def _heuristic_decision(equity: float, pot: int, stack: int) -> dict:
    """Fallback decision based purely on hand strength estimate."""
    if equity >= 0.75:
        amount = min(int(pot * 0.75), stack)
        return {"action": "bet", "amount": amount, "equity": equity}
    if equity >= 0.50:
        amount = min(int(pot * 0.40), stack)
        return {"action": "bet", "amount": amount, "equity": equity}
    if equity >= 0.30:
        return {"action": "check", "amount": 0, "equity": equity}
    return {"action": "fold", "amount": 0, "equity": equity}
