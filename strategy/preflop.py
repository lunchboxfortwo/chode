import json
import random
from config import RANGES_DIR

RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {r: i for i, r in enumerate(RANK_ORDER, 2)}

_range_cache: dict = {}


def _load_range(position: str) -> dict:
    key = position.lower()
    if key not in _range_cache:
        path = RANGES_DIR / f"{key}.json"
        with open(path) as f:
            _range_cache[key] = json.load(f)
    return _range_cache[key]


def hand_category(c1: str, c2: str) -> str:
    r1, s1 = c1[0].upper(), c1[1].lower()
    r2, s2 = c2[0].upper(), c2[1].lower()
    if RANK_VALUE.get(r1, 0) < RANK_VALUE.get(r2, 0):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return f"{r1}{r2}"
    return f"{r1}{r2}s" if s1 == s2 else f"{r1}{r2}o"


def should_open(position: str, c1: str, c2: str) -> bool:
    data = _load_range(position)
    cat = hand_category(c1, c2)
    return cat in data.get("open", [])


def preflop_action(position: str, c1: str, c2: str, facing_raise: bool = False, raise_position: str = "btn") -> str:
    """Returns 'raise', 'call', or 'fold'."""
    data = _load_range(position)
    cat = hand_category(c1, c2)

    if facing_raise:
        if position == "bb":
            defense_key = f"vs_{raise_position}_open_call"
            if cat in data.get("vs_open_3bet", []):
                return "raise"
            if cat in data.get(defense_key, data.get("vs_btn_open_call", [])):
                return "call"
            return "fold"
        else:
            if cat in data.get("vs_raise_3bet", []):
                return "raise"
            if cat in data.get("vs_raise_call", []):
                return "call"
            return "fold"

    if cat in data.get("open", []):
        return "raise"
    return "fold"


def open_raise_size(stack: int, bb: int = 100) -> int:
    return min(bb * 3, stack)


def three_bet_size(last_raise: int, stack: int) -> int:
    return min(last_raise * 3, stack)
