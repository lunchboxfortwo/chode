"""
Preflop chart data layer.

Loads compact pre-extracted MCCFR policies (data/preflop_tables/*_preflop_policy.pkl)
and exposes them in a shape suitable for a 13x13 hand-grid UI in the
GTOWizard style — with positions, scenarios (action histories) and per-hand
mixed-strategy probabilities for {fold, call, bet, allin}.

Storage format reminder (see strategy/preflop_solver.py):
    key   = (hand_category, player_idx, action_history_tuple)
    value = {action_key: float16_bytes}      # canonical entries
          | (hand_category, player_idx, ah)  # reference to canonical entry

We keep the table in memory in *raw* form (so we can introspect it cheaply)
and dequantize lazily per query — this matches what PreflopSolver does.
"""
from __future__ import annotations

import gzip
import logging
import pickle
import struct
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

TABLES_DIR = Path(__file__).parent.parent / "data" / "preflop_tables"

RANKS = "AKQJT98765432"  # display order: top/left = A, bottom/right = 2
ACTIONS = ("fold", "call", "bet", "squeeze", "allin")

# Stack-depth suffixes that may appear on disk
DEPTHS = (30, 50, 75, 100, 150, 200)


# ─── Position labels ──────────────────────────────────────────────────────────
#
# OpenSpiel universal_poker player_idx layout (0-indexed from SB), matching
# strategy/preflop_solver.py:
#   2p :  BTN(SB)=0, BB=1
#   3p :  SB=0, BB=1, BTN=2
#   4p+:  SB=0, BB=1, UTG=2, ..., BTN=N-1

def position_labels(n_players: int) -> list[str]:
    """Return a position label for each player_idx in [0, n_players).

    Matches preflop_fixed_train.py player ordering:
    Player 0 = first to act preflop, last two = SB, BB.
    """
    if n_players == 2:
        return ["SB", "BB"]       # SB=BTN acts first
    if n_players == 3:
        return ["BTN", "SB", "BB"]
    if n_players == 4:
        return ["CO", "BTN", "SB", "BB"]
    if n_players == 5:
        return ["HJ", "CO", "BTN", "SB", "BB"]
    if n_players == 6:
        return ["LJ", "HJ", "CO", "BTN", "SB", "BB"]
    if n_players == 7:
        return ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
    if n_players == 8:
        return ["UTG1", "UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
    # Fallback for larger tables
    n_early = n_players - 4
    early = [f"EP{i+1}" for i in range(n_early)]
    return [*early, "LJ", "CO", "BTN", "SB", "BB"]


# ─── Hand grid ───────────────────────────────────────────────────────────────

def hand_at(row: int, col: int) -> str:
    """
    Map (row, col) in a 13x13 grid to a poker hand category.
        rows top->bottom and cols left->right both run A,K,Q,...,2.
        diagonal (row==col) -> pairs
        upper triangle (row<col) -> suited (row rank > col rank in poker terms)
        lower triangle (row>col) -> offsuit
    """
    if row == col:
        return f"{RANKS[row]}{RANKS[row]}"
    if row < col:
        return f"{RANKS[row]}{RANKS[col]}s"
    return f"{RANKS[col]}{RANKS[row]}o"


def grid_hands() -> list[list[str]]:
    return [[hand_at(r, c) for c in range(13)] for r in range(13)]


# ─── Policy loading + decoding ───────────────────────────────────────────────

_table_cache: dict[tuple[int, int], dict] = {}


def _policy_path(n_players: int, stack_bb: int) -> Path:
    suffix = "" if stack_bb == 100 else f"_{stack_bb}bb"
    return TABLES_DIR / f"{n_players}p{suffix}_preflop_policy.pkl"


def _progress_path(n_players: int, stack_bb: int) -> Path:
    suffix = "" if stack_bb == 100 else f"_{stack_bb}bb"
    return TABLES_DIR / f"{n_players}p{suffix}_solver.progress.json"


def _load_raw(n_players: int, stack_bb: int) -> dict | None:
    key = (n_players, stack_bb)
    if key in _table_cache:
        return _table_cache[key]
    p = _policy_path(n_players, stack_bb)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        try:
            with gzip.open(p, "rb") as f:
                raw = pickle.load(f)
        except OSError:
            with open(p, "rb") as f:
                raw = pickle.load(f)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", p, exc)
        return None
    _table_cache[key] = raw
    return raw


def _decode_entry(raw: dict, entry) -> dict[str, float] | None:
    """Resolve a reference, dequantize, and renormalize."""
    if isinstance(entry, tuple):
        entry = raw.get(entry)
    if not isinstance(entry, dict):
        return None
    out: dict[str, float] = {}
    for k, v in entry.items():
        if isinstance(v, (bytes, bytearray)):
            out[k] = float(struct.unpack("e", v)[0])
        else:
            out[k] = float(v)
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    # Ensure all four keys present for stable UI
    for a in ACTIONS:
        out.setdefault(a, 0.0)
    return out


# ─── Public API ──────────────────────────────────────────────────────────────

def list_configs() -> list[dict]:
    """All (n_players, stack_bb) combos that have a usable compact policy on disk."""
    out = []
    for n in range(2, 10):
        for bb in DEPTHS:
            p = _policy_path(n, bb)
            if p.exists() and p.stat().st_size > 0:
                out.append({
                    "n_players": n,
                    "stack_bb": bb,
                    "size_bytes": p.stat().st_size,
                })
    return out


def list_spots(n_players: int, stack_bb: int) -> list[dict] | None:
    """
    Enumerate distinct decision spots in this policy.

    Returns one entry per (player_idx, action_history) seen in the table,
    annotated with a human-readable label.
    """
    raw = _load_raw(n_players, stack_bb)
    if raw is None:
        return None

    pos = position_labels(n_players)
    seen: dict[tuple[int, tuple], int] = {}
    for k in raw.keys():
        if not isinstance(k, tuple) or len(k) != 3:
            continue
        _, pidx, hist = k
        if not isinstance(pidx, int):
            continue
        seen[(pidx, tuple(hist) if hist else ())] = seen.get((pidx, tuple(hist) if hist else ()), 0) + 1

    spots = []
    for (pidx, hist), n_hands in sorted(seen.items(), key=lambda kv: (len(kv[0][1]), kv[0][0], kv[0][1])):
        if pidx >= len(pos):
            continue
        spots.append({
            "player_idx": pidx,
            "position": pos[pidx],
            "history": list(hist),
            "depth": len(hist),
            "label": _spot_label(pos, pidx, hist),
            "n_hands": n_hands,
        })
    return spots


def get_chart(
    n_players: int, stack_bb: int, player_idx: int, history: Iterable[str]
) -> dict | None:
    """Return a 13x13 grid of {action: prob} dicts for the requested spot."""
    raw = _load_raw(n_players, stack_bb)
    if raw is None:
        return None
    hist_t = tuple(history)

    grid: list[list[dict[str, float] | None]] = []
    found = 0
    for r in range(13):
        row = []
        for c in range(13):
            hand = hand_at(r, c)
            entry = raw.get((hand, player_idx, hist_t))
            decoded = _decode_entry(raw, entry) if entry is not None else None
            if decoded is not None:
                found += 1
            row.append(decoded)
        grid.append(row)

    if found == 0:
        return None

    pos = position_labels(n_players)
    return {
        "n_players": n_players,
        "stack_bb": stack_bb,
        "player_idx": player_idx,
        "position": pos[player_idx] if player_idx < len(pos) else f"P{player_idx}",
        "history": list(hist_t),
        "label": _spot_label(pos, player_idx, hist_t),
        "actions": list(ACTIONS),
        "hands": grid_hands(),  # 13x13 string grid for client convenience
        "grid": grid,           # 13x13 of {action: prob} or null
        "n_hands_decoded": found,
    }


def progress_summary() -> list[dict]:
    """One row per known (n, bb) with current iteration count + policy presence."""
    import json
    rows = []
    for n in range(2, 10):
        for bb in DEPTHS:
            prog_p = _progress_path(n, bb)
            pol_p  = _policy_path(n, bb)
            if not prog_p.exists() and not pol_p.exists():
                continue
            iters = None
            ckpt = None
            if prog_p.exists():
                try:
                    j = json.loads(prog_p.read_text())
                    iters = j.get("iterations_done")
                    ckpt = j.get("checkpoint_time")
                except Exception:
                    pass
            rows.append({
                "n_players": n,
                "stack_bb": bb,
                "iterations": iters,
                "checkpoint_time": ckpt,
                "has_policy": pol_p.exists() and pol_p.stat().st_size > 0,
                "policy_bytes": pol_p.stat().st_size if pol_p.exists() else 0,
            })
    return rows


# ─── Spot labelling ──────────────────────────────────────────────────────────

_AC_PRETTY = {
    "fold": "fold",
    "call": "call",
    "bet":  "raise",
    "squeeze": "squeeze",
    "allin": "allin",
}


def _spot_label(positions: list[str], pidx: int, hist: tuple) -> str:
    """
    Build a short human-readable label like:
        'UTG opens (RFI)'
        'BTN — facing fold,fold,fold'
        'BB vs CO open'
    """
    pos = positions[pidx] if pidx < len(positions) else f"P{pidx}"
    if not hist:
        return f"{pos} — first to act"

    n = len(positions)
    # Reconstruct who acted: action order is UTG-first preflop.
    # Action 0 corresponds to player (first_to_act); subsequent actions advance through
    # the betting order. We approximate by walking positions in order of action.
    # Preflop action order: first-to-act, ..., SB, BB.
    # Trainer index mapping: player 0 = first-to-act preflop, player N-1 = BB.
    #   2p: BTN(=SB)=0, BB=1
    #   3p: BTN=0, SB=1, BB=2
    #   4p: UTG=0, ..., BTN=N-3, SB=N-2, BB=N-1
    if n == 2:
        order = [0, 1]                      # BTN(SB), BB
    elif n == 3:
        order = [0, 1, 2]                   # BTN, SB, BB
    else:
        order = list(range(n - 2)) + [n - 2, n - 1]  # UTG..BTN, SB, BB

    # Build actor sequence, skipping positions that have folded.
    # Once a position folds, it is removed from the rotation.
    live = list(order)
    actor_seq = []
    for a in hist:
        if not live:
            break
        actor = live.pop(0)
        actor_seq.append(actor)
        if a == "fold":
            pass  # already removed from live
        else:
            live.append(actor)  # still in the hand

    # Was there a raise/all-in already?
    raised = any(a in ("bet", "squeeze", "allin") for a in hist)
    only_folds = all(a == "fold" for a in hist)

    if only_folds:
        # Pure fold-around. Pidx is next to act.
        return f"{pos} — folds to you"

    if raised:
        # Find the most recent raiser
        last_raiser_pos = None
        for actor, action in zip(actor_seq, hist):
            if action in ("bet", "squeeze", "allin"):
                last_raiser_pos = positions[actor] if actor < len(positions) else f"P{actor}"
        h = ",".join(_AC_PRETTY.get(a, a) for a in hist)
        if last_raiser_pos:
            return f"{pos} vs {last_raiser_pos} ({h})"
        return f"{pos} — facing {h}"

    # All calls (limps)
    h = ",".join(_AC_PRETTY.get(a, a) for a in hist)
    return f"{pos} — after {h}"
