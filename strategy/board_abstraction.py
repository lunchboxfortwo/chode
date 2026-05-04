"""
Board texture abstraction for postflop blueprint.

Maps any flop (3 cards) to a canonical texture ID used to index
the postflop strategy table.

Texture dimensions:
  high_card  : 0=A-high, 1=K-high, 2=Q-high, 3=J-high, 4=T-high, 5=low(≤9)
  paired     : 0=unpaired, 1=paired
  suit_tex   : 0=rainbow, 1=two-tone, 2=monotone
  connected  : 0=connected(gap≤1), 1=semi(gap≤2), 2=disconnected

64 canonical textures (6 × 2 × 3 × 2, with connectedness merged to 2 buckets
to keep table size manageable).
"""

RANKS = "23456789TJQKA"
RANK_VAL = {r: i for i, r in enumerate(RANKS)}  # '2'→0 … 'A'→12


def board_texture(cards: list[str]) -> dict:
    """
    Return a texture dict for a 3-card flop.
    cards: list of card strings like ['Ah', 'Kd', '2c']
    """
    ranks = sorted([RANK_VAL[c[0].upper()] for c in cards], reverse=True)
    suits = [c[1].lower() for c in cards]

    high = _high_bucket(ranks[0])
    paired = int(ranks[0] == ranks[1] or ranks[1] == ranks[2])
    suit_tex = _suit_bucket(suits)
    connected = _connect_bucket(ranks)

    return {
        "high": high,
        "paired": paired,
        "suit": suit_tex,
        "connected": connected,
    }


def texture_id(cards: list[str]) -> int:
    """Compact integer ID for the board texture (0–63)."""
    t = board_texture(cards)
    return t["high"] * 12 + t["paired"] * 6 + t["suit"] * 2 + t["connected"]


def texture_label(cards: list[str]) -> str:
    t = board_texture(cards)
    high_names = ["A", "K", "Q", "J", "T", "low"]
    suit_names = ["rainbow", "two-tone", "monotone"]
    conn_names = ["connected", "disconnected"]
    paired_str = "paired" if t["paired"] else ""
    parts = [high_names[t["high"]], paired_str, suit_names[t["suit"]], conn_names[t["connected"]]]
    return "-".join(p for p in parts if p)


# ─── Hand equity bucket ───────────────────────────────────────────────────────

def equity_bucket(equity: float) -> int:
    """Map [0,1] equity to bucket 0–3 (strong/medium/weak/air)."""
    if equity >= 0.70:
        return 0  # strong
    if equity >= 0.50:
        return 1  # medium
    if equity >= 0.30:
        return 2  # weak
    return 3       # air


# ─── Dynamic bet sizing ──────────────────────────────────────────────────────

def bet_fraction(board: list[str], street: str, equity: float) -> float:
    """
    Return fraction-of-pot bet size based on board texture, street, and equity.

    Dry boards  → smaller sizing (less protection needed, narrower value range)
    Wet boards  → larger sizing (charge draws, leverage range advantage)
    River       → polarize: strong hands overbet, thin value underbet
    """
    if not board or len(board) < 3:
        return 0.67

    t = board_texture(board[:3])

    # Wetness score 0-3
    wetness = t["suit"] + (1 - t["connected"])  # suit: 0-2, connected bonus: 1 if disconnected
    # paired boards are usually drier (top of range is clearer, less draw equity)

    if t["paired"]:
        base = 0.40
    elif wetness >= 3:   # monotone or connected two-tone
        base = 0.75
    elif wetness == 2:
        base = 0.60
    elif wetness == 1:
        base = 0.50
    else:                # dry rainbow disconnected
        base = 0.40

    if street == "river":
        # Polarize: overbets with monsters, small bets for thin value, bluff sizing matches value
        if equity >= 0.80:
            return 1.20   # overbet
        if equity >= 0.60:
            return 0.33   # thin value / blocking bet
        return base       # bluff / give up

    if street == "turn":
        base = min(base + 0.10, 1.0)   # slightly larger on turn (protection + leverage)

    return base


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _high_bucket(top_rank: int) -> int:
    if top_rank == 12: return 0   # A
    if top_rank == 11: return 1   # K
    if top_rank == 10: return 2   # Q
    if top_rank == 9:  return 3   # J
    if top_rank == 8:  return 4   # T
    return 5                       # 9 or lower


def _suit_bucket(suits: list[str]) -> int:
    unique = len(set(suits))
    if unique == 1: return 2   # monotone
    if unique == 2: return 1   # two-tone
    return 0                    # rainbow


def _connect_bucket(ranks: list[int]) -> int:
    """0=connected (all gaps ≤ 2), 1=disconnected."""
    gaps = [ranks[i] - ranks[i + 1] for i in range(len(ranks) - 1)]
    max_gap = max(gaps) if gaps else 0
    return 0 if max_gap <= 3 else 1
