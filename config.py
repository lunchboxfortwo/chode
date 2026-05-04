from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RANGES_DIR = DATA_DIR / "ranges"
HISTORY_DIR = DATA_DIR / "hand_histories"

BUY_IN = 10_000
SMALL_BLIND = 50
BIG_BLIND = 100
NUM_PLAYERS = 6   # default; game supports 2–8

# Default 6-max bot lineup (5 bots + 1 human)
DEFAULT_BOT_CONFIGS = [
    {"name": "Atlas",   "type": "gto"},
    {"name": "Maverick","type": "whale"},
    {"name": "Niles",   "type": "nit"},
    {"name": "Oracle",  "type": "adaptive"},
    {"name": "Echo",    "type": "gto"},
]

# All available bot profiles for UI selection
BOT_PROFILES = {
    "gto":      {"label": "GTO",      "desc": "Game Theory Optimal — follows solver output"},
    "adaptive": {"label": "Adaptive", "desc": "GTO + exploitation overlay based on your tendencies"},
    "whale":    {"label": "Whale",    "desc": "Loose-aggressive, high VPIP, large bets"},
    "nit":      {"label": "Nit",      "desc": "Extremely tight — only top 5% of hands"},
}

POSITIONS_6MAX = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
