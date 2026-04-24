from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RANGES_DIR = DATA_DIR / "ranges"
HISTORY_DIR = DATA_DIR / "hand_histories"

BUY_IN = 10_000
SMALL_BLIND = 50
BIG_BLIND = 100
NUM_PLAYERS = 6

SOLVER_PATH = BASE_DIR / "TexasSolver" / "console_solver"
SOLVER_THREADS = 4
SOLVER_RAM_GB = 8
SOLVER_ITERATIONS = 100
SOLVER_TIME_LIMIT = 1.0

BOT_NAMES = ["Atlas (GTO)", "Maverick (Whale)", "Niles (Nit)", "Oracle (Adaptive)", "Echo (GTO)"]
BOT_TYPES = ["gto", "whale", "nit", "adaptive", "gto"]

POSITIONS_6MAX = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
