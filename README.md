# Chode 🃏

Neural-network poker AI for heads-up and multi-player No-Limit Texas Hold'em. Trained via MCCFR distillation with online fine-tuning, inspired by DeepStack.

## What It Does

- **Play poker** against GTO-trained bots via a browser UI (WebSocket)
- **Preflop charts** — interactive 13×13 grid for any (players, stack, position, scenario) combo
- **Postflop decisions** — neural net queries for flop/turn/river spots
- **Live training** — watch NN distillation and fine-tuning progress in real time

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  MCCFR      │────▶│  Distillation│────▶│  Neural Net │
│  Tabular    │     │  (supervised)│     │  (Preflop + │
│  Solver     │     │              │     │   Postflop)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                          ▲                       │
                          │                       ▼
                   ┌──────────────┐     ┌─────────────┐
                   │  Online MCCFR│◀────│  GTOBot     │
                   │  Fine-Tuning │     │  (gameplay)  │
                   └──────────────┘     └─────────────┘
```

**Phase 1 — Tabular MCCFR**: External-sampling CFR on fixed action abstractions. Runs for millions of iterations until strategies converge. Outputs PKL/NPZ policy tables.

**Phase 2 — Distillation**: Neural net learns to predict the CFR solver's strategies. Supervised training on (state → strategy) pairs extracted from the tabular solutions.

**Phase 3 — Online fine-tuning**: MCCFR traversals with the NN in the loop. Regret updates correct the NN beyond what distillation alone achieves.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the test suite
make test

# Start the server
python3 main.py
# → http://localhost:8765
```

### Training

```bash
# Preflop NN — distillation from tabular solver data
python3 solver_training/train_preflop_nn.py --phase distill --iters 1000

# Preflop NN — online MCCFR fine-tuning
python3 solver_training/train_preflop_nn.py --phase online --iters 100000 --resume

# Postflop NN — distillation from NPZ policy tables
python3 solver_training/train_postflop_nn.py --phase distill --iters 1000

# Postflop NN — online MCCFR fine-tuning
python3 solver_training/train_postflop_nn.py --phase online --iters 100000 --resume
```

Training progress is saved to `data/{preflop,postflop}_nn/training_progress.json` and viewable at `/progress`.

### Tabular Solver (Prerequisite for Distillation)

```bash
# Run MCCFR to generate tabular solutions (takes hours/days)
python3 solver_training/preflop_fixed_train.py --players 2 --stack-bb 30
python3 solver_training/postflop_fixed_train.py --players 2 --stack-bb 100
```

Outputs PKL (full regret tables) and NPZ (compact policy arrays) to `data/{preflop,postflop}_tables/`.

## Project Structure

```
chode/
├── bots/                   # Bot implementations
│   └── gto.py              # GTOBot — uses PreflopNN + PostflopNN
├── engine/                 # Game engine (deal, bet, showdown)
│   ├── game.py             # Main game loop, WebSocket handler
│   └── display.py          # Terminal/card rendering
├── strategy/               # Strategy modules
│   ├── preflop_nn.py       # Preflop neural net (encode, query, chart)
│   ├── postflop_nn.py      # Postflop neural net (encode, query)
│   ├── board_abstraction.py # Flop texture hashing
│   └── training_progress.py # Training status API
├── solver_training/        # Training scripts
│   ├── preflop_fixed_train.py  # MCCFR solver (preflop)
│   ├── postflop_fixed_train.py # MCCFR solver (postflop)
│   ├── train_preflop_nn.py     # NN distillation + online (preflop)
│   └── train_postflop_nn.py    # NN distillation + online (postflop)
├── static/                 # Web UI
│   ├── index.html          # Game table
│   ├── charts.html         # Preflop charts (13×13 grid)
│   └── progress.html       # Training progress dashboard
├── server.py               # FastAPI server + WebSocket + REST API
├── tests/                  # Test suite (242 tests)
│   ├── test_preflop_encode.py  # Key encoding, card mapping, labels
│   ├── test_postflop_nn.py     # Feature encoding, NN status
│   └── test_server.py          # API endpoints, chart sanity
├── Makefile                # make test, make check
└── main.py                 # Entrypoint
```

## Web UI

| Page | URL | Description |
|------|-----|-------------|
| Play | `/` | Live poker game vs bots |
| Charts | `/charts` | Interactive preflop strategy grid |
| Progress | `/progress` | NN training status, checkpoints, system stats |

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/preflop/nn/chart` | Full 169-hand preflop chart for a spot |
| `GET /api/preflop/nn/query` | Single-hand strategy query |
| `GET /api/preflop/nn/status` | Preflop NN model status |
| `GET /api/postflop/nn/status` | Postflop NN model status |
| `GET /api/progress` | Training jobs + system state |
| `WS /ws` | WebSocket game channel |

## Neural Network Details

### Preflop NN
- **Input**: 56-dim feature vector (hand category, suit, position, stack depth, action history)
- **Output**: 5-dim strategy (fold, call, bet, squeeze, all-in) + value head
- **Training data**: ~14K unique (hand, position, history) entries per config from tabular MCCFR
- **Configs**: 2p–6p × 30bb–100bb

### Postflop NN
- **Input**: 268-dim feature vector (hand category, position, street, board texture, pot/stack/facing sizes, action history)
- **Output**: 6-dim strategy (fold, check/call, bet 33%, bet 50%, bet 75%, all-in) + value head
- **Training data**: ~500K examples generated by simulating states and looking up in NPZ policy tables
- **Board abstraction**: 32-class K-means texture clustering on flop equity distributions

## Testing

```bash
make test          # Run all 242 tests (~3s)
make test-quick    # Encode + NN tests only
make test-server   # API tests only
make test-engine   # Game engine integration tests
make check         # test + confirmation
```

Tests run automatically on `git commit` via the pre-commit hook. Bypass with `--no-verify`.

## Key Design Decisions

- **Fixed action abstraction** for tabular MCCFR (not continuous sizing) — makes the game tree finite and tractable
- **MD5-hashed info keys** in NPZ files — enables columnar storage and binary search lookup during postflop distillation
- **Simulate-and-lookup** for postflop NN training — generate random states, hash them, look up in NPZ (~8% hit rate)
- **Mild class balancing** via `1/sqrt(freq)` weighting — prevents the NN from ignoring rare actions (fold) without overcompensating
- **On-demand NN loading** — server loads models lazily and auto-reloads when new checkpoints appear

## License

Private project. All rights reserved.
