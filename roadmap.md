# Chode Poker — Project Handoff

> Last updated: 2026-05-03  
> Project dir: `~/chode`

---

## Quick Status

| Component | Status |
|-----------|--------|
| **Server** | `python main.py` on port 8765 |
| **Watchdog** | `solver_training/training_watchdog.sh` (cron guard every 5 min) |
| **Active training** | NN Preflop — distillation → online MCCRF pipeline running. Tabular training killed. |
| **All postflop** | ✅ Complete (2p + 3p) |
| **All other preflop** | ✅ Complete (2p/3p/4p/5p at 30bb, 2p at 100bb) |
| **Tests** | 191 pass (60 postflop + 78 preflop + 22 optim + 31 NN) — `pytest solver_training/test_*.py` |

---

## Architecture

```
~/chode/
├── main.py              # Server entry (uvicorn, port 8765)
├── config.py            # Game params, dirs, bot profiles
├── server.py            # HTTP endpoints
├── engine/game.py       # Core game loop (deal, betting, showdown)
├── strategy/
│   ├── solver.py            # MC equity fallback
│   ├── postflop_solver.py   # Fixed-tree NPZ/PKL lookups + compact solver
│   ├── preflop_charts.py    # Stack-depth-aware GTO preflop queries
│   ├── preflop_nn.py        # Neural preflop strategy (continuous stack depth)
│   └── board_abstraction.py # Board texture, equity buckets, bet sizing
├── solver_training/
│   ├── postflop_fixed_train.py   # Postflop MCCFR trainer (flat-array Solver)
│   ├── preflop_fixed_train.py    # Preflop MCCFR trainer (optimized 2.2×)
│   ├── train_preflop_nn.py       # Neural preflop trainer (distill + online MCCFR)
│   ├── parallel_train.py         # [DEPRECATED] Parallel tabular MCCFR
│   ├── nash_conv.py              # NashConv audit tool
│   ├── training_watchdog.sh      # Orchestrates concurrent training
│   ├── guard.sh                  # Cron job restarts watchdog if dead
│   ├── extract_policy.py         # Standalone policy extraction from checkpoints
│   ├── test_postflop_fixed.py    # 60 tests
│   ├── test_preflop_fixed.py     # 78 tests
│   ├── test_preflop_optimizations.py  # 22 tests
│   └── test_preflop_nn.py        # 31 tests
├── data/
│   ├── postflop_tables/   # *.npz policy + *.pkl solver checkpoints + logs
│   └── preflop_tables/    # Same structure
└── roadmap.md
```

---

## Training Data (current)

### Postflop Fixed-Tree MCCFR

| Config | Iters | Info States | Policy File | Size |
|--------|-------|-------------|-------------|------|
| 2p | 2.14M | 236M | `2p_postflop_fixed_policy.npz` | 1.8 GB |
| 3p | 3.0M | 254M | `3p_postflop_fixed_policy.npz` | 1.8 GB |

### Preflop Fixed-Tree MCCFR

| Config | Iters Done | Target Iters | NashConv (bb/hand) | Status |
|--------|-----------|--------------|-------------------|--------|
| 2p 100bb | 50M | 1.7B | ~2.86 | ⏳ Needs 1.65B more |
| 2p 30bb | 30M | 500M | ~3.0 (est) | ⏳ Old format, needs retrain |
| 3p 30bb | 30M | 800M | ~2.55 | ⏳ Needs 770M more |
| 4p 30bb | 30M | 2.9B | ~4.87 | ⏳ Needs 2.87B more |
| 5p 30bb | 30M | 5.9B | ~6.98 | ⏳ Needs 5.87B more |
| 6p 30bb | **26M/30M** 🔄 | 22B | ~13.6 (est) | ⏳ In progress |

**NashConv targets:** < 0.5 bb/hand = strong GTO; < 1.0 = acceptable; < 2.0 = usable

### Neural Preflop Solver

Single network replaces all tabular policies — continuous stack depth, all player counts.

| Item | Detail |
|------|--------|
| Architecture | 4×512 MLP, 890K params |
| Input | hand(169) + position(8) + history(10) + n_players(5) + stack_bb(1) = 193 |
| Output | [fold, call, bet, squeeze, allin] probabilities |
| Training | ReBeL-style MCCFR self-play |
| Status | ✅ Code complete, untrained (random init) |
| Files | `strategy/preflop_nn.py`, `solver_training/train_preflop_nn.py` |
| API | `/api/preflop/nn/chart`, `/api/preflop/nn/query`, `/api/preflop/nn/status` |
| UI | /charts: "Neural Net" source toggle + continuous stack slider |

**Training command:** `python3 solver_training/train_preflop_nn.py --iters 1000000 [--resume]`
**Projected convergence:** ~12 hours to NC < 1.0, ~2.5 days to NC < 0.5 (CPU-only)

**Parallel training:** `python3 solver_training/parallel_train.py --players N --stack-bb 30 --resume`
- 10 workers on 12 cores → ~10× speedup over single-worker
- Optimal schedule: 2p+3p (5 workers each, ~1 day), then 4p (10 workers, ~4 days), then 5p (10 workers, ~12 days)
- **ETA for NC < 0.5: ~17 days** | **NC < 1.0: ~4 days**

**NashConv audit:** `python3 solver_training/nash_conv.py --players 2 3 4 5 --stack-bb 30`
- Uses multi-board showdown averaging (100 boards) for stable measurements
- Single-pass tree traversal computes all players' BR + policy values simultaneously
**Tool:** `python3 solver_training/nash_conv.py --players 2 3 4 5 --stack-bb 30`

---

## Bugs Fixed (Sessions 8-11, 2026-05-01 to 2026-05-03)

### Critical — Game Logic (invalidated all prior training)

1. **Postflop `to_call` overpayment** — Callers put in `to_call` chips instead of `facing_size - invested[p]`. BB overpaid by blind amount every call.
2. **3p+ free checks after call** — Call set `to_call=0`, so players behind got free checks instead of facing the bet.
3. **Wrong raise call amount** — After raise, `new_to_call = rt - to_call` was semantically wrong.
4. **Fold didn't end street** — After fold, remaining bettor was re-asked to act on own bet.

**Fix:** Replaced `to_call` with `facing_size` (total investment to match). Call amount = `facing_size - invested[p]`. All training rebuilt from scratch.

### High — Memory / Reliability

5. **OOM kill during extraction** — `extract_and_save()` built a 222M-entry Python dict (~44 GB) on top of solver's ~30 GB. Total ~74 GB > 62 GB RAM.
6. **PKL key mismatch** — PKL stored `bytes` keys, runtime built `tuple` keys → lookups always returned None. Dead code.
7. **`_free_ram_gb()` over-conservative** — `min(mem_available, swap_free+4)` returned 4 GB when swap was full but 30 GB RAM was free.
8. **Extraction reverse map** — `rev = {v:k for k,v in _key_index.items()}` was 2x larger than needed.

**Fix:** Streaming extraction directly from flat arrays → NPZ. Runtime uses `_fixed_key_bytes()` + hash-based NPZ lookup. `_free_ram_gb()` counts swap at 50%. Reverse map uses list.

### Medium

9. **Runtime `facing_bet` cleared on call** — 3p+ NPZ key mismatches. Fixed: call doesn't clear `facing_bet`.
10. **NPZ no metadata validation** — Added `n_players` check at load.
11. **`preflop_charts.py` dead code** — Duplicate function body after `return`.
12. **Watchdog no RAM checks** — Added `free_ram_gb()` + `need_ram_gb()` before every launch; pauses preflop during 3p postflop.
13. **Watchdog `free_ram_gb()` used psutil** — Rewrote using `/proc/meminfo`.
14. **Array growth 2x → 1.5x** — Saves ~8 GB at 254M info states.
15. **`effective_stack_bb()` wrong positional logic** — Used seat-index exclusion instead of action-order "behind"; 10% term used min(all opponents) instead of min(dangerous stacks behind). Fixed.
16. **`preflop_charts.py` legacy 3bet/4bet sizing** — Used `3×open` and `2.5×facing` instead of spec IP/OOP-aware 9/12bb and 2.3×/2.8×. Fixed.
17. **Postflop 80% dominance dead code** — Conservative `(1-freq)×pot` bound required 99.5% freq to trigger. Fixed: use regret-based EV loss approximation.
18. **`_equity_fallback()` unconstrained sizing** — Used `bet_fraction()` producing overbets and SPR-sensitive jams. Fixed: use fixed 33%/75% pot and raise formula per spec.

---

## Key Design Decisions

- **Solver memory layout:** Flat numpy arrays (`_regrets`, `_strat_sum`) with compact `bytes` keys via `_key_index` dict. Checkpoint format: `flat-v3`.
- **Policy format:** NPZ (compact, hash-based lookup) preferred over PKL (legacy, tuple keys). PKL only generated for small preflop charts (<10K info states).
- **Runtime lookup path:** `_fixed_key_bytes()` → `_hash_key()` → binary search in sorted `hashed_keys` array → index into `probs` array.
- **No side pot support** — Trainer always uses equal stacks. Medium-severity known limitation for 3p+ all-in with unequal stacks.
- **Dominance simplification** (`_simplify`): Hard 95% threshold works; soft 80% path is dead code (requires freq ≥ 0.995 which is stricter than 95%).
- **No LLM integration** — This is a pure game-theory solver. Bot personalities (gto, whale, nit, adaptive) are rule-based, configured in `config.py`.

---

## Known Issues / Tech Debt

| Issue | Severity | Notes |
|-------|----------|-------|
| No side pot payoff in 3p+ | Medium | Rare with equal stacks; no partition planned |
| `_simplify` 80% path dead | Low | Remove or lower `DOMINANCE_EV_THRESHOLD` |
| Server sometimes down after OOM | Low | Manual restart: `cd ~/chode && python main.py` |
| Watchdog spawns multiple instances | Low | `guard.sh` sometimes stacks; `pkill -f training_watchdog` to clean |
| Old buggy data in `data/old_buggy_v1/` | Low | Can delete to save ~30 GB disk |

---

## Quick Reference

```bash
# Start server
cd ~/chode && python main.py &

# Check training progress
cat data/postflop_tables/3p_postflop_fixed_solver.progress.json
cat data/preflop_tables/6p_30bb_preflop_fixed_solver.progress.json

# Monitor training log
tail -f data/postflop_tables/train_3p_fixed.log
tail -f data/preflop_tables/train_6p_30bb_fixed.log

# Run tests
cd ~/chode && pytest solver_training/test_postflop_fixed.py solver_training/test_preflop_fixed.py -v

# Restart watchdog (kill stale instances first)
pkill -f training_watchdog; rm -f solver_training/.preflop_paused
bash solver_training/training_watchdog.sh &

# Extract policy from checkpoint (standalone)
python solver_training/extract_policy.py --type postflop --n-players 3

# Check RAM + processes
free -h; ps aux --sort=-rss | head -8

# Check for OOM kills
sudo dmesg | grep -i oom
```
