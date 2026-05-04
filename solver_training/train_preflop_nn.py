#!/usr/bin/env python3
"""
Neural preflop solver training via tabular distillation + online fine-tuning.

Two-phase approach:
  Phase 1 (Distillation): Train NN to match tabular MCCFR strategies.
    - Extract strategies directly from solver PKL files (fast)
    - Cache as .npz for repeated training rounds
    - Subsample + shuffle each round for SGD
  Phase 2 (Online MCCFR): Fine-tune NN with its own MCCFR traversals.

Usage:
  python3 solver_training/train_preflop_nn.py --iters 100 --phase distill
  python3 solver_training/train_preflop_nn.py --iters 100000 --phase online
"""

import argparse
import json
import logging
import os
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy.preflop_nn import (
    PreflopNet, encode_features, INPUT_DIM, OUTPUT_DIM, N_ACTIONS, ACTION_NAMES,
    MODEL_DIR,
)
from solver_training.preflop_fixed_train import (
    State, legal_actions, apply_action, is_terminal, payoff,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
SOLVER_CONFIGS = [
    (2, 50), (3, 30), (3, 50), (4, 30), (5, 30),
]

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
VALUE_LOSS_WEIGHT = 0.25
CHECKPOINT_EVERY = 50
LOG_EVERY = 10
SAMPLES_PER_ROUND = 50000  # subsample per distillation round
EPOCHS_PER_ROUND = 3

PROGRESS_FILE = MODEL_DIR / "training_progress.json"
DISTILL_CACHE = MODEL_DIR / "distill_data.npz"

_ANAME = {i: ACTION_NAMES[i] for i in range(N_ACTIONS)}


# ─── Data extraction ──────────────────────────────────────────────────────────

def _decode_key(key: bytes) -> tuple[int, int, int, float, int, list[int]]:
    """Decode info key to (r1, r2, suit, pidx, stack_bb, hist_action_indices).

    Key format (from preflop_fixed_train.py info_key):
      Byte 0: player index (0-7)
      Byte 1: hand_cat (r1 << 4) | r2
      Byte 2: suit (0=offsuit, 1=suited, 2=pair)
      Byte 3: n_players (2-8)
      Byte 4: stack_bb // 10
      Byte 5: raise_level (0-4)
      Byte 6: position index (0-7)
      Byte 7: n_cold_callers (0-7)
      Byte 8+: action_hist (1 byte per action, 0-4)
    """
    pidx = key[0]
    r1r2 = key[1]
    suit_byte = key[2]
    n_players = key[3]
    stack_bb_x10 = key[4]
    # raise_level = key[5]  -- not needed for features
    # pos_idx = key[6]       -- not needed for features
    # n_cold = key[7]        -- not needed for features
    r1 = r1r2 >> 4
    r2 = r1r2 & 0xF
    stack_bb = float(stack_bb_x10 * 10)
    hist = list(key[8:])
    return r1, r2, suit_byte, pidx, stack_bb, hist


def build_distill_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Extract training data directly from solver PKL files.
    Much faster than querying the chart API.
    Returns (features, targets) arrays.
    """
    import pickle
    all_feats = []
    all_targets = []

    for n_p, s_bb in SOLVER_CONFIGS:
        # Try both naming conventions: "2p_100bb_..." and "2p_..." (legacy)
        path = f'data/preflop_tables/{n_p}p_{s_bb}bb_preflop_fixed_solver.pkl'
        if not os.path.exists(path):
            path = f'data/preflop_tables/{n_p}p_preflop_fixed_solver.pkl'
        if not os.path.exists(path):
            logger.warning(f"Solver not found for {n_p}p {s_bb}bb")
            continue

        with open(path, 'rb') as f:
            d = pickle.load(f)

        if d.get('format') != 'preflop-v1':
            logger.warning(f"Skipping non-v1 checkpoint: {path}")
            continue

        keys = d['keys']
        strat = d['strat_sums']
        n = 0

        for i in range(len(keys)):
            probs = strat[i].copy()
            # Skip rows with inf/nan (overflow from very long training runs)
            if not np.all(np.isfinite(probs)):
                continue
            probs = np.maximum(probs, 0.0)
            total = probs.sum()
            if total < 1e-8:
                continue
            probs = probs / total

            r1, r2, suit, pidx, stack_bb, hist = _decode_key(keys[i])
            hist_names = [ACTION_NAMES[a] for a in hist if a < len(ACTION_NAMES)]

            # Convert rank indices to card indices for encode_features
            # suit: 0=offsuit, 1=suited, 2=pair
            if suit == 2:  # pair
                c1, c2 = r1 * 4, r1 * 4 + 1  # different suits
            elif suit == 1:  # suited
                c1, c2 = r1 * 4, r2 * 4  # same suit
            else:  # offsuit
                c1, c2 = r1 * 4, r2 * 4 + 1  # different suits

            try:
                feat = encode_features(n_p, float(s_bb), pidx, (c1, c2), hist_names)
            except (IndexError, ValueError):
                continue

            target = np.zeros(OUTPUT_DIM, dtype=np.float32)
            for a in range(N_ACTIONS):
                target[a] = probs[a]

            all_feats.append(feat.numpy())
            all_targets.append(target)
            n += 1

        logger.info(f"  {n_p}p {s_bb}bb: {n} examples")

    X = np.array(all_feats, dtype=np.float32)
    Y = np.array(all_targets, dtype=np.float32)
    return X, Y


def load_or_build_distill_data() -> tuple[np.ndarray, np.ndarray]:
    """Load cached distillation data, or build and cache it."""
    if DISTILL_CACHE.exists():
        logger.info(f"Loading cached distillation data from {DISTILL_CACHE}")
        data = np.load(DISTILL_CACHE)
        return data['features'], data['targets']

    logger.info("Building distillation dataset from solver checkpoints...")
    t0 = time.time()
    X, Y = build_distill_dataset()
    t1 = time.time()
    logger.info(f"Built {len(X)} examples in {t1-t0:.1f}s")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DISTILL_CACHE, features=X, targets=Y)
    logger.info(f"Cached to {DISTILL_CACHE} ({os.path.getsize(DISTILL_CACHE)/1024/1024:.1f} MB)")
    return X, Y


# ─── Phase 1: Distillation ───────────────────────────────────────────────────

def distill(net: PreflopNet, optimizer, device, total_rounds: int):
    """Train NN to match tabular MCCFR strategies."""
    X, Y = load_or_build_distill_data()
    n_examples = len(X)
    logger.info(f"Distillation: {n_examples:,} examples, {total_rounds} rounds")

    # Compute mild class-balanced weights using sqrt inverse frequency
    action_freq = Y.mean(axis=0)  # [5] average target probability per action
    action_freq = np.clip(action_freq, 0.005, None)  # floor at 0.5%
    action_weights = 1.0 / np.sqrt(action_freq)  # sqrt for milder balancing
    action_weights = action_weights / action_weights.sum() * len(action_weights)  # normalize
    action_weights_t = torch.from_numpy(action_weights.astype(np.float32)).to(device)
    logger.info(f"Action weights: {[f'{ACTION_NAMES[i]}={action_weights[i]:.2f}' for i in range(N_ACTIONS)]}")

    step = 0
    t0 = time.time()

    for round_num in range(1, total_rounds + 1):
        # Subsample for this round
        idx = np.random.choice(n_examples, size=min(SAMPLES_PER_ROUND, n_examples), replace=False)
        X_round = torch.from_numpy(X[idx]).to(device)
        Y_round = torch.from_numpy(Y[idx]).to(device)
        M_round = (Y_round > 0.001).float()

        net.train()
        for epoch in range(EPOCHS_PER_ROUND):
            perm = torch.randperm(len(X_round))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X_round), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(X_round))
                bi = perm[start:end]
                x = X_round[bi]
                y = Y_round[bi]
                m = M_round[bi]

                logits, pred_v = net(x)
                log_probs = F.log_softmax(logits.masked_fill(m < 0.5, -1e9), dim=-1)
                # Weighted cross-entropy: rare actions get more weight
                per_action = -(y * m * log_probs) * action_weights_t.unsqueeze(0)
                policy_loss = per_action.sum(dim=-1).mean()

                # Value target: weighted action index as proxy
                value_target = (y * torch.arange(OUTPUT_DIM, dtype=torch.float32, device=device).unsqueeze(0)).sum(dim=-1)
                value_loss = F.mse_loss(pred_v.squeeze(-1), value_target)

                loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            step += 1
            if step % LOG_EVERY == 0:
                elapsed = time.time() - t0
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info(
                    f"Step {step:,} | round {round_num}/{total_rounds} | "
                    f"loss={avg_loss:.4f} | {elapsed:.0f}s"
                )
                _save_progress(step, round_num, step * BATCH_SIZE, t0, total_rounds, phase="distill")

            if step % CHECKPOINT_EVERY == 0:
                _save_checkpoint(net, step)
                _save_progress(step, round_num, step * BATCH_SIZE, t0, total_rounds, phase="distill")

        if round_num % 5 == 0:
            logger.info(f"Round {round_num} done, loss={epoch_loss/max(n_batches,1):.4f}")

    _save_checkpoint(net, step)
    _save_progress(step, total_rounds, step * BATCH_SIZE, t0, total_rounds, phase="distill")
    logger.info(f"Distillation complete: {step} steps")


# ─── Phase 2: Online MCCFR ──────────────────────────────────────────────────

TRAIN_CONFIGS = [
    (2, 50), (3, 30), (4, 30), (5, 30),
]

CONFIG_WEIGHTS = {2: 2.0, 3: 2.0, 4: 1.5, 5: 1.0}


def train_online(net: PreflopNet, optimizer, device, total_iters: int):
    """Online MCCFR training with NN in the loop."""
    buf_features = np.zeros((BATCH_SIZE * 4, INPUT_DIM), dtype=np.float32)
    buf_targets = np.zeros((BATCH_SIZE * 4, OUTPUT_DIM), dtype=np.float32)
    buf_masks = np.zeros((BATCH_SIZE * 4, OUTPUT_DIM), dtype=np.float32)
    buf_values = np.zeros(BATCH_SIZE * 4, dtype=np.float32)
    buf_pos = 0

    t0 = time.time()
    games_played = 0
    step = 0

    logger.info(f"Online training: {total_iters:,} iterations")

    for iteration in range(1, total_iters + 1):
        n_players, stack_bb = _sample_config()

        feats, targets, masks, values = _traverse_mccfr(
            net, n_players, float(stack_bb), device,
        )
        games_played += 1

        for i in range(len(feats)):
            if buf_pos >= len(buf_features):
                _train_step(
                    net, optimizer,
                    buf_features[:buf_pos], buf_targets[:buf_pos],
                    buf_masks[:buf_pos], buf_values[:buf_pos],
                    device,
                )
                step += 1
                buf_pos = 0

                if step % LOG_EVERY == 0:
                    elapsed = time.time() - t0
                    gps = games_played / elapsed
                    logger.info(f"Step {step:,} | {gps:.1f} games/s | iter {iteration:,}/{total_iters:,}")
                    _save_progress(step, iteration, games_played, t0, total_iters, phase="online")

                if step % CHECKPOINT_EVERY == 0:
                    _save_checkpoint(net, step)
                    _save_progress(step, iteration, games_played, t0, total_iters, phase="online")

            if i < len(feats):
                buf_features[buf_pos] = feats[i]
                buf_targets[buf_pos] = targets[i]
                buf_masks[buf_pos] = masks[i]
                buf_values[buf_pos] = values[i]
                buf_pos += 1

    if buf_pos > 0:
        _train_step(net, optimizer, buf_features[:buf_pos], buf_targets[:buf_pos],
                    buf_masks[:buf_pos], buf_values[:buf_pos], device)
        step += 1

    _save_checkpoint(net, step)
    _save_progress(step, total_iters, games_played, t0, total_iters, phase="online")


def _sample_config() -> tuple[int, int]:
    total_w = sum(CONFIG_WEIGHTS.get(c[0], 1.0) for c in TRAIN_CONFIGS)
    r = random.random() * total_w
    cumul = 0
    for c in TRAIN_CONFIGS:
        cumul += CONFIG_WEIGHTS.get(c[0], 1.0)
        if r <= cumul:
            return c
    return TRAIN_CONFIGS[-1]


def _traverse_mccfr(net, n_players, stack_bb, device):
    result_feat, result_target, result_mask = [], [], []

    state = State(n_players=n_players, stack_bb=int(stack_bb))
    deck = list(range(52))
    random.shuffle(deck)
    for p in range(n_players):
        state.hole[p] = [deck[p * 2], deck[p * 2 + 1]]

    for updating in range(n_players):
        _traverse_rec(net, state, updating, [],
                      result_feat, result_target, result_mask,
                      n_players, float(stack_bb), device)

    n = len(result_feat)
    if n == 0:
        return (np.zeros((0, INPUT_DIM), np.float32),) * 3 + (np.zeros(0, np.float32),)

    return (np.array(result_feat, np.float32), np.array(result_target, np.float32),
            np.array(result_mask, np.float32), np.zeros(n, np.float32))


def _traverse_rec(net, s, updating, hist, out_f, out_t, out_m, n_p, sb, dev):
    if is_terminal(s):
        return payoff(s)[updating]

    p = s.acting
    legal = legal_actions(s)
    hist_names = [_ANAME[a] for a in hist]

    with torch.no_grad():
        feat = encode_features(n_p, sb, p, (s.hole[p][0], s.hole[p][1]), hist_names).unsqueeze(0).to(dev)
        mask = torch.zeros(1, OUTPUT_DIM, dtype=torch.bool, device=dev)
        for a in legal:
            mask[0, a] = True
        probs, _ = net.predict(feat, mask)
        strat = probs[0].cpu().numpy()

    if p == updating:
        values = np.zeros(N_ACTIONS, np.float32)
        for a in legal:
            values[a] = _traverse_rec(net, apply_action(s, a), updating, hist + [a],
                                      out_f, out_t, out_m, n_p, sb, dev)

        ev = sum(strat[a] * values[a] for a in legal)
        regrets = np.zeros(N_ACTIONS, np.float32)
        for a in legal:
            regrets[a] = values[a] - ev
        pos_reg = np.maximum(regrets, 0.0)
        pos_sum = pos_reg.sum()
        target = pos_reg / pos_sum if pos_sum > 0 else np.zeros(N_ACTIONS, np.float32)
        if pos_sum == 0:
            for a in legal:
                target[a] = 1.0 / len(legal)

        target_full = np.zeros(OUTPUT_DIM, np.float32)
        mask_full = np.zeros(OUTPUT_DIM, np.float32)
        for a in legal:
            target_full[a] = target[a]
            mask_full[a] = 1.0

        out_f.append(feat[0].cpu().numpy())
        out_t.append(target_full)
        out_m.append(mask_full)
        return ev
    else:
        weights = [strat[a] for a in legal]
        a = random.choices(legal, weights=weights, k=1)[0]
        return _traverse_rec(net, apply_action(s, a), updating, hist + [a],
                             out_f, out_t, out_m, n_p, sb, dev)


# ─── Shared utilities ────────────────────────────────────────────────────────

def _train_step(net, optimizer, features, targets, masks, values, device):
    n = len(features)
    idx = np.random.permutation(n)
    features = features[idx]; targets = targets[idx]
    masks = masks[idx]; values = values[idx]

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        x = torch.from_numpy(features[start:end]).to(device)
        t = torch.from_numpy(targets[start:end]).to(device)
        m = torch.from_numpy(masks[start:end]).to(device)
        v = torch.from_numpy(values[start:end]).to(device)

        logits, pred_v = net(x)
        log_probs = F.log_softmax(logits.masked_fill(m < 0.5, -1e9), dim=-1)
        policy_loss = -(t * m * log_probs).sum(dim=-1).mean()
        value_loss = F.mse_loss(pred_v.squeeze(-1), v)
        loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()


def _save_checkpoint(net, step):
    path = MODEL_DIR / f"preflop_nn_{step:08d}.pt"
    torch.save({"model": net.state_dict(), "step": step}, path)
    ckpts = sorted(MODEL_DIR.glob("preflop_nn_*.pt"))
    for old in ckpts[:-3]:
        old.unlink()
    logger.info(f"Checkpoint: {path.name}")


def _save_progress(step, iteration, games, t0, total_iters, phase="distill"):
    elapsed = time.time() - t0
    progress = {
        "step": step,
        "iteration": iteration,
        "games_played": games,
        "target_games": total_iters,
        "elapsed_seconds": elapsed,
        "games_per_second": games / max(elapsed, 1),
        "phase": phase,
        "configs": [list(c) for c in SOLVER_CONFIGS],
        "timestamp": time.time(),
        "last_update": time.time(),
        "last_checkpoint_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train neural preflop solver")
    parser.add_argument("--iters", type=int, default=100,
                        help="Distillation rounds or online iterations")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase", choices=["distill", "online"], default="distill")
    args = parser.parse_args()

    device = torch.device("cpu")
    net = PreflopNet()

    if args.resume:
        ckpts = sorted(MODEL_DIR.glob("preflop_nn_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            data = torch.load(latest, map_location=device, weights_only=False)
            net.load_state_dict(data["model"])
            logger.info(f"Resumed from {latest.name} (step {data.get('step', 0)})")

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    if args.phase == "distill":
        distill(net, optimizer, device, args.iters)
    else:
        train_online(net, optimizer, device, args.iters)


if __name__ == "__main__":
    main()
