#!/usr/bin/env python3
"""AlphaZero training script for Connect Four.

Usage:
    uv run main.py                          # Train with defaults
    uv run main.py --iterations 50          # Custom iteration count
    uv run main.py --resume                 # Resume from latest checkpoint (default)
    uv run main.py --no-resume              # Start fresh training

See --help for all options.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.model import AlphaZeroNet
from src.self_play import run_self_play
from src.trainer import ReplayBuffer, Trainer
from src.utils import (
    get_device,
    get_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaZero Connect Four Training"
    )
    # Training
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of training iterations (default: 100)",
    )
    parser.add_argument(
        "--games-per-iteration", type=int, default=100,
        help="Self-play games per iteration (default: 100)",
    )
    parser.add_argument(
        "--simulations", type=int, default=200,
        help="MCTS simulations per move (default: 200)",
    )
    parser.add_argument(
        "--epochs", type=int, default=4,
        help="Training epochs per iteration (default: 4)",
    )
    parser.add_argument(
        "--batches-per-epoch", type=int, default=None,
        help="Mini-batches per epoch (default: auto-scale to buffer size)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Training batch size (default: 256)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)",
    )

    # Model
    parser.add_argument(
        "--res-blocks", type=int, default=10,
        help="Number of residual blocks (default: 10)",
    )
    parser.add_argument(
        "--filters", type=int, default=128,
        help="Number of convolutional filters (default: 128)",
    )



    # Checkpoints
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory for checkpoints (default: checkpoints/)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh instead of resuming from checkpoint",
    )

    # Replay buffer
    parser.add_argument(
        "--buffer-capacity", type=int, default=200_000,
        help="Replay buffer capacity (default: 200000)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"🎮 AlphaZero Connect Four Training")
    print(f"   Device: {device}")
    print(f"   Iterations: {args.iterations}")
    print(f"   Games/iteration: {args.games_per_iteration}")
    print(f"   MCTS simulations: {args.simulations}")
    print(f"   Model: {args.res_blocks} res blocks, {args.filters} filters")
    print()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = AlphaZeroNet(
        num_res_blocks=args.res_blocks, num_filters=args.filters
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        total_iterations=args.iterations,
    )

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)

    # Resume from checkpoint if available
    start_iteration = 0
    if not args.no_resume:
        latest = get_latest_checkpoint(checkpoint_dir)
        if latest is not None:
            print(f"   Resuming from: {latest}")
            ckpt = load_checkpoint(latest, model, trainer.optimizer, device)
            start_iteration = ckpt["iteration"]
            if "replay_buffer" in ckpt and ckpt["replay_buffer"] is not None:
                replay_buffer.load_state(ckpt["replay_buffer"])
            if "scheduler_state_dict" in ckpt:
                trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print(f"   Resumed at iteration {start_iteration}, buffer size: {len(replay_buffer)}")
    print()



    # Main training loop
    for iteration in range(start_iteration, args.iterations):
        iter_start = time.time()
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        print(f"━━━ Iteration {iteration + 1}/{args.iterations} (lr={current_lr:.6f}) ━━━")

        # ── Self-play phase ──
        print(f"  🎲 Self-play: {args.games_per_iteration} games, {args.simulations} sims/move ...")
        model.eval()

        sp_start = time.time()
        examples = run_self_play(
            model=model,
            num_games=args.games_per_iteration,
            num_simulations=args.simulations,
        )
        sp_time = time.time() - sp_start

        replay_buffer.add(examples)
        print(f"     Generated {len(examples)} examples in {sp_time:.1f}s "
              f"(buffer: {len(replay_buffer)})")

        # ── Training phase ──
        batches_per_epoch = args.batches_per_epoch or max(1, len(replay_buffer) // args.batch_size)
        print(f"  🧠 Training: {args.epochs} epochs × {batches_per_epoch} batches ...")
        model.to(device)

        train_start = time.time()
        total_losses = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        for epoch in tqdm(range(args.epochs), desc="  Training", unit="epoch"):
            losses = trainer.train_epoch(
                replay_buffer, num_batches=args.batches_per_epoch
            )
            for k in total_losses:
                total_losses[k] += losses[k]

        train_time = time.time() - train_start
        avg_losses = {k: v / args.epochs for k, v in total_losses.items()}

        print(f"     Loss: policy={avg_losses['policy_loss']:.4f}, "
              f"value={avg_losses['value_loss']:.4f}, "
              f"total={avg_losses['total_loss']:.4f} "
              f"({train_time:.1f}s)")

        # Step LR scheduler
        trainer.step_scheduler()

        # ── Save checkpoint ──
        ckpt_path = checkpoint_dir / f"checkpoint_{iteration + 1:04d}.pt"
        save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            iteration=iteration + 1,
            replay_buffer=replay_buffer.get_state(),
            path=ckpt_path,
            scheduler=trainer.scheduler,
        )

        iter_time = time.time() - iter_start
        print(f"  💾 Saved: {ckpt_path} ({iter_time:.1f}s total)")
        print()


    print("✅ Training complete!")
    print(f"   Latest checkpoint: {checkpoint_dir / f'checkpoint_{args.iterations:04d}.pt'}")


if __name__ == "__main__":
    main()
