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
import time
from pathlib import Path

from tqdm import tqdm

from src.model import AlphaZeroNet
from src.self_play import run_self_play
from src.trainer import ReplayBuffer, Trainer
from src.evaluate import evaluate_dataset, evaluate_opening_move, play_batched_arena
from src.utils import (
    get_device,
    get_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaZero Connect Four Training")
    # Training
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)",
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        default=500,
        help="Self-play games per iteration (default: 500)",
    )
    parser.add_argument(
        "--training-simulations",
        type=int,
        default=2400,
        help="MCTS simulations per move for training (default: 2400)",
    )
    parser.add_argument(
        "--eval-simulations",
        type=int,
        default=100,
        help="MCTS simulations per move for evaluation (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Training epochs per iteration (default: 2)",
    )
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=None,
        help="Mini-batches per epoch (default: auto-scale to buffer size)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )

    # Model
    parser.add_argument(
        "--res-blocks",
        type=int,
        default=10,
        help="Number of residual blocks (default: 10)",
    )
    parser.add_argument(
        "--filters",
        type=int,
        default=128,
        help="Number of convolutional filters (default: 128)",
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints (default: checkpoints/)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from checkpoint",
    )

    # Replay buffer
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=100_000,
        help="Replay buffer capacity (default: 100000)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print("🎮 AlphaZero Connect Four Training")
    print(f"   Device: {device}")
    print(f"   Iterations: {args.iterations}")
    print(f"   Games/iteration: {args.games_per_iteration}")
    print(f"   MCTS simulations: {args.training_simulations}")
    print(f"   Model: {args.res_blocks} res blocks, {args.filters} filters")
    print()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = AlphaZeroNet(num_res_blocks=args.res_blocks, num_filters=args.filters).to(
        device
    )

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
            print(
                f"   Resumed at iteration {start_iteration}, buffer size: {len(replay_buffer)}"
            )
    print()

    # Main training loop
    for iteration in range(start_iteration, args.iterations):
        iter_start = time.time()
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        print(
            f"━━━ Iteration {iteration + 1}/{args.iterations} (lr={current_lr:.6f}) ━━━"
        )

        # ── Self-play phase ──
        print(
            f"  🎲 Self-play: {args.games_per_iteration} games, {args.training_simulations} sims/move ..."
        )
        model.eval()

        sp_start = time.time()
        examples = run_self_play(
            model=model,
            num_games=args.games_per_iteration,
            num_simulations=args.training_simulations,
        )
        sp_time = time.time() - sp_start

        replay_buffer.add(examples)
        print(
            f"     Generated {len(examples)} examples in {sp_time:.1f}s "
            f"(buffer: {len(replay_buffer)})"
        )

        # Save previous model for Arena
        previous_model = AlphaZeroNet(args.res_blocks, args.filters).to(device)
        previous_model.load_state_dict(model.state_dict())
        previous_model.eval()

        # ── Training phase ──
        # When batches_per_epoch is None, trainer auto-scales to buffer size
        effective_batches = args.batches_per_epoch or max(
            1, len(replay_buffer) // args.batch_size
        )
        print(f"  🧠 Training: {args.epochs} epochs × {effective_batches} batches ...")
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

        print(
            f"     Loss: policy={avg_losses['policy_loss']:.4f}, "
            f"value={avg_losses['value_loss']:.4f}, "
            f"total={avg_losses['total_loss']:.4f} "
            f"({train_time:.1f}s)"
        )

        # Step LR scheduler
        trainer.step_scheduler()

        # ── Evaluation phase ──
        print("  📊 Evaluation ...")
        best_move, center_pct = evaluate_opening_move(model, device)
        status = "✅" if best_move == 3 else "❌"
        print(
            f"     {status} Opening Move: played column {best_move + 1} (Center visits: {center_pct:.1f}%)"
        )

        dataset_results = {}
        testdata_dir = Path("testdata")
        if testdata_dir.exists():
            for file_path in sorted(testdata_dir.glob("Test_*")):
                with open(file_path, "r") as f:
                    lines = f.readlines()
                result = evaluate_dataset(model, device, lines)
                dataset_results[file_path.name] = result

        if dataset_results:
            print("     Dataset Evaluation:")
            for filename, result in dataset_results.items():
                print(
                    f"       - {filename}: "
                    f"acc={result['accuracy']:.1f}% "
                    f"mae={result['mae']:.3f} "
                    f"r={result['correlation']:.3f}"
                )

        # ── Arena Evaluation ──
        print("  ⚔️ Arena: New Model vs Previous Model (40 games) ...")
        m1_w_p1, m1_w_p2, m2_w_p1, m2_w_p2, draws = play_batched_arena(
            model1=model,
            model2=previous_model,
            device=device,
            num_games=40,
            num_simulations=args.eval_simulations,
        )
        m1_wins = m1_w_p1 + m1_w_p2
        m2_wins = m2_w_p1 + m2_w_p2
        win_rate = (m1_wins + 0.5 * draws) / 40 * 100
        print(
            f"     Result: New Model {m1_wins} - {m2_wins} Previous Model (Draws: {draws})"
        )
        print(f"     Breakdown (New Model): {m1_w_p1} wins as P1, {m1_w_p2} wins as P2")
        print(
            f"     Breakdown (Prev Model): {m2_w_p1} wins as P1, {m2_w_p2} wins as P2"
        )
        print(f"     New Model Winrate: {win_rate:.1f}%")

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
    print(
        f"   Latest checkpoint: {checkpoint_dir / f'checkpoint_{args.iterations:04d}.pt'}"
    )


if __name__ == "__main__":
    main()
