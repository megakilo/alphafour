#!/usr/bin/env python3
"""AlphaZero training script for Connect Four.

Usage:
    uv run main.py                          # Train with defaults
    uv run main.py --iterations 50          # Custom iteration count
    uv run main.py                          # Resume from latest checkpoint (default)
    uv run main.py --no-resume              # Start fresh training
    uv run main.py --compile                # Enable torch.compile on CUDA

See --help for all options.
"""

from __future__ import annotations

import argparse
import inspect
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import src.utils as utils_mod
from src.evaluate import (
    compute_elo_diff,
    evaluate_dataset,
    evaluate_opening_move,
    get_dataset_child_wdl,
    play_anchor_arena,
    play_batched_arena,
)
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
    parser = argparse.ArgumentParser(description="AlphaZero Connect Four Training")
    # Training
    parser.add_argument(
        "--iterations",
        type=int,
        default=150,
        help="Number of training iterations (default: 150)",
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        default=1000,
        help="Self-play games per iteration (default: 1000)",
    )
    parser.add_argument(
        "--training-simulations",
        type=int,
        default=600,
        help="MCTS simulations per move for training (default: 600)",
    )
    parser.add_argument(
        "--eval-simulations",
        type=int,
        default=400,
        help="MCTS simulations per move for periodic evaluation & Arena (default: 400)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="Run MCTS dataset eval, Arena, and Anchor Elo every N iterations (default: 5)",
    )
    parser.add_argument(
        "--arena-games",
        type=int,
        default=40,
        help="Paired-opening games per Arena / Anchor match (default: 40)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Training epochs per iteration after warmup phase (default: 2)",
    )
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=None,
        help="Mini-batches per epoch (default: auto-scale up to 500)",
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
        help="Peak learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.compile + channels_last on CUDA for self-play, eval, and training (default: --compile)",
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

    # Checkpoints & Replay Buffer
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
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=300_000,
        help="Replay buffer capacity (default: 300000)",
    )

    return parser.parse_args()


def _fmt_pct(val: float | None) -> str:
    if val is None or math.isnan(val):
        return "N/A"
    return f"{val:.1f}%"


def _save_replay_buffer_compat(state: Any, path: Path) -> None:
    save_fn = getattr(utils_mod, "save_replay_buffer", None)
    if callable(save_fn):
        save_fn(state, path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


def _load_replay_buffer_compat(path: Path) -> Any:
    load_fn = getattr(utils_mod, "load_replay_buffer", None)
    if callable(load_fn):
        return load_fn(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _print_dataset_summary(
    label: str,
    datasets: list[tuple[str, list[str], np.ndarray | None]],
    model: AlphaZeroNet,
    device: torch.device,
    num_simulations: int,
) -> None:
    if not datasets:
        return

    print(f"     Dataset Evaluation ({label}):")
    total_n = 0
    weighted_acc = 0.0
    weighted_mse = 0.0
    total_draws = 0
    weighted_draw_hits = 0.0
    total_opt_n = 0
    weighted_opt_hits = 0.0

    for filename, lines, child_wdl in datasets:
        res = evaluate_dataset(
            model,
            device,
            lines,
            num_simulations=num_simulations,
            child_wdl=child_wdl,
        )
        n_pos = sum(1 for ln in lines if len(ln.strip().split()) == 2)
        n_draws = sum(
            1
            for ln in lines
            if len(ln.strip().split()) == 2 and int(ln.strip().split()[1]) == 0
        )

        total_n += n_pos
        weighted_acc += res["wdl_accuracy"] * n_pos
        weighted_mse += res["value_mse"] * n_pos

        draw_rec = res["draw_recall"]
        if n_draws > 0 and draw_rec is not None and not math.isnan(draw_rec):
            total_draws += n_draws
            weighted_draw_hits += (draw_rec / 100.0) * n_draws

        opt_pct = res["optimal_move_pct"]
        if opt_pct is not None and not math.isnan(opt_pct):
            total_opt_n += n_pos
            weighted_opt_hits += (opt_pct / 100.0) * n_pos

        print(
            f"       - {filename:<10}: "
            f"wdl_acc={res['wdl_accuracy']:5.1f}%  "
            f"draw_recall={_fmt_pct(draw_rec):>6}  "
            f"mse={res['value_mse']:.3f}  "
            f"opt_move={_fmt_pct(opt_pct):>6}"
        )

    if total_n > 0:
        overall_acc = weighted_acc / total_n
        overall_mse = weighted_mse / total_n
        overall_draw = (
            (weighted_draw_hits / total_draws * 100.0)
            if total_draws > 0
            else float("nan")
        )
        overall_opt = (
            (weighted_opt_hits / total_opt_n * 100.0) if total_opt_n > 0 else None
        )
        print(
            f"       * {'OVERALL':<10}: "
            f"wdl_acc={overall_acc:5.1f}%  "
            f"draw_recall={_fmt_pct(overall_draw):>6}  "
            f"mse={overall_mse:.3f}  "
            f"opt_move={_fmt_pct(overall_opt):>6}"
        )


def main() -> None:
    args = parse_args()
    device = get_device()
    print("🎮 AlphaZero Connect Four Training")
    print(f"   Device: {device}")
    print(f"   Iterations: {args.iterations}")
    print(f"   Games/iteration: {args.games_per_iteration}")
    print(f"   Training simulations: {args.training_simulations}")
    print(
        f"   Eval simulations: {args.eval_simulations} (every {args.eval_interval} iters)"
    )
    print(
        f"   Model: {args.res_blocks} res blocks, {args.filters} filters "
        f"(compile={args.compile})"
    )
    print()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    replay_buffer_path = checkpoint_dir / "replay_buffer.pt"

    # Initialize model (with channels_last + dynamic torch.compile on CUDA when enabled)
    model = AlphaZeroNet(num_res_blocks=args.res_blocks, num_filters=args.filters)
    if hasattr(model, "optimize_for_device"):
        model.optimize_for_device(device, use_compile=args.compile)
    else:
        model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,}")

    # Initialize trainer (with graceful signature detection for use_compile)
    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "device": device,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "total_iterations": args.iterations,
    }
    if "use_compile" in trainer_params:
        trainer_kwargs["use_compile"] = args.compile
    trainer = Trainer(**trainer_kwargs)

    # Initialize replay buffer on target device when supported
    rb_params = inspect.signature(ReplayBuffer.__init__).parameters
    if "device" in rb_params:
        replay_buffer = ReplayBuffer(capacity=args.buffer_capacity, device=device)
    else:
        replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)

    # Resume from checkpoint + separate replay_buffer.pt (or legacy inline buffer)
    start_iteration = 0
    if not args.no_resume:
        latest = get_latest_checkpoint(checkpoint_dir)
        if latest is not None:
            print(f"   Resuming from: {latest}")
            ckpt = load_checkpoint(latest, model, trainer.optimizer, device)
            start_iteration = ckpt["iteration"]
            if replay_buffer_path.exists():
                buf_state = _load_replay_buffer_compat(replay_buffer_path)
                replay_buffer.load_state(buf_state)
            elif "replay_buffer" in ckpt and ckpt["replay_buffer"] is not None:
                replay_buffer.load_state(ckpt["replay_buffer"])
            trainer.restore_scheduler(
                start_iteration, ckpt.get("scheduler_state_dict")
            )
            print(
                f"   Resumed at iteration {start_iteration}, buffer size: {len(replay_buffer)}"
            )
    print()

    # Preload evaluation datasets and cached solver ground-truth child W/D/L once
    datasets: list[tuple[str, list[str], np.ndarray | None]] = []
    testdata_dir = Path("testdata")
    if testdata_dir.exists():
        for file_path in sorted(testdata_dir.glob("Test_*")):
            if file_path.is_file():
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                child_wdl = get_dataset_child_wdl(lines)
                datasets.append((file_path.name, lines, child_wdl))

    sp_params = inspect.signature(run_self_play).parameters
    sp_supports_arrays = "return_arrays" in sp_params and hasattr(
        replay_buffer, "add_batch"
    )
    sp_supports_uint8 = "return_uint8" in sp_params

    # Main training loop
    for iteration in range(start_iteration, args.iterations):
        iter_start = time.time()
        iter_num = iteration + 1
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        print(f"━━━ Iteration {iter_num}/{args.iterations} (lr={current_lr:.6f}) ━━━")

        # ── Self-play phase ──
        print(
            f"  🎲 Self-play: {args.games_per_iteration} games, "
            f"{args.training_simulations} sims/move ..."
        )
        model.eval()

        sp_start = time.time()
        if sp_supports_arrays:
            sp_kwargs: dict[str, Any] = {"return_arrays": True}
            if sp_supports_uint8:
                sp_kwargs["return_uint8"] = True
            states_arr, policies_arr, values_arr = run_self_play(
                model=model,
                num_games=args.games_per_iteration,
                num_simulations=args.training_simulations,
                **sp_kwargs,
            )
            replay_buffer.add_batch(states_arr, policies_arr, values_arr)
            num_generated = len(states_arr)
        else:
            examples = run_self_play(
                model=model,
                num_games=args.games_per_iteration,
                num_simulations=args.training_simulations,
            )
            replay_buffer.add(examples)
            num_generated = len(examples)
        sp_time = time.time() - sp_start

        print(
            f"     Generated {num_generated} examples in {sp_time:.1f}s "
            f"(buffer: {len(replay_buffer)})"
        )

        # ── Training phase ──
        num_epochs = 4 if iteration < 20 else args.epochs
        effective_batches = args.batches_per_epoch or min(
            500, max(1, len(replay_buffer) // args.batch_size)
        )
        print(f"  🧠 Training: {num_epochs} epochs × {effective_batches} batches ...")
        model.to(device)

        train_start = time.time()
        total_losses = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        for _ in tqdm(range(num_epochs), desc="  Training", unit="epoch"):
            losses = trainer.train_epoch(
                replay_buffer, num_batches=effective_batches
            )
            for k in total_losses:
                total_losses[k] += losses[k]

        train_time = time.time() - train_start
        avg_losses = {k: v / num_epochs for k, v in total_losses.items()}

        print(
            f"     Loss: policy={avg_losses['policy_loss']:.4f}, "
            f"value={avg_losses['value_loss']:.4f}, "
            f"total={avg_losses['total_loss']:.4f} "
            f"({train_time:.1f}s)"
        )

        # Step LR scheduler
        trainer.step_scheduler()

        # ── Fast 0-sim Evaluation (every iteration) ──
        print("  📊 Evaluation (0-sim raw network) ...")
        best_move_0, center_pol_pct = evaluate_opening_move(
            model, device, num_simulations=0
        )
        status_0 = "✅" if best_move_0 == 3 else "❌"
        print(
            f"     {status_0} Opening Move (0-sim): played column {best_move_0 + 1} "
            f"(Center policy: {center_pol_pct:.1f}%)"
        )
        _print_dataset_summary(
            label="0-sim",
            datasets=datasets,
            model=model,
            device=device,
            num_simulations=0,
        )

        # ── Periodic MCTS Evaluation + Paired-Opening Arena + Anchor Elo ──
        run_periodic = (
            args.eval_interval > 0
            and (iter_num % args.eval_interval == 0 or iter_num == args.iterations)
        )
        if run_periodic and args.eval_simulations > 0:
            print(
                f"  🔍 MCTS Evaluation ({args.eval_simulations} sims/move) ..."
            )
            best_move_mcts, center_visit_pct = evaluate_opening_move(
                model, device, num_simulations=args.eval_simulations
            )
            status_mcts = "✅" if best_move_mcts == 3 else "❌"
            print(
                f"     {status_mcts} Opening Move ({args.eval_simulations}-sim): "
                f"played column {best_move_mcts + 1} "
                f"(Center visits: {center_visit_pct:.1f}%)"
            )
            _print_dataset_summary(
                label=f"{args.eval_simulations}-sim MCTS",
                datasets=datasets,
                model=model,
                device=device,
                num_simulations=args.eval_simulations,
            )

            # Anchor Elo vs fixed non-neural PureMCTSBaseline
            anchor_sims = min(200, args.eval_simulations)
            print(
                f"  ⚓ Anchor Arena: Current vs PureMCTSBaseline "
                f"({args.arena_games} paired games, {args.eval_simulations} vs {anchor_sims} sims, temp=0) ..."
            )
            anchor_res = play_anchor_arena(
                model=model,
                device=device,
                num_games=args.arena_games,
                num_simulations=args.eval_simulations,
                anchor_simulations=anchor_sims,
                paired_openings=True,
            )
            print(
                f"     Result: Current {anchor_res['model_wins']} - "
                f"{anchor_res['anchor_wins']} Anchor (Draws: {anchor_res['draws']}) | "
                f"Winrate: {anchor_res['win_rate_pct']:.1f}% | "
                f"Anchor Elo: {anchor_res['elo_diff']:+.0f}"
            )

            # Relative Checkpoint Arena (vs checkpoint from lookback iterations ago)
            arena_lookback = max(args.eval_interval, 10)
            if iter_num > arena_lookback:
                opponent_iter = iter_num - arena_lookback
                opponent_path = checkpoint_dir / f"checkpoint_{opponent_iter:04d}.pt"
                if opponent_path.exists():
                    opponent_model = AlphaZeroNet(args.res_blocks, args.filters)
                    if hasattr(opponent_model, "optimize_for_device"):
                        opponent_model.optimize_for_device(
                            device, use_compile=args.compile
                        )
                    else:
                        opponent_model.to(device)
                    try:
                        load_checkpoint(opponent_path, opponent_model, device=device)
                        opponent_model.eval()
                        print(
                            f"  ⚔️ Arena: Current vs Iter-{opponent_iter} "
                            f"({args.arena_games} paired games, {args.eval_simulations} sims, temp=0) ..."
                        )
                        m1_w_p1, m1_w_p2, m2_w_p1, m2_w_p2, draws = (
                            play_batched_arena(
                                model1=model,
                                model2=opponent_model,
                                device=device,
                                num_games=args.arena_games,
                                num_simulations=args.eval_simulations,
                                temperature=0.0,
                                paired_openings=True,
                            )
                        )
                        m1_wins = m1_w_p1 + m1_w_p2
                        m2_wins = m2_w_p1 + m2_w_p2
                        total_g = max(1, m1_wins + m2_wins + draws)
                        win_rate_frac = (m1_wins + 0.5 * draws) / total_g
                        rel_elo = compute_elo_diff(win_rate_frac)
                        print(
                            f"     Result: Current {m1_wins} - {m2_wins} Iter-{opponent_iter} "
                            f"(Draws: {draws})"
                        )
                        print(
                            f"     Breakdown (Current): {m1_w_p1} wins as P1, {m1_w_p2} wins as P2"
                        )
                        print(
                            f"     Breakdown (Iter-{opponent_iter}): "
                            f"{m2_w_p1} wins as P1, {m2_w_p2} wins as P2"
                        )
                        print(
                            f"     Current Winrate: {win_rate_frac * 100:.1f}% "
                            f"(ΔElo: {rel_elo:+.0f})"
                        )
                    except Exception as e:
                        print(
                            f"  ⚔️ Arena: skipped (failed to load iter-{opponent_iter}: {e})"
                        )
                else:
                    print(
                        f"  ⚔️ Arena: skipped (no checkpoint for iter-{opponent_iter})"
                    )

        # ── Save lightweight checkpoint + separate overwritten replay_buffer.pt ──
        ckpt_path = checkpoint_dir / f"checkpoint_{iter_num:04d}.pt"
        save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            iteration=iter_num,
            replay_buffer=None,
            path=ckpt_path,
            scheduler=trainer.scheduler,
        )
        _save_replay_buffer_compat(replay_buffer.get_state(), replay_buffer_path)

        iter_time = time.time() - iter_start
        print(
            f"  💾 Saved: {ckpt_path} + {replay_buffer_path.name} ({iter_time:.1f}s total)"
        )
        print()

    print("✅ Training complete!")
    print(
        f"   Latest checkpoint: {checkpoint_dir / f'checkpoint_{args.iterations:04d}.pt'}"
    )


if __name__ == "__main__":
    main()
