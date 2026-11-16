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
import os
import time
from pathlib import Path
from typing import Any

# Partition Rayon CPU threads before importing `alphafour_engine` under `torchrun`
_env_ws = int(os.environ.get("WORLD_SIZE", "1"))
if _env_ws > 1:
    os.environ.setdefault(
        "RAYON_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // _env_ws))
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_ONLINE_SOFTMAX", "0")

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

import src.utils as utils_mod
from src.distributed import (
    barrier,
    broadcast_object,
    cleanup_distributed,
    configure_rayon_threads,
    find_free_port,
    gather_self_play_arrays,
    get_comm_device,
    init_distributed_process,
    seed_rank,
    shard_count,
)
from src.evaluate import (
    compute_elo_diff,
    evaluate_dataset_distributed,
    evaluate_opening_move,
    get_dataset_child_wdl,
    play_anchor_arena_distributed,
    play_batched_arena_distributed,
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
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of CUDA GPUs for DDP training (default: auto-detect all visible GPUs; 1 = single-GPU)",
    )
    parser.add_argument(
        "--self-play-backend",
        type=str,
        choices=["auto", "rust", "cython", "python"],
        default="auto",
        help="Self-play MCTS engine backend (default: auto)",
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
    is_main: bool = True,
) -> None:
    if not datasets:
        return

    if is_main:
        print(f"     Dataset Evaluation ({label}):")
    total_n = 0
    weighted_acc = 0.0
    weighted_mse = 0.0
    total_draws = 0
    weighted_draw_hits = 0.0
    total_opt_n = 0
    weighted_opt_hits = 0.0

    for filename, lines, child_wdl in datasets:
        res = evaluate_dataset_distributed(
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

        if is_main:
            print(
                f"       - {filename:<10}: "
                f"wdl_acc={res['wdl_accuracy']:5.1f}%  "
                f"draw_recall={_fmt_pct(draw_rec):>6}  "
                f"mse={res['value_mse']:.3f}  "
                f"opt_move={_fmt_pct(opt_pct):>6}"
            )

    if is_main and total_n > 0:
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


def _run_training(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    is_main = rank == 0
    rayon_threads = configure_rayon_threads(world_size)
    base_seed = int(broadcast_object(int(time.time()) if is_main else 0, src=0))
    seed_rank(base_seed, rank)

    if is_main:
        print("🎮 AlphaZero Connect Four Training")
        if world_size > 1:
            print(
                f"   Device: {device} ({world_size} GPUs via DDP, Rayon={rayon_threads} threads/rank)"
            )
        else:
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
        print(f"   Self-play backend: {args.self_play_backend}")
        print()

    checkpoint_dir = Path(args.checkpoint_dir)
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    replay_buffer_path = checkpoint_dir / "replay_buffer.pt"

    # Initialize model (with channels_last + dynamic torch.compile on CUDA when enabled)
    model = AlphaZeroNet(num_res_blocks=args.res_blocks, num_filters=args.filters)
    if hasattr(model, "optimize_for_device"):
        model.optimize_for_device(device, use_compile=args.compile)
    else:
        model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"   Parameters: {param_count:,}")

    # Initialize trainer (with graceful signature detection for use_compile & distributed)
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
    if "distributed" in trainer_params:
        trainer_kwargs["distributed"] = world_size > 1
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
        latest_str: str | None = None
        if is_main:
            latest = get_latest_checkpoint(checkpoint_dir)
            latest_str = str(latest) if latest is not None else None
        latest_str = broadcast_object(latest_str, src=0)

        if latest_str is not None:
            latest_path = Path(latest_str)
            if is_main:
                print(f"   Resuming from: {latest_path}")
            ckpt = load_checkpoint(latest_path, model, trainer.optimizer, device)
            start_iteration = int(ckpt["iteration"])

            rb_exists = bool(
                broadcast_object(
                    replay_buffer_path.exists() if is_main else False, src=0
                )
            )
            if rb_exists:
                buf_state = _load_replay_buffer_compat(replay_buffer_path)
                replay_buffer.load_state(buf_state)
            elif "replay_buffer" in ckpt and ckpt["replay_buffer"] is not None:
                replay_buffer.load_state(ckpt["replay_buffer"])

            trainer.restore_scheduler(
                start_iteration, ckpt.get("scheduler_state_dict")
            )
            if is_main:
                print(
                    f"   Resumed at iteration {start_iteration}, buffer size: {len(replay_buffer)}"
                )
    if is_main:
        print()

    # Preload evaluation datasets and cached solver ground-truth child W/D/L once
    datasets: list[tuple[str, list[str], np.ndarray | None]] = []
    testdata_dir = Path("testdata")

    def _load_testdata() -> list[tuple[str, list[str], np.ndarray | None]]:
        loaded: list[tuple[str, list[str], np.ndarray | None]] = []
        if testdata_dir.exists():
            for file_path in sorted(testdata_dir.glob("Test_*")):
                if file_path.is_file():
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    child_wdl = get_dataset_child_wdl(lines)
                    loaded.append((file_path.name, lines, child_wdl))
        return loaded

    if world_size > 1:
        if is_main:
            datasets = _load_testdata()
        barrier()
        if not is_main:
            datasets = _load_testdata()
    else:
        datasets = _load_testdata()

    sp_params = inspect.signature(run_self_play).parameters
    sp_supports_arrays = "return_arrays" in sp_params and hasattr(
        replay_buffer, "add_batch"
    )
    sp_supports_uint8 = "return_uint8" in sp_params
    sp_supports_backend = "backend" in sp_params

    # Main training loop
    for iteration in range(start_iteration, args.iterations):
        iter_start = time.time()
        iter_num = iteration + 1
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        if is_main:
            print(f"━━━ Iteration {iter_num}/{args.iterations} (lr={current_lr:.6f}) ━━━")

        # ── Self-play phase ──
        local_games = shard_count(args.games_per_iteration, rank, world_size)
        if is_main:
            shard_note = (
                f" (~{local_games}/GPU × {world_size})" if world_size > 1 else ""
            )
            print(
                f"  🎲 Self-play: {args.games_per_iteration} games{shard_note}, "
                f"{args.training_simulations} sims/move (backend={args.self_play_backend}) ..."
            )
        model.eval()

        sp_start = time.time()
        sp_backend_arg = getattr(args, "self_play_backend", "auto")
        if sp_supports_arrays:
            sp_kwargs: dict[str, Any] = {"return_arrays": True}
            if sp_supports_uint8:
                sp_kwargs["return_uint8"] = True
            if sp_supports_backend:
                sp_kwargs["backend"] = sp_backend_arg
            states_arr, policies_arr, values_arr = run_self_play(
                model=model,
                num_games=local_games,
                num_simulations=args.training_simulations,
                **sp_kwargs,
            )
            if world_size > 1:
                states_gathered, policies_gathered, values_gathered = (
                    gather_self_play_arrays(
                        states_arr, policies_arr, values_arr, device
                    )
                )
                replay_buffer.add_batch(
                    states_gathered, policies_gathered, values_gathered
                )
                num_generated = int(states_gathered.shape[0])
            else:
                replay_buffer.add_batch(states_arr, policies_arr, values_arr)
                num_generated = len(states_arr)
        else:
            legacy_sp_kwargs: dict[str, Any] = {}
            if sp_supports_backend:
                legacy_sp_kwargs["backend"] = sp_backend_arg
            examples = run_self_play(
                model=model,
                num_games=local_games,
                num_simulations=args.training_simulations,
                **legacy_sp_kwargs,
            )
            if world_size > 1:
                if examples:
                    ex_states = np.stack([ex[0] for ex in examples], axis=0)
                    ex_policies = np.stack([ex[1] for ex in examples], axis=0)
                    ex_values = np.asarray([ex[2] for ex in examples], dtype=np.float32)
                else:
                    ex_states = np.empty((0, 3, 6, 7), dtype=np.float32)
                    ex_policies = np.empty((0, 7), dtype=np.float32)
                    ex_values = np.empty((0,), dtype=np.float32)
                states_gathered, policies_gathered, values_gathered = (
                    gather_self_play_arrays(
                        ex_states, ex_policies, ex_values, device
                    )
                )
                replay_buffer.add_batch(
                    states_gathered, policies_gathered, values_gathered
                )
                num_generated = int(states_gathered.shape[0])
            else:
                replay_buffer.add(examples)
                num_generated = len(examples)
        sp_time = time.time() - sp_start

        if is_main:
            print(
                f"     Generated {num_generated} examples in {sp_time:.1f}s "
                f"(buffer: {len(replay_buffer)})"
            )

        # ── Training phase ──
        num_epochs = 4 if iteration < 20 else args.epochs
        if args.batches_per_epoch is not None:
            effective_batches = args.batches_per_epoch
        else:
            single_gpu_batches = min(500, max(1, len(replay_buffer) // args.batch_size))
            effective_batches = max(
                1, (single_gpu_batches + world_size - 1) // world_size
            )
        if is_main:
            ddp_note = (
                f"/GPU (global batch={args.batch_size * world_size})"
                if world_size > 1
                else ""
            )
            print(
                f"  🧠 Training: {num_epochs} epochs × {effective_batches} batches{ddp_note} ..."
            )
        model.to(device)

        train_start = time.time()
        total_losses = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        for _ in tqdm(
            range(num_epochs),
            desc="  Training",
            unit="epoch",
            disable=not is_main,
        ):
            losses = trainer.train_epoch(
                replay_buffer, num_batches=effective_batches
            )
            for k in total_losses:
                total_losses[k] += losses[k]

        train_time = time.time() - train_start
        avg_losses = {k: v / num_epochs for k, v in total_losses.items()}

        if is_main:
            print(
                f"     Loss: policy={avg_losses['policy_loss']:.4f}, "
                f"value={avg_losses['value_loss']:.4f}, "
                f"total={avg_losses['total_loss']:.4f} "
                f"({train_time:.1f}s)"
            )

        # Step LR scheduler
        trainer.step_scheduler()

        # ── Fast 0-sim Evaluation (every iteration) ──
        if is_main:
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
            is_main=is_main,
        )

        # ── Periodic MCTS Evaluation + Paired-Opening Arena + Anchor Elo ──
        run_periodic = (
            args.eval_interval > 0
            and (iter_num % args.eval_interval == 0 or iter_num == args.iterations)
        )
        if run_periodic and args.eval_simulations > 0:
            if is_main:
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
                is_main=is_main,
            )

            # Anchor Elo vs fixed non-neural PureMCTSBaseline
            anchor_sims = min(200, args.eval_simulations)
            if is_main:
                print(
                    f"  ⚓ Anchor Arena: Current vs PureMCTSBaseline "
                    f"({args.arena_games} paired games, {args.eval_simulations} vs {anchor_sims} sims, temp=0) ..."
                )
            anchor_res = play_anchor_arena_distributed(
                model=model,
                device=device,
                num_games=args.arena_games,
                num_simulations=args.eval_simulations,
                anchor_simulations=anchor_sims,
                paired_openings=True,
            )
            if is_main:
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
                opp_exists = bool(
                    broadcast_object(
                        opponent_path.exists() if is_main else False, src=0
                    )
                )
                if opp_exists:
                    opponent_model = AlphaZeroNet(args.res_blocks, args.filters)
                    if hasattr(opponent_model, "optimize_for_device"):
                        opponent_model.optimize_for_device(
                            device, use_compile=args.compile
                        )
                    else:
                        opponent_model.to(device)
                    load_err: str | None = None
                    try:
                        load_checkpoint(opponent_path, opponent_model, device=device)
                        opponent_model.eval()
                    except Exception as e:
                        load_err = str(e)

                    local_ok = 1 if load_err is None else 0
                    if world_size > 1:
                        comm_dev = get_comm_device(device)
                        ok_t = torch.tensor(
                            [local_ok], dtype=torch.int64, device=comm_dev
                        )
                        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
                        all_ok = int(ok_t.item()) == 1
                    else:
                        all_ok = local_ok == 1

                    if all_ok:
                        if is_main:
                            print(
                                f"  ⚔️ Arena: Current vs Iter-{opponent_iter} "
                                f"({args.arena_games} paired games, {args.eval_simulations} sims, temp=0) ..."
                            )
                        m1_w_p1, m1_w_p2, m2_w_p1, m2_w_p2, draws = (
                            play_batched_arena_distributed(
                                model1=model,
                                model2=opponent_model,
                                device=device,
                                num_games=args.arena_games,
                                num_simulations=args.eval_simulations,
                                temperature=0.0,
                                paired_openings=True,
                            )
                        )
                        if is_main:
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
                    elif is_main:
                        print(
                            f"  ⚔️ Arena: skipped (failed to load iter-{opponent_iter}: {load_err or 'worker error'})"
                        )
                elif is_main:
                    print(
                        f"  ⚔️ Arena: skipped (no checkpoint for iter-{opponent_iter})"
                    )

        # ── Save lightweight checkpoint + separate overwritten replay_buffer.pt ──
        ckpt_path = checkpoint_dir / f"checkpoint_{iter_num:04d}.pt"
        if is_main:
            save_checkpoint(
                model=model,
                optimizer=trainer.optimizer,
                iteration=iter_num,
                replay_buffer=None,
                path=ckpt_path,
                scheduler=trainer.scheduler,
            )
            _save_replay_buffer_compat(replay_buffer.get_state(), replay_buffer_path)
        barrier()

        iter_time = time.time() - iter_start
        if is_main:
            print(
                f"  💾 Saved: {ckpt_path} + {replay_buffer_path.name} ({iter_time:.1f}s total)"
            )
            print()

    if is_main:
        print("✅ Training complete!")
        print(
            f"   Latest checkpoint: {checkpoint_dir / f'checkpoint_{args.iterations:04d}.pt'}"
        )


def _ddp_spawn_worker(
    rank: int,
    world_size: int,
    master_port: int,
    args: argparse.Namespace,
) -> None:
    """Worker entrypoint for multi-GPU `torch.multiprocessing.spawn` DDP runs."""
    configure_rayon_threads(world_size)
    device = init_distributed_process(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        master_addr="127.0.0.1",
        master_port=master_port,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    )
    try:
        _run_training(args=args, rank=rank, world_size=world_size, device=device)
    finally:
        cleanup_distributed()


def main() -> None:
    args = parse_args()

    # Standard `torchrun --nproc_per_node=... main.py` launch
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        configure_rayon_threads(env_world_size)
        device = init_distributed_process(
            rank=rank,
            world_size=env_world_size,
            local_rank=local_rank,
        )
        try:
            _run_training(
                args=args, rank=rank, world_size=env_world_size, device=device
            )
        finally:
            cleanup_distributed()
        return

    # Auto-detect multi-GPU DDP via `mp.spawn`, or fall back to single device
    avail_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.gpus is not None and args.gpus > 1 and avail_gpus < args.gpus:
        raise ValueError(
            f"Requested --gpus {args.gpus}, but only {avail_gpus} CUDA GPU(s) visible."
        )

    target_gpus = avail_gpus if (args.gpus is None or args.gpus <= 0) else args.gpus
    if target_gpus >= 2:
        configure_rayon_threads(target_gpus)
        master_port = find_free_port()
        mp.spawn(
            _ddp_spawn_worker,
            args=(target_gpus, master_port, args),
            nprocs=target_gpus,
            join=True,
        )
        return

    device = torch.device("cuda:0") if (target_gpus == 1 and avail_gpus >= 1) else get_device()
    _run_training(args=args, rank=0, world_size=1, device=device)


if __name__ == "__main__":
    main()
