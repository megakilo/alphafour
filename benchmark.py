#!/usr/bin/env python3
"""Benchmark script to evaluate AlphaZero Connect Four checkpoints on gamesolver.org datasets."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.evaluate import (
    compute_elo_diff,
    evaluate_dataset,
    evaluate_opening_move,
    get_dataset_child_wdl,
    play_anchor_arena,
    play_batched_arena,
)
from src.model import AlphaZeroNet
from src.utils import get_device, get_latest_checkpoint, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaZero Connect Four Benchmark")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: latest in --checkpoint-dir)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to search when --checkpoint is omitted (default: checkpoints/)",
    )
    parser.add_argument(
        "--eval-simulations",
        type=int,
        default=400,
        help="MCTS simulations per move for search evaluation (default: 400)",
    )
    parser.add_argument(
        "--both",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate both 0-sim raw network intuition and --eval-simulations MCTS (default: --both)",
    )
    parser.add_argument(
        "--testdata-dir",
        type=str,
        default="testdata",
        help="Directory containing Test_L*_R* dataset files (default: testdata/)",
    )
    parser.add_argument(
        "--test-files",
        nargs="*",
        default=None,
        help="Optional explicit test file path(s) to evaluate instead of all Test_* in --testdata-dir",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of positions per dataset file",
    )
    parser.add_argument(
        "--arena-anchor",
        action="store_true",
        help="Run duplicate paired-opening Arena against PureMCTSBaseline anchor and report Elo",
    )
    parser.add_argument(
        "--arena-vs",
        type=str,
        default=None,
        help="Optional opponent checkpoint path for paired-opening head-to-head Arena",
    )
    parser.add_argument(
        "--arena-games",
        type=int,
        default=40,
        help="Number of paired-opening games for Arena evaluation (default: 40)",
    )
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
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.compile + channels_last on CUDA (default: --compile)",
    )
    return parser.parse_args()


def _fmt_pct(val: float | None, width: int = 7) -> str:
    if val is None or math.isnan(val):
        return f"{'N/A':>{width}}"
    return f"{val:>{width - 1}.1f}%"


def _load_model_weights(
    ckpt_path: str | Path, model: AlphaZeroNet, device: torch.device
) -> int | str:
    """Load either a full training checkpoint or a raw state_dict."""
    try:
        ckpt = load_checkpoint(ckpt_path, model, optimizer=None, device=device)
        return ckpt.get("iteration", "?")
    except KeyError:
        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(raw)
        return "?"


def _run_suite_eval(
    model: AlphaZeroNet,
    device: torch.device,
    datasets: list[tuple[str, list[str], np.ndarray | None]],
    num_simulations: int,
) -> None:
    mode_title = (
        "0-sim Raw Network Intuition"
        if num_simulations == 0
        else f"{num_simulations}-sim MCTS Search"
    )
    center_label = "Center policy" if num_simulations == 0 else "Center visits"

    t0 = time.perf_counter()
    print(f"\n  📊 {mode_title}")

    best_move, center_pct = evaluate_opening_move(
        model, device, num_simulations=num_simulations
    )
    status = "✅" if best_move == 3 else "❌"
    print(
        f"     {status} Opening Move: column {best_move + 1} "
        f"({center_label}: {center_pct:.1f}%)"
    )

    if not datasets:
        return

    header = (
        f"     {'Dataset':<12} | {'N':>5} | {'WDL Acc%':>8} | "
        f"{'Win Rec%':>8} | {'Draw Rec%':>9} | {'Loss Rec%':>9} | "
        f"{'Opt Move%':>9} | {'Blunder%':>8} | {'Value MSE':>9}"
    )
    sep = "     " + "-" * (len(header) - 5)
    print()
    print(header)
    print(sep)

    total_n = 0
    total_win = 0
    total_draw = 0
    total_loss = 0
    total_opt_n = 0
    total_non_lost = 0

    correct_wdl = 0.0
    correct_win = 0.0
    correct_draw = 0.0
    correct_loss = 0.0
    sum_sq_err = 0.0
    optimal_hits = 0.0
    blunder_hits = 0.0

    for name, lines, child_wdl in datasets:
        res = evaluate_dataset(
            model,
            device,
            lines,
            num_simulations=num_simulations,
            child_wdl=child_wdl,
        )

        scores = [
            int(ln.strip().split()[1])
            for ln in lines
            if len(ln.strip().split()) == 2
        ]
        n_pos = len(scores)
        n_win = sum(1 for s in scores if s > 0)
        n_draw = sum(1 for s in scores if s == 0)
        n_loss = sum(1 for s in scores if s < 0)

        total_n += n_pos
        correct_wdl += (res["wdl_accuracy"] / 100.0) * n_pos
        sum_sq_err += res["value_mse"] * n_pos

        if n_win > 0 and not math.isnan(res["win_recall"]):
            total_win += n_win
            correct_win += (res["win_recall"] / 100.0) * n_win
        if n_draw > 0 and not math.isnan(res["draw_recall"]):
            total_draw += n_draw
            correct_draw += (res["draw_recall"] / 100.0) * n_draw
        if n_loss > 0 and not math.isnan(res["loss_recall"]):
            total_loss += n_loss
            correct_loss += (res["loss_recall"] / 100.0) * n_loss

        opt_pct = res["optimal_move_pct"]
        blunder_pct = res["blunder_pct"]
        if opt_pct is not None and not math.isnan(opt_pct):
            total_opt_n += n_pos
            optimal_hits += (opt_pct / 100.0) * n_pos
        if (
            child_wdl is not None
            and blunder_pct is not None
            and not math.isnan(blunder_pct)
        ):
            n_non_lost = int(np.sum(child_wdl.max(axis=1) >= 0))
            total_non_lost += n_non_lost
            blunder_hits += (blunder_pct / 100.0) * n_non_lost

        print(
            f"     {name:<12} | {n_pos:>5} | "
            f"{_fmt_pct(res['wdl_accuracy'], 8)} | "
            f"{_fmt_pct(res['win_recall'], 8)} | "
            f"{_fmt_pct(res['draw_recall'], 9)} | "
            f"{_fmt_pct(res['loss_recall'], 9)} | "
            f"{_fmt_pct(opt_pct, 9)} | "
            f"{_fmt_pct(blunder_pct, 8)} | "
            f"{res['value_mse']:>9.4f}"
        )

    if total_n > 0:
        ov_wdl = (correct_wdl / total_n) * 100.0
        ov_win = (correct_win / total_win * 100.0) if total_win > 0 else float("nan")
        ov_draw = (
            (correct_draw / total_draw * 100.0) if total_draw > 0 else float("nan")
        )
        ov_loss = (
            (correct_loss / total_loss * 100.0) if total_loss > 0 else float("nan")
        )
        ov_opt = (optimal_hits / total_opt_n * 100.0) if total_opt_n > 0 else None
        ov_blunder = (
            (blunder_hits / total_non_lost * 100.0) if total_non_lost > 0 else None
        )
        ov_mse = sum_sq_err / total_n

        print(sep)
        print(
            f"     {'OVERALL':<12} | {total_n:>5} | "
            f"{_fmt_pct(ov_wdl, 8)} | "
            f"{_fmt_pct(ov_win, 8)} | "
            f"{_fmt_pct(ov_draw, 9)} | "
            f"{_fmt_pct(ov_loss, 9)} | "
            f"{_fmt_pct(ov_opt, 9)} | "
            f"{_fmt_pct(ov_blunder, 8)} | "
            f"{ov_mse:>9.4f}"
        )

    elapsed = time.perf_counter() - t0
    print(f"\n     Completed in {elapsed:.2f}s")


def main() -> None:
    args = parse_args()
    device = get_device()
    print("📊 AlphaZero Connect Four Benchmark")
    print(f"   Device: {device}")

    checkpoint_path: str | Path | None = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path is None:
            print(f"❌ No checkpoint found in {args.checkpoint_dir}/ directory.")
            return

    print(f"   Loading checkpoint: {checkpoint_path}")
    model = AlphaZeroNet(num_res_blocks=args.res_blocks, num_filters=args.filters)
    if hasattr(model, "optimize_for_device"):
        model.optimize_for_device(device, use_compile=args.compile)
    else:
        model.to(device)

    try:
        iteration = _load_model_weights(checkpoint_path, model, device=device)
        print(f"   Checkpoint iteration: {iteration}")
    except Exception as e:
        print(f"   ❌ Failed to load checkpoint: {e}")
        return

    model.eval()

    # Load test datasets and solver ground-truth cache
    file_paths: list[Path] = []
    if args.test_files:
        file_paths = [Path(p) for p in args.test_files]
    else:
        testdata_dir = Path(args.testdata_dir)
        if testdata_dir.exists():
            file_paths = [
                p for p in sorted(testdata_dir.glob("Test_*")) if p.is_file()
            ]
        else:
            print(f"   ⚠️ Testdata directory '{args.testdata_dir}' not found.")

    datasets: list[tuple[str, list[str], np.ndarray | None]] = []
    for fp in file_paths:
        if not fp.exists():
            print(f"   ⚠️ Skipping missing test file: {fp}")
            continue
        with open(fp, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if args.limit is not None and args.limit > 0:
            lines = lines[: args.limit]
        child_wdl = get_dataset_child_wdl(lines)
        datasets.append((fp.name, lines, child_wdl))

    # Determine simulation modes to evaluate
    sim_modes: list[int] = []
    if args.both and args.eval_simulations > 0:
        sim_modes = [0, args.eval_simulations]
    else:
        sim_modes = [max(0, args.eval_simulations)]

    for sims in sim_modes:
        _run_suite_eval(model, device, datasets, num_simulations=sims)

    # Optional Anchor Elo check
    arena_sims = max(1, args.eval_simulations)
    if args.arena_anchor:
        anchor_sims = min(200, arena_sims)
        print(
            f"\n  ⚓ Anchor Arena vs PureMCTSBaseline "
            f"({args.arena_games} paired games, {arena_sims} vs {anchor_sims} sims, temp=0) ..."
        )
        t_anc = time.perf_counter()
        anc = play_anchor_arena(
            model=model,
            device=device,
            num_games=args.arena_games,
            num_simulations=arena_sims,
            anchor_simulations=anchor_sims,
            paired_openings=True,
        )
        dt_anc = time.perf_counter() - t_anc
        print(
            f"     Model {anc['model_wins']} - {anc['anchor_wins']} Anchor "
            f"(Draws: {anc['draws']}) | "
            f"P1 wins: {anc['model_wins_p1']}, P2 wins: {anc['model_wins_p2']}"
        )
        print(
            f"     Winrate: {anc['win_rate_pct']:.1f}% | "
            f"Anchor Elo: {anc['elo_diff']:+.1f} ({dt_anc:.2f}s)"
        )

    # Optional head-to-head checkpoint Arena
    if args.arena_vs:
        opp_path = Path(args.arena_vs)
        if not opp_path.exists():
            print(f"\n  ⚠️ Opponent checkpoint not found: {opp_path}")
        else:
            opp_model = AlphaZeroNet(
                num_res_blocks=args.res_blocks, num_filters=args.filters
            )
            if hasattr(opp_model, "optimize_for_device"):
                opp_model.optimize_for_device(device, use_compile=args.compile)
            else:
                opp_model.to(device)
            try:
                _load_model_weights(opp_path, opp_model, device=device)
                opp_model.eval()
                print(
                    f"\n  ⚔️ Head-to-Head Arena vs {opp_path.name} "
                    f"({args.arena_games} paired games, {arena_sims} sims, temp=0) ..."
                )
                t_vs = time.perf_counter()
                m1_p1, m1_p2, m2_p1, m2_p2, draws = play_batched_arena(
                    model1=model,
                    model2=opp_model,
                    device=device,
                    num_games=args.arena_games,
                    num_simulations=arena_sims,
                    temperature=0.0,
                    paired_openings=True,
                )
                dt_vs = time.perf_counter() - t_vs
                m1_w = m1_p1 + m1_p2
                m2_w = m2_p1 + m2_p2
                tot = max(1, m1_w + m2_w + draws)
                wr = (m1_w + 0.5 * draws) / tot
                elo = compute_elo_diff(wr)
                print(
                    f"     Model {m1_w} - {m2_w} Opponent (Draws: {draws}) | "
                    f"P1 wins: {m1_p1}, P2 wins: {m1_p2}"
                )
                print(
                    f"     Winrate: {wr * 100.0:.1f}% | ΔElo: {elo:+.1f} ({dt_vs:.2f}s)"
                )
            except Exception as e:
                print(f"\n  ❌ Failed head-to-head Arena vs {opp_path}: {e}")


if __name__ == "__main__":
    main()
