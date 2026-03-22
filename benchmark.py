#!/usr/bin/env python3
"""Benchmark AlphaZero checkpoint against gamesolver.org test positions.

Test data format (from http://blog.gamesolver.org/solving-connect-four/02-test-protocol/):
  Each line: "<moves> <expected_score>"
  - moves: sequence of 1-indexed column digits played alternately by both players
  - score: positive = current player wins, negative = loses, 0 = draw
    Magnitude = 22 - (stones played by winner at game end)

Usage:
    uv run benchmark.py                                          # Run all tests
    uv run benchmark.py --test-files testdata/Test_L3_R1         # Specific file
    uv run benchmark.py --limit 100 --simulations 50             # Quick check
    uv run benchmark.py --checkpoint checkpoints/checkpoint_0050.pt
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.game import ConnectFour, COLS
from src.mcts import MCTS, MCTSNode
from src.model import AlphaZeroNet
from src.utils import get_device, get_latest_checkpoint


# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AlphaZero checkpoint against test positions"
    )
    parser.add_argument(
        "--test-dir", type=str, default="testdata",
        help="Directory containing test files (default: testdata/)",
    )
    parser.add_argument(
        "--test-files", nargs="+", type=str, default=None,
        help="Specific test file(s) to evaluate (overrides --test-dir)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max positions to evaluate per test file",
    )
    parser.add_argument(
        "--simulations", type=int, default=200,
        help="MCTS simulations per move (default: 200)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory to search for latest checkpoint (default: checkpoints/)",
    )
    parser.add_argument(
        "--res-blocks", type=int, default=10,
        help="Number of residual blocks in model (default: 10)",
    )
    parser.add_argument(
        "--filters", type=int, default=128,
        help="Number of convolutional filters (default: 128)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-position details",
    )
    return parser.parse_args()


def parse_position(moves_str: str) -> ConnectFour:
    """Replay a sequence of 1-indexed column moves to build a game state."""
    game = ConnectFour()
    for ch in moves_str:
        col = int(ch) - 1  # Convert 1-indexed to 0-indexed
        assert 0 <= col < COLS, f"Invalid column {ch} in move sequence"
        game.make_move(col)
    return game


def sign(x: int | float) -> int:
    """Return sign of a number: +1, 0, or -1."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def evaluate_position(
    mcts: MCTS, game: ConnectFour
) -> float:
    """Evaluate a position using MCTS and return the value estimate.

    Returns value from the current player's perspective in [-1, 1].
    """
    if game.is_terminal():
        result = game.get_result()
        return result if result is not None else 0.0

    root = MCTSNode(game)
    policy, root_value = mcts._evaluate(game)
    root.expand(policy)

    if not root.children:
        return root_value

    # Run simulations (no noise for evaluation)
    for _ in range(mcts.num_simulations):
        node = root
        while node.is_expanded and node.children:
            node = node.select_child(mcts.c_puct)
        result = node.game.get_result()
        if result is not None:
            node.backpropagate(result)
            continue
        p, v = mcts._evaluate(node.game)
        node.expand(p)
        node.backpropagate(v)

    # Value from the most-visited child (negated: child Q is opponent's pov)
    best_child = max(root.children.values(), key=lambda c: c.visit_count)
    return -best_child.q_value


def load_test_file(path: Path, limit: int | None = None) -> list[tuple[str, int]]:
    """Load test positions from a file.

    Returns list of (moves_string, expected_score) tuples.
    """
    positions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            moves_str, score_str = parts
            positions.append((moves_str, int(score_str)))
            if limit is not None and len(positions) >= limit:
                break
    return positions


# ── Multiprocessing worker functions ──

def _worker_init(
    model_path: str, num_res_blocks: int, num_filters: int, num_simulations: int
) -> None:
    """Initialize worker process with model and MCTS."""
    global _worker_mcts
    model = AlphaZeroNet(num_res_blocks=num_res_blocks, num_filters=num_filters)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    _worker_mcts = MCTS(
        model=model, num_simulations=num_simulations, device=torch.device("cpu")
    )


def _worker_evaluate(args: tuple[str, int]) -> tuple[str, int, float]:
    """Worker function to evaluate a single position.

    Args:
        args: (moves_str, expected_score)

    Returns:
        (moves_str, expected_score, predicted_value)
    """
    global _worker_mcts
    moves_str, expected_score = args
    game = parse_position(moves_str)
    value = evaluate_position(_worker_mcts, game)
    return (moves_str, expected_score, value)


def run_benchmark(
    mcts: MCTS | None,
    test_file: Path,
    limit: int | None = None,
    verbose: bool = False,
    num_workers: int = 1,
    model_path: str | None = None,
    num_res_blocks: int = 10,
    num_filters: int = 128,
    num_simulations: int = 200,
) -> dict:
    """Run benchmark on a single test file.

    Returns dict with accuracy stats.
    """
    positions = load_test_file(test_file, limit)
    if not positions:
        print(f"  {RED}No positions found in {test_file}{RESET}")
        return {"total": 0, "correct": 0, "correct_win": 0, "correct_draw": 0,
                "correct_loss": 0, "total_win": 0, "total_draw": 0, "total_loss": 0,
                "time": 0.0}

    total = len(positions)
    start_time = time.time()

    # Evaluate all positions (parallel or sequential)
    if num_workers > 1 and model_path is not None:
        chunk_size = max(1, total // (num_workers * 4))
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(model_path, num_res_blocks, num_filters, num_simulations),
        ) as pool:
            results = list(tqdm(
                pool.imap(  # preserve order for verbose output
                    _worker_evaluate, positions, chunksize=chunk_size,
                ),
                total=total,
                desc=f"    {test_file.name}",
                unit="pos",
                leave=False,
            ))
    else:
        # Sequential mode
        results = []
        for pos in tqdm(positions, desc=f"    {test_file.name}", unit="pos", leave=False):
            moves_str, expected_score = pos
            game = parse_position(moves_str)
            value = evaluate_position(mcts, game)
            results.append((moves_str, expected_score, value))

    elapsed = time.time() - start_time

    # Tally results
    correct = 0
    correct_win = correct_draw = correct_loss = 0
    total_win = total_draw = total_loss = 0

    for moves_str, expected_score, value in results:
        expected_sign = sign(expected_score)
        predicted_sign = sign(value)

        if expected_sign > 0:
            total_win += 1
        elif expected_sign < 0:
            total_loss += 1
        else:
            total_draw += 1

        is_correct = predicted_sign == expected_sign
        if is_correct:
            correct += 1
            if expected_sign > 0:
                correct_win += 1
            elif expected_sign < 0:
                correct_loss += 1
            else:
                correct_draw += 1

        if verbose:
            status = f"{GREEN}✓{RESET}" if is_correct else f"{RED}✗{RESET}"
            print(f"  {status} {moves_str:30s}  expected={expected_score:+3d}  "
                  f"value={value:+.3f}  ({predicted_sign:+d} vs {expected_sign:+d})")

    return {
        "total": total,
        "correct": correct,
        "correct_win": correct_win,
        "correct_draw": correct_draw,
        "correct_loss": correct_loss,
        "total_win": total_win,
        "total_draw": total_draw,
        "total_loss": total_loss,
        "time": elapsed,
    }


def print_results(name: str, stats: dict) -> None:
    """Print formatted results for a test file."""
    total = stats["total"]
    if total == 0:
        return

    correct = stats["correct"]
    accuracy = correct / total * 100
    rate = total / stats["time"] if stats["time"] > 0 else 0

    # Color based on accuracy
    if accuracy >= 70:
        color = GREEN
    elif accuracy >= 50:
        color = YELLOW
    else:
        color = RED

    print(f"\n  {BOLD}{name}{RESET}")
    print(f"    Accuracy: {color}{accuracy:6.1f}%{RESET}  ({correct}/{total})")

    # Per-category breakdown
    parts = []
    if stats["total_win"] > 0:
        win_acc = stats["correct_win"] / stats["total_win"] * 100
        parts.append(f"Win: {win_acc:.0f}% ({stats['correct_win']}/{stats['total_win']})")
    if stats["total_draw"] > 0:
        draw_acc = stats["correct_draw"] / stats["total_draw"] * 100
        parts.append(f"Draw: {draw_acc:.0f}% ({stats['correct_draw']}/{stats['total_draw']})")
    if stats["total_loss"] > 0:
        loss_acc = stats["correct_loss"] / stats["total_loss"] * 100
        parts.append(f"Loss: {loss_acc:.0f}% ({stats['correct_loss']}/{stats['total_loss']})")

    if parts:
        print(f"    {DIM}{' | '.join(parts)}{RESET}")

    print(f"    {DIM}Time: {stats['time']:.1f}s ({rate:.1f} pos/s){RESET}")


def main() -> None:
    args = parse_args()

    num_workers = args.workers or max(1, mp.cpu_count() - 1)

    print(f"\n{BOLD}{'═' * 50}{RESET}")
    print(f"{BOLD}  🎯 AlphaZero Connect Four Benchmark{RESET}")
    print(f"{BOLD}{'═' * 50}{RESET}")

    # --- Load model ---
    device = get_device()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = get_latest_checkpoint(args.checkpoint_dir)

    if ckpt_path is None:
        print(f"\n{RED}No checkpoint found!{RESET}")
        print(f"Specify one with --checkpoint or train first with: {CYAN}uv run main.py{RESET}")
        sys.exit(1)

    print(f"\n  {DIM}Checkpoint:   {ckpt_path}{RESET}")
    print(f"  {DIM}Device:       {device}{RESET}")
    print(f"  {DIM}Simulations:  {args.simulations}{RESET}")
    print(f"  {DIM}Workers:      {num_workers}{RESET}")

    model = AlphaZeroNet(
        num_res_blocks=args.res_blocks, num_filters=args.filters
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        iteration = checkpoint.get("iteration", "?")
        print(f"  {DIM}Iteration:    {iteration}{RESET}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Save model weights for worker processes
    worker_model_path = Path(args.checkpoint_dir) / "_benchmark_worker_model.pt"
    worker_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, worker_model_path)

    # Create MCTS for single-process fallback
    mcts = MCTS(model=model, num_simulations=args.simulations, device=device)

    # --- Discover test files ---
    if args.test_files:
        test_files = [Path(f) for f in args.test_files]
    else:
        test_dir = Path(args.test_dir)
        if not test_dir.exists():
            print(f"\n{RED}Test directory '{test_dir}' not found!{RESET}")
            sys.exit(1)
        test_files = sorted(
            f for f in test_dir.iterdir()
            if f.is_file() and f.name.startswith("Test_")
        )

    if not test_files:
        print(f"\n{RED}No test files found!{RESET}")
        sys.exit(1)

    if args.limit:
        print(f"  {DIM}Limit:        {args.limit} positions per file{RESET}")

    print(f"\n  Running {len(test_files)} test file(s)...")
    print(f"  {DIM}{'─' * 46}{RESET}")

    # --- Run benchmarks ---
    all_stats = {}
    for test_file in test_files:
        name = test_file.name
        stats = run_benchmark(
            mcts=mcts,
            test_file=test_file,
            limit=args.limit,
            verbose=args.verbose,
            num_workers=num_workers,
            model_path=str(worker_model_path),
            num_res_blocks=args.res_blocks,
            num_filters=args.filters,
            num_simulations=args.simulations,
        )
        all_stats[name] = stats
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {BOLD}{name}{RESET}: {acc:.1f}% ({stats['correct']}/{stats['total']})")

    # Cleanup temp model file
    if worker_model_path.exists():
        worker_model_path.unlink()

    # --- Print summary ---
    print(f"\n{BOLD}{'═' * 50}{RESET}")
    print(f"{BOLD}  Results{RESET}")
    print(f"{'═' * 50}")

    for name, stats in all_stats.items():
        print_results(name, stats)

    # Overall
    totals = {
        "total": sum(s["total"] for s in all_stats.values()),
        "correct": sum(s["correct"] for s in all_stats.values()),
        "correct_win": sum(s["correct_win"] for s in all_stats.values()),
        "correct_draw": sum(s["correct_draw"] for s in all_stats.values()),
        "correct_loss": sum(s["correct_loss"] for s in all_stats.values()),
        "total_win": sum(s["total_win"] for s in all_stats.values()),
        "total_draw": sum(s["total_draw"] for s in all_stats.values()),
        "total_loss": sum(s["total_loss"] for s in all_stats.values()),
        "time": sum(s["time"] for s in all_stats.values()),
    }

    if totals["total"] > 0:
        print(f"\n  {'─' * 46}")
        print_results("OVERALL", totals)

    print()


if __name__ == "__main__":
    main()
