#!/usr/bin/env python3
"""Benchmark script to test datasets directly for AlphaZero Connect Four models."""

import argparse
from pathlib import Path

from src.model import AlphaZeroNet
from src.evaluate import evaluate_dataset, evaluate_opening_move
from src.utils import get_device, load_checkpoint, get_latest_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaZero Connect Four Benchmark")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file. If not provided, gets the latest checkpoint from checkpoints/ directory.",
    )
    parser.add_argument(
        "--eval-simulations",
        type=int,
        default=100,
        help="MCTS simulations per move for dataset and opening move evaluation (default: 100)",
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
        "--testdata-dir",
        type=str,
        default="testdata",
        help="Directory containing the Test_*.txt dataset files (default: testdata/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print("📊 AlphaZero Connect Four Benchmark")
    print(f"   Device: {device}")

    # Determine checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint("checkpoints")
        if checkpoint_path is None:
            print("❌ No checkpoint found in checkpoints/ directory.")
            return

    print(f"   Loading checkpoint: {checkpoint_path}")

    model = AlphaZeroNet(num_res_blocks=args.res_blocks, num_filters=args.filters).to(
        device
    )

    try:
        load_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    except Exception as e:
        print(f"   ❌ Failed to load checkpoint: {e}")
        return

    model.eval()

    print(f"\n  📊 Evaluation with {args.eval_simulations} simulations ...")

    # Evaluate Opening Move
    best_move, center_pct = evaluate_opening_move(
        model, device, num_simulations=args.eval_simulations
    )
    status = "✅" if best_move == 3 else "❌"
    print(
        f"     {status} Opening Move: played column {best_move + 1} (Center visits: {center_pct:.1f}%)"
    )

    # Evaluate Dataset
    dataset_results = {}
    testdata_dir = Path(args.testdata_dir)

    if testdata_dir.exists():
        for file_path in sorted(testdata_dir.glob("Test_*")):
            with open(file_path, "r") as f:
                lines = f.readlines()
            result = evaluate_dataset(
                model, device, lines, num_simulations=args.eval_simulations
            )
            dataset_results[file_path.name] = result
    else:
        print(f"     ⚠️ Testdata directory '{args.testdata_dir}' not found.")

    if dataset_results:
        print("     Dataset Evaluation:")
        for filename, result in dataset_results.items():
            print(
                f"       - {filename}: "
                f"acc={result['accuracy']:.1f}% "
                f"mae={result['mae']:.3f} "
                f"r={result['correlation']:.3f}"
            )


if __name__ == "__main__":
    main()
