import argparse
import os

import torch

from src.checkpoint import build_standalone_weights, load_checkpoint


def default_output_path(checkpoint_path):
    if checkpoint_path.endswith(".pth.tar"):
        return checkpoint_path[:-8] + "_weights.pth"

    root, _ = os.path.splitext(checkpoint_path)
    return root + "_weights.pth"


def main():
    parser = argparse.ArgumentParser(
        description="Extract standalone model weights from a training checkpoint."
    )
    parser.add_argument(
        "--cp",
        required=True,
        help="Path to the training checkpoint to read",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the standalone weights file "
        "(default: <checkpoint>_weights.pth)",
    )
    args = parser.parse_args()

    checkpoint = load_checkpoint(
        args.cp,
        map_location="cpu",
        allow_unsafe_fallback=True,
    )
    weights = build_standalone_weights(checkpoint)

    output_path = args.output or default_output_path(args.cp)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(weights, output_path)

    print(f"Wrote standalone weights to: {output_path}")
    print(
        "Architecture: "
        f"{weights.get('num_resBlocks', 5)} res blocks, "
        f"{weights.get('num_hidden', 128)} hidden channels"
    )
    if "iteration" in weights:
        print(f"Source iteration: {weights['iteration']}")
    print("This file can be used with eval.py and play.py.")


if __name__ == "__main__":
    main()
