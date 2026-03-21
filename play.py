#!/usr/bin/env python3
"""Play Connect Four against the AlphaZero AI.

Usage:
    uv run play.py                          # Human goes first
    uv run play.py --computer-first         # Computer goes first
    uv run play.py --hints                  # Show move value hints
    uv run play.py --simulations 1600       # Stronger AI (more thinking)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.game import ConnectFour, COLS
from src.mcts import MCTS
from src.model import AlphaZeroNet
from src.utils import get_device, get_latest_checkpoint


# ANSI color codes
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
RED_BG = "\033[41m"
BLUE = "\033[94m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Connect Four vs AlphaZero AI")
    parser.add_argument(
        "--computer-first", action="store_true",
        help="Let the computer make the first move",
    )
    parser.add_argument(
        "--hints", action="store_true",
        help="Show win probability for each column",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (default: latest in checkpoints/)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory to search for checkpoints (default: checkpoints/)",
    )
    parser.add_argument(
        "--simulations", type=int, default=800,
        help="MCTS simulations per move (default: 800, higher = stronger)",
    )
    parser.add_argument(
        "--res-blocks", type=int, default=10,
        help="Number of residual blocks in model (default: 10)",
    )
    parser.add_argument(
        "--filters", type=int, default=128,
        help="Number of convolutional filters (default: 128)",
    )
    return parser.parse_args()


def display_hints(mcts: MCTS, game: ConnectFour) -> None:
    """Display win probability hints above the board."""
    values = mcts.get_move_values(game)
    valid = game.get_valid_moves()

    # Column labels (matching board grid alignment)
    col_line = "  "
    for col in range(COLS):
        col_line += f"  {BOLD}{col + 1}{RESET} "

    # Win% values (aligned to same 4-char cells)
    hint_line = "  "
    for col in range(COLS):
        if valid[col] and col in values:
            pct = values[col]
            if pct >= 60:
                color = GREEN
            elif pct >= 40:
                color = YELLOW
            else:
                color = RED
            hint_line += f"{color}{pct:4.0f}{RESET}"
        elif not valid[col]:
            hint_line += f"{DIM}   ×{RESET}"
        else:
            hint_line += "   ?"

    print(f"\n{DIM}  Win%:{RESET}")
    print(col_line)
    print(hint_line)


def get_human_move(game: ConnectFour) -> int:
    """Prompt the human player for a move."""
    valid = game.get_valid_moves()
    valid_cols = [str(c + 1) for c in range(COLS) if valid[c]]

    while True:
        try:
            inp = input(f"\n{BOLD}Your move (column {', '.join(valid_cols)}): {RESET}").strip()
            if inp.lower() in ("q", "quit", "exit"):
                print("\nGoodbye! 👋")
                sys.exit(0)
            col = int(inp) - 1  # Convert to 0-indexed
            if 0 <= col < COLS and valid[col]:
                return col
            print(f"{RED}Invalid move. Choose from columns: {', '.join(valid_cols)}{RESET}")
        except ValueError:
            print(f"{RED}Please enter a number (1-{COLS}) or 'q' to quit.{RESET}")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            sys.exit(0)


def get_ai_move(mcts: MCTS, game: ConnectFour) -> int:
    """Get the AI's move using MCTS."""
    action_probs, visit_counts = mcts.get_action_probs(
        game, temperature=0, add_noise=False
    )
    action = int(np.argmax(action_probs))

    # Show thinking info
    total_visits = int(visit_counts.sum())
    top_visits = int(visit_counts[action])
    confidence = top_visits / total_visits * 100 if total_visits > 0 else 0
    print(f"  {DIM}AI plays column {action + 1} "
          f"({top_visits}/{total_visits} visits, {confidence:.0f}% confident){RESET}")

    return action


def print_board(game: ConnectFour) -> None:
    """Print the current board state."""
    print()
    print(game.display())
    print()


def main() -> None:
    args = parse_args()

    print(f"\n{BOLD}{'═' * 40}{RESET}")
    print(f"{BOLD}  🎮 Connect Four vs AlphaZero AI{RESET}")
    print(f"{BOLD}{'═' * 40}{RESET}")

    # Load model
    device = get_device()

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = get_latest_checkpoint(args.checkpoint_dir)

    if ckpt_path is None:
        print(f"\n{RED}No checkpoint found!{RESET}")
        print(f"Train a model first: {CYAN}uv run main.py{RESET}")
        sys.exit(1)

    print(f"\n  {DIM}Loading model from: {ckpt_path}{RESET}")
    print(f"  {DIM}Device: {device}{RESET}")
    print(f"  {DIM}MCTS simulations: {args.simulations}{RESET}")

    model = AlphaZeroNet(
        num_res_blocks=args.res_blocks, num_filters=args.filters
    ).to(device)

    # Load weights (support both full checkpoint and model-only saves)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        iteration = checkpoint.get("iteration", "?")
        print(f"  {DIM}Checkpoint iteration: {iteration}{RESET}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    mcts = MCTS(model=model, num_simulations=args.simulations, device=device)

    # Determine who goes first
    human_player = -1 if args.computer_first else 1
    ai_player = -human_player

    if human_player == 1:
        print(f"\n  You play {RED}Red ●{RESET} (first)")
        print(f"  AI plays {YELLOW}Yellow ●{RESET}")
    else:
        print(f"\n  AI plays {RED}Red ●{RESET} (first)")
        print(f"  You play {YELLOW}Yellow ●{RESET}")

    print(f"  {DIM}Enter column number (1-7) to play, 'q' to quit{RESET}")

    if args.hints:
        print(f"  {GREEN}Hints enabled:{RESET} showing win% for each column")

    # Game loop
    game = ConnectFour()

    while True:
        print_board(game)

        if game.is_terminal():
            result = game.get_result()
            # result is from current player's perspective
            # if result == -1, the last mover won
            if result == -1.0:
                # Last mover won — who was the last mover?
                last_mover = -game.current_player
                if last_mover == human_player:
                    print(f"  {GREEN}{BOLD}🎉 You win! Congratulations!{RESET}")
                else:
                    print(f"  {RED}{BOLD}💻 AI wins! Better luck next time.{RESET}")
            else:
                print(f"  {YELLOW}{BOLD}🤝 It's a draw!{RESET}")
            break

        is_human_turn = game.current_player == human_player

        if is_human_turn:
            if args.hints:
                display_hints(mcts, game)

            col = get_human_move(game)
            game.make_move(col)
        else:
            print(f"  {DIM}AI is thinking...{RESET}", end="", flush=True)
            col = get_ai_move(mcts, game)
            game.make_move(col)

    # Ask to play again
    print()
    try:
        again = input(f"{BOLD}Play again? (y/n): {RESET}").strip().lower()
        if again in ("y", "yes"):
            print()
            main()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye! 👋")


if __name__ == "__main__":
    main()
