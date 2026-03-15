import torch
import numpy as np
from src.checkpoint import load_checkpoint
from src.game import ConnectFour
from src.model import AlphaZeroNet
from src.mcts import MCTS


def play(checkpoint_path, cpuct=1.0):
    game = ConnectFour()

    # Load the checkpoint and auto-detect architecture
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", allow_unsafe_fallback=True)
    num_resBlocks = checkpoint.get('num_resBlocks', 5)
    num_hidden = checkpoint.get('num_hidden', 128)

    nnet = AlphaZeroNet(game, num_resBlocks=num_resBlocks, num_hidden=num_hidden)
    nnet.load_state_dict(checkpoint['state_dict'])
    nnet.eval()
    print(f"Model: {num_resBlocks} res blocks, {num_hidden} hidden channels")

    # MCTS settings for the AI
    # Increase num_simulations to make the AI stronger (and slower)
    mcts = MCTS(game, nnet, num_simulations=100, c_puct=cpuct, dirichlet_epsilon=0)

    board = game.get_initial_state()
    cur_player = 1  # 1 for Human, -1 for AI (or vice-versa)

    print("\n--- Connect Four vs AlphaZero ---")
    print("Human: X, AI: O")

    while True:
        print_board(board)

        if cur_player == 1:
            # Human Turn
            valids = game.get_valid_moves(board)
            print(f"Valid moves: {[i for i, v in enumerate(valids) if v]}")
            action = -1
            while action not in [i for i, v in enumerate(valids) if v]:
                try:
                    action = int(input("Enter column (0-6): "))
                except ValueError:
                    continue
                except (KeyboardInterrupt, EOFError):
                    print("\nExiting game.")
                    import sys
                    sys.exit(0)
        else:
            # AI Turn
            print("AI is thinking...")
            canonical_board = game.get_canonical_form(board, cur_player)
            # Use temp=0 for the best move in competitive play
            probs = mcts.get_action_prob(canonical_board, temp=0)
            action = np.argmax(probs)
            print(f"AI chose column: {action}")

        board = game.get_next_state(board, cur_player, action)

        result = game.get_game_ended(board, cur_player, action)
        if result != 0:
            print_board(board)
            if result == 1:
                print(f"Player {'Human' if cur_player == 1 else 'AI'} wins!")
            else:
                print("It's a draw!")
            break

        cur_player = -cur_player


def print_board(board):
    print("\n 0 1 2 3 4 5 6")
    print("-" * 15)
    for r in range(board.shape[0]):
        row_str = "|"
        for c in range(board.shape[1]):
            val = board[r][c]
            if val == 1:
                row_str += "X "
            elif val == -1:
                row_str += "O "
            else:
                row_str += ". "
        print(row_str + "|")
    print("-" * 15)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Play Connect Four against a checkpoint file.')
    parser.add_argument('--cp', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--cpuct', type=float, default=1.0, help='PUCT exploration constant')

    args = parser.parse_args()

    play(args.cp, cpuct=args.cpuct)
