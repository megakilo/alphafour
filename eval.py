import os
import torch
import numpy as np
from tqdm import tqdm
from src.game import ConnectFour
from src.model import AlphaZeroNet
from src.mcts import MCTS
import multiprocessing as mp

def play_game(game, mcts1, mcts2):
    """
    Plays a single game between two MCTS instances.
    mcts1: Player 1 (Starts first)
    mcts2: Player 2 (Starts second)
    Returns: 1 if player 1 wins, -1 if player 2 wins, 1e-4 if draw.
    """
    board = game.get_initial_state()
    cur_player = 1
    players = {1: mcts1, -1: mcts2}
    
    while True:
        canonical_board = game.get_canonical_form(board, cur_player)
        # Use temp=0 for purely competitive play (best move)
        probs = players[cur_player].get_action_prob(canonical_board, temp=0)
        action = np.argmax(probs)
        
        board = game.get_next_state(board, cur_player, action)
        
        result = game.get_game_ended(board, cur_player, action)
        if result != 0:
            return result if cur_player == 1 else -result
            
        cur_player = -cur_player

def _evaluation_worker(args_tuple):
    """
    Worker function for the process pool. Plays a single evaluation game.
    """
    game, model1_state, model2_state, mcts_sims, model1_starts = args_tuple

    model1 = AlphaZeroNet(game)
    model1.load_state_dict(model1_state)
    model1.eval()

    model2 = AlphaZeroNet(game)
    model2.load_state_dict(model2_state)
    model2.eval()

    mcts1 = MCTS(game, model1, num_simulations=mcts_sims)
    mcts2 = MCTS(game, model2, num_simulations=mcts_sims)

    if model1_starts:
        res = play_game(game, mcts1, mcts2)
    else:
        res = play_game(game, mcts2, mcts1)
        if res == 1: res = -1
        elif res == -1: res = 1

    return res


def evaluate(cp1_path, cp2_path, num_games=200, mcts_sims=50, num_workers=None):
    game = ConnectFour()
    if num_workers is None:
        num_workers = os.cpu_count()

    # Load Model 1
    model1 = AlphaZeroNet(game, num_resBlocks=5, num_hidden=128)
    print(f"Loading Model 1: {cp1_path}")
    checkpoint1 = torch.load(cp1_path, map_location=model1.device)
    model1.load_state_dict(checkpoint1['state_dict'])
    model1_state = {k: v.cpu() for k, v in model1.state_dict().items()}

    # Load Model 2
    model2 = AlphaZeroNet(game, num_resBlocks=5, num_hidden=128)
    print(f"Loading Model 2: {cp2_path}")
    checkpoint2 = torch.load(cp2_path, map_location=model2.device)
    model2.load_state_dict(checkpoint2['state_dict'])
    model2_state = {k: v.cpu() for k, v in model2.state_dict().items()}

    wins1 = 0
    wins2 = 0
    draws = 0

    # We want half games where Model 1 starts, half where Model 2 starts
    total_games = (num_games // 2) * 2
    half = total_games // 2

    worker_args = [
        (game, model1_state, model2_state, mcts_sims, i < half)
        for i in range(total_games)
    ]

    with mp.Pool(processes=num_workers) as pool:
        pbar = tqdm(total=total_games, desc="Evaluation (Pool)")
        for res in pool.imap_unordered(_evaluation_worker, worker_args):
            if res == 1:
                wins1 += 1
            elif res == -1:
                wins2 += 1
            else:
                draws += 1
            pbar.update(1)
        pbar.close()
    
    print(f"\n--- Evaluation Results ({total_games} games) ---")
    print(f"Model 1 ({os.path.basename(cp1_path)}): {wins1} wins ({wins1/total_games:.2%})")
    print(f"Model 2 ({os.path.basename(cp2_path)}): {wins2} wins ({wins2/total_games:.2%})")
    print(f"Draws: {draws} ({draws/total_games:.2%})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate two AlphaZero checkpoints.')
    parser.add_argument('--cp1', type=str, required=True, help='Path to first checkpoint')
    parser.add_argument('--cp2', type=str, required=True, help='Path to second checkpoint')
    parser.add_argument('--games', type=int, default=200, help='Number of games to play (default: 200)')
    parser.add_argument('--sims', type=int, default=50, help='MCTS simulations per move')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: cpu_count)')
    
    args = parser.parse_args()
    
    evaluate(args.cp1, args.cp2, num_games=args.games, mcts_sims=args.sims, num_workers=args.workers)
