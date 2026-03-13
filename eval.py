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
    game, model1_state, model2_state, mcts_sims, cpuct, num_resBlocks, num_hidden, model1_starts = args_tuple

    model1 = AlphaZeroNet(game, num_resBlocks=num_resBlocks, num_hidden=num_hidden)
    model1.load_state_dict(model1_state)
    model1.eval()

    model2 = AlphaZeroNet(game, num_resBlocks=num_resBlocks, num_hidden=num_hidden)
    model2.load_state_dict(model2_state)
    model2.eval()

    mcts1 = MCTS(game, model1, num_simulations=mcts_sims, c_puct=cpuct, dirichlet_epsilon=0)
    mcts2 = MCTS(game, model2, num_simulations=mcts_sims, c_puct=cpuct, dirichlet_epsilon=0)

    if model1_starts:
        res = play_game(game, mcts1, mcts2)
    else:
        res = play_game(game, mcts2, mcts1)
        if res == 1: res = -1
        elif res == -1: res = 1

    return res, model1_starts


def evaluate(cp1_path, cp2_path, num_games=200, mcts_sims=50, cpuct=1.0, num_resBlocks=5, num_hidden=128, num_workers=None):
    game = ConnectFour()
    if num_workers is None:
        num_workers = os.cpu_count()

    # Load Model 1
    model1 = AlphaZeroNet(game, num_resBlocks=num_resBlocks, num_hidden=num_hidden)
    print(f"Loading Model 1: {cp1_path}")
    checkpoint1 = torch.load(cp1_path, map_location=model1.device)
    model1.load_state_dict(checkpoint1['state_dict'])
    model1_state = {k: v.cpu() for k, v in model1.state_dict().items()}

    # Load Model 2
    model2 = AlphaZeroNet(game, num_resBlocks=num_resBlocks, num_hidden=num_hidden)
    print(f"Loading Model 2: {cp2_path}")
    checkpoint2 = torch.load(cp2_path, map_location=model2.device)
    model2.load_state_dict(checkpoint2['state_dict'])
    model2_state = {k: v.cpu() for k, v in model2.state_dict().items()}

    # Stats: [wins1, wins2, draws] indexed by who went first
    stats = {
        'model1_first': {'wins1': 0, 'wins2': 0, 'draws': 0},
        'model2_first': {'wins1': 0, 'wins2': 0, 'draws': 0},
    }

    # We want half games where Model 1 starts, half where Model 2 starts
    total_games = (num_games // 2) * 2
    half = total_games // 2

    worker_args = [
        (game, model1_state, model2_state, mcts_sims, cpuct, num_resBlocks, num_hidden, i < half)
        for i in range(total_games)
    ]

    with mp.Pool(processes=num_workers) as pool:
        pbar = tqdm(total=total_games, desc="Evaluation (Pool)")
        for res, m1_started in pool.imap_unordered(_evaluation_worker, worker_args):
            bucket = 'model1_first' if m1_started else 'model2_first'
            if res == 1:
                stats[bucket]['wins1'] += 1
            elif res == -1:
                stats[bucket]['wins2'] += 1
            else:
                stats[bucket]['draws'] += 1
            pbar.update(1)
        pbar.close()

    m1f = stats['model1_first']
    m2f = stats['model2_first']
    total_w1 = m1f['wins1'] + m2f['wins1']
    total_w2 = m1f['wins2'] + m2f['wins2']
    total_d  = m1f['draws'] + m2f['draws']

    m1_name = os.path.basename(cp1_path)
    m2_name = os.path.basename(cp2_path)

    print(f"\n{'=' * 60}")
    print(f"  Evaluation Results ({total_games} games)")
    print(f"{'=' * 60}")
    print(f"  Model 1: {m1_name}")
    print(f"  Model 2: {m2_name}")
    print(f"{'-' * 60}")
    print(f"  {'':>20s} {'M1 Wins':>10s} {'M2 Wins':>10s} {'Draws':>10s}")
    print(f"  {'M1 goes first':>20s} {m1f['wins1']:>10d} {m1f['wins2']:>10d} {m1f['draws']:>10d}")
    print(f"  {'M2 goes first':>20s} {m2f['wins1']:>10d} {m2f['wins2']:>10d} {m2f['draws']:>10d}")
    print(f"{'-' * 60}")
    print(f"  {'Overall':>20s} {total_w1:>10d} {total_w2:>10d} {total_d:>10d}")
    print(f"{'=' * 60}")
    print(f"  Model 1 win rate: {total_w1}/{total_games} ({total_w1/total_games:.2%})")
    print(f"  Model 2 win rate: {total_w2}/{total_games} ({total_w2/total_games:.2%})")
    print(f"  Draw rate:        {total_d}/{total_games} ({total_d/total_games:.2%})")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate two AlphaZero checkpoints.')
    parser.add_argument('--cp1', type=str, required=True, help='Path to first checkpoint')
    parser.add_argument('--cp2', type=str, required=True, help='Path to second checkpoint')
    parser.add_argument('--games', type=int, default=20, help='Number of games to play (should be even)')
    parser.add_argument('--sims', type=int, default=1000, help='MCTS simulations per move')
    parser.add_argument('--cpuct', type=float, default=1.0, help='PUCT exploration constant')
    parser.add_argument('--num-res-blocks', type=int, default=5, help='Number of residual blocks')
    parser.add_argument('--num-hidden', type=int, default=128, help='Number of hidden channels')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: cpu_count)')
    
    args = parser.parse_args()
    
    evaluate(args.cp1, args.cp2, num_games=args.games, mcts_sims=args.sims, cpuct=args.cpuct, num_resBlocks=args.num_res_blocks, num_hidden=args.num_hidden, num_workers=args.workers)
