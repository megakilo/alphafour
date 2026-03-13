import torch
import os
import argparse
from src.game import ConnectFour
from src.model import AlphaZeroNet
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='AlphaZero Connect Four Training')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume training from the latest checkpoint')
    parser.add_argument('--iters', type=int, default=20, help='Number of iterations')
    parser.add_argument('--eps', type=int, default=300, help='Episodes per iteration')
    parser.add_argument('--sims', type=int, default=400, help='MCTS simulations per move')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of parallel workers for self-play')
    parser.add_argument('--cpuct', type=float, default=1.0, help='PUCT exploration constant')
    parser.add_argument('--checkpoint', type=str, default='./temp/', help='Directory to save checkpoints')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--temp-threshold', type=int, default=10, help='Temperature threshold for MCTS')
    parser.add_argument('--lr-milestones', type=int, nargs='*', default=[], help='Iterations at which to drop LR (e.g. --lr-milestones 10 15)')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='LR decay factor at each milestone')
    
    args_cli = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("MPS is available. Hardware acceleration enabled.")
    elif torch.cuda.is_available():
        print("CUDA is available. Hardware acceleration enabled.")
    else:
        print("Hardware acceleration not available. Using CPU.")

    args = {
        'num_iters': args_cli.iters,
        'num_eps': args_cli.eps,            
        'temp_threshold': args_cli.temp_threshold,    
        'num_mcts_sims': args_cli.sims,     
        'cpuct': args_cli.cpuct,
        'checkpoint': args_cli.checkpoint,
        'lr': args_cli.lr,
        'epochs': args_cli.epochs,
        'batch_size': args_cli.batch_size,
        'num_workers': args_cli.workers,
        'lr_milestones': args_cli.lr_milestones,
        'lr_gamma': args_cli.lr_gamma,
    }

    print("Initializing Game...")
    g = ConnectFour()

    print("Initializing Model...")
    # Using the 5-block, 128-filter ResNet as requested previously
    nnet = AlphaZeroNet(g, num_resBlocks=5, num_hidden=128)
    print(f"Total Parameters: {nnet.get_num_parameters():,}")

    print("Initializing Trainer...")
    trainer = Trainer(g, nnet, args)

    start_iter = 1
    if args_cli.resume:
        if not os.path.exists(args['checkpoint']):
            os.makedirs(args['checkpoint'])
            
        checkpoints = [f for f in os.listdir(args['checkpoint']) if f.startswith('checkpoint_') and f.endswith('.pth.tar')]
        if checkpoints:
            # Sort checkpoints by iteration number
            latest_cp = sorted(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
            start_iter = int(latest_cp.split('_')[1].split('.')[0]) + 1
            print(f"Resuming from checkpoint: {latest_cp} (Next Iteration: {start_iter})")
            trainer.load_checkpoint(folder=args['checkpoint'], filename=latest_cp)
        else:
            print("No checkpoints found in 'temp/' to resume from. Starting from scratch.")

    print(f"Starting Training from Iteration {start_iter}...")
    trainer.learn(start_iter=start_iter)

if __name__ == "__main__":
    main()
