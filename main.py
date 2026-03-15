import torch
import os
import argparse
from src.checkpoint import load_checkpoint
from src.game import ConnectFour
from src.model import AlphaZeroNet
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='AlphaZero Connect Four Training')
    parser.set_defaults(resume=True)
    parser.add_argument('--resume', action='store_true', dest='resume', help='Resume training from the latest checkpoint')
    parser.add_argument('--no-resume', action='store_false', dest='resume', help='Start training from scratch')
    parser.add_argument('--iters', type=int, default=50, help='Number of iterations')
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
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for AdamW')
    parser.add_argument('--replay-buffer-iters', type=int, default=10, help='How many recent iterations to keep in replay')
    parser.add_argument('--train-samples', type=int, default=65536, help='Max replay examples to sample per iteration')
    parser.add_argument('--sim-batch-size', type=int, default=8, help='How many leaf evaluations to batch inside MCTS')
    parser.add_argument('--fpu-reduction', type=float, default=0.25, help='First-play urgency reduction for unvisited edges')
    parser.add_argument('--dirichlet-alpha', type=float, default=1.4, help='Dirichlet alpha for root exploration noise')
    parser.add_argument('--dirichlet-epsilon', type=float, default=0.25, help='Mix factor for root Dirichlet noise')
    parser.add_argument('--res-blocks', type=int, default=5, help='Number of residual blocks in the network')
    parser.add_argument('--hidden-channels', type=int, default=128, help='Hidden channel width in the network')
    parser.add_argument('--self-play-device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps', 'auto'], help='Device to use inside self-play workers')
    
    args_cli = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("MPS is available. Hardware acceleration enabled.")
    elif torch.cuda.is_available():
        print("CUDA is available. Hardware acceleration enabled.")
    else:
        print("Hardware acceleration not available. Using CPU.")

    os.makedirs(args_cli.checkpoint, exist_ok=True)

    latest_checkpoint = None
    start_iter = 1
    res_blocks = args_cli.res_blocks
    hidden_channels = args_cli.hidden_channels
    if args_cli.resume:
        checkpoints = [
            filename for filename in os.listdir(args_cli.checkpoint)
            if filename.startswith('checkpoint_') and filename.endswith('.pth.tar')
        ]
        if checkpoints:
            latest_checkpoint = sorted(
                checkpoints,
                key=lambda name: int(name.split('_')[1].split('.')[0]),
            )[-1]
            latest_cp_path = os.path.join(args_cli.checkpoint, latest_checkpoint)
            checkpoint_meta = load_checkpoint(
                latest_cp_path,
                map_location='cpu',
                allow_unsafe_fallback=True,
            )
            res_blocks = checkpoint_meta.get('num_resBlocks', res_blocks)
            hidden_channels = checkpoint_meta.get('num_hidden', hidden_channels)
            start_iter = checkpoint_meta.get(
                'iteration',
                int(latest_checkpoint.split('_')[1].split('.')[0]),
            ) + 1
            print(f"Resuming from checkpoint: {latest_checkpoint} (Next Iteration: {start_iter})")
            print(f"Checkpoint architecture: {res_blocks} res blocks, {hidden_channels} hidden channels")
        else:
            print(f"No checkpoints found in '{args_cli.checkpoint}' to resume from. Starting from scratch.")

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
        'weight_decay': args_cli.weight_decay,
        'replay_buffer_iters': args_cli.replay_buffer_iters,
        'train_samples': args_cli.train_samples,
        'sim_batch_size': args_cli.sim_batch_size,
        'fpu_reduction': args_cli.fpu_reduction,
        'dirichlet_alpha': args_cli.dirichlet_alpha,
        'dirichlet_epsilon': args_cli.dirichlet_epsilon,
        'self_play_device': args_cli.self_play_device,
    }

    print("Initializing Game...")
    g = ConnectFour()

    print("Initializing Model...")
    nnet = AlphaZeroNet(g, num_resBlocks=res_blocks, num_hidden=hidden_channels)
    print(f"Total Parameters: {nnet.get_num_parameters():,}")
    print(f"Architecture: {res_blocks} res blocks, {hidden_channels} hidden channels")

    print("Initializing Trainer...")
    trainer = Trainer(g, nnet, args)

    if latest_checkpoint is not None:
        trainer.load_checkpoint(folder=args['checkpoint'], filename=latest_checkpoint)

    print(f"Starting Training from Iteration {start_iter}...")
    trainer.learn(start_iter=start_iter)

if __name__ == "__main__":
    main()
