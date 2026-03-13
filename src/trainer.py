import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from .mcts import MCTS
import multiprocessing as mp


def _self_play_worker(args_tuple):
    """
    Worker function for the process pool. Each worker process calls this
    repeatedly, reusing the same process for multiple episodes.
    """
    game, model_state_dict, args = args_tuple
    from .model import AlphaZeroNet

    model = AlphaZeroNet(game)
    model.load_state_dict(model_state_dict)
    model.eval()

    trainer = Trainer(game, model, args)
    return trainer.execute_episode()


class Trainer:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'])
        # LR schedule: drop LR by gamma at specified iteration milestones
        milestones = self.args.get('lr_milestones', [])
        gamma = self.args.get('lr_gamma', 0.1)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

    def execute_episode(self):
        """
        Executes one episode of self-play, starting with player 1.
        Returns a list of examples (canonicalBoard, pi, v) for the training data.
        """
        train_examples = []
        board = self.game.get_initial_state()
        # Create a fresh MCTS for each episode
        mcts = MCTS(self.game, self.model, self.args['num_mcts_sims'], self.args['cpuct'])
        cur_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, cur_player)
            temp = int(episode_step < self.args['temp_threshold'])

            pi = mcts.get_action_prob(canonical_board, temp=temp)
            sym = self.get_symmetries(canonical_board, pi)
            for b, p in sym:
                train_examples.append([b, cur_player, p])

            action = np.random.choice(len(pi), p=pi)
            board = self.game.get_next_state(board, cur_player, action)
            
            r = self.game.get_game_ended(board, cur_player, action)
            
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples]

            cur_player = -cur_player

    def get_symmetries(self, board, pi):
        """
        Connect4 board is symmetric horizontally.
        """
        assert(len(pi) == self.game.cols)
        l = [(board, pi)]
        pi_rev = pi[::-1]
        board_rev = np.fliplr(board)
        l.append((board_rev, pi_rev))
        return l

    def learn(self, start_iter=1):
        """
        Performs num_iters iterations with num_eps episodes of self-play in each
        iteration. After every iteration, it trains the network.
        """
        num_workers = self.args.get('num_workers', os.cpu_count())
        
        for i in range(start_iter, self.args['num_iters'] + 1):
            print(f'------ITER {i}------')
            train_examples = []
            
            self.model.eval()
            model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
            
            episodes_to_run = self.args['num_eps']
            worker_args = [(self.game, model_state_dict, self.args)] * episodes_to_run

            with mp.Pool(processes=num_workers) as pool:
                pbar = tqdm(total=episodes_to_run, desc="Self Play (Pool)")
                for episode_data in pool.imap_unordered(_self_play_worker, worker_args):
                    train_examples.extend(episode_data)
                    pbar.update(1)
                pbar.close()
            
            # Standard training phase
            np.random.shuffle(train_examples)
            self.train(train_examples)
            
            # Step LR scheduler (per iteration)
            self.scheduler.step()
            print(f'LR: {self.scheduler.get_last_lr()[0]:.6f}')
            
            # Save checkpoint
            self.save_checkpoint(folder=self.args['checkpoint'], filename=f'checkpoint_{i}.pth.tar')

    def train(self, examples):
        """
        Trains the neural network using the examples from self-play.
        """
        batch_size = self.args['batch_size']
        
        for epoch in range(self.args['epochs']):
            print(f'EPOCH ::: {epoch+1}')
            self.model.train()
            
            # Shuffle indices for proper epoch (each example seen exactly once)
            perm = np.random.permutation(len(examples))
            batch_count = int(len(examples) / batch_size)
            pbar = tqdm(range(batch_count), desc="Training")
            
            for batch_idx in pbar:
                sample_ids = perm[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                
                boards = torch.tensor(np.array(boards), dtype=torch.float32, device=self.model.device).unsqueeze(1)
                target_pis = torch.tensor(np.array(pis), dtype=torch.float32, device=self.model.device)
                target_vs = torch.tensor(np.array(vs), dtype=torch.float32, device=self.model.device).unsqueeze(1)

                # predict
                out_pi, out_v = self.model(boards)
                
                # loss
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # backprop
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                pbar.set_postfix({'loss_pi': l_pi.item(), 'loss_v': l_v.item()})

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * F.log_softmax(outputs, dim=1)) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")
        checkpoint = torch.load(filepath, map_location=self.model.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
