import os
import tempfile
from collections import deque

import multiprocessing as mp
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .checkpoint import (
    deserialize_replay_buffer,
    load_checkpoint,
    serialize_replay_buffer,
)
from .mcts import MCTS


_SELF_PLAY_CONTEXT = {}


def get_symmetries(game, board, pi):
    """
    Connect Four is symmetric across the vertical axis.
    """
    assert len(pi) == game.cols
    pi_rev = np.array(pi[::-1], copy=True)
    board_rev = np.fliplr(board).copy()
    return [(board, pi), (board_rev, pi_rev)]


def execute_self_play_episode(game, model, args):
    """
    Executes one episode of self-play and returns training examples.
    """
    train_examples = []
    board = game.get_initial_state()
    mcts = MCTS(
        game,
        model,
        num_simulations=args["num_mcts_sims"],
        c_puct=args["cpuct"],
        dirichlet_alpha=args.get("dirichlet_alpha", 1.4),
        dirichlet_epsilon=args.get("dirichlet_epsilon", 0.25),
        sim_batch_size=args.get("sim_batch_size", 1),
        fpu_reduction=args.get("fpu_reduction", 0.25),
    )
    cur_player = 1
    episode_step = 0
    last_action = None

    while True:
        episode_step += 1
        canonical_board = game.get_canonical_form(board, cur_player)
        temp = int(episode_step < args["temp_threshold"])

        pi = np.asarray(
            mcts.get_action_prob(canonical_board, temp=temp, last_action=last_action),
            dtype=np.float64,
        )
        pi_sum = pi.sum()
        if not np.isfinite(pi_sum) or pi_sum <= 0:
            valid_moves = game.get_valid_moves(canonical_board).astype(np.float64)
            pi = valid_moves / valid_moves.sum()
        else:
            pi /= pi_sum

        for sym_board, sym_pi in get_symmetries(game, canonical_board, pi):
            train_examples.append((sym_board, cur_player, sym_pi))

        action = np.random.choice(len(pi), p=pi)
        board = game.get_next_state(board, cur_player, action)
        last_action = action

        result = game.get_game_ended(board, cur_player, action)
        if result != 0:
            if abs(result - game.draw_value) < 1e-8:
                result = 0.0
            return [
                (example_board, example_pi, result * ((-1) ** (example_player != cur_player)))
                for example_board, example_player, example_pi in train_examples
            ]

        cur_player = -cur_player


def _init_self_play_worker(game, checkpoint_path, args, num_res_blocks, num_hidden):
    from .model import AlphaZeroNet

    device = args.get("self_play_device", "cpu")
    if device == "auto":
        device = None

    model = AlphaZeroNet(
        game,
        num_resBlocks=num_res_blocks,
        num_hidden=num_hidden,
        device=device,
    )
    checkpoint = load_checkpoint(checkpoint_path, map_location=model.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    global _SELF_PLAY_CONTEXT
    _SELF_PLAY_CONTEXT = {
        "game": game,
        "model": model,
        "args": args,
    }


def _self_play_worker(_):
    return execute_self_play_episode(
        _SELF_PLAY_CONTEXT["game"],
        _SELF_PLAY_CONTEXT["model"],
        _SELF_PLAY_CONTEXT["args"],
    )


class Trainer:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.replay_buffer = deque(maxlen=self.args.get("replay_buffer_iters", 10))
        self.optimizer = self._build_optimizer()
        milestones = self.args.get("lr_milestones", [])
        gamma = self.args.get("lr_gamma", 0.1)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=milestones,
            gamma=gamma,
        )

    def _build_optimizer(self):
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias") or "bn" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.args.get("weight_decay", 1e-4)},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.args["lr"],
        )

    def execute_episode(self):
        return execute_self_play_episode(self.game, self.model, self.args)

    def get_symmetries(self, board, pi):
        return get_symmetries(self.game, board, pi)

    def learn(self, start_iter=1):
        """
        Performs num_iters iterations with num_eps episodes of self-play in each
        iteration. After every iteration, it trains the network.
        """
        for iteration in range(start_iter, self.args["num_iters"] + 1):
            print(f"------ITER {iteration}------")
            iter_examples = self._run_self_play()

            self.replay_buffer.append(iter_examples)

            all_examples = [
                example
                for replay_examples in self.replay_buffer
                for example in replay_examples
            ]
            train_examples = self._sample_training_examples(all_examples)
            print(
                f"Training on {len(train_examples)} sampled examples "
                f"from {len(all_examples)} total across {len(self.replay_buffer)} iteration(s)"
            )
            self.train(train_examples)

            self.scheduler.step()
            print(f"LR: {self.scheduler.get_last_lr()[0]:.6f}")

            self.save_checkpoint(
                folder=self.args["checkpoint"],
                filename=f"checkpoint_{iteration}.pth.tar",
                iteration=iteration,
            )

    def _run_self_play(self):
        num_workers = max(1, self.args.get("num_workers", os.cpu_count()) or 1)
        episodes_to_run = self.args["num_eps"]
        iter_examples = []
        self.model.eval()

        if num_workers == 1:
            pbar = tqdm(range(episodes_to_run), desc="Self Play")
            for _ in pbar:
                iter_examples.extend(execute_self_play_episode(self.game, self.model, self.args))
            return iter_examples

        worker_checkpoint = self._write_worker_checkpoint()
        ctx = mp.get_context("spawn")

        try:
            with ctx.Pool(
                processes=num_workers,
                initializer=_init_self_play_worker,
                initargs=(
                    self.game,
                    worker_checkpoint,
                    self.args,
                    self.model.num_resBlocks,
                    self.model.num_hidden,
                ),
            ) as pool:
                pbar = tqdm(total=episodes_to_run, desc="Self Play (Pool)")
                for episode_data in pool.imap_unordered(_self_play_worker, range(episodes_to_run), chunksize=1):
                    iter_examples.extend(episode_data)
                    pbar.update(1)
                pbar.close()
        finally:
            if os.path.exists(worker_checkpoint):
                os.remove(worker_checkpoint)

        return iter_examples

    def _write_worker_checkpoint(self):
        os.makedirs(self.args["checkpoint"], exist_ok=True)
        fd, filepath = tempfile.mkstemp(
            prefix="self_play_model_",
            suffix=".pth.tar",
            dir=self.args["checkpoint"],
        )
        os.close(fd)

        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "num_resBlocks": self.model.num_resBlocks,
                "num_hidden": self.model.num_hidden,
            },
            filepath,
        )
        return filepath

    def _sample_training_examples(self, all_examples):
        if not all_examples:
            return []

        max_samples = self.args.get("train_samples")
        if max_samples is None or len(all_examples) <= max_samples:
            train_examples = list(all_examples)
            np.random.shuffle(train_examples)
            return train_examples

        indices = np.random.choice(len(all_examples), size=max_samples, replace=False)
        train_examples = [all_examples[index] for index in indices]
        np.random.shuffle(train_examples)
        return train_examples

    def train(self, examples):
        """
        Trains the neural network using the examples from self-play.
        """
        if not examples:
            print("No training examples available; skipping optimization step.")
            return

        from .model import AlphaZeroNet

        batch_size = self.args["batch_size"]
        all_boards = AlphaZeroNet.encode_boards([example[0] for example in examples])
        all_pis = np.asarray([example[1] for example in examples], dtype=np.float32)
        all_vs = np.asarray([example[2] for example in examples], dtype=np.float32)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(all_boards, dtype=torch.float32),
            torch.tensor(all_pis, dtype=torch.float32),
            torch.tensor(all_vs, dtype=torch.float32),
        )

        pin_memory = self.model.device.type == "cuda"
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
        )

        for epoch in range(self.args["epochs"]):
            print(f"EPOCH ::: {epoch + 1}")
            self.model.train()
            pbar = tqdm(loader, desc="Training")

            for batch_boards, batch_pis, batch_vs in pbar:
                boards = batch_boards.to(self.model.device, non_blocking=pin_memory)
                target_pis = batch_pis.to(self.model.device, non_blocking=pin_memory)
                target_vs = batch_vs.to(self.model.device, non_blocking=pin_memory).unsqueeze(1)

                out_pi, out_v = self.model(boards)

                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                pbar.set_postfix({"loss_pi": l_pi.item(), "loss_v": l_v.item()})

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * F.log_softmax(outputs, dim=1)) / targets.size(0)

    def loss_v(self, targets, outputs):
        return torch.mean((targets - outputs) ** 2)

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar", iteration=None):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "num_resBlocks": self.model.num_resBlocks,
                "num_hidden": self.model.num_hidden,
                "replay_buffer": serialize_replay_buffer(self.replay_buffer),
                "iteration": iteration,
            },
            filepath,
        )

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")

        checkpoint = load_checkpoint(
            filepath,
            map_location=self.model.device,
            allow_unsafe_fallback=True,
        )
        self.model.load_state_dict(checkpoint["state_dict"])

        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except ValueError as exc:
                print(f"Skipping optimizer state restore: {exc}")

        scheduler_state = checkpoint.get("scheduler")
        if scheduler_state is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state)
            except ValueError as exc:
                print(f"Skipping scheduler state restore: {exc}")

        replay_examples = checkpoint.get("replay_buffer")
        if replay_examples is not None:
            self.replay_buffer = deque(
                deserialize_replay_buffer(replay_examples),
                maxlen=self.args.get("replay_buffer_iters", 10),
            )
