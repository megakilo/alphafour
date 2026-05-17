"""Training loop with replay buffer for AlphaZero."""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .model import AlphaZeroNet
from .game import ROWS


class ReplayBuffer:
    """Fixed-size replay buffer with recency-weighted sampling.

    More recently added examples are sampled with higher probability,
    preventing the model from overfitting to stale data as the self-play
    distribution evolves during training.
    """

    def __init__(self, capacity: int = 200_000) -> None:
        self.buffer: deque[tuple[np.ndarray, np.ndarray, float]] = deque(
            maxlen=capacity
        )

    def add(self, examples: list[tuple[np.ndarray, np.ndarray, float]]) -> None:
        """Add a batch of examples."""
        self.buffer.extend(examples)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch with recency-weighted probabilities.

        Uses linearly increasing weights: the newest example has 2× the
        sampling probability of the oldest. This keeps the training signal
        fresh while still using older data for stability.

        Returns:
            states: (batch, 3, 6, 7) float32
            policies: (batch, 7) float32
            values: (batch,) float32
        """
        n = len(self.buffer)
        actual_size = min(batch_size, n)

        # Linear weights: oldest=1.0, newest=2.0
        weights = np.linspace(1.0, 2.0, n)
        weights /= weights.sum()

        indices = np.random.choice(n, size=actual_size, replace=False, p=weights)

        buf_list = list(self.buffer)  # Required for indexed access
        states = np.array([buf_list[i][0] for i in indices], dtype=np.float32)
        policies = np.array([buf_list[i][1] for i in indices], dtype=np.float32)
        values = np.array([buf_list[i][2] for i in indices], dtype=np.float32)
        return states, policies, values

    def __len__(self) -> int:
        return len(self.buffer)

    def get_state(self) -> list:
        """Serialize buffer state for checkpointing."""
        return list(self.buffer)

    def load_state(self, state: list) -> None:
        """Restore buffer from checkpoint."""
        self.buffer.clear()
        self.buffer.extend(state)


class Trainer:
    """AlphaZero trainer with warm-restart LR schedule."""

    def __init__(
        self,
        model: AlphaZeroNet,
        device: torch.device,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        total_iterations: int = 100,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # Warm restarts: cycle length T_0=20 iters, doubling each restart.
        # This periodically bumps LR back up, helping the model adapt to
        # the evolving self-play distribution rather than overfitting
        # to stale buffer data.
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-5
        )

    def train_epoch(
        self, replay_buffer: ReplayBuffer, num_batches: int | None = None
    ) -> dict[str, float]:
        """Train for one epoch (multiple mini-batches).

        Args:
            replay_buffer: Source of training data.
            num_batches: Number of mini-batches per epoch. If None, auto-scales
                to see each example roughly once (buffer_size // batch_size).

        Returns:
            Dict with 'policy_loss', 'value_loss', 'total_loss'.
        """
        self.model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        actual_batches = 0

        if len(replay_buffer) < 2:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        effective_batch_size = min(self.batch_size, len(replay_buffer))

        # Auto-scale: one pass through the buffer per epoch
        if num_batches is None:
            num_batches = max(1, len(replay_buffer) // effective_batch_size)

        for _ in range(num_batches):
            states, target_policies, target_values = replay_buffer.sample(
                self.batch_size
            )

            states_t = torch.from_numpy(states).to(self.device)
            target_policies_t = torch.from_numpy(target_policies).to(self.device)
            target_values_t = torch.from_numpy(target_values).to(self.device)

            # Forward pass
            policy_logits, pred_values = self.model(states_t)
            pred_values = pred_values.squeeze(1)

            # Mask invalid moves (Issue 1 Fix)
            occupied = states_t[:, 0] + states_t[:, 1]
            heights = occupied.sum(dim=1)
            valid_moves = heights < ROWS
            policy_logits = policy_logits.masked_fill(~valid_moves, -1e9)

            # Policy loss: cross-entropy with MCTS policy
            log_probs = F.log_softmax(policy_logits, dim=1)

            policy_loss = -torch.sum(target_policies_t * log_probs, dim=1).mean()

            # Value loss: MSE
            value_loss = F.mse_loss(pred_values, target_values_t)

            # Total loss
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            actual_batches += 1

        if actual_batches == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        avg_policy = total_policy_loss / actual_batches
        avg_value = total_value_loss / actual_batches
        return {
            "policy_loss": avg_policy,
            "value_loss": avg_value,
            "total_loss": avg_policy + avg_value,
        }

    def step_scheduler(self) -> None:
        """Step the learning rate scheduler."""
        self.scheduler.step()
