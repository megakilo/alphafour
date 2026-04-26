"""AlphaZero neural network for Connect Four.

ResNet-based architecture with dual heads:
- Policy head: probability distribution over 7 columns
- Value head: scalar evaluation of position (-1 to 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import ROWS, COLS


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Learns to weight channels based on global board information,
    helping the network reason about whole-board features like
    piece counts, symmetry, and positional density.
    """

    def __init__(self, num_filters: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(num_filters // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(num_filters, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, num_filters, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.squeeze(x).view(b, c)
        scale = self.excite(scale).view(b, c, 1, 1)
        return x * scale


class ResBlock(nn.Module):
    """Residual block with two conv layers, batch norm, and SE attention."""

    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.se = SEBlock(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """AlphaZero-style neural network for Connect Four.

    Args:
        num_res_blocks: Number of residual blocks (default 10).
        num_filters: Number of convolutional filters (default 128).
    """

    def __init__(self, num_res_blocks: int = 10, num_filters: int = 128) -> None:
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters

        # Initial convolution block
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * ROWS * COLS, COLS),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(ROWS * COLS, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 6, 7).

        Returns:
            policy_logits: Raw logits of shape (batch, 7). NOT softmaxed.
            value: Scalar evaluation of shape (batch, 1), range [-1, 1].
        """
        x = self.conv_block(x)
        x = self.res_blocks(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def predict(
        self, state: torch.Tensor, valid_moves: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy and value for a single state or batch.

        Args:
            state: Board encoding, shape (3, 6, 7) or (batch, 3, 6, 7).
            valid_moves: Boolean mask of valid moves, shape (7,) or (batch, 7).

        Returns:
            policy: Probability distribution over moves, shape (7,) or (batch, 7).
            value: Scalar evaluation, shape () or (batch,).
        """
        single = state.dim() == 3
        if single:
            state = state.unsqueeze(0)
            if valid_moves is not None:
                valid_moves = valid_moves.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            policy_logits, value = self(state)

            # Mask invalid moves
            if valid_moves is not None:
                # Set invalid move logits to very negative number
                policy_logits = policy_logits.masked_fill(~valid_moves, float("-inf"))

            policy = F.softmax(policy_logits, dim=1)

        if single:
            return policy.squeeze(0), value.squeeze(0).squeeze(0)
        return policy, value.squeeze(1)
