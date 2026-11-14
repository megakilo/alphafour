"""AlphaZero neural network for Connect Four.

ResNet-based architecture with Squeeze-and-Excitation attention and dual heads:
- Policy head: probability distribution over 7 columns
- Value head: 16-channel spatial bottleneck with 3-logit WDL output [Loss, Draw, Win]
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
    """AlphaZero-style neural network for Connect Four with 3-logit WDL value head.

    Args:
        num_res_blocks: Number of residual blocks (default 10).
        num_filters: Number of convolutional filters (default 128).
        value_channels: Number of spatial channels in the value head (default 16).
    """

    def __init__(
        self,
        num_res_blocks: int = 10,
        num_filters: int = 128,
        value_channels: int = 16,
    ) -> None:
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.value_channels = value_channels

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

        # Policy head (32 filters for tactical pattern capacity)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * ROWS * COLS, COLS),
        )

        # Value head: 16-channel spatial feature preservation + 3 WDL logits
        # Ordered [Loss (-1), Draw (0), Win (+1)]
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, value_channels, 1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_channels * ROWS * COLS, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 6, 7).

        Returns:
            policy_logits: Raw column logits of shape (batch, 7).
            wdl_logits: Raw WDL logits of shape (batch, 3) ordered [Loss, Draw, Win].
        """
        x = self.conv_block(x)
        x = self.res_blocks(x)
        policy_logits = self.policy_head(x)
        wdl_logits = self.value_head(x)
        return policy_logits, wdl_logits

    @staticmethod
    def wdl_logits_to_value(wdl_logits: torch.Tensor) -> torch.Tensor:
        """Convert WDL logits [Loss (-1), Draw (0), Win (+1)] to scalar value P(Win) - P(Loss)."""
        logits = (
            wdl_logits.float()
            if wdl_logits.dtype in (torch.float16, torch.bfloat16)
            else wdl_logits
        )
        probs = F.softmax(logits, dim=-1)
        return probs[..., 2] - probs[..., 0]

    def predict_wdl(
        self, state: torch.Tensor, valid_moves: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict policy, WDL probabilities, and expected value for a single state or batch.

        Note: caller is responsible for calling model.eval() before inference.

        Args:
            state: Board encoding, shape (3, 6, 7) or (batch, 3, 6, 7).
            valid_moves: Boolean mask of valid moves, shape (7,) or (batch, 7).

        Returns:
            policy: Probability distribution over moves, shape (7,) or (batch, 7), float32.
            wdl_probs: WDL probabilities [P(Loss), P(Draw), P(Win)], shape (3,) or (batch, 3), float32.
            value: Scalar evaluation P(Win) - P(Loss) in [-1, 1], shape () or (batch,), float32.
        """
        single = state.dim() == 3
        if single:
            state = state.unsqueeze(0)

        with torch.no_grad():
            policy_logits, wdl_logits = self(state)

            if valid_moves is not None:
                valid_mask = torch.as_tensor(
                    valid_moves, dtype=torch.bool, device=policy_logits.device
                )
                if single and valid_mask.dim() == 1:
                    valid_mask = valid_mask.unsqueeze(0)
                fill_value = torch.finfo(policy_logits.dtype).min
                policy_logits = policy_logits.masked_fill(~valid_mask, fill_value)

            policy = F.softmax(policy_logits.float(), dim=-1)
            wdl_probs = F.softmax(wdl_logits.float(), dim=-1)
            value = wdl_probs[..., 2] - wdl_probs[..., 0]

        if single:
            return policy.squeeze(0), wdl_probs.squeeze(0), value.squeeze(0)
        return policy, wdl_probs, value

    def predict(
        self, state: torch.Tensor, valid_moves: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy and scalar value for a single state or batch.

        Note: caller is responsible for calling model.eval() before inference.

        Args:
            state: Board encoding, shape (3, 6, 7) or (batch, 3, 6, 7).
            valid_moves: Boolean mask of valid moves, shape (7,) or (batch, 7).

        Returns:
            policy: Probability distribution over moves, shape (7,) or (batch, 7).
            value: Scalar evaluation P(Win) - P(Loss), shape () or (batch,).
        """
        policy, _, value = self.predict_wdl(state, valid_moves)
        return policy, value
