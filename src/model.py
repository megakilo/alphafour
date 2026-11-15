"""AlphaZero neural network for Connect Four.

ResNet-based architecture with Squeeze-and-Excitation attention and dual heads:
- Policy head: probability distribution over 7 columns
- Value head: 16-channel spatial bottleneck with 3-logit WDL output [Loss, Draw, Win]
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import ROWS, COLS
from .utils import configure_torch_compiler


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
        scale = x.mean(dim=(2, 3))
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

        # Non-module runtime acceleration flags (pure bools so AlphaZeroNet stays 100% picklable)
        self._use_channels_last: bool = False
        self._use_compiled_infer: bool = False

    def optimize_for_device(
        self, device: torch.device | str, use_compile: bool = False
    ) -> AlphaZeroNet:
        """Configure memory format (`channels_last`) and optional dynamic `torch.compile` for CUDA inference."""
        dev = torch.device(device)
        self.to(dev)
        if dev.type == "cuda":
            # Keep cudnn.benchmark disabled: variable MCTS leaf batch sizes B otherwise
            # trigger exhaustive per-shape cuDNN autotuning stalls across hundreds of B values.
            torch.backends.cudnn.benchmark = False
            torch.set_float32_matmul_precision("high")
            self.to(memory_format=torch.channels_last)
            self._use_channels_last = True
            if use_compile:
                try:
                    configure_torch_compiler()
                    _ensure_compiled_infer_fns()
                    self._use_compiled_infer = True
                except Exception:
                    self._use_compiled_infer = False
            else:
                self._use_compiled_infer = False
        else:
            self._use_channels_last = False
            self._use_compiled_infer = False
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 6, 7).

        Returns:
            policy_logits: Raw column logits of shape (batch, 7).
            wdl_logits: Raw WDL logits of shape (batch, 3) ordered [Loss, Draw, Win].
        """
        if x.dtype == torch.uint8:
            x = x.float()
        if self._use_channels_last and x.ndim == 4 and x.device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv_block(x)
        x = self.res_blocks(x)
        policy_logits = self.policy_head(x)
        wdl_logits = self.value_head(x)
        return policy_logits, wdl_logits

    def _forward_packed_impl(
        self, states: torch.Tensor, valid_moves: torch.Tensor
    ) -> torch.Tensor:
        """Fused batch inference returning contiguous `float32[B, 8]` (`[policy_0..6, value]`)."""
        policy_logits, wdl_logits = self(states)
        masked_logits = policy_logits.float().masked_fill(~valid_moves, -1e9)
        policy = F.softmax(masked_logits, dim=-1)
        wdl_probs = F.softmax(wdl_logits.float(), dim=-1)
        value = (wdl_probs[:, 2] - wdl_probs[:, 0]).unsqueeze(-1)
        return torch.cat((policy, value), dim=-1)

    def _forward_wdl_WithMask_impl(
        self, state: torch.Tensor, valid_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_logits, wdl_logits = self(state)
        logits_f32 = policy_logits.float()
        if valid_mask is not None:
            logits_f32 = logits_f32.masked_fill(~valid_mask, -1e9)
        policy = F.softmax(logits_f32, dim=-1)
        wdl_probs = F.softmax(wdl_logits.float(), dim=-1)
        value = wdl_probs[..., 2] - wdl_probs[..., 0]
        return policy, wdl_probs, value

    def predict_packed(
        self, states: torch.Tensor, valid_moves: torch.Tensor
    ) -> torch.Tensor:
        """Fast batched inference for MCTS returning `float32[B, 8]` (`[policy_0..6, value]`).

        Accepts `uint8` or `float32` `states` of shape `(B, 3, 6, 7)` and `bool` `valid_moves` `(B, 7)`.
        Automatically runs under `no_grad` + BF16/FP16 autocast on CUDA and uses dynamic `torch.compile`
        when enabled via `optimize_for_device(..., use_compile=True)`.
        """
        dev = states.device
        if dev.type == "cuda":
            amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                # Route B >= 2 through the single dynamic compiled graph (skipping B == 1 avoids
                # TorchDynamo's separate s0 == 1 specialization compile for tiny single-leaf tails).
                if (
                    self._use_compiled_infer
                    and _COMPILED_PACKED_FN is not None
                    and states.shape[0] > 1
                ):
                    try:
                        return _COMPILED_PACKED_FN(self, states, valid_moves)
                    except Exception:
                        self._use_compiled_infer = False
                return self._forward_packed_impl(states, valid_moves)

        with torch.no_grad():
            return self._forward_packed_impl(states, valid_moves)

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
            state: Board encoding (`uint8` or `float32`), shape (3, 6, 7) or (batch, 3, 6, 7).
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
            valid_mask: torch.Tensor | None = None
            if valid_moves is not None:
                valid_mask = torch.as_tensor(
                    valid_moves, dtype=torch.bool, device=state.device
                )
                if single and valid_mask.dim() == 1:
                    valid_mask = valid_mask.unsqueeze(0)

            if (
                self._use_compiled_infer
                and _COMPILED_WDL_FN is not None
                and state.device.type == "cuda"
                and state.shape[0] > 1
            ):
                try:
                    policy, wdl_probs, value = _COMPILED_WDL_FN(self, state, valid_mask)
                except Exception:
                    self._use_compiled_infer = False
                    policy, wdl_probs, value = self._forward_wdl_WithMask_impl(
                        state, valid_mask
                    )
            else:
                policy, wdl_probs, value = self._forward_wdl_WithMask_impl(
                    state, valid_mask
                )

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


def _forward_packed_fn(
    model: AlphaZeroNet, states: torch.Tensor, valid_moves: torch.Tensor
) -> torch.Tensor:
    return model._forward_packed_impl(states, valid_moves)


def _forward_wdl_fn(
    model: AlphaZeroNet, state: torch.Tensor, valid_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return model._forward_wdl_WithMask_impl(state, valid_mask)


_COMPILED_PACKED_FN: (
    Callable[[AlphaZeroNet, torch.Tensor, torch.Tensor], torch.Tensor] | None
) = None
_COMPILED_WDL_FN: (
    Callable[
        [AlphaZeroNet, torch.Tensor, torch.Tensor | None],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
    | None
) = None


def _ensure_compiled_infer_fns() -> None:
    global _COMPILED_PACKED_FN, _COMPILED_WDL_FN
    configure_torch_compiler()
    if _COMPILED_PACKED_FN is None:
        _COMPILED_PACKED_FN = torch.compile(_forward_packed_fn, dynamic=True)
    if _COMPILED_WDL_FN is None:
        _COMPILED_WDL_FN = torch.compile(_forward_wdl_fn, dynamic=True)
