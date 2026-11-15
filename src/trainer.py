"""Single-GPU training loop and contiguous ring-buffer replay memory for AlphaZero."""

from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .distributed import get_comm_device, get_world_size, is_dist_initialized
from .game import COLS, ROWS
from .model import AlphaZeroNet
from .utils import _align_optimizer_state, configure_torch_compiler


class ReplayBuffer:
    """Contiguous ring-buffer replay memory with O(B) inverse-CDF recency sampling.

    Stores board planes compactly as uint8 (0/1) — ~47 MB total for 300,000 examples —
    either on CPU or directly in GPU VRAM, avoiding Python tuple/deque overhead and
    per-batch numpy/PCIe conversion bottlenecks.
    """

    def __init__(
        self,
        capacity: int = 300_000,
        device: torch.device | str | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive.")
        self.capacity = int(capacity)
        self.buf_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

        self.states_buf = torch.zeros(
            (self.capacity, 3, ROWS, COLS),
            dtype=torch.uint8,
            device=self.buf_device,
        )
        self.policies_buf = torch.zeros(
            (self.capacity, COLS),
            dtype=torch.float32,
            device=self.buf_device,
        )
        self.values_buf = torch.zeros(
            (self.capacity,),
            dtype=torch.float32,
            device=self.buf_device,
        )

        self._write_pos: int = 0
        self._size: int = 0

    def to(self, device: torch.device | str) -> ReplayBuffer:
        """Move ring-buffer backing tensors to the target device."""
        target = torch.device(device)
        if target != self.buf_device:
            self.states_buf = self.states_buf.to(target)
            self.policies_buf = self.policies_buf.to(target)
            self.values_buf = self.values_buf.to(target)
            self.buf_device = target
        return self

    def add_batch(
        self,
        states: np.ndarray | torch.Tensor,
        policies: np.ndarray | torch.Tensor,
        values: np.ndarray | torch.Tensor,
    ) -> None:
        """Vectorized ring-buffer write without Python loops."""
        m = int(states.shape[0])
        if m == 0:
            return

        if m > self.capacity:
            states = states[-self.capacity :]
            policies = policies[-self.capacity :]
            values = values[-self.capacity :]
            m = self.capacity

        if isinstance(states, torch.Tensor):
            if states.dtype == torch.uint8:
                s_t = states.to(device=self.buf_device, dtype=torch.uint8)
            else:
                s_t = (states > 0.5).to(device=self.buf_device, dtype=torch.uint8)
        else:
            states_arr = np.asarray(states)
            if states_arr.dtype == np.uint8:
                u8_arr = np.ascontiguousarray(states_arr)
            else:
                u8_arr = np.ascontiguousarray(states_arr > 0.5).view(np.uint8)
            s_t = torch.as_tensor(u8_arr, device=self.buf_device)

        if isinstance(policies, torch.Tensor):
            p_t = policies.to(device=self.buf_device, dtype=torch.float32)
        else:
            p_t = torch.as_tensor(
                np.ascontiguousarray(policies, dtype=np.float32),
                device=self.buf_device,
            )

        if isinstance(values, torch.Tensor):
            v_t = values.reshape(-1).to(device=self.buf_device, dtype=torch.float32)
        else:
            v_t = torch.as_tensor(
                np.ascontiguousarray(values, dtype=np.float32).reshape(-1),
                device=self.buf_device,
            )

        end = self._write_pos + m
        if end <= self.capacity:
            self.states_buf[self._write_pos : end] = s_t
            self.policies_buf[self._write_pos : end] = p_t
            self.values_buf[self._write_pos : end] = v_t
        else:
            first = self.capacity - self._write_pos
            rem = m - first
            self.states_buf[self._write_pos :] = s_t[:first]
            self.states_buf[:rem] = s_t[first:]
            self.policies_buf[self._write_pos :] = p_t[:first]
            self.policies_buf[:rem] = p_t[first:]
            self.values_buf[self._write_pos :] = v_t[:first]
            self.values_buf[:rem] = v_t[first:]

        self._write_pos = end % self.capacity
        self._size = min(self.capacity, self._size + m)

    def add(
        self,
        examples: (
            tuple[
                np.ndarray | torch.Tensor,
                np.ndarray | torch.Tensor,
                np.ndarray | torch.Tensor,
            ]
            | list[tuple[np.ndarray, np.ndarray, float]]
        ),
    ) -> None:
        """Add examples from either an array tuple (states, policies, values) or legacy tuple list."""
        if isinstance(examples, tuple) and len(examples) == 3:
            states, policies, values = examples
            self.add_batch(states, policies, values)
            return

        examples_list = examples if isinstance(examples, list) else list(examples)
        if not examples_list:
            return

        states_arr = np.stack([ex[0] for ex in examples_list], axis=0)
        policies_arr = np.stack([ex[1] for ex in examples_list], axis=0)
        values_arr = np.asarray([ex[2] for ex in examples_list], dtype=np.float32)
        self.add_batch(states_arr, policies_arr, values_arr)

    def sample_tensors(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a recency-weighted batch directly as tensors on `device`.

        Analytical inverse-CDF for linear weights w(x) = 1 + x on [0, 1]
        (oldest weight = 1.0, newest weight = 2.0):
            F(x) = (2x + x^2) / 3 = u  =>  x = sqrt(1 + 3u) - 1.
        """
        n = self._size
        if n == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer.")

        actual_size = min(batch_size, n)
        out_device = torch.device(device) if device is not None else self.buf_device

        u = torch.rand(actual_size, device=self.buf_device, dtype=torch.float32)
        rank = ((torch.sqrt(1.0 + 3.0 * u) - 1.0) * n).long().clamp_(0, n - 1)
        oldest_idx = (self._write_pos - n) % self.capacity
        idx = (oldest_idx + rank) % self.capacity

        states = self.states_buf[idx].to(device=out_device, dtype=torch.float32)
        policies = self.policies_buf[idx].to(device=out_device)
        values = self.values_buf[idx].to(device=out_device)
        return states, policies, values

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a recency-weighted batch as CPU float32 numpy arrays."""
        states, policies, values = self.sample_tensors(
            batch_size, device=torch.device("cpu")
        )
        return states.numpy(), policies.numpy(), values.numpy()

    def __len__(self) -> int:
        return self._size

    def get_state(self) -> dict[str, torch.Tensor]:
        """Serialize compact buffer state on CPU in chronological (oldest -> newest) order."""
        n = self._size
        if n == 0:
            return {
                "states": torch.empty((0, 3, ROWS, COLS), dtype=torch.uint8),
                "policies": torch.empty((0, COLS), dtype=torch.float32),
                "values": torch.empty((0,), dtype=torch.float32),
            }

        oldest_idx = (self._write_pos - n) % self.capacity
        if oldest_idx + n <= self.capacity:
            end = oldest_idx + n
            states = self.states_buf[oldest_idx:end].detach().to("cpu").contiguous().clone()
            policies = self.policies_buf[oldest_idx:end].detach().to("cpu").contiguous().clone()
            values = self.values_buf[oldest_idx:end].detach().to("cpu").contiguous().clone()
        else:
            states = torch.cat(
                (self.states_buf[oldest_idx:], self.states_buf[: self._write_pos]),
                dim=0,
            ).to("cpu")
            policies = torch.cat(
                (self.policies_buf[oldest_idx:], self.policies_buf[: self._write_pos]),
                dim=0,
            ).to("cpu")
            values = torch.cat(
                (self.values_buf[oldest_idx:], self.values_buf[: self._write_pos]),
                dim=0,
            ).to("cpu")

        return {
            "states": states,
            "policies": policies,
            "values": values,
        }

    def load_state(self, state: dict[str, Any] | list[Any]) -> None:
        """Restore buffer from either compact tensor dict or legacy list of tuples."""
        self._write_pos = 0
        self._size = 0
        if not state:
            return

        if isinstance(state, dict):
            self.add_batch(state["states"], state["policies"], state["values"])
        else:
            self.add(state)


class _AlphaFourSequentialLR(SequentialLR):
    """SequentialLR subclass that excludes runtime restore hooks from state_dict()."""

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        sd.pop("_alphafour_restore", None)
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        cleaned = {k: v for k, v in state_dict.items() if k != "_alphafour_restore"}
        super().load_state_dict(cleaned)


class Trainer:
    """AlphaZero trainer (single-GPU or DDP) with BF16 AMP, WDL loss, and warmup+cosine LR."""

    def __init__(
        self,
        model: AlphaZeroNet,
        device: torch.device,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        total_iterations: int = 150,
        use_compile: bool = False,
        distributed: bool | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        if hasattr(self.model, "optimize_for_device"):
            self.model.optimize_for_device(self.device, use_compile=use_compile)
        self.batch_size = batch_size
        self.lr = float(lr)
        self.total_iterations = max(1, int(total_iterations))

        if distributed is None:
            self.is_distributed = is_dist_initialized() and get_world_size() > 1
        else:
            if distributed and not is_dist_initialized():
                raise RuntimeError(
                    "Trainer(distributed=True) requires torch.distributed to be initialized."
                )
            self.is_distributed = bool(distributed)
        self.world_size = get_world_size() if self.is_distributed else 1

        # Exclude 1D parameters (BatchNorm weights/biases and Linear biases) from weight decay
        decay_params = [
            p for p in self.model.parameters() if p.requires_grad and p.ndim >= 2
        ]
        no_decay_params = [
            p for p in self.model.parameters() if p.requires_grad and p.ndim < 2
        ]
        param_groups = [
            g
            for g in (
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            )
            if g["params"]
        ]

        use_fused = self.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            fused=use_fused,
        )

        # Wrap unwrapped model in DistributedDataParallel when multi-GPU DDP is active
        if self.is_distributed:
            if self.device.type == "cuda":
                ddp_idx = (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
                self._base_train_model: torch.nn.Module = DistributedDataParallel(
                    self.model,
                    device_ids=[ddp_idx],
                    output_device=ddp_idx,
                    broadcast_buffers=True,
                    gradient_as_bucket_view=False,
                )
            else:
                self._base_train_model = DistributedDataParallel(
                    self.model,
                    broadcast_buffers=True,
                    gradient_as_bucket_view=False,
                )
        else:
            self._base_train_model = self.model

        # Optional torch.compile on CUDA with safe fallback to eager base/DDP module
        self.use_compile = bool(use_compile and self.device.type == "cuda")
        self._train_model: torch.nn.Module = self._base_train_model
        if self.use_compile:
            try:
                configure_torch_compiler()
                self._train_model = torch.compile(self._base_train_model)
            except Exception:
                self.use_compile = False
                self._train_model = self._base_train_model

        # Mixed precision configuration: BF16 preferred on modern CUDA, FP16+GradScaler fallback
        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                self._use_amp = True
                self._amp_dtype = torch.bfloat16
                self._use_scaler = False
            else:
                self._use_amp = True
                self._amp_dtype = torch.float16
                self._use_scaler = True
        else:
            self._use_amp = False
            self._amp_dtype = torch.float32
            self._use_scaler = False

        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_scaler)

        # Linear warmup + single monotonic CosineAnnealingLR over total_iterations (no restart spikes)
        self._warmup_iters = min(5, max(1, self.total_iterations // 20))
        self._cosine_iters = max(1, self.total_iterations - self._warmup_iters)
        self._max_scheduler_steps = self._warmup_iters + self._cosine_iters
        self._eta_min = min(1e-5, self.lr)
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self) -> SequentialLR:
        """Build a fresh LinearLR + CosineAnnealingLR schedule over `total_iterations`.

        Resets `param_groups` `lr` and `initial_lr` to `self.lr` while leaving any
        existing AdamW momentum/variance state in `self.optimizer.state` untouched.
        """
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.lr
            pg["initial_lr"] = self.lr

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.2,
            end_factor=1.0,
            total_iters=self._warmup_iters,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self._cosine_iters,
            eta_min=self._eta_min,
        )
        sched = _AlphaFourSequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self._warmup_iters],
        )
        sched._alphafour_restore = self.restore_scheduler  # type: ignore[attr-defined]
        return sched

    def _scheduler_state_matches_horizon(
        self, scheduler_state_dict: dict[str, Any] | None
    ) -> bool:
        """Return True iff `scheduler_state_dict` matches this Trainer's horizon and base LR."""
        if not isinstance(scheduler_state_dict, dict):
            return False
        if scheduler_state_dict.get("_milestones") != [self._warmup_iters]:
            return False
        child_sds = scheduler_state_dict.get("_schedulers")
        if not isinstance(child_sds, (list, tuple)) or len(child_sds) != 2:
            return False
        warmup_sd, cosine_sd = child_sds
        if not isinstance(warmup_sd, dict) or not isinstance(cosine_sd, dict):
            return False
        if warmup_sd.get("total_iters") != self._warmup_iters:
            return False
        if cosine_sd.get("T_max") != self._cosine_iters:
            return False
        if not math.isclose(
            float(cosine_sd.get("eta_min", -1.0)),
            self._eta_min,
            rel_tol=1e-6,
            abs_tol=1e-12,
        ):
            return False
        base_lrs = warmup_sd.get("base_lrs")
        if not isinstance(base_lrs, (list, tuple)) or not base_lrs:
            return False
        if not all(
            math.isclose(float(b), self.lr, rel_tol=1e-6, abs_tol=1e-12)
            for b in base_lrs
        ):
            return False
        saved_epoch = scheduler_state_dict.get("last_epoch")
        if not isinstance(saved_epoch, int) or not (
            0 <= saved_epoch <= self._max_scheduler_steps
        ):
            return False
        return True

    def restore_scheduler(
        self,
        iteration: int,
        scheduler_state_dict: dict[str, Any] | None = None,
    ) -> None:
        """Safely restore or rebuild the LR schedule when resuming from a checkpoint.

        - When `scheduler_state_dict` matches this Trainer's `total_iterations` horizon
          (`_milestones` + `T_max`), base LR, and valid `last_epoch`, loads it directly
          and syncs `optimizer.param_groups` `lr` to `scheduler.get_last_lr()`.
        - When `--iterations` or `--lr` changed on resume (e.g. `50 -> 100` or
          `100 -> 50`) or `scheduler_state_dict` is absent (legacy checkpoint), rebuilds
          `SequentialLR` over `total_iterations` and fast-forwards cleanly to
          `min(max(0, iteration), self._max_scheduler_steps)` without warnings or
          cosine `T_max` overshoot spikes.
        """
        if self._scheduler_state_matches_horizon(scheduler_state_dict):
            assert scheduler_state_dict is not None
            self.scheduler.load_state_dict(scheduler_state_dict)
            self.scheduler._alphafour_restore = self.restore_scheduler  # type: ignore[attr-defined]
            for pg, lr_val in zip(
                self.optimizer.param_groups, self.scheduler.get_last_lr()
            ):
                pg["initial_lr"] = self.lr
                pg["lr"] = float(lr_val)
            return

        target_steps = min(max(0, int(iteration)), self._max_scheduler_steps)
        fresh_scheduler = self._build_scheduler()
        if target_steps > 0:
            self.optimizer._opt_called = True  # type: ignore[attr-defined]
            for _ in range(target_steps):
                fresh_scheduler.step()

        self.scheduler.__dict__.update(fresh_scheduler.__dict__)
        self.scheduler._alphafour_restore = self.restore_scheduler  # type: ignore[attr-defined]
        for pg, lr_val in zip(
            self.optimizer.param_groups, self.scheduler.get_last_lr()
        ):
            pg["initial_lr"] = self.lr
            pg["lr"] = float(lr_val)

    @staticmethod
    def _align_grad_layouts(model: torch.nn.Module) -> None:
        """Ensure parameter gradients match each parameter's stride/layout for fused AdamW."""
        for p in model.parameters():
            if p.grad is not None and p.grad.stride() != p.stride():
                aligned_grad = torch.empty_like(p, memory_format=torch.preserve_format)
                aligned_grad.copy_(p.grad)
                p.grad = aligned_grad

    def _run_batch_step(
        self,
        model_fn: torch.nn.Module,
        states_t: torch.Tensor,
        target_policies_t: torch.Tensor,
        target_values_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one forward + backward + optimizer step on `model_fn`."""
        self.optimizer.zero_grad(set_to_none=True)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self._amp_dtype)
            if self._use_amp
            else nullcontext()
        )
        with amp_ctx:
            policy_logits, wdl_logits = model_fn(states_t)

            # Mask full columns safely (-1e4 avoids FP16 overflow and 0 * -inf NaN)
            occupied = states_t[:, 0] + states_t[:, 1]
            valid_moves = occupied.sum(dim=1) < ROWS
            policy_logits = policy_logits.masked_fill(~valid_moves, -1e4)

            # Policy loss: soft cross-entropy against MCTS visit distribution
            log_probs = F.log_softmax(policy_logits.float(), dim=1)
            policy_loss = -torch.sum(target_policies_t * log_probs, dim=1).mean()

            # Value loss: WDL cross-entropy ([Loss (-1), Draw (0), Win (+1)])
            # Exact one-hot on {-1, 0, +1} and smooth calibrated soft WDL on [-1, 1]
            v_clamped = target_values_t.clamp(-1.0, 1.0)
            wdl_targets = torch.stack(
                (F.relu(-v_clamped), 1.0 - v_clamped.abs(), F.relu(v_clamped)),
                dim=1,
            )
            wdl_log_probs = F.log_softmax(wdl_logits.float(), dim=1)
            value_loss = -torch.sum(wdl_targets * wdl_log_probs, dim=1).mean()

            loss = policy_loss + value_loss

        if self._use_scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self._align_grad_layouts(self.model)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self._align_grad_layouts(self.model)
            self.optimizer.step()

        return policy_loss.detach(), value_loss.detach()

    def train_epoch(
        self, replay_buffer: ReplayBuffer, num_batches: int | None = None
    ) -> dict[str, float]:
        """Train for one epoch (multiple mini-batches).

        Returns:
            Dict with 'policy_loss', 'value_loss', 'total_loss'.
        """
        self.model.train()
        self._base_train_model.train()
        if (
            self._train_model is not self._base_train_model
            and hasattr(self._train_model, "train")
        ):
            self._train_model.train()
        _align_optimizer_state(self.optimizer)

        buf_len = len(replay_buffer)
        if buf_len < 2:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        effective_batch_size = min(self.batch_size, buf_len)
        if num_batches is None:
            num_batches = max(1, buf_len // effective_batch_size)
        if num_batches <= 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        total_policy_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        total_value_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        actual_batches = 0

        for _ in range(num_batches):
            if hasattr(replay_buffer, "sample_tensors"):
                states_t, target_policies_t, target_values_t = (
                    replay_buffer.sample_tensors(effective_batch_size, self.device)
                )
            else:
                states_np, policies_np, values_np = replay_buffer.sample(
                    effective_batch_size
                )
                states_t = torch.from_numpy(states_np).to(self.device)
                target_policies_t = torch.from_numpy(policies_np).to(self.device)
                target_values_t = torch.from_numpy(values_np).to(self.device)

            if self.use_compile:
                try:
                    p_loss, v_loss = self._run_batch_step(
                        self._train_model,
                        states_t,
                        target_policies_t,
                        target_values_t,
                    )
                except Exception:
                    self.use_compile = False
                    self._train_model = self._base_train_model
                    p_loss, v_loss = self._run_batch_step(
                        self._base_train_model,
                        states_t,
                        target_policies_t,
                        target_values_t,
                    )
            else:
                p_loss, v_loss = self._run_batch_step(
                    self._base_train_model,
                    states_t,
                    target_policies_t,
                    target_values_t,
                )

            total_policy_loss += p_loss
            total_value_loss += v_loss
            actual_batches += 1

        if actual_batches == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        loss_vec = torch.stack((total_policy_loss, total_value_loss))
        if self.is_distributed:
            comm_dev = get_comm_device(self.device)
            # Keep BatchNorm running stats identical across ranks after the final step
            for buf in self.model.buffers():
                if buf.data.device == comm_dev:
                    dist.broadcast(buf.data, src=0)
                else:
                    buf_comm = buf.data.to(comm_dev)
                    dist.broadcast(buf_comm, src=0)
                    buf.data.copy_(buf_comm)

            loss_comm = loss_vec.to(comm_dev)
            dist.all_reduce(loss_comm, op=dist.ReduceOp.SUM)
            avg_losses_cpu = (loss_comm / (actual_batches * self.world_size)).cpu()
        else:
            # Single device-to-host synchronization at epoch end
            avg_losses_cpu = (loss_vec / actual_batches).cpu()

        avg_policy = avg_losses_cpu[0].item()
        avg_value = avg_losses_cpu[1].item()
        return {
            "policy_loss": avg_policy,
            "value_loss": avg_value,
            "total_loss": avg_policy + avg_value,
        }

    def _effective_scheduler_horizon(self) -> int:
        """Return safe step horizon bounded by both `_max_scheduler_steps` and live `T_max`."""
        horizon = self._max_scheduler_steps
        milestones = getattr(self.scheduler, "_milestones", None)
        sub_scheds = getattr(self.scheduler, "_schedulers", None)
        if (
            isinstance(milestones, (list, tuple))
            and len(milestones) == 1
            and isinstance(sub_scheds, (list, tuple))
            and len(sub_scheds) == 2
            and hasattr(sub_scheds[1], "T_max")
        ):
            horizon = min(horizon, int(milestones[0]) + int(sub_scheds[1].T_max))
        return horizon

    def step_scheduler(self) -> None:
        """Step the LR scheduler, clamping safely at eta_min once total_iterations is reached."""
        horizon = self._effective_scheduler_horizon()
        if self.scheduler.last_epoch < horizon:
            self.optimizer._opt_called = True  # type: ignore[attr-defined]
            self.scheduler.step()
        else:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self._eta_min
            self.scheduler._last_lr = [
                self._eta_min for _ in self.optimizer.param_groups
            ]
