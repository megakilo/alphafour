"""Utility functions: single-GPU device selection, clean checkpoint & replay buffer I/O."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch


def get_device() -> torch.device:
    """Return the best available single-GPU device: MPS (Metal) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap torch.compile / DDP / DataParallel wrappers (in any nesting order) to the base nn.Module."""
    ddp_types = (
        torch.nn.parallel.DistributedDataParallel,
        torch.nn.DataParallel,
    )
    while True:
        if hasattr(model, "_orig_mod"):
            model = getattr(model, "_orig_mod")
        elif isinstance(model, ddp_types):
            model = model.module
        else:
            break
    return model


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip any '_orig_mod.' and 'module.' prefixes produced by torch.compile or DDP."""
    prefixes = ("_orig_mod.", "module.")
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        clean_key = key
        stripped = True
        while stripped:
            stripped = False
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    stripped = True
        cleaned[clean_key] = value
    return cleaned


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    replay_buffer: Any = None,
    path: str | Path | None = None,
    scheduler: Any = None,
) -> None:
    """Save a lightweight training checkpoint atomically.

    When `replay_buffer` is None (default), omits the 'replay_buffer' key so
    periodic checkpoints remain small (~36 MB). Supports both keyword and
    positional invocation.
    """
    if path is None and isinstance(replay_buffer, (str, Path)):
        path = replay_buffer
        replay_buffer = None
    if path is None:
        raise ValueError("`path` must be provided to save_checkpoint.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_model = _unwrap_model(model)
    data: dict[str, Any] = {
        "iteration": iteration,
        "model_state_dict": _clean_state_dict(base_model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if replay_buffer is not None:
        data["replay_buffer"] = replay_buffer
    if scheduler is not None:
        sched_sd = scheduler.state_dict()
        if isinstance(sched_sd, dict) and "_alphafour_restore" in sched_sd:
            sched_sd = {k: v for k, v in sched_sd.items() if k != "_alphafour_restore"}
        data["scheduler_state_dict"] = sched_sd

    tmp_path = path.with_suffix(".tmp")
    torch.save(data, tmp_path)
    tmp_path.replace(path)


def _restore_scheduler_from_checkpoint(
    scheduler: Any, checkpoint: dict[str, Any]
) -> None:
    """Safely restore `scheduler` from `checkpoint` without mismatched `T_max` spikes."""
    iteration = int(checkpoint.get("iteration", 0))
    sched_sd = checkpoint.get("scheduler_state_dict")

    if hasattr(scheduler, "restore_scheduler"):
        scheduler.restore_scheduler(iteration, sched_sd)
        return
    if hasattr(scheduler, "_alphafour_restore"):
        scheduler._alphafour_restore(iteration, sched_sd)
        return

    milestones = getattr(scheduler, "_milestones", None)
    sub_scheds = getattr(scheduler, "_schedulers", None)
    if (
        isinstance(milestones, (list, tuple))
        and len(milestones) == 1
        and isinstance(sub_scheds, (list, tuple))
        and len(sub_scheds) == 2
        and hasattr(sub_scheds[0], "total_iters")
        and hasattr(sub_scheds[1], "T_max")
    ):
        warmup_iters = int(milestones[0])
        cosine_iters = int(sub_scheds[1].T_max)
        horizon = warmup_iters + cosine_iters
        child_sds = sched_sd.get("_schedulers") if isinstance(sched_sd, dict) else None
        saved_epoch = sched_sd.get("last_epoch") if isinstance(sched_sd, dict) else None

        if (
            isinstance(sched_sd, dict)
            and sched_sd.get("_milestones") == list(milestones)
            and isinstance(child_sds, (list, tuple))
            and len(child_sds) == 2
            and isinstance(child_sds[0], dict)
            and isinstance(child_sds[1], dict)
            and child_sds[0].get("total_iters") == warmup_iters
            and child_sds[1].get("T_max") == cosine_iters
            and isinstance(saved_epoch, int)
            and 0 <= saved_epoch <= horizon
        ):
            scheduler.load_state_dict(sched_sd)
            for pg, lr_val in zip(
                scheduler.optimizer.param_groups, scheduler.get_last_lr()
            ):
                pg["lr"] = float(lr_val)
            return

        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        opt = scheduler.optimizer
        base_lrs = [float(b) for b in sub_scheds[0].base_lrs]
        for pg, base_lr in zip(opt.param_groups, base_lrs):
            pg["lr"] = base_lr
            pg["initial_lr"] = base_lr

        fresh = SequentialLR(
            opt,
            schedulers=[
                LinearLR(
                    opt,
                    start_factor=float(getattr(sub_scheds[0], "start_factor", 0.2)),
                    end_factor=float(getattr(sub_scheds[0], "end_factor", 1.0)),
                    total_iters=warmup_iters,
                ),
                CosineAnnealingLR(
                    opt,
                    T_max=cosine_iters,
                    eta_min=float(getattr(sub_scheds[1], "eta_min", 1e-5)),
                ),
            ],
            milestones=[warmup_iters],
        )
        target_steps = min(max(0, iteration), horizon)
        if target_steps > 0:
            opt._opt_called = True  # type: ignore[attr-defined]
            for _ in range(target_steps):
                fresh.step()
        scheduler.__dict__.update(fresh.__dict__)
        for pg, lr_val in zip(opt.param_groups, scheduler.get_last_lr()):
            pg["lr"] = float(lr_val)
        return

    if isinstance(sched_sd, dict):
        scheduler.load_state_dict(sched_sd)
        if hasattr(scheduler, "get_last_lr") and hasattr(scheduler, "optimizer"):
            for pg, lr_val in zip(
                scheduler.optimizer.param_groups, scheduler.get_last_lr()
            ):
                pg["lr"] = float(lr_val)


def _align_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Align optimizer state tensors to each parameter's device, dtype, and memory format/strides.

    CUDA fused AdamW requires `params`, `grads`, `exp_avg`, and `exp_avg_sq` to share the exact
    same dtype, device, and layout/strides (e.g. when resuming an NCHW checkpoint into a
    `channels_last` NHWC model, or vice versa, or across CPU <-> CUDA).
    """
    default_fused = bool(optimizer.defaults.get("fused", False))
    for group in optimizer.param_groups:
        group_fused = bool(group.get("fused", default_fused))
        for p in group["params"]:
            state = optimizer.state.get(p)
            if not state:
                continue
            for key, val in list(state.items()):
                if not torch.is_tensor(val):
                    continue
                if val.shape == p.shape and p.ndim > 0:
                    if (
                        val.device != p.device
                        or val.dtype != p.dtype
                        or val.stride() != p.stride()
                    ):
                        aligned = torch.empty_like(
                            p, memory_format=torch.preserve_format
                        )
                        aligned.copy_(val)
                        state[key] = aligned
                elif key == "step":
                    target_dev = p.device if group_fused else torch.device("cpu")
                    if val.device != target_dev or val.dtype != torch.float32:
                        state[key] = val.to(device=target_dev, dtype=torch.float32)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
    scheduler: Any = None,
) -> dict[str, Any]:
    """Load a checkpoint into `model` (compiled or uncompiled) and return the checkpoint dict."""
    map_location = device or torch.device("cpu")
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    base_model = _unwrap_model(model)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        base_model.load_state_dict(_clean_state_dict(checkpoint["model_state_dict"]))
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _align_optimizer_state(optimizer)
        if scheduler is not None:
            _restore_scheduler_from_checkpoint(scheduler, checkpoint)
    elif isinstance(checkpoint, dict):
        base_model.load_state_dict(_clean_state_dict(checkpoint))

    return checkpoint


def save_replay_buffer(buffer_state: Any, path: str | Path) -> None:
    """Atomically save serialized replay buffer state (or a ReplayBuffer instance) to disk."""
    if hasattr(buffer_state, "get_state"):
        buffer_state = buffer_state.get_state()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(buffer_state, tmp_path)
    tmp_path.replace(path)


def load_replay_buffer(path: str | Path) -> Any:
    """Load serialized replay buffer state from disk onto CPU."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def get_latest_checkpoint(directory: str | Path) -> Path | None:
    """Find the most recent checkpoint file in a directory.

    Expects files named like 'checkpoint_0001.pt'.
    """
    directory = Path(directory)
    if not directory.exists():
        return None

    pattern = re.compile(r"checkpoint_(\d+)\.pt$")
    best: tuple[int, Path] | None = None

    for f in directory.iterdir():
        m = pattern.match(f.name)
        if m:
            iteration = int(m.group(1))
            if best is None or iteration > best[0]:
                best = (iteration, f)

    return best[1] if best else None


def save_model_for_play(model: torch.nn.Module, path: str | Path) -> None:
    """Save clean model weights atomically for use in play.py."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_model = _unwrap_model(model)
    tmp_path = path.with_suffix(".tmp")
    torch.save(_clean_state_dict(base_model.state_dict()), tmp_path)
    tmp_path.replace(path)
