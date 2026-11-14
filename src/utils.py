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
    """Unwrap a torch.compile OptimizedModule (or nested wrapper) to its base nn.Module."""
    while hasattr(model, "_orig_mod"):
        model = getattr(model, "_orig_mod")
    return model


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip any '_orig_mod.' prefixes produced by torch.compile for clean compatibility."""
    prefix = "_orig_mod."
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        clean_key = key
        while clean_key.startswith(prefix):
            clean_key = clean_key[len(prefix) :]
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
        data["scheduler_state_dict"] = scheduler.state_dict()

    tmp_path = path.with_suffix(".tmp")
    torch.save(data, tmp_path)
    tmp_path.replace(path)


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
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
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
