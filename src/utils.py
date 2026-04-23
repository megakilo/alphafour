"""Utility functions: device selection, checkpoint I/O."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch


def get_device() -> torch.device:
    """Return the best available device: MPS (Metal) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    replay_buffer: Any,
    path: str | Path,
    scheduler: Any = None,
) -> None:
    """Save a training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay_buffer": replay_buffer,
    }
    if scheduler is not None:
        data["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(data, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load a checkpoint and return the checkpoint dict.

    Restores model and optionally optimizer state.
    Returns dict with 'iteration' and 'replay_buffer' keys.
    """
    map_location = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


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
    """Save just the model weights for use in play.py."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
