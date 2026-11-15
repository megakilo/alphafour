"""Distributed (DDP) runtime, sharding, and collective synchronization primitives."""

from __future__ import annotations

import datetime
import inspect
import os
import random
import socket
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .game import COLS, ROWS


def is_dist_initialized() -> bool:
    """Return True iff `torch.distributed` is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the global rank (0 in single-process runs)."""
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    """Return the global world size (1 in single-process runs)."""
    return dist.get_world_size() if is_dist_initialized() else 1


def is_main_process() -> bool:
    """Return True on rank 0 (or in single-process runs)."""
    return get_rank() == 0


def get_comm_device(device: torch.device) -> torch.device:
    """Return the collective communication device matching the active distributed backend."""
    if is_dist_initialized() and dist.get_backend() == "nccl" and device.type == "cuda":
        return device
    return torch.device("cpu")


def configure_rayon_threads(world_size: int) -> int:
    """Partition `RAYON_NUM_THREADS` evenly across `world_size` ranks before Rust initialization."""
    ws = max(1, int(world_size))
    threads = max(1, (os.cpu_count() or 1) // ws)
    if ws > 1:
        os.environ.setdefault("RAYON_NUM_THREADS", str(threads))
    return int(os.environ.get("RAYON_NUM_THREADS", str(threads)))


def seed_rank(base_seed: int, rank: int) -> None:
    """Seed Python, NumPy, and PyTorch CPU/CUDA RNGs with distinct streams per rank."""
    rank_seed = (int(base_seed) + int(rank) * 100_003) & 0xFFFFFFFF
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)


def find_free_port() -> int:
    """Bind an ephemeral localhost TCP port and return its port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def init_distributed_process(
    rank: int,
    world_size: int,
    local_rank: int,
    master_addr: str = "127.0.0.1",
    master_port: int | None = None,
    backend: str | None = None,
) -> torch.device:
    """Initialize the default `torch.distributed` process group and return this rank's device."""
    os.environ.setdefault("MASTER_ADDR", master_addr)
    if master_port is not None:
        os.environ.setdefault("MASTER_PORT", str(master_port))
    else:
        os.environ.setdefault("MASTER_PORT", "29500")

    chosen_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if chosen_backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if not is_dist_initialized():
        init_kwargs: dict[str, Any] = {
            "backend": chosen_backend,
            "init_method": "env://",
            "rank": rank,
            "world_size": world_size,
            "timeout": datetime.timedelta(minutes=30),
        }
        init_sig = inspect.signature(dist.init_process_group).parameters
        if "device_id" in init_sig and device.type == "cuda":
            init_kwargs["device_id"] = device
        dist.init_process_group(**init_kwargs)

    return device


def cleanup_distributed() -> None:
    """Destroy the active distributed process group if initialized."""
    if is_dist_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronize all ranks when distributed execution is active."""
    if not is_dist_initialized() or get_world_size() <= 1:
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        barrier_sig = inspect.signature(dist.barrier).parameters
        if "device_ids" in barrier_sig:
            dist.barrier(device_ids=[torch.cuda.current_device()])
            return
    dist.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a picklable Python object from `src` to all ranks."""
    if not is_dist_initialized() or get_world_size() <= 1:
        return obj
    obj_list = [obj if get_rank() == src else None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def shard_count(total: int, rank: int, world_size: int) -> int:
    """Partition `total` items evenly across `world_size` ranks with exact sum."""
    base, rem = divmod(max(0, int(total)), max(1, int(world_size)))
    return base + (1 if rank < rem else 0)


def _to_uint8_states_tensor(
    states: np.ndarray | torch.Tensor, device: torch.device
) -> torch.Tensor:
    if isinstance(states, torch.Tensor):
        if states.numel() == 0:
            return torch.empty((0, 3, ROWS, COLS), dtype=torch.uint8, device=device)
        if states.dtype == torch.uint8:
            return states.to(device=device, dtype=torch.uint8)
        return (states > 0.5).to(device=device, dtype=torch.uint8)

    states_arr = np.asarray(states)
    if states_arr.shape[0] == 0:
        return torch.empty((0, 3, ROWS, COLS), dtype=torch.uint8, device=device)
    if states_arr.dtype == np.uint8:
        u8_arr = np.ascontiguousarray(states_arr)
    else:
        u8_arr = np.ascontiguousarray(states_arr > 0.5).view(np.uint8)
    return torch.as_tensor(u8_arr, device=device)


def _to_float32_tensor(
    arr: np.ndarray | torch.Tensor, empty_shape: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        if arr.numel() == 0:
            return torch.empty(empty_shape, dtype=torch.float32, device=device)
        return arr.to(device=device, dtype=torch.float32)

    np_arr = np.asarray(arr, dtype=np.float32)
    if np_arr.shape[0] == 0:
        return torch.empty(empty_shape, dtype=torch.float32, device=device)
    return torch.as_tensor(np.ascontiguousarray(np_arr), device=device)


def gather_self_play_arrays(
    states: np.ndarray | torch.Tensor,
    policies: np.ndarray | torch.Tensor,
    values: np.ndarray | torch.Tensor,
    device: torch.device,
    as_numpy: bool = False,
) -> tuple[Any, Any, Any]:
    """Gather variable-length self-play arrays across all ranks in deterministic rank order.

    Returns `(states, policies, values)` across all ranks so every rank's `ReplayBuffer`
    ingests the exact same chronological batch.
    """
    world_size = get_world_size()
    if world_size <= 1:
        if as_numpy:
            s_out = (
                states.detach().cpu().numpy()
                if isinstance(states, torch.Tensor)
                else states
            )
            p_out = (
                policies.detach().cpu().numpy()
                if isinstance(policies, torch.Tensor)
                else policies
            )
            v_out = (
                values.detach().cpu().numpy()
                if isinstance(values, torch.Tensor)
                else values
            )
            return s_out, p_out, v_out
        return states, policies, values

    comm_device = get_comm_device(device)
    s_t = _to_uint8_states_tensor(states, comm_device)
    p_t = _to_float32_tensor(policies, (0, COLS), comm_device)
    v_t = _to_float32_tensor(values, (0,), comm_device).reshape(-1)
    local_n = int(s_t.shape[0])

    local_count_t = torch.tensor([local_n], dtype=torch.int64, device=comm_device)
    gathered_counts = [
        torch.zeros((1,), dtype=torch.int64, device=comm_device)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered_counts, local_count_t)
    counts = [int(c.item()) for c in gathered_counts]
    max_n = max(counts)

    if max_n == 0:
        empty_s = torch.empty((0, 3, ROWS, COLS), dtype=torch.uint8, device=comm_device)
        empty_p = torch.empty((0, COLS), dtype=torch.float32, device=comm_device)
        empty_v = torch.empty((0,), dtype=torch.float32, device=comm_device)
        if as_numpy:
            return empty_s.cpu().numpy(), empty_p.cpu().numpy(), empty_v.cpu().numpy()
        return empty_s, empty_p, empty_v

    if local_n < max_n:
        pad = max_n - local_n
        s_pad = torch.cat(
            (
                s_t,
                torch.zeros((pad, 3, ROWS, COLS), dtype=torch.uint8, device=comm_device),
            ),
            dim=0,
        )
        p_pad = torch.cat(
            (p_t, torch.zeros((pad, COLS), dtype=torch.float32, device=comm_device)),
            dim=0,
        )
        v_pad = torch.cat(
            (v_t, torch.zeros((pad,), dtype=torch.float32, device=comm_device)),
            dim=0,
        )
    else:
        s_pad = s_t.contiguous()
        p_pad = p_t.contiguous()
        v_pad = v_t.contiguous()

    s_list = [torch.empty_like(s_pad) for _ in range(world_size)]
    p_list = [torch.empty_like(p_pad) for _ in range(world_size)]
    v_list = [torch.empty_like(v_pad) for _ in range(world_size)]
    dist.all_gather(s_list, s_pad)
    dist.all_gather(p_list, p_pad)
    dist.all_gather(v_list, v_pad)

    s_all = torch.cat(
        [s_list[r][: counts[r]] for r in range(world_size) if counts[r] > 0], dim=0
    )
    p_all = torch.cat(
        [p_list[r][: counts[r]] for r in range(world_size) if counts[r] > 0], dim=0
    )
    v_all = torch.cat(
        [v_list[r][: counts[r]] for r in range(world_size) if counts[r] > 0], dim=0
    )

    if as_numpy:
        return (
            np.ascontiguousarray(s_all.cpu().numpy(), dtype=np.uint8),
            np.ascontiguousarray(p_all.cpu().numpy(), dtype=np.float32),
            np.ascontiguousarray(v_all.cpu().numpy(), dtype=np.float32),
        )
    return s_all, p_all, v_all
