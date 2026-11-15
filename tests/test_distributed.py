"""Unit and multi-process DDP integration tests for src/distributed.py and multi-GPU training."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

import main as main_cli
from src.distributed import (
    barrier,
    cleanup_distributed,
    configure_rayon_threads,
    find_free_port,
    gather_self_play_arrays,
    get_rank,
    get_world_size,
    init_distributed_process,
    is_dist_initialized,
    is_main_process,
    seed_rank,
    shard_count,
)
from src.evaluate import (
    evaluate_dataset,
    evaluate_dataset_distributed,
    play_anchor_arena_distributed,
    play_batched_arena_distributed,
)
from src.game import COLS, ROWS
from src.model import AlphaZeroNet
from src.self_play import run_self_play
from src.trainer import ReplayBuffer, Trainer
from src.utils import (
    _BenignTorchCompileLogFilter,
    _clean_state_dict,
    _unwrap_model,
    configure_torch_compiler,
    load_checkpoint,
    save_checkpoint,
)


class _FakeCompiledWrapper(torch.nn.Module):
    """Lightweight stand-in mimicking torch.compile's `_orig_mod` attribute."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = inner


# ── 1. Checkpoint & Wrapper Unwrapping Unit Tests ───────────────────────────


def test_unwrap_model_and_clean_state_dict_nested_wrappers() -> None:
    """`_unwrap_model` and `_clean_state_dict` strip `_orig_mod` and `module` in any order."""
    base = AlphaZeroNet(num_res_blocks=2, num_filters=32)
    dp_wrapped = torch.nn.DataParallel(base)
    compiled_dp = _FakeCompiledWrapper(dp_wrapped)
    double_wrapped = _FakeCompiledWrapper(compiled_dp)

    assert _unwrap_model(base) is base
    assert _unwrap_model(dp_wrapped) is base
    assert _unwrap_model(compiled_dp) is base
    assert _unwrap_model(double_wrapped) is base

    dp_nested = torch.nn.DataParallel(_FakeCompiledWrapper(base))
    assert _unwrap_model(dp_nested) is base

    base_sd = base.state_dict()
    prefixed_sd = {
        (
            f"module._orig_mod.{k}"
            if idx % 2 == 0
            else f"_orig_mod.module.module.{k}"
        ): v
        for idx, (k, v) in enumerate(base_sd.items())
    }
    cleaned = _clean_state_dict(prefixed_sd)
    assert set(cleaned.keys()) == set(base_sd.keys())
    for key in base_sd:
        assert torch.equal(cleaned[key], base_sd[key])


def test_checkpoint_roundtrip_between_ddp_prefixed_and_unwrapped(tmp_path: Path) -> None:
    """Prefixed (`module.` / `_orig_mod.`) and plain checkpoints cross-load cleanly."""
    model_a = AlphaZeroNet(num_res_blocks=2, num_filters=32)
    dp_a = torch.nn.DataParallel(_FakeCompiledWrapper(model_a))
    trainer_a = Trainer(
        model=model_a,
        device=torch.device("cpu"),
        lr=0.001,
        batch_size=4,
        total_iterations=10,
        use_compile=False,
    )

    ckpt_path = tmp_path / "ckpt_ddp.pt"
    save_checkpoint(
        model=dp_a,
        optimizer=trainer_a.optimizer,
        iteration=3,
        replay_buffer=None,
        path=ckpt_path,
        scheduler=trainer_a.scheduler,
    )

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert not any(
        k.startswith(("module.", "_orig_mod.")) for k in raw_ckpt["model_state_dict"]
    )

    # Simulate legacy checkpoint that saved explicit `module.` prefixes
    legacy_ckpt_path = tmp_path / "ckpt_legacy_module.pt"
    raw_ckpt["model_state_dict"] = {
        f"module.{k}": v for k, v in raw_ckpt["model_state_dict"].items()
    }
    torch.save(raw_ckpt, legacy_ckpt_path)

    model_b = AlphaZeroNet(num_res_blocks=2, num_filters=32)
    load_checkpoint(legacy_ckpt_path, model_b, device=torch.device("cpu"))
    for k, v in model_a.state_dict().items():
        assert torch.equal(model_b.state_dict()[k], v)


# ── 2. Pure Distributed Utility & CLI Tests ─────────────────────────────────


def test_shard_count_exact_partition() -> None:
    """`shard_count` partitions any total across any world_size with exact sum."""
    for total in (0, 1, 2, 3, 7, 17, 1000):
        for world_size in (1, 2, 3, 4):
            counts = [shard_count(total, r, world_size) for r in range(world_size)]
            assert sum(counts) == total
            assert max(counts) - min(counts) <= 1


def test_configure_rayon_threads_and_seed_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """`configure_rayon_threads` partitions cores and respects explicit env overrides."""
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    threads = configure_rayon_threads(2)
    assert threads == 8
    assert os.environ["RAYON_NUM_THREADS"] == "8"

    # Explicit user override should not be overwritten
    monkeypatch.setenv("RAYON_NUM_THREADS", "4")
    assert configure_rayon_threads(2) == 4
    assert os.environ["RAYON_NUM_THREADS"] == "4"

    seed_rank(1234, 0)
    r0_np = float(np.random.random())
    r0_th = float(torch.rand(1).item())
    seed_rank(1234, 1)
    r1_np = float(np.random.random())
    r1_th = float(torch.rand(1).item())
    assert r0_np != r1_np
    assert r0_th != r1_th


def test_main_gpus_cli_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """`main.py` accepts default auto `--gpus` (`None`) and explicit integer values."""
    monkeypatch.setattr(sys, "argv", ["main.py"])
    assert main_cli.parse_args().gpus is None

    monkeypatch.setattr(sys, "argv", ["main.py", "--gpus", "1"])
    assert main_cli.parse_args().gpus == 1

    monkeypatch.setattr(sys, "argv", ["main.py", "--gpus", "2"])
    assert main_cli.parse_args().gpus == 2


def test_configure_torch_compiler_suppresses_benign_warnings_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`configure_torch_compiler` disables small-tensor online softmax and filters benign compile noise."""
    import torch._dynamo.config as dyn_cfg
    import torch._inductor.config as ind_cfg

    configure_torch_compiler()
    configure_torch_compiler()  # Must be idempotent
    assert ind_cfg.online_softmax is False

    for name in ("torch._inductor.utils", "torch._dynamo.variables.ctx_manager"):
        filters = [
            f
            for f in logging.getLogger(name).filters
            if isinstance(f, _BenignTorchCompileLogFilter)
        ]
        assert len(filters) == 1

    flt = _BenignTorchCompileLogFilter()
    monkeypatch.setattr(ind_cfg, "max_autotune", False)
    monkeypatch.setattr(ind_cfg, "max_autotune_gemm", False)
    monkeypatch.setattr(dyn_cfg, "capture_profiler_record_function", False)

    def _rec(msg: str) -> logging.LogRecord:
        return logging.LogRecord("test", logging.WARNING, __file__, 1, msg, (), None)

    assert not flt.filter(_rec("Not enough SMs to use max_autotune_gemm mode"))
    assert not flt.filter(
        _rec("Profiler record function <class 'x'> will be ignored")
    )
    assert flt.filter(_rec("Graph break in user code"))

    # If user explicitly requests max_autotune, do not silence the SM warning
    monkeypatch.setattr(ind_cfg, "max_autotune", True)
    assert flt.filter(_rec("Not enough SMs to use max_autotune_gemm mode"))


# ── 3. Multi-Process Gloo DDP End-to-End Parity Tests ───────────────────────


_TEST_DATASET_LINES = [
    "4 1",
    "44 -1",
    "443 1",
    "123 0",
    "3344 0",
    "43434 1",
    "2233 0",
]


def _gloo_e2e_worker(
    rank: int,
    world_size: int,
    master_port: int,
    artifact_dir_str: str,
) -> None:
    """2-rank CPU/gloo DDP test verifying gather, Trainer sync, eval & arena exactness."""
    os.environ["RAYON_NUM_THREADS"] = "2"
    artifact_dir = Path(artifact_dir_str)
    device = init_distributed_process(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        master_addr="127.0.0.1",
        master_port=master_port,
        backend="gloo",
    )
    try:
        assert is_dist_initialized()
        assert get_rank() == rank
        assert get_world_size() == world_size
        assert is_main_process() == (rank == 0)
        seed_rank(777, rank)

        # (a) gather_self_play_arrays: uneven lengths (5 vs 2)
        local_n = 5 if rank == 0 else 2
        s_local = np.full((local_n, 3, ROWS, COLS), rank, dtype=np.uint8)
        p_local = np.full((local_n, COLS), float(rank + 1) / COLS, dtype=np.float32)
        v_local = np.full((local_n,), 1.0 if rank == 0 else -1.0, dtype=np.float32)
        s_g, p_g, v_g = gather_self_play_arrays(
            s_local, p_local, v_local, device, as_numpy=True
        )
        assert s_g.shape == (7, 3, ROWS, COLS) and s_g.dtype == np.uint8
        assert p_g.shape == (7, COLS)
        assert v_g.shape == (7,)
        assert np.all(s_g[:5] == 0) and np.all(s_g[5:] == 1)
        assert np.allclose(v_g[:5], 1.0) and np.allclose(v_g[5:], -1.0)

        # (b) gather_self_play_arrays: one rank empty (3 vs 0), and both empty (0 vs 0)
        e_n = 3 if rank == 0 else 0
        es, ep, ev = gather_self_play_arrays(
            np.ones((e_n, 3, ROWS, COLS), dtype=np.float32),
            np.full((e_n, COLS), 1.0 / COLS, dtype=np.float32),
            np.zeros((e_n,), dtype=np.float32),
            device,
            as_numpy=True,
        )
        assert len(es) == 3 and es.dtype == np.uint8 and np.all(es == 1)

        zs, zp, zv = gather_self_play_arrays(
            np.empty((0, 3, ROWS, COLS), dtype=np.uint8),
            np.empty((0, COLS), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            device,
            as_numpy=True,
        )
        assert len(zs) == 0 and len(zp) == 0 and len(zv) == 0

        # (c) Load identical initial weights and verify evaluate_dataset_distributed exactness
        init_ckpt = torch.load(
            artifact_dir / "init_model.pt", map_location="cpu", weights_only=False
        )
        model = AlphaZeroNet(num_res_blocks=2, num_filters=32)
        model.load_state_dict(init_ckpt["state_dict"])
        model.eval()

        expected_eval0 = init_ckpt["expected_eval0"]
        expected_eval4 = init_ckpt["expected_eval4"]
        child_wdl = init_ckpt["child_wdl"]

        dist_eval0 = evaluate_dataset_distributed(
            model, device, _TEST_DATASET_LINES, num_simulations=0, child_wdl=child_wdl
        )
        dist_eval4 = evaluate_dataset_distributed(
            model, device, _TEST_DATASET_LINES, num_simulations=4, child_wdl=child_wdl
        )
        for exp_d, got_d in ((expected_eval0, dist_eval0), (expected_eval4, dist_eval4)):
            for k in (
                "wdl_accuracy",
                "win_recall",
                "draw_recall",
                "loss_recall",
                "value_mse",
                "optimal_move_pct",
                "blunder_pct",
            ):
                if isinstance(exp_d[k], float) and np.isnan(exp_d[k]):
                    assert np.isnan(got_d[k])
                else:
                    assert got_d[k] == pytest.approx(exp_d[k], abs=1e-5)

        # (d) Paired-opening symmetry + anchor arena distributed
        clone = AlphaZeroNet(num_res_blocks=2, num_filters=32)
        clone.load_state_dict(model.state_dict())
        m1_p1, m1_p2, m2_p1, m2_p2, draws = play_batched_arena_distributed(
            model1=model,
            model2=clone,
            device=device,
            num_games=6,
            num_simulations=4,
            temperature=0.0,
            paired_openings=True,
            seed=99,
        )
        assert m1_p1 + m1_p2 + m2_p1 + m2_p2 + draws == 6
        assert m1_p1 == m2_p1 and m1_p2 == m2_p2

        anc_res = play_anchor_arena_distributed(
            model=model,
            device=device,
            num_games=4,
            num_simulations=4,
            anchor_simulations=4,
            seed=42,
        )
        assert anc_res["model_wins"] + anc_res["anchor_wins"] + anc_res["draws"] == 4

        # (e) Sharded self-play -> gather -> bit-identical ReplayBuffer -> DDP Trainer step
        s_sp, p_sp, v_sp = run_self_play(
            model=model,
            num_games=2,
            num_simulations=4,
            return_arrays=True,
            return_uint8=True,
        )
        sg, pg, vg = gather_self_play_arrays(s_sp, p_sp, v_sp, device)
        rb = ReplayBuffer(capacity=10_000, device=device)
        rb.add_batch(sg, pg, vg)
        assert len(rb) > 0

        trainer = Trainer(
            model=model,
            device=device,
            lr=0.005,
            batch_size=8,
            total_iterations=5,
            use_compile=False,
            distributed=True,
        )
        assert trainer.is_distributed is True
        losses = trainer.train_epoch(rb, num_batches=3)
        assert losses["total_loss"] > 0.0

        # Save each rank's full state_dict + buffer slice and verify bit-identity across barrier
        torch.save(
            {
                "model": model.state_dict(),
                "rb": rb.get_state(),
                "losses": losses,
            },
            artifact_dir / f"rank_{rank}_after.pt",
        )
        if rank == 0:
            save_checkpoint(
                model=model,
                optimizer=trainer.optimizer,
                iteration=1,
                replay_buffer=None,
                path=artifact_dir / "rank0_ckpt.pt",
                scheduler=trainer.scheduler,
            )
        barrier()

        r0 = torch.load(
            artifact_dir / "rank_0_after.pt", map_location="cpu", weights_only=False
        )
        r1 = torch.load(
            artifact_dir / "rank_1_after.pt", map_location="cpu", weights_only=False
        )
        assert r0["losses"] == pytest.approx(r1["losses"], abs=1e-6)
        for key in ("states", "policies", "values"):
            assert torch.equal(r0["rb"][key], r1["rb"][key])
        for key in r0["model"]:
            assert torch.equal(r0["model"][key], r1["model"][key]), f"Mismatch on {key}"

        if rank == 1:
            fresh = AlphaZeroNet(num_res_blocks=2, num_filters=32)
            load_checkpoint(artifact_dir / "rank0_ckpt.pt", fresh, device=device)
            for k, v in fresh.state_dict().items():
                assert torch.equal(v, model.state_dict()[k])
    finally:
        cleanup_distributed()


def test_gloo_ddp_end_to_end_parity(tmp_path: Path) -> None:
    """Spawn 2 gloo DDP ranks and verify gather, Trainer sync, eval & arena exactness."""
    torch.manual_seed(101)
    init_model = AlphaZeroNet(num_res_blocks=2, num_filters=32)
    init_model.eval()
    child_wdl = np.array(
        [
            [-1, 0, 1, 1, 0, -1, -1],
            [-1, -1, -1, 0, -1, -1, -1],
            [0, 0, 1, 1, 0, 0, 0],
            [-1, 0, 0, 0, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, -1, 1, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    expected_eval0 = evaluate_dataset(
        init_model,
        torch.device("cpu"),
        _TEST_DATASET_LINES,
        num_simulations=0,
        child_wdl=child_wdl,
    )
    expected_eval4 = evaluate_dataset(
        init_model,
        torch.device("cpu"),
        _TEST_DATASET_LINES,
        num_simulations=4,
        child_wdl=child_wdl,
    )
    torch.save(
        {
            "state_dict": init_model.state_dict(),
            "child_wdl": child_wdl,
            "expected_eval0": expected_eval0,
            "expected_eval4": expected_eval4,
        },
        tmp_path / "init_model.pt",
    )

    port = find_free_port()
    mp.spawn(
        _gloo_e2e_worker,
        args=(2, port, str(tmp_path)),
        nprocs=2,
        join=True,
    )


# ── 4. Multi-GPU NCCL + torch.compile End-to-End Test ───────────────────────


def _nccl_cuda_worker(
    rank: int,
    world_size: int,
    master_port: int,
    artifact_dir_str: str,
) -> None:
    """Lightweight 2-GPU NCCL + torch.compile DDP test (`AlphaZeroNet(2, 32)`)."""
    os.environ["RAYON_NUM_THREADS"] = "4"
    artifact_dir = Path(artifact_dir_str)
    device = init_distributed_process(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        master_addr="127.0.0.1",
        master_port=master_port,
        backend="nccl",
    )
    try:
        assert device == torch.device(f"cuda:{rank}")
        seed_rank(2026, rank)

        init_ckpt = torch.load(
            artifact_dir / "init_cuda_model.pt",
            map_location=device,
            weights_only=False,
        )
        model = AlphaZeroNet(num_res_blocks=2, num_filters=32)
        model.load_state_dict(init_ckpt["state_dict"])
        model.optimize_for_device(device, use_compile=True)

        trainer = Trainer(
            model=model,
            device=device,
            lr=0.002,
            batch_size=8,
            total_iterations=5,
            use_compile=True,
            distributed=True,
        )
        assert trainer.is_distributed is True

        # Sharded self-play + NCCL gather
        s_sp, p_sp, v_sp = run_self_play(
            model=model,
            num_games=2,
            num_simulations=4,
            return_arrays=True,
            return_uint8=True,
        )
        sg, pg, vg = gather_self_play_arrays(s_sp, p_sp, v_sp, device)
        assert sg.device == device
        rb = ReplayBuffer(capacity=4096, device=device)
        rb.add_batch(sg, pg, vg)
        assert len(rb) > 0

        losses = trainer.train_epoch(rb, num_batches=3)
        assert losses["total_loss"] > 0.0

        eval_res = evaluate_dataset_distributed(
            model,
            device,
            _TEST_DATASET_LINES,
            num_simulations=4,
            child_wdl=init_ckpt["child_wdl"],
        )
        assert 0.0 <= eval_res["wdl_accuracy"] <= 100.0

        clone = AlphaZeroNet(num_res_blocks=2, num_filters=32)
        clone.load_state_dict(model.state_dict())
        clone.optimize_for_device(device, use_compile=True)
        arena_res = play_batched_arena_distributed(
            model1=model,
            model2=clone,
            device=device,
            num_games=4,
            num_simulations=4,
            temperature=0.0,
            paired_openings=True,
            seed=11,
        )
        assert sum(arena_res) == 4

        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "losses": losses,
            },
            artifact_dir / f"cuda_rank_{rank}_after.pt",
        )
        barrier()

        c0 = torch.load(
            artifact_dir / "cuda_rank_0_after.pt",
            map_location="cpu",
            weights_only=False,
        )
        c1 = torch.load(
            artifact_dir / "cuda_rank_1_after.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert c0["losses"] == pytest.approx(c1["losses"], abs=1e-5)
        for key in c0["model"]:
            assert torch.allclose(
                c0["model"][key].float(), c1["model"][key].float(), atol=1e-6, rtol=1e-6
            ), f"CUDA DDP parameter/buffer mismatch on {key}"
    finally:
        cleanup_distributed()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires 2+ CUDA GPUs",
)
def test_nccl_dual_gpu_compile_e2e(tmp_path: Path) -> None:
    """Verify real 2-GPU NCCL DDP + channels_last + torch.compile self-play, train & eval."""
    torch.manual_seed(303)
    init_model = AlphaZeroNet(num_res_blocks=2, num_filters=32)
    child_wdl = np.zeros((len(_TEST_DATASET_LINES), COLS), dtype=np.int8)
    torch.save(
        {
            "state_dict": init_model.state_dict(),
            "child_wdl": child_wdl,
        },
        tmp_path / "init_cuda_model.pt",
    )

    port = find_free_port()
    mp.spawn(
        _nccl_cuda_worker,
        args=(2, port, str(tmp_path)),
        nprocs=2,
        join=True,
    )
