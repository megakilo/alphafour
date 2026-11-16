#!/usr/bin/env python3
"""Throughput benchmark for the Cython Pure Python Mode self-play engine (`src.self_play`).

Benchmarks:
1. Mode 1 — Pure Engine CPU Throughput (Zero Neural-Net Cost):
   Drives `CythonBatchedSelfPlay` directly through the barrier-free
   `collect_continuous_u8` / `apply_continuous_packed` / `take_completed_u8`
   interface using a preallocated contiguous `float32[K, 8]` evaluation buffer
   across workloads `(32, 100)`, `(64, 200)`, and `(128, 400)`.
2. Mode 2A — End-to-End Self-Play with `AlphaZeroNet` (10 res blocks, 128 filters) on GPU:
   Evaluates `run_self_play(..., return_arrays=True, return_uint8=True)` under both
   CUDA BF16/FP16 Eager (`channels_last` + `predict_packed`) and CUDA
   `torch.compile(dynamic=True)` across `(32, 100)`, `(64, 200)`, and `(128, 400)`.
3. Mode 2B — End-to-End Self-Play with `AlphaZeroNet` on CPU.
4. Mode 3 — Compiled Cython (`.so`) vs Uncompiled Pure Python Mode (`src/self_play.py`)
   speedup comparison on identical Cython Pure Python Mode source code.

Usage:
    uv run python benchmark_selfplay.py
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cython
import numpy as np
import torch

import src.self_play as self_play_mod
from src.game import COLS
from src.mcts import DEFAULT_C_FPU, DEFAULT_DIRICHLET_ALPHA
from src.model import AlphaZeroNet
from src.self_play import (
    CYTHON_COMPILED,
    RANDOM_OPENING_PLIES,
    TEMP_THRESHOLD,
    CythonBatchedSelfPlay,
    is_cython_compiled,
    run_self_play,
)

REPO_ROOT = Path(__file__).resolve().parent

# Shared AlphaZero Connect Four self-play hyperparameters
BENCH_C_PUCT = 1.5
BENCH_DIRICHLET_ALPHA = DEFAULT_DIRICHLET_ALPHA  # 0.35
BENCH_DIRICHLET_EPSILON = 0.25
BENCH_TEMP_THRESHOLD = TEMP_THRESHOLD            # 10
BENCH_RANDOM_OPENING_PLIES = RANDOM_OPENING_PLIES  # 4
BENCH_C_FPU = DEFAULT_C_FPU                      # 0.20


@dataclass
class TrialMetrics:
    elapsed_s: float
    num_games: int
    num_simulations: int
    aug_examples: int      # 2x mirrored training positions emitted
    raw_plies: int         # aug_examples // 2 (actual game moves played)
    leaf_evals: int = 0    # exact leaf states evaluated (Mode 1)
    batches: int = 0       # number of collect/apply batches (Mode 1)

    @property
    def games_per_sec(self) -> float:
        return self.num_games / self.elapsed_s

    @property
    def positions_per_sec(self) -> float:
        """Augmented training positions/sec (2x mirrored pairs)."""
        return self.aug_examples / self.elapsed_s

    @property
    def plies_per_sec(self) -> float:
        return self.raw_plies / self.elapsed_s

    @property
    def sims_per_sec(self) -> float:
        """Effective MCTS simulations/sec (`raw_plies * num_simulations / elapsed_s`)."""
        return (self.raw_plies * self.num_simulations) / self.elapsed_s

    @property
    def leaves_per_sec(self) -> float:
        return self.leaf_evals / self.elapsed_s if self.leaf_evals > 0 else 0.0

    @property
    def us_per_batch(self) -> float:
        return (self.elapsed_s * 1e6) / self.batches if self.batches > 0 else 0.0


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _fmt_ms(mean: float, std: float, decimals: int = 1) -> str:
    return f"{mean:,.{decimals}f} ± {std:,.{decimals}f}"


def _cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "Unknown CPU"


def _sync_if_cuda(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_pure_engine_once(
    engine_cls: Callable[..., object],
    num_games: int,
    num_simulations: int,
    mock_packed: np.ndarray,
) -> TrialMetrics:
    """Drive `CythonBatchedSelfPlay` with zero neural-network overhead."""
    t0 = time.perf_counter()
    engine = engine_cls(
        num_games,
        num_simulations,
        BENCH_C_PUCT,
        BENCH_DIRICHLET_ALPHA,
        BENCH_DIRICHLET_EPSILON,
        BENCH_TEMP_THRESHOLD,
        BENCH_RANDOM_OPENING_PLIES,
        BENCH_C_FPU,
    )
    total_leaves = 0
    batches = 0
    while not engine.is_done():  # type: ignore[attr-defined]
        _s_u8, _valid, count, _fin_c = engine.collect_continuous_u8()  # type: ignore[attr-defined]
        if count > 0:
            batches += 1
            total_leaves += int(count)
            engine.apply_continuous_packed(mock_packed[:count])  # type: ignore[attr-defined]
        elif engine.is_done():  # type: ignore[attr-defined]
            break

    s_out, _p_out, _v_out = engine.take_completed_u8()  # type: ignore[attr-defined]
    elapsed = time.perf_counter() - t0
    aug_examples = int(len(s_out))
    return TrialMetrics(
        elapsed_s=elapsed,
        num_games=num_games,
        num_simulations=num_simulations,
        aug_examples=aug_examples,
        raw_plies=aug_examples // 2,
        leaf_evals=total_leaves,
        batches=batches,
    )


def _run_e2e_once(
    sp_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    model: AlphaZeroNet,
    device: torch.device,
    num_games: int,
    num_simulations: int,
) -> TrialMetrics:
    """Run one timed end-to-end `run_self_play` trial with exact CUDA synchronization."""
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    states_u8, _policies, _values = sp_fn(
        model=model,
        num_games=num_games,
        num_simulations=num_simulations,
        c_puct=BENCH_C_PUCT,
        dirichlet_alpha=BENCH_DIRICHLET_ALPHA,
        dirichlet_epsilon=BENCH_DIRICHLET_EPSILON,
        temp_threshold=BENCH_TEMP_THRESHOLD,
        random_opening_plies=BENCH_RANDOM_OPENING_PLIES,
        c_fpu=BENCH_C_FPU,
        return_arrays=True,
        return_uint8=True,
    )
    _sync_if_cuda(device)
    elapsed = time.perf_counter() - t0
    aug_examples = int(len(states_u8))
    return TrialMetrics(
        elapsed_s=elapsed,
        num_games=num_games,
        num_simulations=num_simulations,
        aug_examples=aug_examples,
        raw_plies=aug_examples // 2,
    )


def benchmark_mode1_pure_engine(
    workloads: list[tuple[int, int]],
    warmup_trials: int,
    timed_trials: int,
) -> None:
    print("=" * 100)
    print(
        "MODE 1: Pure Engine CPU Throughput (Isolated MCTS + Bitboard + Solver + Buffer Transfer, Zero NN Cost)"
    )
    print(
        "  API: collect_continuous_u8() -> preallocated float32[K, 8] -> apply_continuous_packed() -> take_completed_u8()"
    )
    print(
        f"  Trials: {warmup_trials} warmup + {timed_trials} timed | "
        "OpenMP prange | positions/s = 2x mirrored examples/s"
    )
    print("=" * 100)
    header = (
        f"{'Workload':<12} | {'Games/s':<17} | {'Positions/s (2x)':<20} | "
        f"{'MCTS Sims/s':<21} | {'us/batch':<15}"
    )
    print(header)
    print("-" * len(header))

    max_games = max(g for g, _ in workloads)
    mock_packed = np.zeros((max_games * 32 + 64, COLS + 1), dtype=np.float32)
    mock_packed[:, :COLS] = 1.0 / COLS
    mock_packed[:, COLS] = 0.0

    for num_games, num_sims in workloads:
        for _ in range(warmup_trials):
            _run_pure_engine_once(CythonBatchedSelfPlay, num_games, num_sims, mock_packed)

        trials: list[TrialMetrics] = []
        for _ in range(timed_trials):
            gc.collect()
            trials.append(
                _run_pure_engine_once(
                    CythonBatchedSelfPlay, num_games, num_sims, mock_packed
                )
            )

        gps_m, gps_s = _mean_std([x.games_per_sec for x in trials])
        pps_m, pps_s = _mean_std([x.positions_per_sec for x in trials])
        sps_m, sps_s = _mean_std([x.sims_per_sec for x in trials])
        ub_m, ub_s = _mean_std([x.us_per_batch for x in trials])
        wl_str = f"({num_games}, {num_sims})"

        print(
            f"{wl_str:<12} | {_fmt_ms(gps_m, gps_s, 1):<17} | "
            f"{_fmt_ms(pps_m, pps_s, 0):<20} | {_fmt_ms(sps_m, sps_s, 0):<21} | "
            f"{_fmt_ms(ub_m, ub_s, 1):<15}"
        )
    print("-" * len(header))
    print()


def benchmark_mode2_end_to_end(
    title: str,
    device: torch.device,
    use_compile: bool,
    workloads: list[tuple[int, int]],
    warmup_trials: int,
    timed_trials: int,
    res_blocks: int = 10,
    filters: int = 128,
) -> None:
    print("=" * 100)
    print(f"MODE 2: {title}")
    print(
        f"  Model: AlphaZeroNet(res_blocks={res_blocks}, filters={filters}) on {device} "
        f"(compile={use_compile}, return_arrays=True, return_uint8=True)"
    )
    print(
        f"  Trials: {warmup_trials} warmup + {timed_trials} timed | "
        "positions/s = 2x mirrored examples/s"
    )
    print("=" * 100)
    header = (
        f"{'Workload':<12} | {'Games/s':<17} | {'Positions/s (2x)':<20} | "
        f"{'MCTS Sims/s':<21} | {'Avg Plies':<10}"
    )
    print(header)
    print("-" * len(header))

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model = AlphaZeroNet(num_res_blocks=res_blocks, num_filters=filters)
    model.optimize_for_device(device, use_compile=use_compile)
    model.eval()

    for num_games, num_sims in workloads:
        for _ in range(warmup_trials):
            _run_e2e_once(run_self_play, model, device, num_games, num_sims)

        trials: list[TrialMetrics] = []
        for _ in range(timed_trials):
            gc.collect()
            trials.append(
                _run_e2e_once(run_self_play, model, device, num_games, num_sims)
            )

        gps_m, gps_s = _mean_std([x.games_per_sec for x in trials])
        pps_m, pps_s = _mean_std([x.positions_per_sec for x in trials])
        sps_m, sps_s = _mean_std([x.sims_per_sec for x in trials])
        avg_plies = statistics.mean([x.raw_plies / x.num_games for x in trials])
        wl_str = f"({num_games}, {num_sims})"

        print(
            f"{wl_str:<12} | {_fmt_ms(gps_m, gps_s, 2):<17} | "
            f"{_fmt_ms(pps_m, pps_s, 0):<20} | {_fmt_ms(sps_m, sps_s, 0):<21} | "
            f"{avg_plies:10.1f}"
        )
    print("-" * len(header))
    print()


def benchmark_mode3_compiled_vs_uncompiled(device: torch.device) -> None:
    """Compare compiled Cython `.so` against uncompiled Pure Python execution of `src/self_play.py`."""
    num_games, num_sims = 8, 32
    print("=" * 100)
    print(
        f"MODE 3: Compiled Cython (.so) vs Uncompiled Pure Python Mode (.py) ({num_games} games, {num_sims} sims)"
    )
    print("=" * 100)
    header = (
        f"{'Mode':<22} | {'Elapsed (s)':<12} | {'Games/s':<12} | "
        f"{'Positions/s (2x)':<18} | {'MCTS Sims/s':<16} | {'Speedup':<10}"
    )
    print(header)
    print("-" * len(header))

    src_path = REPO_ROOT / "src" / "self_play.py"
    spec = importlib.util.spec_from_file_location(
        "_self_play_uncompiled_bench", src_path
    )
    assert spec is not None and spec.loader is not None
    pure_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pure_mod
    try:
        spec.loader.exec_module(pure_mod)
        assert pure_mod.is_cython_compiled() is False
        pure_mod._disable_tqdm = lambda: True

        torch.manual_seed(42)
        model = AlphaZeroNet(num_res_blocks=4, num_filters=64)
        model.optimize_for_device(device, use_compile=False)
        model.eval()

        _run_e2e_once(pure_mod.run_self_play, model, device, 2, 8)
        _run_e2e_once(run_self_play, model, device, 2, 8)

        py_m = _run_e2e_once(pure_mod.run_self_play, model, device, num_games, num_sims)
        cy_m = _run_e2e_once(run_self_play, model, device, num_games, num_sims)

        for label, m in (
            ("Pure Python (.py)", py_m),
            ("Compiled Cython (.so)", cy_m),
        ):
            speedup = (
                m.games_per_sec / py_m.games_per_sec if py_m.games_per_sec > 0 else 0.0
            )
            print(
                f"{label:<22} | {m.elapsed_s:12.3f} | {m.games_per_sec:12.2f} | "
                f"{m.positions_per_sec:18,.0f} | {m.sims_per_sec:16,.0f} | {speedup:8.2f}x"
            )
    finally:
        sys.modules.pop(spec.name, None)

    print("-" * len(header))
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Compiled Cython (Pure Python Mode) Self-Play Engine"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of timed trials per workload in end-to-end benchmarks (default: 3)",
    )
    parser.add_argument(
        "--engine-trials",
        type=int,
        default=5,
        help="Number of timed trials per workload in Mode 1 pure engine benchmark (default: 5)",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip CUDA torch.compile benchmark section",
    )
    parser.add_argument(
        "--skip-cpu-e2e",
        action="store_true",
        help="Skip CPU AlphaZeroNet end-to-end section",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assert CYTHON_COMPILED and is_cython_compiled() is True, (
        "Cython extension is NOT compiled (`is_cython_compiled() != True`)! "
        "Build first with: uv run python setup_cython.py build_ext --inplace"
    )

    orig_disable_tqdm = self_play_mod._disable_tqdm
    self_play_mod._disable_tqdm = lambda: True

    try:
        cpu_count = os.cpu_count() or 1
        omp_env = os.environ.get("OMP_NUM_THREADS", f"auto ({cpu_count} logical CPUs)")
        has_cuda = torch.cuda.is_available()
        gpu_desc = (
            f"{torch.cuda.get_device_name(0)} (BF16={'yes' if torch.cuda.is_bf16_supported() else 'no'})"
            if has_cuda
            else "None (CPU only)"
        )

        print("=" * 100)
        print("ALPHAZERO CONNECT FOUR — CYTHON (PURE PYTHON MODE .SO) SELF-PLAY BENCHMARK")
        print("=" * 100)
        print(f"  Python         : {platform.python_version()} ({platform.platform()})")
        print(f"  Cython         : {cython.__version__} (is_cython_compiled={is_cython_compiled()})")
        print(f"  PyTorch / NumPy: {torch.__version__} / {np.__version__}")
        print(f"  CPU            : {_cpu_model_name()} ({cpu_count} logical cores, OMP_NUM_THREADS={omp_env})")
        print(f"  GPU (cuda:0)   : {gpu_desc}")
        print(
            f"  Hyperparameters: c_puct={BENCH_C_PUCT}, dirichlet=({BENCH_DIRICHLET_ALPHA}, {BENCH_DIRICHLET_EPSILON}), "
            f"temp_threshold={BENCH_TEMP_THRESHOLD}, opening_plies={BENCH_RANDOM_OPENING_PLIES}, c_fpu={BENCH_C_FPU}"
        )
        print()

        benchmark_mode1_pure_engine(
            workloads=[(32, 100), (64, 200), (128, 400)],
            warmup_trials=1,
            timed_trials=args.engine_trials,
        )

        if has_cuda:
            cuda_dev = torch.device("cuda:0")
            benchmark_mode2_end_to_end(
                title="End-to-End Self-Play on GPU — Eager BF16/FP16 + channels_last + predict_packed",
                device=cuda_dev,
                use_compile=False,
                workloads=[(32, 100), (64, 200), (128, 400)],
                warmup_trials=1,
                timed_trials=args.trials,
            )
            if not args.skip_compile:
                benchmark_mode2_end_to_end(
                    title="End-to-End Self-Play on GPU — torch.compile(dynamic=True) + BF16/FP16 + predict_packed",
                    device=cuda_dev,
                    use_compile=True,
                    workloads=[(32, 100), (64, 200), (128, 400)],
                    warmup_trials=2,
                    timed_trials=args.trials,
                )

        if not args.skip_cpu_e2e:
            benchmark_mode2_end_to_end(
                title="End-to-End Self-Play on CPU — AlphaZeroNet(4, 64) FP32",
                device=torch.device("cpu"),
                use_compile=False,
                workloads=[(8, 50), (16, 100)],
                warmup_trials=1,
                timed_trials=args.trials,
                res_blocks=4,
                filters=64,
            )

        ref_dev = torch.device("cuda:0") if has_cuda else torch.device("cpu")
        benchmark_mode3_compiled_vs_uncompiled(device=ref_dev)

    finally:
        self_play_mod._disable_tqdm = orig_disable_tqdm


if __name__ == "__main__":
    main()
