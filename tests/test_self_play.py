"""Comprehensive unit, tactical solver, and parity tests for the Cython self-play engine (`src/self_play.py`).

Verifies:
- Strict Cython Pure Python Mode (`src/self_play.py`, zero `.pyx`/`.pxd` files)
- Compiled `.so` execution (`is_cython_compiled() is True`) AND uncompiled `.py` execution
- `inspect.signature(run_self_play)` introspection compatibility on compiled Cython extension
- Shapes, C-contiguity, dtypes (`uint8` and `float32`), `return_arrays=True/False`
- Valid probability distributions (`sum == 1.0`, zero on full columns) & value set `{-1, 0, +1}`
- Exact horizontal mirror pair symmetry & zero double-augmentation in `run_self_play`
- `num_games=0` handling across all return formats
- Opening ply distribution (`50% 0, 25% 1, 25% 2..=K`) & soft policy target decoupling
- Exact MCTS-Solver tactical Mate-in-1 win, Mate-in-2 fork, and Mate-in-1 forced block
- Deterministic & statistical parity between compiled Cython (`.so`), uncompiled Pure Python (`.py`), and `src.mcts.MCTS`
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import torch

from src.game import COLS, ROWS, ConnectFour
from src.mcts import MCTS
from src.model import AlphaZeroNet
from src.self_play import (
    CYTHON_COMPILED,
    BatchedSelfPlay,
    CythonBatchedSelfPlay,
    is_cython_compiled,
    play_batched_games,
    run_self_play,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class UniformDummyModel:
    """Fast CPU dummy model for deterministic/fast self-play tests."""

    def __init__(
        self,
        priors: list[float] | np.ndarray | None = None,
        value: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self._param = torch.zeros(1, device=device)
        if priors is None:
            self.priors = np.full(COLS, 1.0 / COLS, dtype=np.float32)
        else:
            arr = np.asarray(priors, dtype=np.float32)
            self.priors = arr / arr.sum()
        self.value = float(value)

    def parameters(self):
        return iter([self._param])

    def eval(self):
        return self

    def predict(
        self, state: torch.Tensor, valid_moves: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single = state.dim() == 3
        state_4d = state.unsqueeze(0) if single else state
        batch = state_4d.shape[0]
        pol = (
            torch.tensor(self.priors, dtype=torch.float32, device=state.device)
            .unsqueeze(0)
            .expand(batch, -1)
            .clone()
        )
        if valid_moves is not None:
            vm = valid_moves.unsqueeze(0) if single else valid_moves
            pol = pol.masked_fill(~vm, 0.0)
            sums = pol.sum(dim=1, keepdim=True).clamp(min=1e-8)
            pol = pol / sums
        val = torch.full((batch,), self.value, dtype=torch.float32, device=state.device)
        if single:
            return pol.squeeze(0), val.squeeze(0)
        return pol, val


# ── 1. Compilation & Pure Python Mode Verification ──────────────────────────


class TestCythonPurePythonMode:
    def test_compiled_extension_is_active_and_no_pyx_or_pxd_files_exist(self) -> None:
        assert CYTHON_COMPILED is True
        assert is_cython_compiled() is True
        assert BatchedSelfPlay is CythonBatchedSelfPlay

        sig_params = inspect.signature(run_self_play).parameters
        assert "return_arrays" in sig_params
        assert "return_uint8" in sig_params
        assert "backend" not in sig_params

        source_file = REPO_ROOT / "src" / "self_play.py"
        assert source_file.is_file(), "Missing Pure Python Mode source src/self_play.py"
        assert not (REPO_ROOT / "src" / "cython_self_play.py").exists()
        assert not (REPO_ROOT / "rust").exists()

        pyx_files = [
            p
            for p in REPO_ROOT.rglob("*.pyx")
            if ".venv" not in p.parts and "build" not in p.parts
        ]
        pxd_files = [
            p
            for p in REPO_ROOT.rglob("*.pxd")
            if ".venv" not in p.parts and "build" not in p.parts
        ]
        assert pyx_files == [], f"Disallowed .pyx files found: {pyx_files}"
        assert pxd_files == [], f"Disallowed .pxd files found: {pxd_files}"

    def test_uncompiled_pure_python_execution_via_source_loader(self) -> None:
        """Load `src/self_play.py` directly via SourceFileLoader and run both engine & `run_self_play`."""
        src_path = REPO_ROOT / "src" / "self_play.py"
        spec = importlib.util.spec_from_file_location(
            "_self_play_uncompiled_test", src_path
        )
        assert spec is not None and spec.loader is not None
        pure_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pure_mod
        try:
            spec.loader.exec_module(pure_mod)
            assert pure_mod.is_cython_compiled() is False
            assert pure_mod.CYTHON_COMPILED is False

            engine = pure_mod.CythonBatchedSelfPlay(
                2,
                6,
                1.5,
                0.35,
                0.25,
                10,
                2,
                0.20,
            )
            assert engine.num_active() == 2
            assert not engine.is_done()

            finished = 0
            steps = 0
            while not engine.is_done() and steps < 500:
                s_u8, valid, count, fin_c = engine.collect_continuous_u8()
                finished += fin_c
                if count > 0:
                    assert s_u8.shape == (count, 3, ROWS, COLS) and s_u8.dtype == np.uint8
                    assert valid.shape == (count, COLS) and valid.dtype == bool
                    packed = np.zeros((count, COLS + 1), dtype=np.float32)
                    packed[:, :COLS] = np.where(valid, 1.0, 0.0)
                    row_sums = np.maximum(packed[:, :COLS].sum(axis=1, keepdims=True), 1e-8)
                    packed[:, :COLS] /= row_sums
                    finished += engine.apply_continuous_packed(packed)
                steps += 1

            assert engine.is_done()
            assert finished == 2
            st_u8, pol, val = engine.take_completed_u8()
            assert len(st_u8) > 0 and len(st_u8) % 2 == 0
            assert st_u8.dtype == np.uint8
            assert pol.dtype == np.float32
            assert val.dtype == np.float32
            npt.assert_allclose(pol.sum(axis=1), 1.0, atol=1e-5)

            # Also verify high-level `run_self_play` works end-to-end in uncompiled Pure Python mode
            dummy = UniformDummyModel()
            s_sp, p_sp, v_sp = pure_mod.run_self_play(
                model=dummy,
                num_games=2,
                num_simulations=4,
                random_opening_plies=0,
                return_arrays=True,
                return_uint8=True,
            )
            assert len(s_sp) > 0 and len(s_sp) % 2 == 0
            assert s_sp.dtype == np.uint8
            npt.assert_allclose(p_sp.sum(axis=1), 1.0, atol=1e-5)
            assert np.all(np.isin(v_sp, np.array([-1.0, 0.0, 1.0], dtype=np.float32)))
        finally:
            sys.modules.pop(spec.name, None)


# ── 2. Output Shapes, Dtypes, Symmetry, Full-Column Masking & num_games=0 ───


class TestCythonShapesDtypesAndSymmetry:
    def test_num_games_zero_across_all_return_modes(self) -> None:
        dummy = UniformDummyModel()

        ex_list = run_self_play(
            model=dummy,
            num_games=0,
            num_simulations=8,
            return_arrays=False,
        )
        assert ex_list == []

        s_f32, p_f32, v_f32 = run_self_play(
            model=dummy,
            num_games=0,
            num_simulations=8,
            return_arrays=True,
            return_uint8=False,
        )
        assert s_f32.shape == (0, 3, ROWS, COLS) and s_f32.dtype == np.float32
        assert p_f32.shape == (0, COLS) and p_f32.dtype == np.float32
        assert v_f32.shape == (0,) and v_f32.dtype == np.float32

        s_u8, p_u8, v_u8 = run_self_play(
            model=dummy,
            num_games=0,
            num_simulations=8,
            return_arrays=True,
            return_uint8=True,
        )
        assert s_u8.shape == (0, 3, ROWS, COLS) and s_u8.dtype == np.uint8
        assert p_u8.shape == (0, COLS) and p_u8.dtype == np.float32
        assert v_u8.shape == (0,) and v_u8.dtype == np.float32

    def test_direct_engine_continuous_api_and_buffer_drain(self) -> None:
        engine = CythonBatchedSelfPlay(6, 16, 1.5, 0.35, 0.25, 10, 4, 0.20)
        assert engine.num_active() == 6
        assert not engine.is_done()

        finished = 0
        steps = 0
        while not engine.is_done() and steps < 2000:
            s_u8, valid, count, fin_c = engine.collect_continuous_u8()
            finished += fin_c
            if count > 0:
                assert s_u8.shape == (count, 3, ROWS, COLS)
                assert s_u8.dtype == np.uint8 and s_u8.flags["C_CONTIGUOUS"]
                assert valid.shape == (count, COLS)
                assert valid.dtype == bool and valid.flags["C_CONTIGUOUS"]
                packed = np.zeros((count, COLS + 1), dtype=np.float32)
                packed[:, :COLS] = 1.0 / COLS
                finished += engine.apply_continuous_packed(packed)
            steps += 1

        assert engine.is_done()
        assert engine.num_active() == 0
        assert finished == 6

        st_u8, pol, val = engine.take_completed_u8()
        assert len(st_u8) > 0 and len(st_u8) % 2 == 0
        assert st_u8.dtype == np.uint8 and st_u8.flags["C_CONTIGUOUS"]
        assert pol.dtype == np.float32 and pol.flags["C_CONTIGUOUS"]
        assert val.dtype == np.float32 and val.flags["C_CONTIGUOUS"]

        # Subsequent drain must be empty
        st_empty, pol_empty, val_empty = engine.take_completed_u8()
        assert len(st_empty) == 0 and len(pol_empty) == 0 and len(val_empty) == 0

    def test_return_arrays_f32_u8_list_and_mirror_symmetry(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(123)
        model = AlphaZeroNet(num_res_blocks=1, num_filters=16).to(device)
        model.eval()

        # 1. uint8 arrays
        s_u8, pol_u8, val_u8 = run_self_play(
            model=model,
            num_games=10,
            num_simulations=12,
            random_opening_plies=4,
            return_arrays=True,
            return_uint8=True,
        )
        assert s_u8.ndim == 4 and s_u8.shape[1:] == (3, ROWS, COLS)
        assert s_u8.dtype == np.uint8 and s_u8.flags["C_CONTIGUOUS"]
        assert pol_u8.shape == (len(s_u8), COLS) and pol_u8.dtype == np.float32
        assert val_u8.shape == (len(s_u8),) and val_u8.dtype == np.float32
        assert np.all((s_u8 == 0) | (s_u8 == 1))
        assert len(s_u8) > 0 and len(s_u8) % 2 == 0

        # Valid probability distributions & zero probability on full columns
        assert np.all(pol_u8 >= 0.0)
        npt.assert_allclose(pol_u8.sum(axis=1), 1.0, atol=1e-5)
        col_heights = (s_u8[:, 0] + s_u8[:, 1]).sum(axis=1)  # shape (N, 7)
        full_col_mask = col_heights == ROWS
        if np.any(full_col_mask):
            npt.assert_array_equal(pol_u8[full_col_mask], 0.0)

        # Value range strictly in {-1.0, 0.0, +1.0}
        assert np.all(np.isin(val_u8, np.array([-1.0, 0.0, 1.0], dtype=np.float32)))

        # Exact horizontal mirror pair symmetry on every consecutive pair
        npt.assert_array_equal(s_u8[1::2], s_u8[0::2, :, :, ::-1])
        npt.assert_allclose(pol_u8[1::2], pol_u8[0::2, ::-1], atol=1e-6)
        npt.assert_array_equal(val_u8[1::2], val_u8[0::2])

        # 2. float32 arrays
        s_f32, pol_f32, val_f32 = run_self_play(
            model=model,
            num_games=4,
            num_simulations=8,
            return_arrays=True,
            return_uint8=False,
        )
        assert s_f32.dtype == np.float32 and s_f32.flags["C_CONTIGUOUS"]
        assert np.all((s_f32 == 0.0) | (s_f32 == 1.0))
        npt.assert_allclose(pol_f32.sum(axis=1), 1.0, atol=1e-5)

        # 3. list[SelfPlayExample]
        examples = run_self_play(
            model=model,
            num_games=3,
            num_simulations=8,
            return_arrays=False,
        )
        assert isinstance(examples, list) and len(examples) > 0 and len(examples) % 2 == 0
        st0, p0, v0 = examples[0]
        st1, p1, v1 = examples[1]
        assert st0.shape == (3, ROWS, COLS) and st0.dtype == np.float32
        assert p0.shape == (COLS,) and p0.dtype == np.float32
        assert isinstance(v0, float) and v0 in (-1.0, 0.0, 1.0)
        npt.assert_array_equal(st1, st0[:, :, ::-1])
        npt.assert_allclose(p1, p0[::-1], atol=1e-6)
        assert v1 == v0

    def test_zero_prob_on_explicitly_full_column(self) -> None:
        """Force column 3 to fill completely (6 stones) and verify policy[:, 3] == 0.0 afterwards."""
        peaked_col3 = UniformDummyModel(
            priors=[0.001, 0.001, 0.01, 0.976, 0.01, 0.001, 0.001]
        )
        s_u8, pol, _ = run_self_play(
            model=peaked_col3,
            num_games=2,
            num_simulations=24,
            dirichlet_epsilon=0.0,
            temp_threshold=0,
            random_opening_plies=0,
            return_arrays=True,
            return_uint8=True,
        )
        col_heights = (s_u8[:, 0] + s_u8[:, 1]).sum(axis=1)
        full_mask = col_heights == ROWS
        assert np.any(full_mask), "Expected column 3 to become full during game"
        npt.assert_array_equal(pol[full_mask], 0.0)
        npt.assert_allclose(pol.sum(axis=1), 1.0, atol=1e-5)

    def test_run_self_play_does_not_double_augment(self) -> None:
        """Verify `run_self_play` emits 2x (NOT 4x) examples per game."""
        dummy = UniformDummyModel()
        s_cy, _, _ = run_self_play(
            model=dummy,
            num_games=6,
            num_simulations=10,
            random_opening_plies=0,
            return_arrays=True,
            return_uint8=True,
        )
        # Move-0 empty boards appear once per game -> exactly 2 * 6 = 12 empty-board examples (NOT 24!).
        piece_counts = (s_cy[:, 0] + s_cy[:, 1]).sum(axis=(1, 2))
        assert int(np.sum(piece_counts == 0)) == 12, (
            f"Expected exactly 2 * 6 = 12 Move-0 examples (no double augmentation), got {int(np.sum(piece_counts == 0))}"
        )


# ── 3. Openings, Policy Decoupling & Tactical MCTS-Solver Sanity ────────────


class TestOpeningsAndTacticalSolver:
    def test_opening_ply_distribution_and_soft_policy_targets(self) -> None:
        dummy = UniformDummyModel()
        s_u8, _, _ = run_self_play(
            model=dummy,
            num_games=160,
            num_simulations=6,
            random_opening_plies=4,
            return_arrays=True,
            return_uint8=True,
        )
        piece_counts = (s_u8[:, 0] + s_u8[:, 1]).sum(axis=(1, 2))
        n_move0_games = int(np.sum(piece_counts == 0)) // 2
        assert 50 <= n_move0_games <= 115, (
            f"Expected ~50% of 160 games to start at Move 0, got {n_move0_games}"
        )
        assert np.any(piece_counts == 1), "Expected Move 1 examples in dataset"

        # Soft policy target decoupling when temp_threshold == 0
        _, pol_greedy, _ = play_batched_games(
            model=dummy,
            num_games=2,
            num_simulations=28,
            temp_threshold=0,
            random_opening_plies=0,
            return_arrays=True,
        )
        assert pol_greedy[0].max() < 0.75
        assert np.count_nonzero(pol_greedy[0] > 0.05) >= 4

    def test_tactical_mate_in_1_win_and_mate_in_1_block_solver(self) -> None:
        """Verify exact MCTS-Solver finds tactical wins (Mate-in-2 -> Mate-in-1) and forced blocks."""
        eng_win = CythonBatchedSelfPlay(
            1,
            24,
            1.5,
            0.35,
            0.0,  # dirichlet_epsilon=0.0
            0,    # temp_threshold=0 (greedy)
            0,    # random_opening_plies=0
            0.20,
        )
        while not eng_win.is_done():
            s_u8, valid, count, _ = eng_win.collect_continuous_u8()
            if count > 0:
                packed = np.zeros((count, COLS + 1), dtype=np.float32)
                for k in range(count):
                    pieces = int((s_u8[k, 0] + s_u8[k, 1]).sum())
                    if pieces < 4:
                        target = {0: 2, 1: 6, 2: 3, 3: 6}[pieces]
                        packed[k, target] = 1.0
                        packed[k, COLS] = -0.95
                    elif pieces == 4:
                        packed[k, :COLS] = [0.05, 0.15, 0.05, 0.05, 0.15, 0.05, 0.50]
                        packed[k, COLS] = -0.95
                    elif pieces == 5:
                        packed[k, :COLS] = 1.0 / COLS
                    else:
                        packed[k, :COLS] = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.994]
                    packed[k, :COLS] *= valid[k]
                    packed[k, :COLS] /= max(float(packed[k, :COLS].sum()), 1e-8)
                eng_win.apply_continuous_packed(packed)

        st_w, pol_w, val_w = eng_win.take_completed_u8()
        assert len(st_w) == 14, f"Expected 7-ply (14-example) tactical win, got {len(st_w)}"
        assert float(pol_w[8, 1] + pol_w[8, 4]) == pytest.approx(1.0, abs=1e-5)
        npt.assert_allclose(
            pol_w[12],
            np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32),
            atol=1e-5,
        )
        assert val_w[12] == pytest.approx(1.0)

        # Case B: Mate-in-1 Forced Block (Defense)
        eng_blk = CythonBatchedSelfPlay(1, 48, 1.5, 0.35, 0.0, 0, 0, 0.20)
        while not eng_blk.is_done():
            s_u8, valid, count, _ = eng_blk.collect_continuous_u8()
            if count > 0:
                packed = np.zeros((count, COLS + 1), dtype=np.float32)
                for k in range(count):
                    pieces = int((s_u8[k, 0] + s_u8[k, 1]).sum())
                    if pieces == 0 or pieces == 2 or pieces == 4:
                        packed[k, :COLS] = 0.001
                        packed[k, 0] = 0.994
                    elif pieces == 1:
                        packed[k, :COLS] = 0.001
                        packed[k, 6] = 0.994
                    elif pieces == 3:
                        packed[k, :COLS] = 0.001
                        packed[k, 5] = 0.994
                    elif pieces == 5:
                        packed[k, :COLS] = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.994]
                    else:
                        packed[k, :COLS] = 1.0 / COLS
                    packed[k, :COLS] *= valid[k]
                    packed[k, :COLS] /= max(float(packed[k, :COLS].sum()), 1e-8)
                eng_blk.apply_continuous_packed(packed)

        st_b, pol_b, _ = eng_blk.take_completed_u8()
        assert len(st_b) > 14, "P2 failed to block P1's Mate-in-1 threat on ply 5!"
        assert int(np.argmax(pol_b[10])) == 0
        assert pol_b[10, 0] == pytest.approx(1.0, abs=1e-5)


# ── 4. Parity: Compiled Cython (.so) vs Uncompiled (.py) & Reference MCTS ───


class TestCompiledVsUncompiledAndReferenceParity:
    def test_deterministic_and_statistical_parity_vs_uncompiled_and_python_mcts(self) -> None:
        """Compare Move-0 policies and full-game distributions across compiled .so, uncompiled .py & MCTS."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(2026)
        model = AlphaZeroNet(num_res_blocks=2, num_filters=32).to(device)
        model.eval()

        # Load uncompiled Pure Python module
        src_path = REPO_ROOT / "src" / "self_play.py"
        spec = importlib.util.spec_from_file_location(
            "_self_play_parity_pure", src_path
        )
        assert spec is not None and spec.loader is not None
        pure_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pure_mod
        try:
            spec.loader.exec_module(pure_mod)
            assert pure_mod.is_cython_compiled() is False

            # 1. Exact deterministic parity (`np.random.seed`) between compiled .so and uncompiled .py
            det_kwargs = dict(
                model=model,
                num_games=2,
                num_simulations=48,
                c_puct=1.5,
                dirichlet_alpha=0.35,
                dirichlet_epsilon=0.0,
                temp_threshold=0,
                random_opening_plies=0,
                c_fpu=0.20,
                return_arrays=True,
                return_uint8=True,
            )
            np.random.seed(777)
            s_cy, p_cy, v_cy = run_self_play(**det_kwargs)
            np.random.seed(777)
            s_un, p_un, v_un = pure_mod.run_self_play(**det_kwargs)

            npt.assert_array_equal(s_cy, s_un)
            npt.assert_allclose(p_cy, p_un, atol=1e-5)
            npt.assert_array_equal(v_cy, v_un)

            # Compare Move-0 policy against reference single-leaf Python MCTS (`src/mcts.py`)
            pieces = (s_cy[:, 0] + s_cy[:, 1]).sum(axis=(1, 2))
            pol0_cy = p_cy[int(np.flatnonzero(pieces == 0)[0])]

            ref_mcts = MCTS(
                model=model, num_simulations=48, c_puct=1.5, c_fpu=0.20, device=device
            )
            vc_ref = ref_mcts.search(ConnectFour(), add_noise=False)
            pol0_ref = vc_ref / vc_ref.sum()
            l1_cy_ref = float(np.abs(pol0_cy - pol0_ref).sum())
            assert l1_cy_ref < 0.35, (
                f"Move-0 policy diverged between Cython {pol0_cy} and reference MCTS {pol0_ref} (L1={l1_cy_ref:.3f})"
            )

            # 2. Stochastic self-play sanity across 24 compiled games vs 8 uncompiled games
            stoch_kwargs = dict(
                model=model,
                num_simulations=20,
                c_puct=1.5,
                dirichlet_alpha=0.35,
                dirichlet_epsilon=0.25,
                temp_threshold=10,
                random_opening_plies=4,
                c_fpu=0.20,
                return_arrays=True,
                return_uint8=True,
            )
            s_cy_s, p_cy_s, v_cy_s = run_self_play(num_games=24, **stoch_kwargs)
            s_un_s, p_un_s, v_un_s = pure_mod.run_self_play(num_games=8, **stoch_kwargs)

            cy_plies = (len(s_cy_s) / 2.0) / 24.0
            un_plies = (len(s_un_s) / 2.0) / 8.0
            assert 7.0 <= cy_plies <= 42.0
            assert abs(cy_plies - un_plies) < 14.0
            assert np.all(np.isin(v_cy_s, np.array([-1.0, 0.0, 1.0], dtype=np.float32)))
            npt.assert_allclose(p_cy_s.sum(axis=1), 1.0, atol=1e-5)
        finally:
            sys.modules.pop(spec.name, None)
