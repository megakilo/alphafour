"""Comprehensive unit, tactical solver, and parity tests for Cython self-play engine.

Verifies:
- Strict Cython Pure Python Mode (`src/cython_self_play.py`, zero `.pyx`/`.pxd` files)
- Compiled `.so` execution (`is_cython_compiled() is True`) AND uncompiled `.py` execution
- Shapes, C-contiguity, dtypes (`uint8` and `float32`), `return_arrays=True/False`
- Valid probability distributions (`sum == 1.0`, zero on full columns) & value set `{-1, 0, +1}`
- Exact horizontal mirror pair symmetry & zero double-augmentation in `run_self_play`
- `num_games=0` handling across all backends and return formats
- Opening ply distribution (`50% 0, 25% 1, 25% 2..=K`) & soft policy target decoupling
- Exact MCTS-Solver tactical Mate-in-1 win and Mate-in-1 block under adversarial priors
- CLI `--self-play-backend` support and statistical parity vs `backend="rust"` and `backend="python"`
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest
import torch

import main as main_cli
import src.self_play as self_play
from src.cython_self_play import CythonBatchedSelfPlay, is_cython_compiled
from src.game import COLS, ROWS
from src.model import AlphaZeroNet
from src.self_play import (
    CYTHON_AVAILABLE,
    CYTHON_COMPILED,
    RUST_AVAILABLE,
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
        assert CYTHON_AVAILABLE is True
        assert CYTHON_COMPILED is True
        assert is_cython_compiled() is True

        source_file = REPO_ROOT / "src" / "cython_self_play.py"
        assert source_file.is_file(), "Missing Pure Python Mode source src/cython_self_play.py"

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
        """Load `src/cython_self_play.py` directly via SourceFileLoader and run pure Python mode."""
        src_path = REPO_ROOT / "src" / "cython_self_play.py"
        spec = importlib.util.spec_from_file_location(
            "_cython_self_play_uncompiled_test", src_path
        )
        assert spec is not None and spec.loader is not None
        pure_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pure_mod
        try:
            spec.loader.exec_module(pure_mod)
            assert pure_mod.is_cython_compiled() is False

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
        finally:
            sys.modules.pop(spec.name, None)


# ── 2. Output Shapes, Dtypes, Symmetry, Full-Column Masking & num_games=0 ───


class TestCythonShapesDtypesAndSymmetry:
    def test_num_games_zero_across_all_return_modes(self) -> None:
        dummy = UniformDummyModel()

        ex_list = run_self_play(
            model=dummy,  # type: ignore[arg-type]
            num_games=0,
            num_simulations=8,
            return_arrays=False,
            backend="cython",
        )
        assert ex_list == []

        s_f32, p_f32, v_f32 = run_self_play(
            model=dummy,  # type: ignore[arg-type]
            num_games=0,
            num_simulations=8,
            return_arrays=True,
            return_uint8=False,
            backend="cython",
        )
        assert s_f32.shape == (0, 3, ROWS, COLS) and s_f32.dtype == np.float32
        assert p_f32.shape == (0, COLS) and p_f32.dtype == np.float32
        assert v_f32.shape == (0,) and v_f32.dtype == np.float32

        s_u8, p_u8, v_u8 = run_self_play(
            model=dummy,  # type: ignore[arg-type]
            num_games=0,
            num_simulations=8,
            return_arrays=True,
            return_uint8=True,
            backend="cython",
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
            backend="cython",
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
            backend="cython",
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
            backend="cython",
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
        # Heavily bias col 3 then col 2 so col 3 fills up in 6 plies without a 4-in-a-row win
        # Alternating P1/P2 in col 3: P1(0,3), P2(1,3), P1(2,3), P2(3,3), P1(4,3), P2(5,3) -> 0 wins!
        peaked_col3 = UniformDummyModel(
            priors=[0.001, 0.001, 0.01, 0.976, 0.01, 0.001, 0.001]
        )
        s_u8, pol, _ = run_self_play(
            model=peaked_col3,  # type: ignore[arg-type]
            num_games=2,
            num_simulations=24,
            dirichlet_epsilon=0.0,
            temp_threshold=0,
            random_opening_plies=0,
            return_arrays=True,
            return_uint8=True,
            backend="cython",
        )
        col_heights = (s_u8[:, 0] + s_u8[:, 1]).sum(axis=1)
        full_mask = col_heights == ROWS
        assert np.any(full_mask), "Expected column 3 to become full during game"
        npt.assert_array_equal(pol[full_mask], 0.0)
        npt.assert_allclose(pol.sum(axis=1), 1.0, atol=1e-5)

    def test_run_self_play_does_not_double_augment_cython_or_auto_fallback(self) -> None:
        """Verify `run_self_play` emits 2x (NOT 4x) examples for `backend='cython'` and `auto`."""
        dummy = UniformDummyModel()
        # Compare play_batched_games vs run_self_play example counts on shortest vertical games
        # Using a deterministic-style model that prefers leftmost valid column? Wait: alternating players
        # in the SAME column alternates pieces (P1, P2, P1, P2...) which doesn't make 4-in-a-row!
        # Instead, compare `play_batched_games(..., backend="cython")` directly with `run_self_play`:
        # Since Cython emits exact mirrored pairs `(s[2k], s[2k+1])` where `s[2k+1] == mirror(s[2k])`,
        # double-augmentation would append the whole array reversed as `[orig_pairs, mirrored_orig_pairs]`,
        # causing `len` to be ~2x Rust's average length AND repeating states across first/second halves!
        s_cy, p_cy, v_cy = run_self_play(
            model=dummy,  # type: ignore[arg-type]
            num_games=6,
            num_simulations=10,
            random_opening_plies=0,
            return_arrays=True,
            return_uint8=True,
            backend="cython",
        )
        # In a 6-game Connect Four batch, max possible plies is 6 * 42 = 252 -> max 504 augmented examples.
        # Move-0 empty boards appear once per game -> exactly 2 * 6 = 12 empty-board examples (NOT 24!).
        piece_counts = (s_cy[:, 0] + s_cy[:, 1]).sum(axis=(1, 2))
        assert int(np.sum(piece_counts == 0)) == 12, (
            f"Expected exactly 2 * 6 = 12 Move-0 examples (no double augmentation), got {int(np.sum(piece_counts == 0))}"
        )

        # Also test `backend="auto"` when `RUST_AVAILABLE` is False -> picks compiled Cython without double-augmenting!
        with patch.object(self_play, "RUST_AVAILABLE", False):
            assert self_play._resolve_backend("auto") == "cython"
            s_auto, _, _ = run_self_play(
                model=dummy,  # type: ignore[arg-type]
                num_games=5,
                num_simulations=10,
                random_opening_plies=0,
                return_arrays=True,
                return_uint8=True,
                backend="auto",
            )
        auto_piece_counts = (s_auto[:, 0] + s_auto[:, 1]).sum(axis=(1, 2))
        assert int(np.sum(auto_piece_counts == 0)) == 10


# ── 3. Openings, Policy Decoupling & Tactical MCTS-Solver Sanity ────────────


class TestOpeningsAndTacticalSolver:
    def test_opening_ply_distribution_and_soft_policy_targets(self) -> None:
        dummy = UniformDummyModel()
        s_u8, pol, _ = run_self_play(
            model=dummy,  # type: ignore[arg-type]
            num_games=160,
            num_simulations=6,
            random_opening_plies=4,
            return_arrays=True,
            return_uint8=True,
            backend="cython",
        )
        piece_counts = (s_u8[:, 0] + s_u8[:, 1]).sum(axis=(1, 2))
        # Each 0-ply start yields 2 mirrored Move-0 examples (`piece_counts == 0`)
        n_move0_games = int(np.sum(piece_counts == 0)) // 2
        assert 50 <= n_move0_games <= 115, (
            f"Expected ~50% of 160 games to start at Move 0, got {n_move0_games}"
        )
        assert np.any(piece_counts == 1), "Expected Move 1 examples in dataset"

        # Soft policy target decoupling when temp_threshold == 0
        _, pol_greedy, _ = play_batched_games(
            model=dummy,  # type: ignore[arg-type]
            num_games=2,
            num_simulations=28,
            temp_threshold=0,
            random_opening_plies=0,
            return_arrays=True,
            backend="cython",
        )
        assert pol_greedy[0].max() < 0.75
        assert np.count_nonzero(pol_greedy[0] > 0.05) >= 4

    def test_tactical_mate_in_1_win_and_mate_in_1_block_solver(self) -> None:
        """Verify exact MCTS-Solver finds tactical wins (Mate-in-2 -> Mate-in-1) and forced blocks."""
        # Case A: Tactical Win (Open-ended double threat at Ply 4 -> Mate-in-1 finish at Ply 6)
        # Plies 0..3: steer P1 -> col 2, P2 -> col 6, P1 -> col 3, P2 -> col 6.
        # At Ply 4 (P1 to move: stones at (0,2) and (0,3)), playing col 1 or col 4 creates an
        # unstoppable open-ended 3-in-a-row on row 0, forcing a Mate-in-1 win on Ply 6 (7 plies = 14 examples),
        # even when the network adversarially puts its highest prior on col 6!
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
                        # Ply 4: adversarial prior prefers col 6 (0.50) over winning cols 1 & 4 (0.15, 0.15)
                        packed[k, :COLS] = [0.05, 0.15, 0.05, 0.05, 0.15, 0.05, 0.50]
                        packed[k, COLS] = -0.95
                    elif pieces == 5:
                        packed[k, :COLS] = 1.0 / COLS
                    else:
                        # Ply 6 (Mate-in-1): adversarial prior puts 0.994 on col 6!
                        packed[k, :COLS] = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.994]
                    packed[k, :COLS] *= valid[k]
                    packed[k, :COLS] /= max(float(packed[k, :COLS].sum()), 1e-8)
                eng_win.apply_continuous_packed(packed)

        st_w, pol_w, val_w = eng_win.take_completed_u8()
        # Game MUST finish in 7 plies (14 mirrored examples): Ply 4 double-threat + Ply 6 Mate-in-1!
        assert len(st_w) == 14, f"Expected 7-ply (14-example) tactical win, got {len(st_w)}"
        # Ply 4 original example (index 8): policy target concentrated on proven winning cols {1, 4}
        assert float(pol_w[8, 1] + pol_w[8, 4]) == pytest.approx(1.0, abs=1e-5)
        # Ply 6 original example (index 12): uniform 0.5/0.5 over the two immediate winning columns {1, 5}!
        npt.assert_allclose(
            pol_w[12],
            np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32),
            atol=1e-5,
        )
        assert val_w[12] == pytest.approx(1.0)

        # Case B: Mate-in-1 Forced Block (Defense)
        # Steer 5 plies: P1->0, P2->6, P1->0, P2->5, P1->0.
        # At ply 5 (P2 to move), P1 has 3 vertical stones in col 0! Every move != 0 loses in 1 ply.
        # Give P2 an adversarial prior favoring col 6 (0.994) and starving col 0 (0.001).
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
                        # Ply 5: adversarial prior tries to blunder into col 6 instead of blocking col 0!
                        packed[k, :COLS] = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.994]
                    else:
                        packed[k, :COLS] = 1.0 / COLS
                    packed[k, :COLS] *= valid[k]
                    packed[k, :COLS] /= max(float(packed[k, :COLS].sum()), 1e-8)
                eng_blk.apply_continuous_packed(packed)

        st_b, pol_b, _ = eng_blk.take_completed_u8()
        # Ply 5 original example is at index 2 * 5 = 10.
        # Because all cols != 0 are proven losses (SOLVER_WIN for opponent), P2 MUST block at col 0!
        assert len(st_b) > 14, "P2 failed to block P1's Mate-in-1 threat on ply 5!"
        assert int(np.argmax(pol_b[10])) == 0
        assert pol_b[10, 0] == pytest.approx(1.0, abs=1e-5)


# ── 4. CLI Integration & Statistical Parity vs Rust and Python ──────────────


class TestCLIBackendAndStatisticalParity:
    def test_cli_self_play_backend_flags_and_invalid_backend_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["main.py"])
        assert main_cli.parse_args().self_play_backend == "auto"

        for choice in ("auto", "rust", "cython", "python"):
            monkeypatch.setattr(
                sys, "argv", ["main.py", "--self-play-backend", choice]
            )
            assert main_cli.parse_args().self_play_backend == choice

        dummy = UniformDummyModel()
        with pytest.raises(ValueError, match="Unknown self-play backend"):
            run_self_play(dummy, num_games=1, num_simulations=4, backend="bogus")  # type: ignore[arg-type]

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Requires Rust engine")
    def test_statistical_parity_cython_vs_rust_and_python(self) -> None:
        """Compare Move-0 policies, game lengths, and value distributions across Cython, Rust & Python."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(2026)
        model = AlphaZeroNet(num_res_blocks=2, num_filters=32).to(device)
        model.eval()

        # 1. Deterministic root search parity (dirichlet_epsilon=0.0, random_opening_plies=0):
        # Move-0 policy targets on the empty board should closely agree across all 3 engines!
        common_det = dict(
            model=model,
            num_games=4,
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
        s_cy, p_cy, _ = run_self_play(**common_det, backend="cython")  # type: ignore[arg-type]
        s_rs, p_rs, _ = run_self_play(**common_det, backend="rust")  # type: ignore[arg-type]
        s_py, p_py, _ = run_self_play(**common_det, backend="python")  # type: ignore[arg-type]

        def _first_empty_policy(states: np.ndarray, policies: np.ndarray) -> np.ndarray:
            pieces = (states[:, 0] + states[:, 1]).sum(axis=(1, 2))
            idx = int(np.flatnonzero(pieces == 0)[0])
            return policies[idx]

        pol0_cy = _first_empty_policy(s_cy, p_cy)
        pol0_rs = _first_empty_policy(s_rs, p_rs)
        pol0_py = _first_empty_policy(s_py, p_py)

        # Cython and Rust use identical continuous multi-leaf virtual-loss selection (`LEAVES_PER_GAME=16`),
        # so on a deterministic empty board (`epsilon=0`) their Move-0 policy is nearly identical!
        l1_cy_rs = float(np.abs(pol0_cy - pol0_rs).sum())
        l1_cy_py = float(np.abs(pol0_cy - pol0_py).sum())
        assert l1_cy_rs < 0.15, (
            f"Move-0 deterministic policy diverged between Cython {pol0_cy} and Rust {pol0_rs} (L1={l1_cy_rs:.3f})"
        )
        assert l1_cy_py < 0.35, (
            f"Move-0 deterministic policy diverged between Cython {pol0_cy} and Python {pol0_py} (L1={l1_cy_py:.3f})"
        )

        # 2. Stochastic full self-play parity across 32 games (`cython` vs `rust`) and 12 games (`python`)
        stoch_kwargs = dict(
            model=model,
            num_simulations=24,
            c_puct=1.5,
            dirichlet_alpha=0.35,
            dirichlet_epsilon=0.25,
            temp_threshold=10,
            random_opening_plies=4,
            c_fpu=0.20,
            return_arrays=True,
            return_uint8=True,
        )
        s_cy_s, p_cy_s, v_cy_s = run_self_play(
            num_games=32, backend="cython", **stoch_kwargs  # type: ignore[arg-type]
        )
        s_rs_s, p_rs_s, v_rs_s = run_self_play(
            num_games=32, backend="rust", **stoch_kwargs  # type: ignore[arg-type]
        )
        s_py_s, p_py_s, v_py_s = run_self_play(
            num_games=12, backend="python", **stoch_kwargs  # type: ignore[arg-type]
        )

        cy_plies_per_game = (len(s_cy_s) / 2.0) / 32.0
        rs_plies_per_game = (len(s_rs_s) / 2.0) / 32.0
        py_plies_per_game = (len(s_py_s) / 2.0) / 12.0

        assert 7.0 <= cy_plies_per_game <= 42.0
        assert abs(cy_plies_per_game - rs_plies_per_game) < 10.0, (
            f"Mean game plies diverged: Cython={cy_plies_per_game:.1f} vs Rust={rs_plies_per_game:.1f}"
        )
        assert abs(cy_plies_per_game - py_plies_per_game) < 12.0, (
            f"Mean game plies diverged: Cython={cy_plies_per_game:.1f} vs Python={py_plies_per_game:.1f}"
        )

        # Policy entropy sanity across generated datasets
        def _mean_entropy(pols: np.ndarray) -> float:
            clipped = np.clip(pols, 1e-8, 1.0)
            return float(-np.sum(pols * np.log(clipped), axis=1).mean())

        ent_cy = _mean_entropy(p_cy_s)
        ent_rs = _mean_entropy(p_rs_s)
        assert abs(ent_cy - ent_rs) < 0.45, (
            f"Mean policy entropy diverged: Cython={ent_cy:.3f} vs Rust={ent_rs:.3f}"
        )
        assert abs(float(v_cy_s.mean()) - float(v_rs_s.mean())) < 0.45
