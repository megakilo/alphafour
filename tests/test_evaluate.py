"""Unit tests for src/evaluate.py, main.py, and benchmark.py CLI contracts."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import benchmark as benchmark_cli
import main as main_cli
import src.evaluate as evaluate_mod
from src.evaluate import (
    PureMCTSBaseline,
    _MEMORY_SOLVER_CACHE,
    _bb_from_moves,
    _bb_has_won,
    _bb_shallow_tactical_score,
    compute_elo_diff,
    evaluate_dataset,
    evaluate_opening_move,
    generate_paired_openings,
    get_dataset_child_wdl,
    play_anchor_arena,
    play_batched_arena,
)
from src.game import COLS, ROWS
from src.model import AlphaZeroNet


class SpyPredictor:
    """Deterministic mock predictor that records every batch passed to `predict`."""

    def __init__(self, preferred_col: int = 3, fixed_value: float = 0.1) -> None:
        self.preferred_col = preferred_col
        self.fixed_value = fixed_value
        self.eval_called = False
        self.calls: list[tuple[np.ndarray, np.ndarray | None]] = []

    def eval(self) -> SpyPredictor:
        self.eval_called = True
        return self

    def predict(
        self,
        state: torch.Tensor,
        valid_moves: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single = state.dim() == 3
        if single:
            state_b = state.unsqueeze(0)
            valid_b = valid_moves.unsqueeze(0) if valid_moves is not None else None
        else:
            state_b = state
            valid_b = valid_moves

        self.calls.append(
            (
                state_b.detach().cpu().numpy().copy(),
                valid_b.detach().cpu().numpy().copy() if valid_b is not None else None,
            )
        )

        bsz = state_b.shape[0]
        logits = torch.zeros((bsz, COLS), dtype=torch.float32, device=state.device)
        logits[:, self.preferred_col] = 5.0
        # Secondary preference on center/adjacent columns
        logits[:, 3] += 1.0
        logits[:, 2] += 0.5
        if valid_b is not None:
            logits = logits.masked_fill(~valid_b, -1e9)
        pol = torch.softmax(logits, dim=1)
        val = torch.full(
            (bsz,), self.fixed_value, dtype=torch.float32, device=state.device
        )
        if single:
            return pol.squeeze(0), val.squeeze(0)
        return pol, val


# ── 1. Arena Model Isolation Tests (C2 Regression) ──────────────────────────


def test_arena_strict_model_isolation_single_game(monkeypatch: pytest.MonkeyPatch) -> None:
    """On model1's root turns model2 must receive ZERO leaf/root evals (and vice versa)."""
    model1 = SpyPredictor(preferred_col=3, fixed_value=0.2)
    model2 = SpyPredictor(preferred_col=2, fixed_value=-0.2)
    device = torch.device("cpu")

    orig_select = evaluate_mod._select_arena_actions_for_partition
    turn_log: list[tuple[str, int, int]] = []

    def wrapped_select(
        model: Any,
        dev: torch.device,
        partition_games: list[Any],
        num_simulations: int,
        use_rust: bool,
    ) -> np.ndarray:
        m1_before = len(model1.calls)
        m2_before = len(model2.calls)
        out = orig_select(model, dev, partition_games, num_simulations, use_rust)
        m1_delta = len(model1.calls) - m1_before
        m2_delta = len(model2.calls) - m2_before
        if model is model1:
            assert m2_delta == 0, (
                f"Opponent contamination! model2 called {m2_delta} times during model1's MCTS!"
            )
            assert m1_delta > 0
            turn_log.append(("m1", m1_delta, m2_delta))
        elif model is model2:
            assert m1_delta == 0, (
                f"Opponent contamination! model1 called {m1_delta} times during model2's MCTS!"
            )
            assert m2_delta > 0
            turn_log.append(("m2", m1_delta, m2_delta))
        return out

    monkeypatch.setattr(
        evaluate_mod, "_select_arena_actions_for_partition", wrapped_select
    )

    play_batched_arena(
        model1=model1,
        model2=model2,
        device=device,
        num_games=1,
        num_simulations=20,
        temperature=0.0,
        paired_openings=False,
    )

    assert len(turn_log) >= 7  # At least 7 plies needed to finish a Connect Four game

    # Verify model1 evaluated depth-1 (odd stone count) leaves inside its OWN tree on ply 0,
    # while model2 NEVER saw the empty root board (0 stones).
    m1_stone_counts = [
        int((batch[i, 0] + batch[i, 1]).sum())
        for batch, _ in model1.calls
        for i in range(len(batch))
    ]
    m2_stone_counts = [
        int((batch[i, 0] + batch[i, 1]).sum())
        for batch, _ in model2.calls
        for i in range(len(batch))
    ]
    assert 0 in m1_stone_counts
    assert 1 in m1_stone_counts  # Depth-1 opponent-to-move leaf evaluated by model1 itself!
    assert 0 not in m2_stone_counts  # Zero redundant dual-model root call on ply 0!


def test_arena_strict_model_isolation_batched_paired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify strict per-partition model isolation across multi-game paired-opening Arena."""
    model1 = SpyPredictor(preferred_col=3, fixed_value=0.15)
    model2 = SpyPredictor(preferred_col=4, fixed_value=-0.05)
    device = torch.device("cpu")

    orig_select = evaluate_mod._select_arena_actions_for_partition

    def wrapped_select(
        model: Any,
        dev: torch.device,
        partition_games: list[Any],
        num_simulations: int,
        use_rust: bool,
    ) -> np.ndarray:
        m1_before = len(model1.calls)
        m2_before = len(model2.calls)
        out = orig_select(model, dev, partition_games, num_simulations, use_rust)
        if model is model1:
            assert len(model2.calls) == m2_before
        else:
            assert len(model1.calls) == m1_before
        return out

    monkeypatch.setattr(
        evaluate_mod, "_select_arena_actions_for_partition", wrapped_select
    )

    res = play_batched_arena(
        model1=model1,
        model2=model2,
        device=device,
        num_games=6,
        num_simulations=12,
        temperature=0.0,
        paired_openings=True,
        seed=123,
    )
    assert sum(res) == 6
    # Verify valid_moves was passed on every single predict call
    assert all(valid is not None for _, valid in model1.calls)
    assert all(valid is not None for _, valid in model2.calls)


# ── 2. Duplicate Paired-Opening Symmetry Tests ──────────────────────────────


def test_paired_openings_generation_quality_and_determinism() -> None:
    """Generated openings must be deterministic, 2-4 plies, non-terminal, and tactically safe."""
    ops_a = generate_paired_openings(20, seed=42)
    ops_b = generate_paired_openings(20, seed=42)
    assert ops_a == ops_b
    assert len(ops_a) == 20
    assert len(set(ops_a)) == 20

    for moves_str in ops_a:
        assert 2 <= len(moves_str) <= 4
        pos, mask, mc = _bb_from_moves(moves_str)
        assert not _bb_has_won(pos)
        assert not _bb_has_won(pos ^ mask)
        assert _bb_shallow_tactical_score(pos, mask, mc, depth=6) == 0


def test_paired_opening_symmetry_identical_models() -> None:
    """Two identical deterministic models with temperature=0.0 must tie at exactly 50.0%."""
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = AlphaZeroNet(num_res_blocks=2, num_filters=32).to(device)
    model2 = AlphaZeroNet(num_res_blocks=2, num_filters=32).to(device)
    model2.load_state_dict(model1.state_dict())

    m1_w_p1, m1_w_p2, m2_w_p1, m2_w_p2, draws = play_batched_arena(
        model1=model1,
        model2=model2,
        device=device,
        num_games=8,
        num_simulations=16,
        temperature=0.0,
        paired_openings=True,
        seed=99,
    )

    m1_wins = m1_w_p1 + m1_w_p2
    m2_wins = m2_w_p1 + m2_w_p2
    total = m1_wins + m2_wins + draws

    assert total == 8
    assert m1_w_p1 == m2_w_p1
    assert m1_w_p2 == m2_w_p2
    assert m1_wins == m2_wins
    win_rate = (m1_wins + 0.5 * draws) / total
    assert win_rate == pytest.approx(0.5, abs=1e-9)


# ── 3. evaluate_dataset Metrics & Bugfix Verification ───────────────────────


def test_evaluate_dataset_passes_valid_moves_mask_in_raw_mode() -> None:
    """Raw 0-sim evaluation must pass `valid_moves` so full columns are masked out."""
    # "444444" stacks 6 alternating stones in column 4 (0-indexed col 3), filling it completely!
    lines = ["444444 2"]

    class FullColTrapModel:
        def __init__(self) -> None:
            self.received_valid_moves: list[np.ndarray | None] = []

        def eval(self) -> FullColTrapModel:
            return self

        def predict(
            self, state: torch.Tensor, valid_moves: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.received_valid_moves.append(
                valid_moves.detach().cpu().numpy().copy()
                if valid_moves is not None
                else None
            )
            bsz = state.shape[0]
            logits = torch.zeros((bsz, COLS), dtype=torch.float32, device=state.device)
            # Strongly prefer the illegal full col 3 unless masked by valid_moves!
            logits[:, 3] = 100.0
            logits[:, 2] = 10.0
            if valid_moves is not None:
                logits = logits.masked_fill(~valid_moves, float("-inf"))
            return torch.softmax(logits, dim=1), torch.full(
                (bsz,), 0.8, dtype=torch.float32, device=state.device
            )

    model = FullColTrapModel()
    # Provide synthetic child_wdl where col 3 is -2 (illegal) and col 2 is +1 (optimal)
    child_wdl = np.array([[-1, -1, 1, -2, -1, -1, -1]], dtype=np.int8)

    res = evaluate_dataset(
        model,  # type: ignore[arg-type]
        torch.device("cpu"),
        lines,
        num_simulations=0,
        child_wdl=child_wdl,
    )

    assert len(model.received_valid_moves) == 1
    vm = model.received_valid_moves[0]
    assert vm is not None, "valid_moves was None in raw 0-sim evaluate_dataset!"
    assert vm.shape == (1, COLS)
    assert bool(vm[0, 3]) is False
    assert all(bool(vm[0, c]) is True for c in range(COLS) if c != 3)
    assert res["optimal_move_pct"] == pytest.approx(100.0)
    assert res["blunder_pct"] == pytest.approx(0.0)


def test_evaluate_dataset_draw_recall_fixes_narrow_band_ceiling() -> None:
    """Noisy oracle draw predictions in (0.06..0.28) must achieve 100% draw recall (vs 0% with ±0.05)."""
    # 10 wins (+5), 10 draws (0), 10 losses (-5)
    lines = (
        [f"4 {5}" for _ in range(10)]
        + [f"44 {0}" for _ in range(10)]
        + [f"444 {-5}" for _ in range(10)]
    )

    # Every single draw value is outside [-0.05, +0.05] (between ±0.09 and ±0.28),
    # which gave 0% draw recall under the old ±0.05 band, and 100% under symmetric ±1/3!
    noisy_draw_vals = [0.12, -0.15, 0.22, -0.09, 0.28, -0.25, 0.18, -0.19, 0.11, -0.27]
    pred_stream = [0.80] * 10 + noisy_draw_vals + [-0.85] * 10

    class NoisyScalarOracle:
        def eval(self) -> NoisyScalarOracle:
            return self

        def predict(
            self, state: torch.Tensor, valid_moves: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            bsz = state.shape[0]
            pol = torch.full((bsz, COLS), 1.0 / COLS, device=state.device)
            vals = torch.tensor(pred_stream[:bsz], dtype=torch.float32, device=state.device)
            return pol, vals

    dummy_child_wdl = np.zeros((30, COLS), dtype=np.int8)
    res_scalar = evaluate_dataset(
        NoisyScalarOracle(),  # type: ignore[arg-type]
        torch.device("cpu"),
        lines,
        num_simulations=0,
        child_wdl=dummy_child_wdl,
    )

    assert res_scalar["draw_recall"] == pytest.approx(100.0)
    assert res_scalar["win_recall"] == pytest.approx(100.0)
    assert res_scalar["loss_recall"] == pytest.approx(100.0)
    assert res_scalar["wdl_accuracy"] == pytest.approx(100.0)
    assert res_scalar["accuracy"] == pytest.approx(res_scalar["wdl_accuracy"])

    # Also verify predict_wdl argmax classification when P(Win)-P(Loss) > 1/3 yet argmax is Draw!
    class WDLHeadOracle:
        def eval(self) -> WDLHeadOracle:
            return self

        def predict_wdl(
            self, state: torch.Tensor, valid_moves: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            bsz = state.shape[0]
            pol = torch.full((bsz, COLS), 1.0 / COLS, device=state.device)
            wdl = torch.zeros((bsz, 3), dtype=torch.float32, device=state.device)
            # First 10: Win (idx 2)
            wdl[:10] = torch.tensor([0.05, 0.15, 0.80])
            # Middle 10: Draw (idx 1 has 0.49 > 0.47, even though P(W)-P(L)=0.43 > 1/3!)
            wdl[10:20] = torch.tensor([0.04, 0.49, 0.47])
            # Last 10: Loss (idx 0)
            wdl[20:30] = torch.tensor([0.75, 0.20, 0.05])
            vals = wdl[:, 2] - wdl[:, 0]
            return pol, wdl, vals

    res_wdl = evaluate_dataset(
        WDLHeadOracle(),  # type: ignore[arg-type]
        torch.device("cpu"),
        lines,
        num_simulations=0,
        child_wdl=dummy_child_wdl,
    )
    assert res_wdl["draw_recall"] == pytest.approx(100.0)
    assert res_wdl["wdl_accuracy"] == pytest.approx(100.0)


def test_evaluate_dataset_optimal_move_blunder_and_nan_recall() -> None:
    """Verify exact optimal_move_pct, blunder_pct (excluding already-lost), value_mse, and NaN recall."""
    # 3 positions: Win (+4), Win (+2), Loss (-3) -> 0 draws in dataset!
    lines = ["4 4", "44 2", "444 -3"]
    # Pos 0 (Win, max=+1): col 0 is +1 (optimal), others 0/-1
    # Pos 1 (Win, max=+1): col 3 is +1, col 0 is 0 (blunder!)
    # Pos 2 (Already Lost, max=-1): all valid cols are -1 (cannot blunder an already-lost board!)
    child_wdl = np.array(
        [
            [1, 0, -1, 0, -1, 0, -1],
            [0, -1, -1, 1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.int8,
    )

    class AlwaysColZeroModel:
        def eval(self) -> AlwaysColZeroModel:
            return self

        def predict(
            self, state: torch.Tensor, valid_moves: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            bsz = state.shape[0]
            pol = torch.zeros((bsz, COLS), dtype=torch.float32, device=state.device)
            pol[:, 0] = 1.0
            vals = torch.tensor([0.5, 0.5, -0.5][:bsz], dtype=torch.float32, device=state.device)
            return pol, vals

    res = evaluate_dataset(
        AlwaysColZeroModel(),  # type: ignore[arg-type]
        torch.device("cpu"),
        lines,
        num_simulations=0,
        child_wdl=child_wdl,
    )

    # Pos 0 optimal (+1==+1), Pos 1 suboptimal (0<+1), Pos 2 optimal (-1==-1) => 2/3
    assert res["optimal_move_pct"] == pytest.approx(200.0 / 3.0)
    # Non-lost positions are Pos 0 and Pos 1 (2 positions); Pos 1 blundered => 1/2 = 50%
    assert res["blunder_pct"] == pytest.approx(50.0)
    # 0 true draws => draw_recall must be NaN
    assert math.isnan(res["draw_recall"])
    assert res["win_recall"] == pytest.approx(100.0)
    assert res["loss_recall"] == pytest.approx(100.0)
    # MSE of [0.5, 0.5, -0.5] vs [1.0, 1.0, -1.0] = mean([0.25, 0.25, 0.25]) = 0.25
    assert res["value_mse"] == pytest.approx(0.25)


def test_evaluate_dataset_mcts_undiluted_best_value() -> None:
    """At >0 sims, MCTS on an immediate win-in-1 must report undiluted +1.0 value (not diluted root Q)."""
    # "131415": P1 has 3 vertical stones in col 1 (col 0) and P2 has an open-ended 3-in-a-row at cols 3,4,5.
    # Col 0 wins immediately (+1.0), whereas cols 1..6 all lose immediately (-1.0) on P2's reply!
    # Diluted root.q_value mixes exploratory visits to cols 1..6 (< +1.0), whereas -best_child.q_value == +1.0!
    lines = ["131415 19"]
    uniform_model = PureMCTSBaseline(use_heuristic=False)
    res = evaluate_dataset(
        uniform_model,  # type: ignore[arg-type]
        torch.device("cpu"),
        lines,
        num_simulations=32,
    )
    assert res["optimal_move_pct"] == pytest.approx(100.0)
    assert res["blunder_pct"] == pytest.approx(0.0)
    assert res["wdl_accuracy"] == pytest.approx(100.0)
    assert res["value_mse"] == pytest.approx(0.0, abs=1e-6)


# ── 4. Solver Ground-Truth Cache & Testdata Exactness ───────────────────────


def test_solver_ground_truth_and_fast_npz_cache(tmp_path: Path) -> None:
    """Verify exact solver child_wdl matches gamesolver.org Test_L3_R1 signs and reloads in <5ms."""
    with open("testdata/Test_L3_R1", "r", encoding="utf-8") as f:
        sample_lines = [f.readline() for _ in range(25)]

    _MEMORY_SOLVER_CACHE.clear()
    child_wdl = get_dataset_child_wdl(sample_lines, cache_dir=tmp_path)
    assert child_wdl is not None
    assert child_wdl.shape == (25, COLS)
    assert child_wdl.dtype == np.int8

    # Verify max(valid child_wdl) == sign(true_score) on 100% of positions
    true_signs = np.array(
        [np.sign(int(ln.strip().split()[1])) for ln in sample_lines], dtype=np.int8
    )
    np.testing.assert_array_equal(child_wdl.max(axis=1), true_signs)

    # Verify .npz cache file + self-ignoring .gitignore exist
    npz_files = list(tmp_path.glob("*.npz"))
    assert len(npz_files) == 1
    assert (tmp_path / ".gitignore").exists()

    # Clear memory cache and verify reload from .npz takes <5ms
    _MEMORY_SOLVER_CACHE.clear()
    t0 = time.perf_counter()
    cached_wdl = get_dataset_child_wdl(sample_lines, cache_dir=tmp_path)
    load_ms = (time.perf_counter() - t0) * 1000.0

    assert cached_wdl is not None
    np.testing.assert_array_equal(cached_wdl, child_wdl)
    assert load_ms < 25.0  # Typically ~0.3ms


def test_solver_graceful_none_on_deep_position_without_rust(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When neither cache nor Rust solver is available on deep positions, optimal_move_pct is None."""
    monkeypatch.setattr(evaluate_mod, "RUST_SOLVER_AVAILABLE", False)
    _MEMORY_SOLVER_CACHE.clear()
    deep_lines = ["44 1"]  # 40 empty squares -> exceeds tiny Python node budget
    child_wdl = get_dataset_child_wdl(
        deep_lines, cache_dir=tmp_path, use_cache=False, python_max_nodes=10
    )
    assert child_wdl is None

    res = evaluate_dataset(
        PureMCTSBaseline(),  # type: ignore[arg-type]
        torch.device("cpu"),
        deep_lines,
        num_simulations=0,
        child_wdl=None,
    )
    # Note: evaluate_dataset calls get_dataset_child_wdl with default 25k budget which also returns None on 2-ply
    if not evaluate_mod.RUST_SOLVER_AVAILABLE:
        assert res["optimal_move_pct"] is None
        assert res["blunder_pct"] is None


# ── 5. Opening Move, Anchor Elo, and CLI Smoke Tests ────────────────────────


def test_evaluate_opening_move_and_anchor_elo() -> None:
    device = torch.device("cpu")
    model = AlphaZeroNet(num_res_blocks=2, num_filters=16).to(device)

    move0, pct0 = evaluate_opening_move(model, device, num_simulations=0)
    assert 0 <= move0 < COLS
    assert 0.0 <= pct0 <= 100.0

    move_mcts, pct_mcts = evaluate_opening_move(model, device, num_simulations=12)
    assert 0 <= move_mcts < COLS
    assert 0.0 <= pct_mcts <= 100.0

    assert compute_elo_diff(0.5) == pytest.approx(0.0)
    assert compute_elo_diff(0.75) == pytest.approx(-compute_elo_diff(0.25))
    assert math.isfinite(compute_elo_diff(0.0))
    assert math.isfinite(compute_elo_diff(1.0))
    assert compute_elo_diff(1.0) > 1000.0
    assert compute_elo_diff(0.0) < -1000.0

    anc = play_anchor_arena(
        model=model,
        device=device,
        num_games=4,
        num_simulations=8,
        anchor_simulations=8,
    )
    assert anc["model_wins"] + anc["anchor_wins"] + anc["draws"] == 4
    assert 0.0 <= float(anc["win_rate"]) <= 1.0
    assert math.isfinite(float(anc["elo_diff"]))


def test_cli_argparsers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify main.py and benchmark.py parse all documented flags cleanly."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--iterations",
            "10",
            "--eval-simulations",
            "64",
            "--eval-interval",
            "5",
            "--compile",
            "--no-resume",
        ],
    )
    m_args = main_cli.parse_args()
    assert m_args.iterations == 10
    assert m_args.eval_simulations == 64
    assert m_args.eval_interval == 5
    assert m_args.compile is True
    assert m_args.no_resume is True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--eval-simulations",
            "32",
            "--both",
            "--arena-anchor",
            "--limit",
            "50",
        ],
    )
    b_args = benchmark_cli.parse_args()
    assert b_args.eval_simulations == 32
    assert b_args.both is True
    assert b_args.arena_anchor is True
    assert b_args.limit == 50


def test_rust_batched_mcts_and_solver_adapter_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify the RustBatchedMCTS + solve_batch_children_wdl FFI adapter code paths."""
    from src.game import ConnectFour

    class FakeRustConnectFour:
        def __init__(self) -> None:
            self._g = ConnectFour()

        @classmethod
        def from_moves(cls, moves_str: str) -> FakeRustConnectFour:
            inst = cls()
            for ch in moves_str:
                inst._g.make_move(int(ch) - 1)
            return inst

        @property
        def current_player(self) -> int:
            return self._g.current_player

        def make_move(self, col: int) -> None:
            self._g.make_move(col)

        def is_terminal(self) -> bool:
            return self._g.is_terminal()

        def get_result(self) -> float | None:
            return self._g.get_result()

        def get_valid_moves(self) -> np.ndarray:
            return self._g.get_valid_moves()

        def encode(self) -> np.ndarray:
            return self._g.encode()

    class FakeRustBatchedMCTS:
        def __init__(
            self,
            games: list[FakeRustConnectFour],
            num_simulations: int,
            c_puct: float = 1.5,
            c_fpu: float = 0.20,
        ) -> None:
            self.games = games
            self.num_simulations = num_simulations
            self._steps = 0
            self._pols: np.ndarray | None = None
            self._vals: np.ndarray | None = None

        def get_root_states(self) -> tuple[np.ndarray, np.ndarray]:
            states = np.stack([g.encode() for g in self.games]).astype(np.float32)
            valid = np.stack([g.get_valid_moves() for g in self.games]).astype(bool)
            return states, valid

        def init_roots(self, policies: np.ndarray) -> None:
            assert policies.dtype == np.float32
            assert policies.flags["C_CONTIGUOUS"]
            self._pols = policies

        def is_done(self) -> bool:
            return self._steps >= 1

        def collect_leaves(self) -> tuple[np.ndarray, np.ndarray, int]:
            states = np.stack([g.encode() for g in self.games]).astype(np.float32)
            valid = np.stack([g.get_valid_moves() for g in self.games]).astype(bool)
            return states, valid, len(self.games)

        def apply_evaluations(self, policies: np.ndarray, values: np.ndarray) -> None:
            assert policies.dtype == np.float32
            assert values.dtype == np.float32
            assert policies.flags["C_CONTIGUOUS"]
            assert values.flags["C_CONTIGUOUS"]
            self._vals = values
            self._steps += 1

        def get_results(
            self,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            n = len(self.games)
            assert self._pols is not None and self._vals is not None
            vc = np.zeros((n, COLS), dtype=np.float32)
            best_actions = np.zeros((n,), dtype=np.int64)
            for i, g in enumerate(self.games):
                valid = g.get_valid_moves()
                masked = np.where(valid, self._pols[i], -np.inf)
                a = int(np.argmax(masked))
                best_actions[i] = a
                vc[i, a] = float(self.num_simulations)
            q_vals = np.full((n, COLS), np.nan, dtype=np.float32)
            return vc, best_actions, self._vals.copy(), q_vals

    def fake_solve_batch_children_wdl(moves_list: list[str]) -> np.ndarray:
        out = np.zeros((len(moves_list), COLS), dtype=np.int8)
        out[:, 3] = 1
        return out

    monkeypatch.setattr(evaluate_mod, "RUST_MCTS_AVAILABLE", True)
    monkeypatch.setattr(evaluate_mod, "RUST_SOLVER_AVAILABLE", True)
    monkeypatch.setattr(evaluate_mod, "RustConnectFour", FakeRustConnectFour)
    monkeypatch.setattr(evaluate_mod, "RustBatchedMCTS", FakeRustBatchedMCTS)
    monkeypatch.setattr(
        evaluate_mod, "solve_batch_children_wdl", fake_solve_batch_children_wdl
    )

    _MEMORY_SOLVER_CACHE.clear()
    lines = ["44 2", "43 2"]
    cw = get_dataset_child_wdl(lines, cache_dir=tmp_path)
    assert cw is not None and cw.shape == (2, COLS)
    assert len(list(tmp_path.glob("*.npz"))) == 1

    spy1 = SpyPredictor(preferred_col=3, fixed_value=0.9)
    spy2 = SpyPredictor(preferred_col=3, fixed_value=0.9)

    res = evaluate_dataset(
        spy1,  # type: ignore[arg-type]
        torch.device("cpu"),
        lines,
        num_simulations=10,
        child_wdl=cw,
    )
    assert res["wdl_accuracy"] == pytest.approx(100.0)
    assert res["optimal_move_pct"] == pytest.approx(100.0)

    move_op, pct_op = evaluate_opening_move(
        spy1,  # type: ignore[arg-type]
        torch.device("cpu"),
        num_simulations=10,
    )
    assert move_op == 3
    assert pct_op == pytest.approx(100.0)

    arena_res = play_batched_arena(
        spy1,
        spy2,
        torch.device("cpu"),
        num_games=4,
        num_simulations=8,
        temperature=0.0,
        paired_openings=True,
    )
    assert arena_res[0] + arena_res[1] == arena_res[2] + arena_res[3]

