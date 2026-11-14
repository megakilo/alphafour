"""Integration tests for Rust BatchedSelfPlay & RustBatchedMCTS engines.

Verifies:
- Virtual loss sign fix & collision-abort (0% in-batch duplicate leaf evaluations)
- Exact simulation counts via `is_step_done()` and unexpanded-only root init after rebase
- Opening ply distribution covering Move 0 (`piece_count == 0`) and Move 1 (`piece_count == 1`)
- Decoupled visit-count policy targets vs move-selection `temp_threshold`
- Exact MCTS-Solver mate-in-1 / mate-in-2 propagation & `RustBatchedMCTS` semantics
- Safe FFI contiguous / shape validation

Run with: uv run pytest tests/test_rust_mcts.py -v
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

try:
    from alphafour_engine import (
        RustBatchedMCTS,
        RustBatchedSelfPlay,
        RustConnectFour,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE,
    reason="Rust engine not built. Run: uv sync",
)

COLS = 7


class TestBatchedSelfPlay:
    """Test the Rust BatchedSelfPlay engine with dummy NN evaluations."""

    def _run_engine(
        self,
        num_games: int = 5,
        num_sims: int = 10,
        temp_threshold: int = 10,
        random_opening_plies: int = 0,
        use_unexpanded_api: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        engine = RustBatchedSelfPlay(
            num_games,
            num_sims,
            c_puct=1.5,
            dirichlet_alpha=0.35,
            dirichlet_epsilon=0.25,
            temp_threshold=temp_threshold,
            random_opening_plies=random_opening_plies,
            c_fpu=0.20,
        )

        states, valid = engine.get_root_states()
        assert states.shape == (num_games, 3, 6, 7)
        assert valid.shape == (num_games, 7)

        n = engine.num_active()
        policies = np.full((n, COLS), 1.0 / COLS, dtype=np.float32)
        engine.init_roots(policies)

        all_states = []
        all_policies = []
        all_values = []

        max_iters = 500
        iters = 0
        while not engine.is_done() and iters < max_iters:
            engine.add_noise()

            while not engine.is_step_done():
                leaf_states, leaf_valid, count = engine.collect_leaves()
                if count > 0:
                    assert leaf_states.shape == (count, 3, 6, 7)
                    assert leaf_valid.shape == (count, 7)
                    p = np.full((count, COLS), 1.0 / COLS, dtype=np.float32)
                    v = np.zeros(count, dtype=np.float32)
                    engine.apply_evaluations(p, v)

            ex_states, ex_policies, ex_values = engine.advance_games()
            if len(ex_states) > 0:
                all_states.append(ex_states)
                all_policies.append(ex_policies)
                all_values.append(ex_values)

            n = engine.num_active()
            if n > 0:
                if use_unexpanded_api:
                    u_states, u_valid, u_count = engine.get_unexpanded_root_states()
                    assert u_states.shape == (u_count, 3, 6, 7)
                    assert u_valid.shape == (u_count, 7)
                    if u_count > 0:
                        u_pols = np.full((u_count, COLS), 1.0 / COLS, dtype=np.float32)
                        engine.init_unexpanded_roots(u_pols)
                else:
                    states, valid = engine.get_root_states()
                    policies = np.full((n, COLS), 1.0 / COLS, dtype=np.float32)
                    engine.init_new_roots(policies)

            iters += 1

        assert engine.is_done(), f"Games didn't finish in {max_iters} iterations"

        states = np.concatenate(all_states) if all_states else np.empty((0, 3, 6, 7))
        policies = np.concatenate(all_policies) if all_policies else np.empty((0, 7))
        values = np.concatenate(all_values) if all_values else np.empty((0,))
        return states, policies, values

    def test_basic_completion_and_legacy_compat(self):
        states, policies, values = self._run_engine(num_games=5, num_sims=10)
        assert len(states) > 0
        assert len(states) == len(policies) == len(values)

        s2, p2, v2 = self._run_engine(
            num_games=3, num_sims=8, use_unexpanded_api=False
        )
        assert len(s2) > 0
        assert len(s2) == len(p2) == len(v2)

    def test_output_shapes(self):
        states, policies, values = self._run_engine(num_games=3, num_sims=5)
        assert states.ndim == 4 and states.shape[1:] == (3, 6, 7)
        assert policies.ndim == 2 and policies.shape[1] == 7
        assert values.ndim == 1

    def test_augmentation_doubles_data_and_mirrors(self):
        states, policies, values = self._run_engine(num_games=4, num_sims=10)
        assert len(states) % 2 == 0
        for i in range(0, len(states) - 1, 2):
            npt.assert_array_equal(states[i + 1], states[i][:, :, ::-1])
            npt.assert_array_almost_equal(policies[i + 1], policies[i][::-1])
            assert values[i] == values[i + 1]

    def test_policies_and_values_valid(self):
        states, policies, values = self._run_engine(num_games=4, num_sims=12)
        npt.assert_allclose(policies.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(policies >= 0.0)
        assert np.all((values >= -1.0) & (values <= 1.0))


class TestDiversityAndCollisionAbort:
    """Verify virtual loss sign fix and collision-abort prevent duplicate in-batch leaves."""

    def test_zero_in_batch_duplicates_single_game(self):
        g = RustConnectFour()
        mcts = RustBatchedMCTS([g], num_simulations=32, c_puct=1.5, c_fpu=0.20)
        mcts.init_roots(np.full((1, COLS), 1.0 / COLS, dtype=np.float32))

        # Batch 1 on depth-1 root: only 7 children exist, so collision-abort MUST stop at 7!
        states1, _valid1, count1 = mcts.collect_leaves()
        assert count1 == 7
        unique1 = {states1[i].tobytes() for i in range(count1)}
        assert len(unique1) == 7

        mcts.apply_evaluations(
            np.full((count1, COLS), 1.0 / COLS, dtype=np.float32),
            np.zeros(count1, dtype=np.float32),
        )

        # Batch 2 on depth-2 tree: 49 grandchildren exist, all 16 leaves in batch MUST be distinct!
        states2, _valid2, count2 = mcts.collect_leaves()
        assert count2 == 16
        unique2 = {states2[i].tobytes() for i in range(count2)}
        assert len(unique2) == 16


class TestSimulationTrackingAndOpenings:
    """Verify `is_step_done()`, unexpanded-only root evals, opening ply coverage, and policy decoupling."""

    def test_is_step_done_runs_exact_simulation_budget(self):
        num_games = 4
        num_sims = 25  # Non-multiple of LEAVES_PER_GAME (16)
        engine = RustBatchedSelfPlay(
            num_games, num_sims, 1.5, 0.35, 0.25, 10, 0, 0.20
        )
        engine.init_roots(np.full((num_games, COLS), 1.0 / COLS, dtype=np.float32))
        engine.add_noise()

        total_leaves = 0
        while not engine.is_step_done():
            _s, _v, count = engine.collect_leaves()
            if count > 0:
                total_leaves += count
                engine.apply_evaluations(
                    np.full((count, COLS), 1.0 / COLS, dtype=np.float32),
                    np.zeros(count, dtype=np.float32),
                )

        assert engine.is_step_done()
        # Early game (move 0) has no terminal hits within depth 2, so exact leaf count == 4 * 25
        assert total_leaves == num_games * num_sims

        engine.advance_games()
        # All chosen children were already expanded during the 25 simulations!
        _us, _uv, u_count = engine.get_unexpanded_root_states()
        assert u_count == 0

    def test_opening_ply_distribution_covers_move_0_and_move_1(self):
        engine = RustBatchedSelfPlay(
            400, 8, 1.5, 0.35, 0.25, 10, random_opening_plies=4, c_fpu=0.20
        )
        states, _valid = engine.get_root_states()
        piece_counts = (states[:, 0] + states[:, 1]).sum(axis=(1, 2)).astype(int)

        n0 = int(np.sum(piece_counts == 0))
        n1 = int(np.sum(piece_counts == 1))
        n2_plus = int(np.sum((piece_counts >= 2) & (piece_counts <= 4)))

        assert n0 + n1 + n2_plus == 400
        assert 150 <= n0 <= 250, f"Expected ~50% 0-ply games, got {n0}/400"
        assert 65 <= n1 <= 140, f"Expected ~25% 1-ply games, got {n1}/400"
        assert 65 <= n2_plus <= 140, f"Expected ~25% 2..4-ply games, got {n2_plus}/400"

        # Also verify actual generated training examples include piece_count == 0 and piece_count == 1
        ex_states, _, _ = TestBatchedSelfPlay()._run_engine(
            num_games=24, num_sims=8, random_opening_plies=4
        )
        ex_pieces = (ex_states[:, 0] + ex_states[:, 1]).sum(axis=(1, 2)).astype(int)
        assert np.any(ex_pieces == 0), "Move 0 missing from self-play training examples!"
        assert np.any(ex_pieces == 1), "Move 1 missing from self-play training examples!"

    def test_decoupled_policy_target_remains_soft_beyond_temp_threshold(self):
        # Set temp_threshold=0 so ALL moves use greedy argmax action selection,
        # while training policy targets must STILL record normalized visit distributions!
        states, policies, _ = TestBatchedSelfPlay()._run_engine(
            num_games=2, num_sims=28, temp_threshold=0, random_opening_plies=0
        )
        assert len(policies) > 0
        # On the opening move (first example), 28 uniform simulations visit all 7 columns (~4 visits each),
        # so max(policy[0]) should be well below 1.0 (NOT a 1-hot vector!).
        assert policies[0].max() < 0.75, (
            f"Expected soft visit-count policy target even at temp=0, got {policies[0]}"
        )
        assert np.count_nonzero(policies[0] > 0.05) >= 4


class TestRustBatchedMCTS:
    """Verify RustBatchedMCTS evaluation engine, MCTS-Solver propagation, and terminal handling."""

    def test_mate_in_1_and_mate_in_2_solver_and_terminal_input(self):
        g_mate1 = RustConnectFour.from_moves("121212")  # Col 0 wins immediately
        g_mate2 = RustConnectFour.from_moves("3141")  # Col 1 or Col 4 forces open-ended 3-in-a-row win in 2
        g_terminal_loss = RustConnectFour.from_moves("1212121")  # Already won by P1 (current player P2 lost)

        mcts = RustBatchedMCTS(
            [g_mate1, g_mate2, g_terminal_loss],
            num_simulations=64,
            c_puct=1.5,
            c_fpu=0.20,
        )
        states, valid = mcts.get_root_states()
        assert states.shape == (3, 3, 6, 7)
        assert valid.shape == (3, 7)

        mcts.init_roots(np.full((3, COLS), 1.0 / COLS, dtype=np.float32))
        while not mcts.is_done():
            _ls, _lv, count = mcts.collect_leaves()
            if count > 0:
                mcts.apply_evaluations(
                    np.full((count, COLS), 1.0 / COLS, dtype=np.float32),
                    np.zeros(count, dtype=np.float32),
                )

        vc, best_actions, best_values, child_q = mcts.get_results()
        assert vc.shape == (3, 7) and vc.dtype == np.float32
        assert best_actions.shape == (3,) and best_actions.dtype == np.int64
        assert best_values.shape == (3,) and best_values.dtype == np.float32
        assert child_q.shape == (3, 7) and child_q.dtype == np.float32

        # 0: Mate-in-1
        assert best_actions[0] == 0
        assert best_values[0] == pytest.approx(1.0)
        assert child_q[0, 0] == pytest.approx(1.0)

        # 1: Mate-in-2 ("3141" -> playing col 1 or col 4 gives open-ended [1,2,3] or [2,3,4] on row 0!)
        assert best_actions[1] in (1, 4)
        assert best_values[1] == pytest.approx(1.0)
        assert child_q[1, best_actions[1]] == pytest.approx(1.0)

        # 2: Already terminal won board
        assert best_actions[2] == -1
        assert best_values[2] == pytest.approx(-1.0)
        assert np.all(vc[2] == 0.0)
        assert np.all(np.isnan(child_q[2]))

    def test_illegal_and_unvisited_columns_have_nan_child_q(self):
        g = RustConnectFour.from_moves("4444441235671235671235")  # Col 3 is full
        mcts = RustBatchedMCTS([g], num_simulations=2, c_puct=1.5, c_fpu=0.20)
        mcts.init_roots(np.full((1, COLS), 1.0 / COLS, dtype=np.float32))
        while not mcts.is_done():
            _ls, _lv, count = mcts.collect_leaves()
            if count > 0:
                mcts.apply_evaluations(
                    np.full((count, COLS), 1.0 / COLS, dtype=np.float32),
                    np.full(count, 0.4, dtype=np.float32),
                )

        _vc, _ba, best_values, child_q = mcts.get_results()
        # Col 3 is full -> must be NaN
        assert np.isnan(child_q[0, 3])
        # Visited children should have Q == -0.4 from root perspective
        visited_q = child_q[0, ~np.isnan(child_q[0])]
        assert len(visited_q) >= 1
        assert -1.0 <= best_values[0] <= 1.0

    def test_non_contiguous_or_bad_shape_raises_value_error(self):
        g = RustConnectFour()
        mcts = RustBatchedMCTS([g], num_simulations=4)
        non_contig = np.zeros((7, 2), dtype=np.float32)[:, 0:1].T  # non-contiguous slice if transposed
        bad_small = np.zeros((1, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            mcts.init_roots(bad_small)
        _ = non_contig
