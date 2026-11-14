"""Unit and integration tests for MCTS-Solver, FPU, Tree Reuse, Self-Play, and play.py."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import torch

import play
import src.self_play as self_play
from src.game import COLS, ROWS, ConnectFour
from src.mcts import (
    DEFAULT_C_FPU,
    DEFAULT_DIRICHLET_ALPHA,
    MCTS,
    SOLVER_DRAW,
    SOLVER_LOSS,
    SOLVER_WIN,
    MCTSNode,
)
from src.model import AlphaZeroNet
from src.self_play import (
    RANDOM_OPENING_PLIES,
    TEMP_THRESHOLD,
    _create_opening_game,
    _play_batched_python,
    _play_batched_rust,
    run_self_play,
)

# Verified 100%-legal 42-move draw game sequence (0-indexed columns)
DRAW_42_MOVES = [
    3, 3, 0, 2, 5, 4, 5, 6, 4, 2, 2, 3, 3, 5,
    5, 2, 2, 4, 3, 3, 1, 1, 2, 1, 5, 5, 0, 4,
    0, 4, 4, 6, 6, 6, 6, 0, 1, 0, 6, 0, 1, 1,
]

# Verified 100%-legal 34-move position with unique forced mate-in-3 (5 plies) at col 3,
# while cols 4 and 5 lose in 2 plies.
MATE_IN_3_MOVES = [
    2, 2, 4, 1, 2, 1, 6, 3, 3, 2, 0, 6, 6, 0, 5, 6, 1,
    1, 3, 0, 1, 5, 6, 0, 3, 6, 5, 1, 2, 3, 5, 2, 0, 0,
]


class DummyModel:
    """Fast customizable dummy model mimicking AlphaZeroNet.predict."""

    def __init__(
        self,
        priors: list[float] | np.ndarray | None = None,
        value: float = 0.0,
        p1_value: float | None = None,
        device: str = "cpu",
    ) -> None:
        self._param = torch.zeros(1, device=device)
        if priors is None:
            self.priors = np.full(COLS, 1.0 / COLS, dtype=np.float32)
        else:
            arr = np.asarray(priors, dtype=np.float32)
            self.priors = arr / arr.sum()
        self.value = float(value)
        self.p1_value = p1_value
        self.eval_calls = 0
        self.total_states_evaluated = 0

    def parameters(self):
        return iter([self._param])

    def eval(self):
        return self

    def predict(
        self, state: torch.Tensor, valid_moves: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval_calls += 1
        single = state.dim() == 3
        state_4d = state.unsqueeze(0) if single else state
        batch = state_4d.shape[0]
        self.total_states_evaluated += batch

        pol = torch.tensor(self.priors, dtype=torch.float32, device=state.device)
        pol = pol.unsqueeze(0).expand(batch, -1).clone()
        if valid_moves is not None:
            vm = valid_moves.unsqueeze(0) if single else valid_moves
            pol = pol.masked_fill(~vm, 0.0)
            sums = pol.sum(dim=1, keepdim=True).clamp(min=1e-8)
            pol = pol / sums

        if self.p1_value is not None:
            # Plane 2 is all 1s when current_player == 1, all 0s when current_player == -1
            is_p1 = state_4d[:, 2, 0, 0] > 0.5
            val = torch.where(
                is_p1,
                torch.tensor(float(self.p1_value), dtype=torch.float32, device=state.device),
                torch.tensor(float(-self.p1_value), dtype=torch.float32, device=state.device),
            )
        else:
            val = torch.full((batch,), self.value, dtype=torch.float32, device=state.device)

        if single:
            return pol.squeeze(0), val.squeeze(0)
        return pol, val


def make_game_from_cols(cols_zero_indexed: list[int]) -> ConnectFour:
    game = ConnectFour()
    for c in cols_zero_indexed:
        game.make_move(c)
    return game


# ── 1. Backpropagate perspective & docstring ─────────────────────────────────


class TestBackpropagatePerspective:
    def test_backpropagate_docstring_and_alternating_signs(self):
        doc = MCTSNode.backpropagate.__doc__ or ""
        assert "current_player" in doc
        assert "just moved" not in doc

        root_game = ConnectFour()
        root = MCTSNode(root_game)
        root.expand(np.full(COLS, 1.0 / COLS, dtype=np.float32))
        child = root.children[3]
        child.expand(np.full(COLS, 1.0 / COLS, dtype=np.float32))
        grandchild = child.children[3]

        # Value +0.75 from grandchild.game.current_player (P1) perspective:
        # grandchild gets +0.75, child (P2) gets -0.75, root (P1) gets +0.75
        grandchild.backpropagate(0.75)

        assert grandchild.visit_count == 1
        assert math.isclose(grandchild.q_value, 0.75)
        assert child.visit_count == 1
        assert math.isclose(child.q_value, -0.75)
        assert root.visit_count == 1
        assert math.isclose(root.q_value, 0.75)


# ── 2. Exact MCTS-Solver (Mate-in-1, Mate-in-2, Mate-in-3, Defense, Draw) ────


class TestMCTSSolver:
    def test_mate_in_1_immediate_propagation_and_zero_leaf_evals(self):
        # P1 at (0,0), (0,1), (0,2); P2 at (1,0), (1,1), (1,2). P1 to move: col 3 wins!
        game = make_game_from_cols([0, 0, 1, 1, 2, 2])
        assert game.current_player == 1

        # Adversarial model puts almost all prior mass on col 6 instead of winning col 3
        adv_model = DummyModel(priors=[0.01, 0.01, 0.01, 0.001, 0.01, 0.01, 0.949])
        mcts = MCTS(model=adv_model, num_simulations=25)

        visit_counts = mcts.search(game, add_noise=False)
        assert mcts.root is not None
        assert mcts.root.solver_state == SOLVER_WIN
        assert mcts.root.children[3].solver_state == SOLVER_LOSS
        assert int(np.argmax(visit_counts)) == 3
        assert visit_counts[3] == 25.0
        # Only 1 NN call to expand root; all 25 sims short-circuit on proven SOLVER_LOSS child!
        assert adv_model.eval_calls == 1

    def test_mate_in_2_double_threat_propagation(self):
        # P1 at (0,2), (0,3); P2 at (0,6), (1,6). P1 to move:
        # Playing col 1 creates open-ended 3-in-a-row (0,1)-(0,3) threatening BOTH col 0 & col 4!
        # Playing col 4 also creates open-ended 3-in-a-row (0,2)-(0,4) threatening BOTH col 1 & col 5!
        game = make_game_from_cols([2, 6, 3, 6])
        assert game.current_player == 1

        adv_model = DummyModel(priors=[0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.65])
        mcts = MCTS(model=adv_model, num_simulations=80)

        visit_counts = mcts.search(game, add_noise=False)
        best_col = int(np.argmax(visit_counts))

        assert mcts.root is not None
        assert mcts.root.solver_state == SOLVER_WIN
        assert best_col in (1, 4)
        assert mcts.root.children[best_col].solver_state == SOLVER_LOSS
        for reply_child in mcts.root.children[best_col].children.values():
            assert reply_child.solver_state == SOLVER_WIN

    def test_mate_in_3_five_ply_exact_propagation(self):
        # 34-ply legal position: valid columns are {3, 4, 5};
        # col 3 forces a win in 5 plies (mate-in-3), while cols 4 and 5 lose in 2 plies!
        game = make_game_from_cols(MATE_IN_3_MOVES)
        assert not game.is_terminal()

        # Adversarial model strongly prefers losing cols 4 & 5 over winning col 3
        adv_model = DummyModel(
            priors=[0.01, 0.01, 0.01, 0.02, 0.48, 0.46, 0.01], value=-0.4
        )
        mcts = MCTS(model=adv_model, num_simulations=40)

        visit_counts, move_values = mcts.search_with_values(game, add_noise=False)
        assert mcts.root is not None
        assert mcts.root.solver_state == SOLVER_WIN
        assert mcts.root.children[3].solver_state == SOLVER_LOSS
        assert mcts.root.children[4].solver_state == SOLVER_WIN
        assert mcts.root.children[5].solver_state == SOLVER_WIN
        assert int(np.argmax(visit_counts)) == 3
        assert move_values == {3: 100.0, 4: 0.0, 5: 0.0}

    def test_mate_in_1_defense_skips_proven_losing_moves(self):
        # (0,0)=P1, (0,1)=P2, (0,2)=P2, (0,3)=empty, (0,4)=P2, (0,5)=P1 => P2 threatens col 3!
        game = make_game_from_cols([0, 1, 0, 2, 5, 4])
        assert game.current_player == 1

        adv_model = DummyModel(priors=[0.10, 0.10, 0.10, 0.005, 0.10, 0.10, 0.495])
        mcts = MCTS(model=adv_model, num_simulations=30)

        action_probs, visit_counts = mcts.get_action_probs(
            game, temperature=0, add_noise=False
        )
        assert int(np.argmax(action_probs)) == 3
        assert mcts.root is not None
        for col in [0, 1, 2, 4, 5, 6]:
            assert mcts.root.children[col].solver_state == SOLVER_WIN
            assert visit_counts[col] == 0.0

    def test_solver_draw_propagation_on_nearly_full_board(self):
        game = make_game_from_cols(DRAW_42_MOVES[:40])
        assert not game.is_terminal()

        model = DummyModel(value=0.8)
        mcts = MCTS(model=model, num_simulations=15)
        _ = mcts.search(game, add_noise=False)

        assert mcts.root is not None
        assert mcts.root.solver_state == SOLVER_DRAW
        vals = mcts.get_move_values(game)
        assert vals == {1: 50.0}


# ── 3. First Play Urgency (FPU) reduction ───────────────────────────────────


class TestFPUReduction:
    def test_fpu_formula_when_parent_q_negative(self):
        game = ConnectFour()
        parent = MCTSNode(game)
        priors = np.array([0.02, 0.03, 0.15, 0.64, 0.10, 0.04, 0.02], dtype=np.float32)
        parent.expand(priors)

        # Simulate 4 visits through col 3 where parent is losing (parent.q_value = -0.5)
        for _ in range(4):
            parent.children[3].backpropagate(0.5)

        assert math.isclose(parent.q_value, -0.5)
        visited_policy = 0.64
        expected_neg_q = -0.5 - 0.20 * math.sqrt(visited_policy)  # -0.66
        expected_expl = 1.5 * priors[0] * math.sqrt(4) / 1.0
        actual_ucb = parent.children[0].ucb_score(
            c_puct=1.5, c_fpu=0.20, is_noisy_root=False
        )
        assert math.isclose(actual_ucb, expected_neg_q + expected_expl, rel_tol=1e-6)

        # Noisy root disables FPU reduction (neg_q = 0.0)
        noisy_ucb_col0 = parent.children[0].ucb_score(
            c_puct=1.5, c_fpu=0.20, is_noisy_root=True
        )
        assert math.isclose(noisy_ucb_col0, 0.0 + expected_expl, rel_tol=1e-6)
        assert parent.select_child(c_puct=1.5, c_fpu=0.20, is_noisy_root=False).action == 3

    def test_fpu_clamping_to_minus_one(self):
        game = ConnectFour()
        parent = MCTSNode(game)
        parent.expand(np.full(COLS, 1.0 / COLS, dtype=np.float32))
        parent.children[3].backpropagate(0.95)  # parent.q_value = -0.95

        fpu_val = parent._unvisited_fpu_value(c_fpu=0.50, is_noisy_root=False)
        assert fpu_val == -1.0


# ── 4. get_move_values at 0 and N simulations ───────────────────────────────


class TestGetMoveValues:
    def test_zero_sims_checks_terminal_before_nn_eval(self):
        # Mate-in-1 at col 3
        game = make_game_from_cols([0, 0, 1, 1, 2, 2])
        model = DummyModel(value=0.2)  # (-0.2 + 1)/2 * 100 = 40% for non-terminal
        mcts = MCTS(model=model, num_simulations=0)

        values = mcts.get_move_values(game)
        assert values[3] == 100.0
        for col in [0, 1, 2, 4, 5, 6]:
            assert math.isclose(values[col], 40.0, abs_tol=1e-4)
        # 1 root eval + 6 non-terminal child evals = 7 (col 3 terminal win NEVER sent to NN!)
        assert model.eval_calls == 7

    def test_zero_sims_terminal_draw_returns_50_without_nn_eval(self):
        game = make_game_from_cols(DRAW_42_MOVES[:41])
        assert not game.is_terminal()

        model = DummyModel(value=0.9)
        mcts = MCTS(model=model, num_simulations=0)
        values = mcts.get_move_values(game)
        assert values == {1: 50.0}
        # Only root expand call (1); terminal draw child was NOT passed to _evaluate!
        assert model.eval_calls == 1

    def test_n_sims_solver_and_unvisited_fpu_move_values(self):
        # Consistent player-aware evaluation: P1 has -0.6 across all depths => root.q_value == -0.6
        game = ConnectFour()
        priors = [0.001, 0.001, 0.10, 0.796, 0.10, 0.001, 0.001]
        model = DummyModel(priors=priors, p1_value=-0.6)
        mcts = MCTS(model=model, num_simulations=5, c_fpu=0.20)

        values = mcts.get_move_values(game)
        assert mcts.root is not None
        assert math.isclose(mcts.root.q_value, -0.6, abs_tol=1e-5)
        assert mcts.root.children[0].visit_count == 0

        expected_fpu_pct = (
            mcts.root._unvisited_fpu_value(c_fpu=0.20, is_noisy_root=False) + 1.0
        ) / 2.0 * 100.0
        assert math.isclose(values[0], expected_fpu_pct, abs_tol=1e-5)
        assert values[0] < 15.0  # ~11.0%, never misleading 50.0%!


# ── 5. Stateful Tree Reuse & play.py loop ───────────────────────────────────


class TestTreeReuseAndPlayCLI:
    def test_search_with_values_and_same_position_caching(self):
        game = ConnectFour()
        model = DummyModel(priors=[0.05, 0.10, 0.15, 0.40, 0.15, 0.10, 0.05])
        mcts = MCTS(model=model, num_simulations=20)

        # Hints call followed by AI move call on the SAME position shares 1 search!
        hint_values = mcts.get_move_values(game)
        calls_after_hints = model.eval_calls
        assert len(hint_values) == 7

        action_probs, visit_counts = mcts.get_action_probs(
            game, temperature=0, add_noise=False
        )
        assert model.eval_calls == calls_after_hints  # ZERO redundant NN evaluations!
        assert visit_counts.sum() == 20.0
        assert math.isclose(action_probs.sum(), 1.0)

    def test_advance_move_and_reset_across_turns(self):
        game = ConnectFour()
        model = DummyModel(priors=[0.02, 0.03, 0.10, 0.70, 0.10, 0.03, 0.02])
        mcts = MCTS(model=model, num_simulations=25)

        vc1, _ = mcts.search_with_values(game, add_noise=False)
        best_col = int(np.argmax(vc1))
        assert best_col == 3
        assert mcts.root is not None
        reused_sub_visits = sum(
            c.visit_count for c in mcts.root.children[best_col].children.values()
        )
        assert reused_sub_visits > 0

        # Mutate game in-place BEFORE advance_move (just as play.py does!)
        game.make_move(best_col)
        mcts.advance_move(best_col)
        assert mcts.root is not None
        assert mcts.root.parent is None

        calls_before_turn2 = model.eval_calls
        vc2 = mcts.search(game, add_noise=False)
        calls_turn2 = model.eval_calls - calls_before_turn2

        assert vc2.sum() == 25.0
        assert calls_turn2 <= 25 - reused_sub_visits

        mcts.reset()
        assert mcts.root is None

    def test_play_py_has_no_recursive_main_and_reuses_model_on_play_again(
        self, tmp_path: Path
    ):
        source = Path(play.__file__).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                recursive_calls = [
                    sub
                    for sub in ast.walk(node)
                    if isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "main"
                ]
                assert recursive_calls == [], "play.main() must not call main() recursively"

        ckpt_path = tmp_path / "test_ckpt.pt"
        tiny_model = AlphaZeroNet(num_res_blocks=1, num_filters=16)
        torch.save({"model_state_dict": tiny_model.state_dict(), "iteration": 1}, ckpt_path)

        def fake_input_gen():
            played_again = False
            col_cycle = ["1", "2", "3", "4", "5", "6", "7"]
            idx = 0
            while True:
                prompt = yield
                if "Play again?" in prompt:
                    if not played_again:
                        played_again = True
                        yield "y"
                    else:
                        yield "n"
                else:
                    yield col_cycle[idx % len(col_cycle)]
                    idx += 1

        gen = fake_input_gen()
        next(gen)

        def fake_input(prompt: str = "") -> str:
            resp = gen.send(prompt)
            next(gen)
            return resp

        load_count = 0
        orig_torch_load = torch.load

        def counting_load(*args, **kwargs):
            nonlocal load_count
            load_count += 1
            return orig_torch_load(*args, **kwargs)

        argv = [
            "play.py",
            "--checkpoint",
            str(ckpt_path),
            "--res-blocks",
            "1",
            "--filters",
            "16",
            "--simulations",
            "4",
            "--hints",
            "--cpu",
        ]
        with (
            patch("sys.argv", argv),
            patch("builtins.input", side_effect=fake_input),
            patch("torch.load", side_effect=counting_load),
        ):
            play.main()

        assert load_count == 1


# ── 6. Self-Play Defaults, Kwargs, Return Arrays, Openings & Policy Targets ──


class TestSelfPlay:
    def test_updated_self_play_defaults(self):
        assert TEMP_THRESHOLD == 10
        assert RANDOM_OPENING_PLIES == 4
        assert DEFAULT_DIRICHLET_ALPHA == 0.35
        assert DEFAULT_C_FPU == 0.20

    def test_run_self_play_forwards_all_kwargs(self):
        captured: dict[str, object] = {}

        def fake_play_batched_games(**kwargs):
            captured.update(kwargs)
            if kwargs.get("return_arrays"):
                return (
                    np.zeros((2, 3, ROWS, COLS), dtype=np.float32),
                    np.full((2, COLS), 1.0 / COLS, dtype=np.float32),
                    np.array([1.0, -1.0], dtype=np.float32),
                )
            return []

        dummy = DummyModel()
        with (
            patch.object(self_play, "RUST_AVAILABLE", False),
            patch.object(self_play, "play_batched_games", side_effect=fake_play_batched_games),
        ):
            states, policies, values = run_self_play(
                model=dummy,  # type: ignore[arg-type]
                num_games=6,
                num_simulations=12,
                c_puct=2.25,
                dirichlet_alpha=0.42,
                dirichlet_epsilon=0.15,
                temp_threshold=8,
                random_opening_plies=5,
                c_fpu=0.18,
                return_arrays=True,
            )

        assert captured["num_games"] == 6
        assert captured["num_simulations"] == 12
        assert captured["c_puct"] == 2.25
        assert captured["dirichlet_alpha"] == 0.42
        assert captured["dirichlet_epsilon"] == 0.15
        assert captured["temp_threshold"] == 8
        assert captured["random_opening_plies"] == 5
        assert captured["c_fpu"] == 0.18
        assert captured["return_arrays"] is True

        assert states.shape == (4, 3, ROWS, COLS)
        assert policies.shape == (4, COLS)
        assert values.shape == (4,)
        assert states.dtype == np.float32 and states.flags["C_CONTIGUOUS"]
        assert policies.dtype == np.float32 and policies.flags["C_CONTIGUOUS"]
        assert values.dtype == np.float32 and values.flags["C_CONTIGUOUS"]

    def test_opening_ply_distribution_covers_move_0_and_move_1_and_rolls_back_terminal(
        self,
    ):
        np.random.seed(12345)
        move_counts = [_create_opening_game(4)._move_count for _ in range(400)]
        count_0 = sum(1 for m in move_counts if m == 0)
        count_1 = sum(1 for m in move_counts if m == 1)
        count_2_to_4 = sum(1 for m in move_counts if 2 <= m <= 4)

        assert 150 <= count_0 <= 250
        assert 65 <= count_1 <= 135
        assert 65 <= count_2_to_4 <= 135

        for _ in range(50):
            g = _create_opening_game(24)
            assert not g.is_terminal()

    def test_python_self_play_return_arrays_and_move0_move1_coverage(self):
        np.random.seed(7)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AlphaZeroNet(num_res_blocks=1, num_filters=16).to(device)
        model.eval()

        with patch.object(self_play, "RUST_AVAILABLE", False):
            states, policies, values = run_self_play(
                model=model,
                num_games=16,
                num_simulations=6,
                random_opening_plies=4,
                return_arrays=True,
            )

        assert isinstance(states, np.ndarray)
        assert states.ndim == 4 and states.shape[1:] == (3, ROWS, COLS)
        assert policies.shape == (len(states), COLS)
        assert values.shape == (len(states),)
        assert states.dtype == np.float32 and states.flags["C_CONTIGUOUS"]
        assert policies.dtype == np.float32 and policies.flags["C_CONTIGUOUS"]
        assert values.dtype == np.float32 and values.flags["C_CONTIGUOUS"]
        assert len(states) % 2 == 0

        piece_counts = (states[:, 0] + states[:, 1]).sum(axis=(1, 2))
        assert np.any(piece_counts == 0), "Expected Move 0 (empty board) examples"
        assert np.any(piece_counts == 1), "Expected Move 1 (1-stone board) examples"
        npt.assert_allclose(policies.sum(axis=1), 1.0, atol=1e-5)

        with patch.object(self_play, "RUST_AVAILABLE", False):
            examples = run_self_play(
                model=model,
                num_games=2,
                num_simulations=4,
                random_opening_plies=2,
                return_arrays=False,
            )
        assert isinstance(examples, list) and len(examples) > 0
        s0, p0, v0 = examples[0]
        assert s0.shape == (3, ROWS, COLS)
        assert p0.shape == (COLS,)
        assert isinstance(v0, float)

    def test_decoupled_policy_targets_remain_smooth_under_greedy_temp_threshold(self):
        np.random.seed(99)
        dummy = DummyModel(priors=[0.12, 0.14, 0.16, 0.16, 0.16, 0.14, 0.12])

        states, policies, _ = _play_batched_python(
            model=dummy,  # type: ignore[arg-type]
            num_games=3,
            num_simulations=14,
            temp_threshold=0,
            random_opening_plies=0,
            return_arrays=True,
        )
        assert len(policies) > 0
        npt.assert_allclose(policies.sum(axis=1), 1.0, atol=1e-5)
        non_one_hot = np.sum(policies > 0.0, axis=1) > 1
        assert np.any(non_one_hot), (
            "Policy targets should be smooth visit distributions, not forced one-hot at >= temp_threshold"
        )

    def test_rust_batched_loop_uses_is_step_done_and_skips_zero_unexpanded_roots(self):
        class FakeRustEngine:
            instances: list[FakeRustEngine] = []

            def __init__(
                self,
                num_games,
                num_simulations,
                c_puct,
                dirichlet_alpha,
                dirichlet_epsilon,
                temp_threshold,
                random_opening_plies,
                c_fpu,
            ):
                self.num_games = num_games
                self.num_simulations = num_simulations
                self.c_fpu = c_fpu
                self.ply = 0
                self.sims_done = 0
                self.collect_calls_per_ply: list[int] = []
                self._cur_collect_calls = 0
                self.init_unexpanded_calls = 0
                FakeRustEngine.instances.append(self)

            def num_active(self) -> int:
                return self.num_games if self.ply < 2 else 0

            def is_done(self) -> bool:
                return self.ply >= 2

            def get_unexpanded_root_states(self):
                if self.ply == 0:
                    return (
                        np.zeros((self.num_games, 3, ROWS, COLS), dtype=np.float32),
                        np.ones((self.num_games, COLS), dtype=bool),
                        self.num_games,
                    )
                return (
                    np.zeros((0, 3, ROWS, COLS), dtype=np.float32),
                    np.ones((0, COLS), dtype=bool),
                    0,
                )

            def init_unexpanded_roots(self, policies: np.ndarray) -> None:
                assert policies.flags["C_CONTIGUOUS"] and policies.dtype == np.float32
                self.init_unexpanded_calls += 1

            def add_noise(self) -> None:
                self.sims_done = 0
                self._cur_collect_calls = 0

            def is_step_done(self) -> bool:
                return self.sims_done >= self.num_simulations

            def collect_leaves(self):
                self._cur_collect_calls += 1
                step = min(4, self.num_simulations - self.sims_done)
                self.sims_done += step
                return (
                    np.zeros((step, 3, ROWS, COLS), dtype=np.float32),
                    np.ones((step, COLS), dtype=bool),
                    step,
                )

            def apply_evaluations(self, policies: np.ndarray, values: np.ndarray) -> None:
                assert policies.flags["C_CONTIGUOUS"] and policies.dtype == np.float32
                assert values.flags["C_CONTIGUOUS"] and values.dtype == np.float32

            def advance_games(self):
                self.collect_calls_per_ply.append(self._cur_collect_calls)
                self.ply += 1
                if self.ply == 2:
                    return (
                        np.zeros((4, 3, ROWS, COLS), dtype=np.float32),
                        np.full((4, COLS), 1.0 / COLS, dtype=np.float32),
                        np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32),
                    )
                return (
                    np.empty((0, 3, ROWS, COLS), dtype=np.float32),
                    np.empty((0, COLS), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dummy = DummyModel(device=device)

        with (
            patch.object(self_play, "RUST_AVAILABLE", True),
            patch.object(self_play, "RustBatchedSelfPlay", FakeRustEngine, create=True),
        ):
            states, policies, values = _play_batched_rust(
                model=dummy,  # type: ignore[arg-type]
                num_games=2,
                num_simulations=20,
                c_fpu=0.20,
                return_arrays=True,
            )

        eng = FakeRustEngine.instances[-1]
        assert eng.collect_calls_per_ply == [5, 5]
        assert eng.init_unexpanded_calls == 1
        assert dummy.eval_calls == 11
        assert states.shape == (4, 3, ROWS, COLS)
        assert policies.shape == (4, COLS)
        assert values.shape == (4,)
