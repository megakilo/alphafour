"""Integration tests: Rust game engine & exact bitboard solver vs Python game engine.

Verifies that the Rust ConnectFour implementation produces identical
results to the Python ConnectFour for:
- Board state after sequences of moves & `from_moves`
- Win detection, draw detection, game results
- Board encoding & batch helpers (zero-copy numpy arrays)
- Exact Pascal-Pons bitboard W/D/L solver (`solve_position_wdl`,
  `solve_children_wdl`, `solve_batch_children_wdl`) verified against `testdata`

Run with: uv run pytest tests/test_rust_game.py -v
"""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import numpy.testing as npt
import pytest

from src.game import ConnectFour as PyConnectFour

try:
    from alphafour_engine import (
        RustConnectFour,
        batch_encode,
        batch_valid_moves,
        solve_batch_children_wdl,
        solve_children_wdl,
        solve_position_wdl,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE,
    reason="Rust engine not built. Run: uv sync",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTDATA_DIR = REPO_ROOT / "testdata"


def play_moves(moves: list[int]) -> tuple[PyConnectFour, RustConnectFour]:
    """Play a sequence of moves on both Python and Rust engines, return both."""
    py_game = PyConnectFour()
    rs_game = RustConnectFour()

    for col in moves:
        py_game.make_move(col)
        rs_game.make_move(col)

    return py_game, rs_game


def assert_games_equal(py_game: PyConnectFour, rs_game: RustConnectFour) -> None:
    """Assert that the Python and Rust game states are identical."""
    assert py_game.current_player == rs_game.current_player
    assert py_game._move_count == rs_game.move_count
    assert py_game.last_move == rs_game.last_move
    assert py_game.is_win() == rs_game.is_win()
    assert py_game.is_draw() == rs_game.is_draw()
    assert py_game.is_terminal() == rs_game.is_terminal()
    assert py_game.get_result() == rs_game.get_result()

    # Compare valid moves
    py_valid = py_game.get_valid_moves()
    rs_valid = np.array(rs_game.get_valid_moves())
    npt.assert_array_equal(py_valid, rs_valid)

    # Compare encoding
    py_encoded = py_game.encode()
    rs_encoded = np.array(rs_game.encode())
    npt.assert_array_almost_equal(py_encoded, rs_encoded)


class TestBasicOperations:
    """Test basic game operations match between Rust and Python."""

    def test_new_game(self):
        py_game, rs_game = play_moves([])
        assert_games_equal(py_game, rs_game)

    def test_single_move(self):
        py_game, rs_game = play_moves([3])
        assert_games_equal(py_game, rs_game)

    def test_two_moves(self):
        py_game, rs_game = play_moves([3, 4])
        assert_games_equal(py_game, rs_game)

    def test_same_column_stacking(self):
        py_game, rs_game = play_moves([0, 0, 0, 0, 0, 0])
        assert_games_equal(py_game, rs_game)

    def test_all_columns(self):
        py_game, rs_game = play_moves([0, 1, 2, 3, 4, 5, 6])
        assert_games_equal(py_game, rs_game)


class TestFromMoves:
    """Test RustConnectFour.from_moves(moves_str) 1-indexed move parser."""

    def test_empty_moves_string(self):
        rs_game = RustConnectFour.from_moves("")
        py_game = PyConnectFour()
        assert_games_equal(py_game, rs_game)

    def test_moves_sequence_matches_python(self):
        moves_str = "44531726"
        rs_game = RustConnectFour.from_moves(moves_str)
        py_game = PyConnectFour()
        for ch in moves_str:
            py_game.make_move(int(ch) - 1)
        assert_games_equal(py_game, rs_game)

    def test_terminal_final_move_allowed_but_post_terminal_rejected(self):
        rs_win = RustConnectFour.from_moves("1212121")
        assert rs_win.is_win()
        assert rs_win.is_terminal()

        with pytest.raises(ValueError):
            RustConnectFour.from_moves("12121212")

    def test_invalid_characters_and_overflow_raise_value_error(self):
        with pytest.raises(ValueError):
            RustConnectFour.from_moves("0")
        with pytest.raises(ValueError):
            RustConnectFour.from_moves("8")
        with pytest.raises(ValueError):
            RustConnectFour.from_moves("4444444")


class TestWinDetection:
    """Test that wins are detected identically."""

    def test_horizontal_win_bottom(self):
        moves = [0, 0, 1, 1, 2, 2, 3]
        py_game, rs_game = play_moves(moves)
        assert_games_equal(py_game, rs_game)
        assert py_game.is_win() is True
        assert rs_game.is_win() is True

    def test_vertical_win(self):
        moves = [0, 1, 0, 1, 0, 1, 0]
        py_game, rs_game = play_moves(moves)
        assert_games_equal(py_game, rs_game)
        assert py_game.is_win() is True

    def test_no_win_yet(self):
        moves = [0, 1, 2]
        py_game, rs_game = play_moves(moves)
        assert_games_equal(py_game, rs_game)
        assert py_game.is_win() is False
        assert rs_game.is_win() is False


class TestEncoding:
    """Test that board encoding matches exactly."""

    def test_empty_board_encoding(self):
        py_game, rs_game = play_moves([])
        py_enc = py_game.encode()
        rs_enc = np.array(rs_game.encode())
        npt.assert_array_equal(py_enc, rs_enc)
        assert py_enc[2].sum() == 42

    def test_encoding_after_one_move(self):
        py_game, rs_game = play_moves([3])
        py_enc = py_game.encode()
        rs_enc = np.array(rs_game.encode())
        npt.assert_array_equal(py_enc, rs_enc)
        assert py_enc[2].sum() == 0

    def test_encoding_symmetry(self):
        py_game, rs_game = play_moves([3, 4])
        py_enc = py_game.encode()
        rs_enc = np.array(rs_game.encode())
        npt.assert_array_equal(py_enc, rs_enc)
        assert py_enc[2].sum() == 42

    def test_encoding_multiple_pieces(self):
        moves = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5]
        py_game, rs_game = play_moves(moves)
        py_enc = py_game.encode()
        rs_enc = np.array(rs_game.encode())
        npt.assert_array_equal(py_enc, rs_enc)


class TestRandomGames:
    """Fuzz test: play many random games and verify Rust matches Python."""

    def test_1000_random_games(self):
        rng = random.Random(42)

        for game_idx in range(1000):
            py_game = PyConnectFour()
            rs_game = RustConnectFour()

            while not py_game.is_terminal():
                py_valid = py_game.get_valid_moves()
                rs_valid = np.array(rs_game.get_valid_moves())
                npt.assert_array_equal(
                    py_valid,
                    rs_valid,
                    err_msg=f"Game {game_idx}, move {py_game._move_count}",
                )

                valid_cols = [c for c in range(7) if py_valid[c]]
                col = rng.choice(valid_cols)

                py_game.make_move(col)
                rs_game.make_move(col)

                assert py_game.is_win() == rs_game.is_win()
                assert py_game.is_terminal() == rs_game.is_terminal()

            assert py_game.get_result() == rs_game.get_result()

    def test_100_random_games_full_encoding(self):
        rng = random.Random(123)

        for game_idx in range(100):
            py_game = PyConnectFour()
            rs_game = RustConnectFour()

            while not py_game.is_terminal():
                py_enc = py_game.encode()
                rs_enc = np.array(rs_game.encode())
                npt.assert_array_equal(
                    py_enc,
                    rs_enc,
                    err_msg=f"Game {game_idx}, move {py_game._move_count}",
                )

                valid_cols = py_game.get_valid_move_indices()
                col = rng.choice(valid_cols)
                py_game.make_move(col)
                rs_game.make_move(col)


class TestCopy:
    """Test that copy produces independent game states."""

    def test_copy_independence(self):
        rs_game = RustConnectFour()
        rs_game.make_move(3)
        rs_game.make_move(4)

        rs_copy = rs_game.copy()
        assert rs_copy.current_player == rs_game.current_player
        assert rs_copy.move_count == rs_game.move_count

        rs_game.make_move(5)
        assert rs_game.move_count != rs_copy.move_count

    def test_copy_encoding_matches(self):
        rs_game = RustConnectFour()
        rs_game.make_move(3)
        rs_copy = rs_game.copy()

        enc1 = np.array(rs_game.encode())
        enc2 = np.array(rs_copy.encode())
        npt.assert_array_equal(enc1, enc2)


class TestBatchOperations:
    """Test batch encode/valid_moves helpers."""

    def test_batch_encode(self):
        games = []
        for col in [0, 3, 6]:
            g = RustConnectFour()
            g.make_move(col)
            games.append(g)

        batch = batch_encode(games)
        assert batch.shape == (3, 3, 6, 7)

        for i, g in enumerate(games):
            individual = np.array(g.encode())
            npt.assert_array_equal(batch[i], individual)

    def test_batch_valid_moves(self):
        games = []
        for _ in range(5):
            g = RustConnectFour()
            g.make_move(0)
            games.append(g)

        batch = batch_valid_moves(games)
        assert batch.shape == (5, 7)

        for i, g in enumerate(games):
            individual = np.array(g.get_valid_moves())
            npt.assert_array_equal(batch[i], individual)

    def test_batch_encode_empty(self):
        batch = batch_encode([])
        assert batch.shape == (0, 3, 6, 7)


class TestExactBitboardSolver:
    """Test exact Connect Four bitboard W/D/L solver and testdata ground-truth accuracy."""

    def test_tactical_win_and_full_column(self):
        assert solve_position_wdl("") == 1
        assert solve_children_wdl("") == [-1, -1, 0, 1, 0, -1, -1]

        assert solve_position_wdl("121212") == 1
        children = solve_children_wdl("121212")
        assert len(children) == 7
        assert children[0] == 1  # Col 1 wins immediately for root mover

        # Full col 4 (0-indexed col 3) returns -2
        full_col = solve_children_wdl("4444441235671235671235")
        assert full_col[3] == -2
        assert max(full_col) == solve_position_wdl("4444441235671235671235")

        # Optional max_nodes budget returns -3 on deep opening when tiny budget is exceeded
        budgeted = solve_batch_children_wdl(["44"], max_nodes=5)
        assert budgeted.shape == (1, 7)
        assert np.any(budgeted == -3)

    def test_mirror_symmetry_and_position_consistency(self):
        samples = ["32164625", "6146", "274552224131661", "5554224333234511764415115"]
        for s in samples:
            mirror_s = "".join(str(8 - int(ch)) for ch in s)
            c1 = solve_children_wdl(s)
            c2 = solve_children_wdl(mirror_s)
            assert c1 == c2[::-1]
            assert max(c1) == solve_position_wdl(s)

    @pytest.mark.parametrize("dataset_name", ["Test_L3_R1", "Test_L1_R1", "Test_L2_R2"])
    def test_solver_exactness_against_testdata(self, dataset_name: str):
        path = TESTDATA_DIR / dataset_name
        if not path.exists():
            pytest.skip(f"{path} not found")

        moves_list: list[str] = []
        expected_signs: list[int] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            m, score_str = line.split()
            score = int(score_str)
            moves_list.append(m)
            expected_signs.append(1 if score > 0 else (-1 if score < 0 else 0))

        batch_wdl = solve_batch_children_wdl(moves_list)
        assert batch_wdl.shape == (len(moves_list), 7)
        assert batch_wdl.dtype == np.int8

        # Every entry is in {-2, -1, 0, 1}, and max over valid cols equals sign(score)!
        assert np.all((batch_wdl >= -2) & (batch_wdl <= 1))
        max_valid_wdl = batch_wdl.max(axis=1)
        expected_arr = np.array(expected_signs, dtype=np.int8)
        npt.assert_array_equal(max_valid_wdl, expected_arr)

    def test_solver_invalid_input_raises_value_error(self):
        with pytest.raises(ValueError):
            solve_position_wdl("invalid")
        with pytest.raises(ValueError):
            solve_children_wdl("999")
        with pytest.raises(ValueError):
            solve_batch_children_wdl(["44", "0"])
