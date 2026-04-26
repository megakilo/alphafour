"""Integration tests: Rust game engine vs Python game engine.

Verifies that the Rust ConnectFour implementation produces identical
results to the Python ConnectFour for:
- Board state after sequences of moves
- Win detection
- Draw detection
- Game results
- Board encoding (numpy arrays)
- Valid moves mask

Run with: pytest tests/test_rust_game.py -v
"""

import numpy as np
import numpy.testing as npt
import pytest
import random

from src.game import ConnectFour as PyConnectFour

try:
    from alphafour_engine import RustConnectFour, batch_encode, batch_valid_moves

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE,
    reason="Rust engine not built. Run: cd rust && maturin develop --release",
)


def play_moves(moves: list[int]):
    """Play a sequence of moves on both Python and Rust engines, return both."""
    py_game = PyConnectFour()
    rs_game = RustConnectFour()

    for col in moves:
        py_game.make_move(col)
        rs_game.make_move(col)

    return py_game, rs_game


def assert_games_equal(py_game: PyConnectFour, rs_game: RustConnectFour):
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


class TestWinDetection:
    """Test that wins are detected identically."""

    def test_horizontal_win_bottom(self):
        moves = [0, 0, 1, 1, 2, 2, 3]  # P1 wins bottom row
        py_game, rs_game = play_moves(moves)
        assert_games_equal(py_game, rs_game)
        assert py_game.is_win() == True
        assert rs_game.is_win() == True

    def test_vertical_win(self):
        moves = [0, 1, 0, 1, 0, 1, 0]  # P1 wins column 0
        py_game, rs_game = play_moves(moves)
        assert_games_equal(py_game, rs_game)
        assert py_game.is_win() == True

    def test_no_win_yet(self):
        moves = [0, 1, 2]
        py_game, rs_game = play_moves(moves)
        assert_games_equal(py_game, rs_game)
        assert py_game.is_win() == False
        assert rs_game.is_win() == False


class TestEncoding:
    """Test that board encoding matches exactly."""

    def test_empty_board_encoding(self):
        py_game, rs_game = play_moves([])
        py_enc = py_game.encode()
        rs_enc = np.array(rs_game.encode())
        npt.assert_array_equal(py_enc, rs_enc)
        # Plane 2 should be all 1s (player 1)
        assert py_enc[2].sum() == 42

    def test_encoding_after_one_move(self):
        py_game, rs_game = play_moves([3])
        py_enc = py_game.encode()
        rs_enc = np.array(rs_game.encode())
        npt.assert_array_equal(py_enc, rs_enc)
        # After P1 moves, current player is P2 (-1)
        # Plane 2 should be all 0s
        assert py_enc[2].sum() == 0

    def test_encoding_symmetry(self):
        """After 2 moves, current player is back to P1."""
        py_game, rs_game = play_moves([3, 4])
        py_enc = py_game.encode()
        rs_enc = np.array(rs_game.encode())
        npt.assert_array_equal(py_enc, rs_enc)
        assert py_enc[2].sum() == 42  # Player 1 indicator

    def test_encoding_multiple_pieces(self):
        moves = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5]
        py_game, rs_game = play_moves(moves)
        py_enc = py_game.encode()
        rs_enc = np.array(rs_game.encode())
        npt.assert_array_equal(py_enc, rs_enc)


class TestRandomGames:
    """Fuzz test: play many random games and verify Rust matches Python."""

    def test_1000_random_games(self):
        """Play 1000 random games, checking state consistency at every move."""
        rng = random.Random(42)  # Fixed seed for reproducibility

        for game_idx in range(1000):
            py_game = PyConnectFour()
            rs_game = RustConnectFour()

            while not py_game.is_terminal():
                # Get valid moves from Python
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

                # Check state matches after every move
                assert py_game.is_win() == rs_game.is_win(), (
                    f"Game {game_idx}: is_win mismatch after move {col}"
                )
                assert py_game.is_terminal() == rs_game.is_terminal(), (
                    f"Game {game_idx}: is_terminal mismatch after move {col}"
                )

            # Final state check
            assert py_game.get_result() == rs_game.get_result(), (
                f"Game {game_idx}: result mismatch"
            )

    def test_100_random_games_full_encoding(self):
        """Play 100 random games, checking full encoding at every move."""
        rng = random.Random(123)

        for game_idx in range(100):
            py_game = PyConnectFour()
            rs_game = RustConnectFour()

            while not py_game.is_terminal():
                # Check encoding matches
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

        # Mutating original shouldn't affect copy
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

        # Verify each game's encoding matches individual encode
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
