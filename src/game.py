"""Connect Four game engine using numpy arrays for board representation."""

from __future__ import annotations

import numpy as np

ROWS = 6
COLS = 7
WIN_LENGTH = 4


class ConnectFour:
    """Connect Four game state.

    Players are 1 and -1. Player 1 moves first.
    Board uses numpy array: 0=empty, 1=player1, -1=player2.
    """

    __slots__ = ("board", "current_player", "last_move", "_heights", "_move_count")

    def __init__(self) -> None:
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player: int = 1
        self.last_move: int | None = None
        # Track the next free row for each column (0 = bottom row)
        self._heights = np.zeros(COLS, dtype=np.int8)
        self._move_count: int = 0

    def copy(self) -> ConnectFour:
        """Return a deep copy of this game state."""
        g = ConnectFour.__new__(ConnectFour)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.last_move = self.last_move
        g._heights = self._heights.copy()
        g._move_count = self._move_count
        return g

    def get_valid_moves(self) -> np.ndarray:
        """Return boolean array of valid columns."""
        return self._heights < ROWS

    def get_valid_move_indices(self) -> list[int]:
        """Return list of valid column indices."""
        return [c for c in range(COLS) if self._heights[c] < ROWS]

    def make_move(self, col: int) -> ConnectFour:
        """Make a move and return self (for chaining). Mutates in-place."""
        row = self._heights[col]
        assert row < ROWS, f"Column {col} is full"
        self.board[row, col] = self.current_player
        self._heights[col] = row + 1
        self.last_move = col
        self._move_count += 1
        self.current_player = -self.current_player
        return self

    def is_win(self) -> bool:
        """Check if the last move created a win."""
        if self.last_move is None:
            return False
        col = self.last_move
        row = self._heights[col] - 1
        player = self.board[row, col]  # The player who just moved
        return self._check_win_at(row, col, player)

    def _check_win_at(self, row: int, col: int, player: int) -> bool:
        """Check if there's a 4-in-a-row through (row, col) for player."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag, anti-diag
        for dr, dc in directions:
            count = 1
            # Check positive direction
            for i in range(1, WIN_LENGTH):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                    count += 1
                else:
                    break
            # Check negative direction
            for i in range(1, WIN_LENGTH):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                    count += 1
                else:
                    break
            if count >= WIN_LENGTH:
                return True
        return False

    def is_draw(self) -> bool:
        """Check if the board is full (draw)."""
        return self._move_count >= ROWS * COLS

    def is_terminal(self) -> bool:
        """Check if the game is over."""
        return self.is_win() or self.is_draw()

    def get_result(self) -> float | None:
        """Get game result from the perspective of the current player.

        Returns:
            1.0 if current player won (shouldn't happen since we switched),
            -1.0 if opponent (whoever just moved) won,
            0.0 for draw,
            None if game is not over.
        """
        if self.is_win():
            # The player who just moved won, so current player lost
            return -1.0
        if self.is_draw():
            return 0.0
        return None

    def get_canonical_board(self) -> np.ndarray:
        """Return board from current player's perspective.

        Current player's pieces are +1, opponent's are -1.
        """
        return self.board * self.current_player

    def encode(self) -> np.ndarray:
        """Encode board as 3-plane tensor for neural network input.

        Plane 0: current player's pieces (1 where current player has a piece)
        Plane 1: opponent's pieces (1 where opponent has a piece)
        Plane 2: constant plane indicating current player (all 1s if player 1, all 0s if player -1)
        """
        canonical = self.get_canonical_board()
        planes = np.zeros((3, ROWS, COLS), dtype=np.float32)
        planes[0] = (canonical == 1).astype(np.float32)
        planes[1] = (canonical == -1).astype(np.float32)
        if self.current_player == 1:
            planes[2] = 1.0
        return planes

    def mirror(self) -> ConnectFour:
        """Return a horizontally mirrored copy of the game state."""
        g = ConnectFour.__new__(ConnectFour)
        g.board = self.board[:, ::-1].copy()
        g.current_player = self.current_player
        g.last_move = COLS - 1 - self.last_move if self.last_move is not None else None
        g._heights = self._heights[::-1].copy()
        g._move_count = self._move_count
        return g

    def display(self, show_col_numbers: bool = True, last_move_highlight: bool = True) -> str:
        """Return a string representation of the board for console display.

        Uses ANSI colors: Red (●) for player 1, Yellow (●) for player -1.
        Last move is highlighted with a red background.
        Grid style:
            1   2   3   4   5   6   7
          +---+---+---+---+---+---+---+
          | · | · | · | · | · | · | · |
          +---+---+---+---+---+---+---+
        """
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RED_BG = "\033[41m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"

        lines: list[str] = []
        separator = "  " + "+---" * COLS + "+"

        # Column numbers header (centered over each cell)
        if show_col_numbers:
            header = "  "
            for c in range(COLS):
                header += f"  {BOLD}{c + 1}{RESET} "
            lines.append(header)

        # Top border
        lines.append(separator)

        # Board rows (top to bottom visually = high row index to low)
        for row in range(ROWS - 1, -1, -1):
            row_str = "  "
            for col in range(COLS):
                piece = self.board[row, col]
                is_last = (
                    last_move_highlight
                    and self.last_move == col
                    and self._heights[col] - 1 == row
                )

                if piece == 1:
                    symbol = f"{RED}●{RESET}"
                elif piece == -1:
                    symbol = f"{YELLOW}●{RESET}"
                else:
                    symbol = f"{DIM}·{RESET}"

                if is_last and piece != 0:
                    row_str += f"|{RED_BG} {symbol}{RED_BG} {RESET}"
                else:
                    row_str += f"| {symbol} "

            row_str += "|"
            lines.append(row_str)
            lines.append(separator)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ConnectFour(move_count={self._move_count}, player={self.current_player})"
