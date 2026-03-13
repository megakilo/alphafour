import numpy as np

class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.action_size = self.cols

    def get_initial_state(self):
        """Returns the initial empty board (6x7)."""
        return np.zeros((self.rows, self.cols), dtype=np.int8)

    def get_next_state(self, board, player, action):
        """
        Applies an action (0-6) by 'player' (1 or -1) to the board.
        Returns the new board state.
        """
        # Create a copy so we don't mutate the original board during MCTS
        new_board = np.copy(board)
        # Find the lowest empty slot in the chosen column
        for r in range(self.rows - 1, -1, -1):
            if new_board[r][action] == 0:
                new_board[r][action] = player
                return new_board
        # If column is full, this shouldn't happen if valid_moves are checked
        raise ValueError("Invalid move: Column is full")

    def get_valid_moves(self, board):
        """
        Returns a boolean array of size 7. 
        True if the column is not full (top row is 0).
        """
        return board[0] == 0

    def check_win(self, board, action):
        """
        Check if the last action resulted in a win.
        We know the last move was in 'action' column. We find the row.
        Returns the winning player (1 or -1), or None if no win.
        """
        if action is None:
            return None
            
        # Find the row where the last piece was placed
        row = -1
        for r in range(self.rows):
            if board[r][action] != 0:
                row = r
                break
        
        if row == -1:
            return None
            
        player = board[row][action]

        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal \
            (1, -1)   # Diagonal /
        ]

        for dr, dc in directions:
            count = 1
            # Check positive direction
            r, c = row + dr, action + dc
            while 0 <= r < self.rows and 0 <= c < self.cols and board[r][c] == player:
                count += 1
                r += dr
                c += dc
                
            # Check negative direction
            r, c = row - dr, action - dc
            while 0 <= r < self.rows and 0 <= c < self.cols and board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
                
            if count >= 4:
                return player

        return None

    def get_canonical_form(self, board, player):
        """
        Returns the board from the perspective of the current player.
        """
        return board * player

    def get_game_ended(self, board, player, action=None):
        """
        Returns:
            0 if not ended
            1 if player won
            -1 if player lost
            1e-4 if draw
        """
        if action is not None:
            # Fast path: only check around the last action
            win_player = self.check_win(board, action)
            if win_player is not None:
                if win_player == player:
                    return 1
                else:
                    return -1
        else:
            # Slow path: scan all columns for a win (used when action is unknown)
            for col in range(self.cols):
                win_player = self.check_win(board, col)
                if win_player is not None:
                    if win_player == player:
                        return 1
                    else:
                        return -1
        
        if not np.any(self.get_valid_moves(board)):
            return 1e-4 # Draw
            
        return 0 # Ongoing

    def string_representation(self, board):
        """Returns a string representation of the board for hashing in MCTS."""
        return board.tobytes()
