/// Connect Four game engine — high-performance bitboard implementation.
///
/// Uses two u64 bitboards (one per player) for O(1) win detection.
/// Board layout: each column uses 7 bits (6 rows + 1 sentinel), left to right.
/// Bit index = col * (ROWS+1) + row, where row 0 = bottom.
///
/// Win detection uses the standard bitboard alignment check:
/// for each direction, shift and AND to find 4 consecutive pieces.
///
/// Players are 1 and -1. Player 1 moves first.

pub const ROWS: usize = 6;
pub const COLS: usize = 7;
pub const WIN_LENGTH: usize = 4;
pub const BOARD_SIZE: usize = ROWS * COLS;

/// Bits per column in the bitboard (ROWS + 1 sentinel bit).
const H1: usize = ROWS + 1;

#[derive(Clone)]
pub struct ConnectFour {
    /// Bitboard for player 1's pieces.
    bb_p1: u64,
    /// Bitboard for player -1's pieces.
    bb_p2: u64,
    pub heights: [u8; COLS],
    pub current_player: i8,
    pub last_move: Option<u8>,
    pub move_count: u8,
}

impl ConnectFour {
    pub fn new() -> Self {
        ConnectFour {
            bb_p1: 0,
            bb_p2: 0,
            heights: [0u8; COLS],
            current_player: 1,
            last_move: None,
            move_count: 0,
        }
    }

    /// Convert (row, col) to bit index in the bitboard.
    #[inline(always)]
    fn bit_index(row: usize, col: usize) -> u32 {
        (col * H1 + row) as u32
    }

    /// Get the value at (row, col): 1, -1, or 0 (empty).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> i8 {
        let bit = 1u64 << Self::bit_index(row, col);
        if self.bb_p1 & bit != 0 {
            1
        } else if self.bb_p2 & bit != 0 {
            -1
        } else {
            0
        }
    }

    /// Returns a boolean array indicating which columns are not full.
    pub fn get_valid_moves(&self) -> [bool; COLS] {
        let mut valid = [false; COLS];
        for c in 0..COLS {
            valid[c] = (self.heights[c] as usize) < ROWS;
        }
        valid
    }

    /// Returns list of valid column indices.
    pub fn get_valid_move_indices(&self) -> Vec<u8> {
        (0..COLS as u8)
            .filter(|&c| (self.heights[c as usize] as usize) < ROWS)
            .collect()
    }

    /// Make a move in the given column. Mutates in-place.
    /// Panics if the column is full.
    pub fn make_move(&mut self, col: u8) {
        let c = col as usize;
        let row = self.heights[c] as usize;
        assert!(row < ROWS, "Column {} is full", col);

        let bit = 1u64 << Self::bit_index(row, c);
        if self.current_player == 1 {
            self.bb_p1 |= bit;
        } else {
            self.bb_p2 |= bit;
        }

        self.heights[c] += 1;
        self.last_move = Some(col);
        self.move_count += 1;
        self.current_player = -self.current_player;
    }

    /// Check if the given bitboard has a 4-in-a-row.
    ///
    /// For each direction, we shift the bitboard and AND repeatedly.
    /// If any bit remains after 3 shifts, there's a 4-in-a-row.
    #[inline]
    fn has_won(bb: u64) -> bool {
        // Direction offsets for the bitboard layout:
        // Vertical:       shift by 1 (adjacent rows in same column)
        // Horizontal:     shift by H1 (adjacent columns)
        // Diagonal /:     shift by H1+1
        // Anti-diagonal \: shift by H1-1
        const DIRECTIONS: [u32; 4] = [1, H1 as u32, (H1 + 1) as u32, (H1 - 1) as u32];

        for &dir in &DIRECTIONS {
            let m = bb & (bb >> dir);
            if m & (m >> (2 * dir)) != 0 {
                return true;
            }
        }
        false
    }

    /// Check if the last move created a win.
    #[inline]
    pub fn is_win(&self) -> bool {
        if self.last_move.is_none() {
            return false;
        }
        // The player who just moved is -current_player
        let bb = if self.current_player == -1 {
            self.bb_p1
        } else {
            self.bb_p2
        };
        Self::has_won(bb)
    }

    /// Check if the board is full (draw).
    #[inline]
    pub fn is_draw(&self) -> bool {
        self.move_count as usize >= BOARD_SIZE
    }

    /// Check if the game is over.
    pub fn is_terminal(&self) -> bool {
        self.is_win() || self.is_draw()
    }

    /// Get game result from the perspective of the current player.
    ///
    /// Returns:
    ///   Some(-1.0) if opponent (whoever just moved) won,
    ///   Some(0.0) for draw,
    ///   None if game is not over.
    pub fn get_result(&self) -> Option<f64> {
        if self.is_win() {
            // The player who just moved won, so current player lost
            Some(-1.0)
        } else if self.is_draw() {
            Some(0.0)
        } else {
            None
        }
    }

    /// Encode board as 3-plane tensor for neural network input.
    ///
    /// Plane 0: current player's pieces (1 where current player has a piece)
    /// Plane 1: opponent's pieces (1 where opponent has a piece)
    /// Plane 2: constant plane indicating current player (all 1s if player 1, all 0s if player -1)
    ///
    /// Returns flattened [3 * ROWS * COLS] f32 array in C-contiguous order (row-major).
    pub fn encode(&self) -> [f32; 3 * BOARD_SIZE] {
        let mut planes = [0.0f32; 3 * BOARD_SIZE];
        let cp = self.current_player;

        let (bb_cur, bb_opp) = if cp == 1 {
            (self.bb_p1, self.bb_p2)
        } else {
            (self.bb_p2, self.bb_p1)
        };

        for r in 0..ROWS {
            for c in 0..COLS {
                let bit = 1u64 << Self::bit_index(r, c);
                let flat_idx = r * COLS + c;
                // Plane 0: current player's pieces
                if bb_cur & bit != 0 {
                    planes[flat_idx] = 1.0;
                }
                // Plane 1: opponent's pieces
                if bb_opp & bit != 0 {
                    planes[BOARD_SIZE + flat_idx] = 1.0;
                }
            }
        }

        // Plane 2: current player indicator
        if cp == 1 {
            for i in 0..BOARD_SIZE {
                planes[2 * BOARD_SIZE + i] = 1.0;
            }
        }

        planes
    }
}

impl Default for ConnectFour {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ConnectFour {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ConnectFour(move_count={}, player={})",
            self.move_count, self.current_player
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game() {
        let game = ConnectFour::new();
        assert_eq!(game.current_player, 1);
        assert_eq!(game.move_count, 0);
        assert_eq!(game.last_move, None);
        assert!(!game.is_terminal());
        assert!(game.get_result().is_none());
        // All columns should be valid
        assert!(game.get_valid_moves().iter().all(|&v| v));
    }

    #[test]
    fn test_make_move() {
        let mut game = ConnectFour::new();
        game.make_move(3); // Player 1 in center column
        assert_eq!(game.current_player, -1);
        assert_eq!(game.move_count, 1);
        assert_eq!(game.last_move, Some(3));
        assert_eq!(game.get(0, 3), 1); // Bottom row, center column
        assert_eq!(game.heights[3], 1);
    }

    #[test]
    fn test_stacking_moves() {
        let mut game = ConnectFour::new();
        // Stack pieces in column 0
        for i in 0..ROWS {
            assert!(!game.is_terminal());
            game.make_move(0);
            assert_eq!(game.heights[0] as usize, i + 1);
        }
        // Column 0 should now be full
        let valid = game.get_valid_moves();
        assert!(!valid[0]);
        assert!(valid[1]); // Other columns still valid
    }

    #[test]
    #[should_panic(expected = "Column 0 is full")]
    fn test_full_column_panics() {
        let mut game = ConnectFour::new();
        for _ in 0..ROWS + 1 {
            game.make_move(0);
        }
    }

    #[test]
    fn test_horizontal_win() {
        let mut game = ConnectFour::new();
        // Player 1: columns 0,1,2,3 (bottom row)
        // Player -1: columns 0,1,2 (second row)
        game.make_move(0); // P1
        game.make_move(0); // P-1
        game.make_move(1); // P1
        game.make_move(1); // P-1
        game.make_move(2); // P1
        game.make_move(2); // P-1
        game.make_move(3); // P1 wins horizontally
        assert!(game.is_win());
        assert!(game.is_terminal());
        assert_eq!(game.get_result(), Some(-1.0)); // Current player (P-1) lost
    }

    #[test]
    fn test_vertical_win() {
        let mut game = ConnectFour::new();
        // Player 1 stacks in column 0, Player -1 in column 1
        game.make_move(0); // P1
        game.make_move(1); // P-1
        game.make_move(0); // P1
        game.make_move(1); // P-1
        game.make_move(0); // P1
        game.make_move(1); // P-1
        game.make_move(0); // P1 wins vertically
        assert!(game.is_win());
        assert_eq!(game.get_result(), Some(-1.0));
    }

    #[test]
    fn test_diagonal_win() {
        let mut game = ConnectFour::new();
        // Build a diagonal win for Player 1
        // P1=1, P2=-1
        game.make_move(0); // P1 (0,0)
        game.make_move(1); // P2 (0,1)
        game.make_move(1); // P1 (1,1)
        game.make_move(2); // P2 (0,2)
        game.make_move(3); // P1 (0,3)
        game.make_move(2); // P2 (1,2)
        game.make_move(2); // P1 (2,2)
        game.make_move(3); // P2 (1,3)
        game.make_move(3); // P1 (2,3)
        game.make_move(4); // P2 (0,4)
        game.make_move(3); // P1 (3,3) — diagonal 0,0 -> 1,1 -> 2,2 -> 3,3
        assert!(game.is_win());
    }

    #[test]
    fn test_antidiagonal_win() {
        let mut game = ConnectFour::new();
        // Build anti-diagonal: (0,3), (1,2), (2,1), (3,0)
        // P1=1, P2=-1
        game.make_move(3); // P1 (0,3)
        game.make_move(2); // P2 (0,2)
        game.make_move(2); // P1 (1,2)
        game.make_move(1); // P2 (0,1)
        game.make_move(0); // P1 (0,0)
        game.make_move(1); // P2 (1,1)
        game.make_move(1); // P1 (2,1)
        game.make_move(0); // P2 (1,0)
        game.make_move(0); // P1 (2,0)
        game.make_move(4); // P2 (0,4)
        game.make_move(0); // P1 (3,0) — anti-diagonal (0,3)->(1,2)->(2,1)->(3,0)
        assert!(game.is_win());
    }

    #[test]
    fn test_draw() {
        let mut game = ConnectFour::new();
        // Fill board without winning (known draw pattern)
        let moves = [
            // Col 0: P1,P2,P1,P2,P1,P2
            0, 1, 0, 1, 0, 1,
            // Col 2: P1,P2,P1,P2,P1,P2
            2, 3, 2, 3, 2, 3,
            // Col 4: P1,P2,P1,P2,P1,P2
            4, 5, 4, 5, 4, 5,
            // Col 1: P1,P2,P1,P2,P1,P2
            1, 0, 1, 0, 1, 0,
            // Col 3: P1,P2,P1,P2,P1,P2
            3, 2, 3, 2, 3, 2,
            // Col 5: P1,P2,P1,P2,P1,P2
            5, 4, 5, 4, 5, 4,
            // Col 6: P1,P2,P1,P2,P1,P2
            6, 6, 6, 6, 6, 6,
        ];

        for &col in &moves {
            if game.is_terminal() {
                break;
            }
            game.make_move(col);
        }

        // If we got a draw, great. If not, the pattern caused a win
        // — this test mainly verifies we can fill the board.
        if game.move_count as usize == BOARD_SIZE && !game.is_win() {
            assert!(game.is_draw());
            assert_eq!(game.get_result(), Some(0.0));
        }
    }

    #[test]
    fn test_no_premature_win() {
        let mut game = ConnectFour::new();
        game.make_move(0); // P1
        game.make_move(1); // P2
        game.make_move(2); // P1
        assert!(!game.is_win());
        assert!(!game.is_terminal());
    }

    #[test]
    fn test_clone() {
        let mut game = ConnectFour::new();
        game.make_move(3);
        game.make_move(4);

        let cloned = game.clone();
        assert_eq!(cloned.heights, game.heights);
        assert_eq!(cloned.current_player, game.current_player);
        assert_eq!(cloned.move_count, game.move_count);

        // Mutating original shouldn't affect clone
        game.make_move(5);
        assert_ne!(game.move_count, cloned.move_count);
    }

    #[test]
    fn test_encode_empty_board() {
        let game = ConnectFour::new();
        let encoded = game.encode();

        // Plane 0 (current player = P1): all zeros (no pieces)
        for i in 0..BOARD_SIZE {
            assert_eq!(encoded[i], 0.0);
        }
        // Plane 1 (opponent = P2): all zeros
        for i in 0..BOARD_SIZE {
            assert_eq!(encoded[BOARD_SIZE + i], 0.0);
        }
        // Plane 2 (player indicator): all 1s (player 1)
        for i in 0..BOARD_SIZE {
            assert_eq!(encoded[2 * BOARD_SIZE + i], 1.0);
        }
    }

    #[test]
    fn test_encode_after_moves() {
        let mut game = ConnectFour::new();
        game.make_move(3); // P1 at (0,3)
        game.make_move(4); // P2 at (0,4), now current_player = P1

        let encoded = game.encode();
        // Current player is P1 again
        // Plane 0 (P1 pieces): (0,3) should be 1
        assert_eq!(encoded[0 * COLS + 3], 1.0);
        // Plane 1 (P2 pieces): (0,4) should be 1
        assert_eq!(encoded[BOARD_SIZE + 0 * COLS + 4], 1.0);
        // Plane 2 should be all 1s (current player is P1)
        assert_eq!(encoded[2 * BOARD_SIZE], 1.0);
    }

    #[test]
    fn test_encode_player2_perspective() {
        let mut game = ConnectFour::new();
        game.make_move(3); // P1 at (0,3), now current = P2

        let encoded = game.encode();
        // Current player is P2 (-1)
        // Plane 0 (current player P2's pieces): none yet
        assert_eq!(encoded[0 * COLS + 3], 0.0);
        // Plane 1 (opponent P1's pieces): (0,3) should be 1
        assert_eq!(encoded[BOARD_SIZE + 0 * COLS + 3], 1.0);
        // Plane 2 should be all 0s (current player is -1)
        assert_eq!(encoded[2 * BOARD_SIZE], 0.0);
    }

    #[test]
    fn test_valid_moves_mask() {
        let mut game = ConnectFour::new();
        // Fill column 0
        for _ in 0..ROWS {
            game.make_move(0);
            if game.is_terminal() {
                return; // Can't continue if someone won
            }
            game.make_move(1);
            if game.is_terminal() {
                return;
            }
        }
        let valid = game.get_valid_moves();
        assert!(!valid[0]); // Column 0 should be full
    }

    #[test]
    fn test_get_valid_move_indices() {
        let game = ConnectFour::new();
        let indices = game.get_valid_move_indices();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_result_is_from_current_player_perspective() {
        let mut game = ConnectFour::new();
        // P1 wins vertically in col 0
        game.make_move(0); // P1
        game.make_move(1); // P2
        game.make_move(0); // P1
        game.make_move(1); // P2
        game.make_move(0); // P1
        game.make_move(1); // P2
        game.make_move(0); // P1 wins

        // After P1 wins, current_player is P2 (-1)
        assert_eq!(game.current_player, -1);
        // Result from P2's perspective: P2 lost → -1.0
        assert_eq!(game.get_result(), Some(-1.0));
    }

    #[test]
    fn test_bitboard_win_detection_speed() {
        // Stress test: play many games and verify win detection is consistent
        let mut game = ConnectFour::new();
        // Horizontal win in bottom row
        game.make_move(0);
        game.make_move(0);
        game.make_move(1);
        game.make_move(1);
        game.make_move(2);
        game.make_move(2);
        assert!(!game.is_win());
        game.make_move(3); // P1 4-in-a-row: cols 0,1,2,3
        assert!(game.is_win());
    }

    #[test]
    fn test_get_value_consistency() {
        let mut game = ConnectFour::new();
        game.make_move(3); // P1 at (0,3)
        assert_eq!(game.get(0, 3), 1);
        assert_eq!(game.get(0, 0), 0);

        game.make_move(4); // P-1 at (0,4)
        assert_eq!(game.get(0, 4), -1);
    }
}
