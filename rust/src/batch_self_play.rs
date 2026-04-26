/// Batched self-play engine for AlphaZero training.
///
/// Manages N concurrent games, each with its own MCTS tree.
/// The Python side calls collect_leaves() → GPU inference → apply_evaluations()
/// in a loop, then advance_games() to sample actions and progress games.

use crate::game::{ConnectFour, BOARD_SIZE, COLS, ROWS};
use crate::mcts::MCTSTree;
use rand::Rng;

/// A single training example: (encoded_state, action_probs, value).
pub struct Example {
    pub state: [f32; 3 * BOARD_SIZE],
    pub policy: [f32; COLS],
    pub value: f32,
}

/// History entry recorded during a game.
struct HistoryEntry {
    state: [f32; 3 * BOARD_SIZE],
    policy: [f32; COLS],
    player: i8,
}

/// Batched self-play engine.
pub struct BatchedSelfPlay {
    // Game states
    games: Vec<ConnectFour>,
    trees: Vec<MCTSTree>,
    histories: Vec<Vec<HistoryEntry>>,
    active: Vec<usize>,

    // Pending leaves awaiting neural network evaluation
    pending: Vec<(usize, usize)>, // (game_index, node_index)

    // Config
    _num_simulations: usize,
    c_puct: f32,
    dirichlet_alpha: f64,
    dirichlet_epsilon: f64,
    temp_threshold: usize,
}

impl BatchedSelfPlay {
    pub fn new(
        num_games: usize,
        _num_simulations: usize,
        c_puct: f32,
        dirichlet_alpha: f64,
        dirichlet_epsilon: f64,
        temp_threshold: usize,
    ) -> Self {
        let mut games = Vec::with_capacity(num_games);
        let mut trees = Vec::with_capacity(num_games);
        let mut histories = Vec::with_capacity(num_games);

        for _ in 0..num_games {
            let game = ConnectFour::new();
            trees.push(MCTSTree::new(game.clone()));
            games.push(game);
            histories.push(Vec::new());
        }

        let active = (0..num_games).collect();

        BatchedSelfPlay {
            games,
            trees,
            histories,
            active,
            pending: Vec::new(),
            _num_simulations,
            c_puct,
            dirichlet_alpha,
            dirichlet_epsilon,
            temp_threshold,
        }
    }

    /// Returns the number of active (non-finished) games.
    pub fn num_active(&self) -> usize {
        self.active.len()
    }

    /// Check if all games are finished.
    pub fn is_done(&self) -> bool {
        self.active.is_empty()
    }

    /// Get encoded states for initial root nodes.
    /// Returns (states, valid_moves) as flat Vec<f32> / Vec<bool>.
    pub fn get_root_states(&self) -> (Vec<f32>, Vec<bool>) {
        let n = self.active.len();
        let mut states = Vec::with_capacity(n * 3 * BOARD_SIZE);
        let mut valid = Vec::with_capacity(n * COLS);

        for &i in &self.active {
            states.extend_from_slice(&self.games[i].encode());
            let v = self.games[i].get_valid_moves();
            valid.extend_from_slice(&v);
        }
        (states, valid)
    }

    /// Initialize root nodes with policies from neural network.
    /// `policies` is a flat array of shape (num_active, COLS).
    pub fn init_roots(&mut self, policies: &[f32]) {
        for (idx, &i) in self.active.iter().enumerate() {
            let policy = &policies[idx * COLS..(idx + 1) * COLS];
            let root = self.trees[i].root();
            self.trees[i].expand(root, policy);
        }
    }

    /// Add Dirichlet noise to all active roots.
    pub fn add_noise(&mut self) {
        for &i in &self.active {
            let root = self.trees[i].root();
            self.trees[i]
                .add_dirichlet_noise(root, self.dirichlet_alpha, self.dirichlet_epsilon);
        }
    }

    /// Run one simulation step: select leaves from all active trees.
    /// Returns encoded states for non-terminal leaves needing NN evaluation.
    /// Terminal leaves are backpropagated immediately.
    /// Returns (states, valid_moves, count).
    pub fn collect_leaves(&mut self) -> (Vec<f32>, Vec<bool>, usize) {
        self.pending.clear();
        let mut states = Vec::new();
        let mut valid = Vec::new();

        for &i in &self.active {
            let root = self.trees[i].root();
            let leaf = self.trees[i].select_leaf(root, self.c_puct);

            let result = self.trees[i].nodes[leaf].game.get_result();
            if let Some(val) = result {
                // Terminal node: backpropagate immediately
                self.trees[i].backpropagate(leaf, val);
            } else {
                // Non-terminal: needs NN evaluation
                states.extend_from_slice(&self.trees[i].nodes[leaf].game.encode());
                let v = self.trees[i].nodes[leaf].game.get_valid_moves();
                valid.extend_from_slice(&v);
                self.pending.push((i, leaf));
            }
        }

        let count = self.pending.len();
        (states, valid, count)
    }

    /// Apply neural network evaluations to pending leaves.
    /// `policies` shape: (count, COLS), `values` shape: (count,).
    pub fn apply_evaluations(&mut self, policies: &[f32], values: &[f32]) {
        for (idx, &(game_i, node_idx)) in self.pending.iter().enumerate() {
            let policy = &policies[idx * COLS..(idx + 1) * COLS];
            self.trees[game_i].expand(node_idx, policy);
            self.trees[game_i]
                .backpropagate(node_idx, values[idx] as f64);
        }
        self.pending.clear();
    }

    /// After all simulations: sample actions, advance games, return finished examples.
    /// Returns training examples (with augmentation) for completed games.
    pub fn advance_games(&mut self) -> Vec<Example> {
        let mut finished_examples = Vec::new();
        let mut new_active = Vec::new();
        let mut rng = rand::thread_rng();

        for &i in &self.active {
            let root = self.trees[i].root();
            let visit_counts = self.trees[i].get_visit_counts(root);
            let move_count = self.games[i].move_count as usize;

            // Temperature-based action selection
            let action_probs = if move_count >= self.temp_threshold {
                // Greedy
                let mut probs = [0.0f32; COLS];
                let best = visit_counts
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                probs[best] = 1.0;
                probs
            } else {
                // Temperature = 1.0: proportional to visit counts
                let total: f32 = visit_counts.iter().sum();
                if total == 0.0 {
                    let valid = self.games[i].get_valid_moves();
                    let n_valid = valid.iter().filter(|&&v| v).count() as f32;
                    let mut probs = [0.0f32; COLS];
                    for c in 0..COLS {
                        if valid[c] {
                            probs[c] = 1.0 / n_valid;
                        }
                    }
                    probs
                } else {
                    let mut probs = [0.0f32; COLS];
                    for c in 0..COLS {
                        probs[c] = visit_counts[c] / total;
                    }
                    probs
                }
            };

            // Record history
            self.histories[i].push(HistoryEntry {
                state: self.games[i].encode(),
                policy: action_probs,
                player: self.games[i].current_player,
            });

            // Sample action
            let r: f32 = rng.gen();
            let mut cumsum = 0.0f32;
            let mut action = 0u8;
            for c in 0..COLS {
                cumsum += action_probs[c];
                if r < cumsum {
                    action = c as u8;
                    break;
                }
                if c == COLS - 1 {
                    action = c as u8;
                }
            }

            self.games[i].make_move(action);

            if self.games[i].is_terminal() {
                let result = self.games[i].get_result().unwrap_or(0.0) as f32;
                let current = self.games[i].current_player;

                // Convert history to training examples
                for entry in &self.histories[i] {
                    let val = if entry.player == current {
                        result
                    } else {
                        -result
                    };

                    // Original example
                    finished_examples.push(Example {
                        state: entry.state,
                        policy: entry.policy,
                        value: val,
                    });

                    // Augmented (horizontally mirrored) example
                    let mut mirrored_state = [0.0f32; 3 * BOARD_SIZE];
                    for plane in 0..3 {
                        for r in 0..ROWS {
                            for c in 0..COLS {
                                mirrored_state[plane * BOARD_SIZE + r * COLS + c] =
                                    entry.state[plane * BOARD_SIZE + r * COLS + (COLS - 1 - c)];
                            }
                        }
                    }
                    let mut mirrored_policy = [0.0f32; COLS];
                    for c in 0..COLS {
                        mirrored_policy[c] = entry.policy[COLS - 1 - c];
                    }
                    finished_examples.push(Example {
                        state: mirrored_state,
                        policy: mirrored_policy,
                        value: val,
                    });
                }

                self.histories[i].clear();
            } else {
                new_active.push(i);
                // Create fresh tree for next move (no tree reuse for simplicity)
                self.trees[i] = MCTSTree::new(self.games[i].clone());
            }
        }

        self.active = new_active;
        finished_examples
    }

    /// Initialize new roots with policies (same as init_roots).
    pub fn init_new_roots(&mut self, policies: &[f32]) {
        self.init_roots(policies)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_engine() {
        let engine = BatchedSelfPlay::new(10, 100, 1.5, 1.0, 0.25, 30);
        assert_eq!(engine.num_active(), 10);
        assert!(!engine.is_done());
    }

    #[test]
    fn test_get_root_states() {
        let engine = BatchedSelfPlay::new(5, 10, 1.5, 1.0, 0.25, 30);
        let (states, valid) = engine.get_root_states();
        assert_eq!(states.len(), 5 * 3 * BOARD_SIZE);
        assert_eq!(valid.len(), 5 * COLS);
    }

    #[test]
    fn test_init_roots() {
        let mut engine = BatchedSelfPlay::new(3, 10, 1.5, 1.0, 0.25, 30);
        let policies = vec![1.0 / COLS as f32; 3 * COLS];
        engine.init_roots(&policies);

        // All roots should be expanded
        for &i in &engine.active {
            assert!(engine.trees[i].nodes[0].is_expanded);
        }
    }

    #[test]
    fn test_collect_and_apply() {
        let mut engine = BatchedSelfPlay::new(3, 10, 1.5, 1.0, 0.25, 30);

        // Init roots
        let policies = vec![1.0 / COLS as f32; 3 * COLS];
        engine.init_roots(&policies);
        engine.add_noise();

        // One simulation step
        let (states, valid, count) = engine.collect_leaves();
        assert!(count > 0);
        assert_eq!(states.len(), count * 3 * BOARD_SIZE);
        assert_eq!(valid.len(), count * COLS);

        // Apply dummy evaluations
        let eval_policies = vec![1.0 / COLS as f32; count * COLS];
        let eval_values = vec![0.0f32; count];
        engine.apply_evaluations(&eval_policies, &eval_values);
    }

    #[test]
    fn test_full_game_loop() {
        let mut engine = BatchedSelfPlay::new(2, 5, 1.5, 1.0, 0.25, 30);
        let mut all_examples = Vec::new();

        // Init roots
        let n = engine.num_active();
        let policies = vec![1.0 / COLS as f32; n * COLS];
        engine.init_roots(&policies);

        let mut iterations = 0;
        while !engine.is_done() && iterations < 1000 {
            engine.add_noise();

            // Simulations
            for _ in 0..5 {
                let (_states, _valid, count) = engine.collect_leaves();
                if count > 0 {
                    let p = vec![1.0 / COLS as f32; count * COLS];
                    let v = vec![0.0f32; count];
                    engine.apply_evaluations(&p, &v);
                }
            }

            // Advance
            let examples = engine.advance_games();
            all_examples.extend(examples);

            // Init new roots
            let n = engine.num_active();
            if n > 0 {
                let (_, _) = engine.get_root_states();
                let p = vec![1.0 / COLS as f32; n * COLS];
                engine.init_new_roots(&p);
            }

            iterations += 1;
        }

        assert!(engine.is_done());
        assert!(!all_examples.is_empty());
        // Each example should have augmented pair
        assert!(all_examples.len() % 2 == 0);
    }
}
