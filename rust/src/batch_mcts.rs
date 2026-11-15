/// Batched MCTS evaluation engine (`RustBatchedMCTS`) for fast Arena & dataset evaluation.
///
/// Runs exact `num_simulations` MCTS searches (with virtual losses, collision-abort,
/// exact MCTS-Solver propagation, and parent-relative FPU reduction) across `N` arbitrary
/// input `ConnectFour` positions in parallel via Rayon.

use crate::batch_self_play::LEAVES_PER_GAME;
use crate::game::{ConnectFour, BOARD_SIZE, COLS};
use crate::mcts::{MCTSTree, SolverState};
use rayon::prelude::*;

struct EvalSlot {
    game: ConnectFour,
    tree: MCTSTree,
    sims_done: usize,
    last_batch_leaves: usize,
    initial_terminal: bool,
    pending_nodes: Vec<usize>,
    batch_offset: usize,
}

pub struct BatchedMCTS {
    slots: Vec<EvalSlot>,
    pending_count: usize,
    num_simulations: usize,
    c_puct: f32,
    c_fpu: f32,
}

impl BatchedMCTS {
    pub fn new(
        games: Vec<ConnectFour>,
        num_simulations: usize,
        c_puct: f32,
        c_fpu: f32,
    ) -> Self {
        let slots = games
            .into_iter()
            .map(|game| {
                let initial_terminal = game.is_terminal();
                let tree = MCTSTree::new(game);
                let sims_done = if initial_terminal { num_simulations } else { 0 };
                EvalSlot {
                    game,
                    tree,
                    sims_done,
                    last_batch_leaves: LEAVES_PER_GAME,
                    initial_terminal,
                    pending_nodes: Vec::with_capacity(LEAVES_PER_GAME),
                    batch_offset: 0,
                }
            })
            .collect();

        BatchedMCTS {
            slots,
            pending_count: 0,
            num_simulations,
            c_puct,
            c_fpu,
        }
    }

    #[inline]
    pub fn num_games(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    pub fn num_pending(&self) -> usize {
        self.pending_count
    }

    /// Encode all `N` root positions as `(states_flat[N*3*6*7], valid_flat[N*7])`.
    /// Note: For a 42-stone full-board draw, sets `valid_flat` row to `[true; 7]` if needed?
    /// Wait: `get_valid_moves()` matches `game.get_valid_moves()`, and `init_roots` safely ignores
    /// terminal positions even if `policies` contains `NaN`s from softmaxing all-false masks.
    pub fn get_root_states(&self) -> (Vec<f32>, Vec<bool>) {
        let n = self.slots.len();
        let mut states = Vec::with_capacity(n * 3 * BOARD_SIZE);
        let mut valid = Vec::with_capacity(n * COLS);

        for slot in &self.slots {
            states.extend_from_slice(&slot.game.encode());
            valid.extend_from_slice(&slot.game.get_valid_moves());
        }
        (states, valid)
    }

    /// Encode all `N` root positions compactly as binary `u8` `(states_u8[N*3*6*7], valid_flat[N*7])`.
    pub fn get_root_states_u8(&self) -> (Vec<u8>, Vec<bool>) {
        let n = self.slots.len();
        let mut states = Vec::with_capacity(n * 3 * BOARD_SIZE);
        let mut valid = Vec::with_capacity(n * COLS);

        for slot in &self.slots {
            states.extend_from_slice(&slot.game.encode_u8());
            valid.extend_from_slice(&slot.game.get_valid_moves());
        }
        (states, valid)
    }

    /// Expand all non-terminal root nodes using `policies` (`shape: (N, 7)`).
    pub fn init_roots(&mut self, policies: &[f32]) {
        let num_sims = self.num_simulations;
        self.slots
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| {
                if slot.initial_terminal {
                    slot.sims_done = num_sims;
                    return;
                }
                let policy = &policies[i * COLS..(i + 1) * COLS];
                let root = slot.tree.root();
                slot.tree.expand(root, policy);
                if slot.tree.nodes[root].solver_state != SolverState::Ongoing {
                    slot.sims_done = num_sims;
                }
            });
    }

    /// True when every game slot has reached `num_simulations` or proven its root.
    pub fn is_done(&self) -> bool {
        let total_slots = self.slots.len();
        let unfinished = self
            .slots
            .iter()
            .filter(|slot| {
                let root = slot.tree.root();
                slot.sims_done < self.num_simulations
                    && slot.tree.nodes[root].solver_state == SolverState::Ongoing
            })
            .count();

        if unfinished == 0 {
            return true;
        }

        // Straggler tail guard on large eval batches (>= 64 positions):
        // If <= 4 positions remain and all of them have >= 75% sims done and <= 2 leaves/batch, finish cleanly.
        if total_slots >= 64 && unfinished <= 4 {
            let min_sims = (self.num_simulations * 3) / 4;
            return self.slots.iter().all(|slot| {
                let root = slot.tree.root();
                slot.sims_done >= self.num_simulations
                    || slot.tree.nodes[root].solver_state != SolverState::Ongoing
                    || (slot.sims_done >= min_sims && slot.last_batch_leaves <= 2)
            });
        }

        false
    }

    /// Collect leaves across all unfinished games in parallel using virtual losses + collision-abort.
    pub fn collect_leaves(&mut self) -> (Vec<f32>, Vec<bool>, usize) {
        let (states_u8, valid, count) = self.collect_leaves_u8();
        let states_f32: Vec<f32> = states_u8.into_iter().map(|b| b as f32).collect();
        (states_f32, valid, count)
    }

    /// Collect leaves compactly as binary `u8` planes (`4x` smaller H2D transfer).
    pub fn collect_leaves_u8(&mut self) -> (Vec<u8>, Vec<bool>, usize) {
        if self.pending_count > 0 {
            for slot in &mut self.slots {
                for &node_idx in &slot.pending_nodes {
                    slot.tree.revert_virtual_loss(node_idx);
                }
                slot.pending_nodes.clear();
            }
            self.pending_count = 0;
        }

        let num_sims = self.num_simulations;
        let c_puct = self.c_puct;
        let c_fpu = self.c_fpu;

        let per_slot: Vec<(Vec<u8>, Vec<bool>)> = self
            .slots
            .par_iter_mut()
            .map(|slot| {
                slot.pending_nodes.clear();
                let root = slot.tree.root();
                if slot.sims_done >= num_sims
                    || slot.tree.nodes[root].solver_state != SolverState::Ongoing
                {
                    slot.sims_done = num_sims;
                    return (Vec::new(), Vec::new());
                }

                let remaining = num_sims - slot.sims_done;
                let budget = LEAVES_PER_GAME.min(remaining);
                let (leaves, terminal_count) =
                    slot.tree
                        .select_multiple_leaves(root, c_puct, c_fpu, false, budget);

                slot.last_batch_leaves = leaves.len() + terminal_count as usize;
                slot.sims_done += slot.last_batch_leaves;
                if slot.tree.nodes[root].solver_state != SolverState::Ongoing {
                    slot.sims_done = num_sims;
                }

                if leaves.is_empty() {
                    return (Vec::new(), Vec::new());
                }

                let mut s_buf = Vec::with_capacity(leaves.len() * 3 * BOARD_SIZE);
                let mut v_buf = Vec::with_capacity(leaves.len() * COLS);
                for &leaf in &leaves {
                    s_buf.extend_from_slice(&slot.tree.nodes[leaf].game.encode_u8());
                    v_buf.extend_from_slice(&slot.tree.nodes[leaf].game.get_valid_moves());
                }
                slot.pending_nodes = leaves;
                (s_buf, v_buf)
            })
            .collect();

        let mut total_count = 0usize;
        for slot in &mut self.slots {
            slot.batch_offset = total_count;
            total_count += slot.pending_nodes.len();
        }

        let mut states = Vec::with_capacity(total_count * 3 * BOARD_SIZE);
        let mut valid = Vec::with_capacity(total_count * COLS);
        for (s_buf, v_buf) in per_slot {
            states.extend_from_slice(&s_buf);
            valid.extend_from_slice(&v_buf);
        }

        self.pending_count = total_count;
        (states, valid, total_count)
    }

    /// Apply neural network evaluations (`policies[K, 7]`, `values[K]`) to pending leaves.
    pub fn apply_evaluations(&mut self, policies: &[f32], values: &[f32]) {
        let num_sims = self.num_simulations;
        self.slots
            .par_iter_mut()
            .filter(|s| !s.pending_nodes.is_empty())
            .for_each(|slot| {
                let offset = slot.batch_offset;
                for (j, &node_idx) in slot.pending_nodes.iter().enumerate() {
                    let k = offset + j;
                    slot.tree.revert_virtual_loss(node_idx);
                    let policy = &policies[k * COLS..(k + 1) * COLS];
                    slot.tree.expand(node_idx, policy);
                    let bval = slot.tree.nodes[node_idx]
                        .solver_state
                        .proven_value()
                        .unwrap_or(values[k] as f64);
                    slot.tree.backpropagate(node_idx, bval);
                }
                slot.pending_nodes.clear();
                let root = slot.tree.root();
                if slot.tree.nodes[root].solver_state != SolverState::Ongoing {
                    slot.sims_done = num_sims;
                }
            });
        self.pending_count = 0;
    }

    /// Apply packed `(pending_count, 8)` evaluations (`row[0..7]` = policy, `row[7]` = value).
    pub fn apply_evaluations_packed(&mut self, evals_packed: &[f32]) {
        let num_sims = self.num_simulations;
        let stride = COLS + 1;
        self.slots
            .par_iter_mut()
            .filter(|s| !s.pending_nodes.is_empty())
            .for_each(|slot| {
                let offset = slot.batch_offset;
                for (j, &node_idx) in slot.pending_nodes.iter().enumerate() {
                    let k = offset + j;
                    let row = &evals_packed[k * stride..(k + 1) * stride];
                    let policy = &row[..COLS];
                    let val = row[COLS];
                    slot.tree.revert_virtual_loss(node_idx);
                    slot.tree.expand(node_idx, policy);
                    let bval = slot.tree.nodes[node_idx]
                        .solver_state
                        .proven_value()
                        .unwrap_or(val as f64);
                    slot.tree.backpropagate(node_idx, bval);
                }
                slot.pending_nodes.clear();
                let root = slot.tree.root();
                if slot.tree.nodes[root].solver_state != SolverState::Ongoing {
                    slot.sims_done = num_sims;
                }
            });
        self.pending_count = 0;
    }

    /// Extract search results for all `N` games:
    /// `(visit_counts[N*7], best_actions[N], best_values[N], child_q_values[N*7])`,
    /// where `best_values` and `child_q_values` are in ROOT `i`'s current player perspective
    /// (`[-1, +1]`, `NaN` for illegal or unvisited non-proven columns; terminal games handled cleanly).
    pub fn get_results(&self) -> (Vec<f32>, Vec<i64>, Vec<f32>, Vec<f32>) {
        let n = self.slots.len();
        let mut visit_counts = vec![0.0f32; n * COLS];
        let mut best_actions = vec![-1i64; n];
        let mut best_values = vec![0.0f32; n];
        let mut child_q_values = vec![f32::NAN; n * COLS];

        for (i, slot) in self.slots.iter().enumerate() {
            if slot.initial_terminal {
                best_actions[i] = -1;
                best_values[i] = slot.game.get_result().unwrap_or(0.0) as f32;
                continue;
            }

            let root = slot.tree.root();
            let root_node = &slot.tree.nodes[root];
            if root_node.children.is_empty() {
                // Uninitialized non-terminal root fallback: pick leftmost valid move
                let valid = slot.game.get_valid_moves();
                if let Some(col) = (0..COLS).find(|&c| valid[c]) {
                    best_actions[i] = col as i64;
                }
                best_values[i] = 0.0;
                continue;
            }

            // 1. Populate child_q_values in root mover's perspective
            for &(col, c_idx) in &root_node.children {
                let child = &slot.tree.nodes[c_idx];
                let q_root_view = match child.solver_state {
                    SolverState::ProvenLoss => 1.0f32,
                    SolverState::ProvenWin => -1.0f32,
                    SolverState::ProvenDraw => 0.0f32,
                    SolverState::Ongoing => {
                        if child.visit_count > 0 {
                            ((-child.total_value / child.visit_count as f64) as f32)
                                .clamp(-1.0, 1.0)
                        } else {
                            f32::NAN
                        }
                    }
                };
                child_q_values[i * COLS + col as usize] = q_root_view;
            }

            // 2. Populate visit_counts, best_actions, and best_values
            let mut counts = slot.tree.get_visit_counts(root);
            let root_solver = root_node.solver_state;

            match root_solver {
                SolverState::ProvenWin => {
                    // Only winning children (ProvenLoss for opponent) are optimal
                    let mut best_col = 0u8;
                    let mut best_vc = 0u32;
                    let mut best_prior = f32::NEG_INFINITY;
                    let mut first = true;

                    for &(col, c_idx) in &root_node.children {
                        let child = &slot.tree.nodes[c_idx];
                        if child.solver_state == SolverState::ProvenLoss {
                            if first
                                || child.visit_count > best_vc
                                || (child.visit_count == best_vc && child.prior > best_prior)
                            {
                                best_col = col;
                                best_vc = child.visit_count;
                                best_prior = child.prior;
                                first = false;
                            }
                        } else {
                            counts[col as usize] = 0.0;
                        }
                    }

                    let sum_win: f32 = counts.iter().sum();
                    if sum_win == 0.0 {
                        counts[best_col as usize] = self.num_simulations.max(1) as f32;
                    }
                    visit_counts[i * COLS..(i + 1) * COLS].copy_from_slice(&counts);
                    best_actions[i] = best_col as i64;
                    best_values[i] = 1.0;
                }
                SolverState::ProvenDraw => {
                    let mut best_col = 0u8;
                    let mut best_vc = 0u32;
                    let mut best_prior = f32::NEG_INFINITY;
                    let mut first = true;

                    for &(col, c_idx) in &root_node.children {
                        let child = &slot.tree.nodes[c_idx];
                        if child.solver_state == SolverState::ProvenDraw {
                            if first
                                || child.visit_count > best_vc
                                || (child.visit_count == best_vc && child.prior > best_prior)
                            {
                                best_col = col;
                                best_vc = child.visit_count;
                                best_prior = child.prior;
                                first = false;
                            }
                        } else {
                            counts[col as usize] = 0.0;
                        }
                    }

                    let sum_draw: f32 = counts.iter().sum();
                    if sum_draw == 0.0 {
                        counts[best_col as usize] = self.num_simulations.max(1) as f32;
                    }
                    visit_counts[i * COLS..(i + 1) * COLS].copy_from_slice(&counts);
                    best_actions[i] = best_col as i64;
                    best_values[i] = 0.0;
                }
                SolverState::ProvenLoss => {
                    let mut best_col = root_node.children[0].0;
                    let mut best_vc = 0u32;
                    let mut best_prior = f32::NEG_INFINITY;
                    let mut first = true;

                    for &(col, c_idx) in &root_node.children {
                        let child = &slot.tree.nodes[c_idx];
                        if first
                            || child.visit_count > best_vc
                            || (child.visit_count == best_vc && child.prior > best_prior)
                        {
                            best_col = col;
                            best_vc = child.visit_count;
                            best_prior = child.prior;
                            first = false;
                        }
                    }

                    let sum_all: f32 = counts.iter().sum();
                    if sum_all == 0.0 && self.num_simulations > 0 {
                        counts[best_col as usize] = self.num_simulations as f32;
                    }
                    visit_counts[i * COLS..(i + 1) * COLS].copy_from_slice(&counts);
                    best_actions[i] = best_col as i64;
                    best_values[i] = -1.0;
                }
                SolverState::Ongoing => {
                    // Prefer children that are NOT proven losses for root (child != ProvenWin)
                    let any_non_losing_visited = root_node.children.iter().any(|&(_, c_idx)| {
                        let c = &slot.tree.nodes[c_idx];
                        c.solver_state != SolverState::ProvenWin && c.visit_count > 0
                    });
                    if any_non_losing_visited {
                        for &(col, c_idx) in &root_node.children {
                            if slot.tree.nodes[c_idx].solver_state == SolverState::ProvenWin {
                                counts[col as usize] = 0.0;
                            }
                        }
                    }

                    let mut best_col = root_node.children[0].0;
                    let mut best_c_idx = root_node.children[0].1;
                    let mut best_vc = 0u32;
                    let mut best_prior = f32::NEG_INFINITY;
                    let mut first = true;

                    for &(col, c_idx) in &root_node.children {
                        let child = &slot.tree.nodes[c_idx];
                        if child.solver_state == SolverState::ProvenWin {
                            continue;
                        }
                        if first
                            || child.visit_count > best_vc
                            || (child.visit_count == best_vc && child.prior > best_prior)
                        {
                            best_col = col;
                            best_c_idx = c_idx;
                            best_vc = child.visit_count;
                            best_prior = child.prior;
                            first = false;
                        }
                    }

                    visit_counts[i * COLS..(i + 1) * COLS].copy_from_slice(&counts);
                    best_actions[i] = best_col as i64;

                    let best_child = &slot.tree.nodes[best_c_idx];
                    best_values[i] = match best_child.solver_state {
                        SolverState::ProvenLoss => 1.0,
                        SolverState::ProvenWin => -1.0,
                        SolverState::ProvenDraw => 0.0,
                        SolverState::Ongoing => {
                            if best_child.visit_count > 0 {
                                ((-best_child.total_value / best_child.visit_count as f64) as f32)
                                    .clamp(-1.0, 1.0)
                            } else {
                                0.0
                            }
                        }
                    };
                }
            }
        }

        (visit_counts, best_actions, best_values, child_q_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batched_mcts_regular_and_terminal_games() {
        let g_open = ConnectFour::new();
        let g_win_in_1 = ConnectFour::from_moves("121212").unwrap();
        let g_already_won = ConnectFour::from_moves("1212121").unwrap();
        assert!(g_already_won.is_terminal());

        let mut mcts = BatchedMCTS::new(vec![g_open, g_win_in_1, g_already_won], 32, 1.5, 0.20);
        let (states, valid) = mcts.get_root_states();
        assert_eq!(states.len(), 3 * 3 * BOARD_SIZE);
        assert_eq!(valid.len(), 3 * COLS);

        let init_pols = vec![1.0 / 7.0f32; 3 * COLS];
        mcts.init_roots(&init_pols);

        while !mcts.is_done() {
            let (_ls, _lv, count) = mcts.collect_leaves();
            if count > 0 {
                let p = vec![1.0 / 7.0f32; count * COLS];
                let v = vec![0.25f32; count];
                mcts.apply_evaluations(&p, &v);
            }
        }

        let (vc, best_actions, best_values, child_q) = mcts.get_results();
        // Game 0: 32 simulations completed
        let sum0: f32 = vc[0..7].iter().sum();
        assert_eq!(sum0, 32.0);
        assert!((0..7).contains(&best_actions[0]));
        assert!(best_values[0] >= -1.0 && best_values[0] <= 1.0);

        // Game 1: Mate-in-1 at col 0 => best_action == 0, best_value == +1.0, child_q[0] == +1.0!
        assert_eq!(best_actions[1], 0);
        assert_eq!(best_values[1], 1.0);
        assert_eq!(child_q[COLS], 1.0);

        // Game 2: Already terminal (won by previous mover) => best_action == -1, best_value == -1.0, all child_q NaN
        assert_eq!(best_actions[2], -1);
        assert_eq!(best_values[2], -1.0);
        assert!(child_q[2 * COLS..3 * COLS].iter().all(|q| q.is_nan()));
    }
}
