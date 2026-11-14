/// Batched self-play engine for AlphaZero training.
///
/// Manages N concurrent games, each with its own MCTS tree.
/// Uses virtual losses + collision-abort for duplicate-free multi-leaf collection,
/// exact MCTS-Solver, parent-relative FPU, per-game simulation tracking (`sims_done`),
/// stochastic opening plies covering Move 0 and Move 1, decoupled visit-count policy
/// targets, unexpanded-only root evaluation after tree rebase, and Rayon parallelism.

use crate::game::{ConnectFour, BOARD_SIZE, COLS, ROWS};
use crate::mcts::{MCTSTree, SolverState};
use rand::Rng;
use rayon::prelude::*;

/// Maximum number of leaves to collect per game per batch.
pub const LEAVES_PER_GAME: usize = 16;

/// A single training example: `(encoded_state, action_probs, value)`.
#[allow(dead_code)]
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

struct SelfPlaySlot {
    game: ConnectFour,
    tree: MCTSTree,
    history: Vec<HistoryEntry>,
    sims_done: usize,
    root_noisy: bool,
    active: bool,
    pending_nodes: Vec<usize>,
    batch_offset: usize,
}

/// Sample non-terminal random opening board according to:
/// - `random_opening_plies == 0`: 0 plies
/// - `random_opening_plies == 1`: 50% 0 plies, 50% 1 ply
/// - `random_opening_plies >= 2`: 50% 0 plies, 25% 1 ply, 25% uniform `2..=random_opening_plies`
/// Resamples / rolls back if any random move creates a terminal board.
fn sample_opening_game<R: Rng>(rng: &mut R, random_opening_plies: usize) -> ConnectFour {
    let max_plies = random_opening_plies.min(BOARD_SIZE - 2);
    if max_plies == 0 {
        return ConnectFour::new();
    }

    let u: f64 = rng.gen();
    let target_plies = if max_plies == 1 {
        if u < 0.50 { 0 } else { 1 }
    } else if u < 0.50 {
        0
    } else if u < 0.75 {
        1
    } else {
        rng.gen_range(2..=max_plies)
    };

    if target_plies == 0 {
        return ConnectFour::new();
    }

    for _attempt in 0..32 {
        let mut game = ConnectFour::new();
        let mut ok = true;
        for _ in 0..target_plies {
            let valid = game.get_valid_move_indices();
            if valid.is_empty() {
                ok = false;
                break;
            }
            let col = valid[rng.gen_range(0..valid.len())];
            game.make_move(col);
            if game.is_terminal() {
                ok = false;
                break;
            }
        }
        if ok {
            return game;
        }
    }

    // Fallback: play up to `target_plies`, rolling back if a move creates a terminal state.
    let mut game = ConnectFour::new();
    for _ in 0..target_plies {
        let valid = game.get_valid_move_indices();
        if valid.is_empty() {
            break;
        }
        let prev = game.clone();
        let col = valid[rng.gen_range(0..valid.len())];
        game.make_move(col);
        if game.is_terminal() {
            game = prev;
            break;
        }
    }
    game
}

/// Batched self-play engine.
pub struct BatchedSelfPlay {
    slots: Vec<SelfPlaySlot>,
    active_count: usize,
    pending_count: usize,

    // Config
    num_simulations: usize,
    c_puct: f32,
    c_fpu: f32,
    dirichlet_alpha: f64,
    dirichlet_epsilon: f64,
    temp_threshold: usize,
}

impl BatchedSelfPlay {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_games: usize,
        num_simulations: usize,
        c_puct: f32,
        dirichlet_alpha: f64,
        dirichlet_epsilon: f64,
        temp_threshold: usize,
        random_opening_plies: usize,
        c_fpu: f32,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut slots = Vec::with_capacity(num_games);

        for _ in 0..num_games {
            let game = sample_opening_game(&mut rng, random_opening_plies);
            let tree = MCTSTree::new(game.clone());
            slots.push(SelfPlaySlot {
                game,
                tree,
                history: Vec::new(),
                sims_done: 0,
                root_noisy: false,
                active: true,
                pending_nodes: Vec::new(),
                batch_offset: 0,
            });
        }

        BatchedSelfPlay {
            slots,
            active_count: num_games,
            pending_count: 0,
            num_simulations,
            c_puct,
            c_fpu,
            dirichlet_alpha,
            dirichlet_epsilon,
            temp_threshold,
        }
    }

    /// Number of active (non-finished) games.
    #[inline]
    pub fn num_active(&self) -> usize {
        self.active_count
    }

    /// Number of leaves currently pending NN evaluation.
    #[inline]
    pub fn num_pending(&self) -> usize {
        self.pending_count
    }

    /// Check if all games are finished.
    #[inline]
    pub fn is_done(&self) -> bool {
        self.active_count == 0
    }

    /// True when every active game has completed `num_simulations` or has a proven root.
    pub fn is_step_done(&self) -> bool {
        self.slots.iter().filter(|s| s.active).all(|s| {
            let root = s.tree.root();
            s.sims_done >= self.num_simulations
                || s.tree.nodes[root].solver_state != SolverState::Ongoing
        })
    }

    /// Count active games whose root node is not yet expanded.
    pub fn num_unexpanded_roots(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| {
                s.active && {
                    let root = s.tree.root();
                    !s.tree.nodes[root].is_expanded
                }
            })
            .count()
    }

    /// Get encoded states for all active root nodes.
    /// Returns `(states_flat, valid_flat)`.
    pub fn get_root_states(&self) -> (Vec<f32>, Vec<bool>) {
        let n = self.active_count;
        let mut states = Vec::with_capacity(n * 3 * BOARD_SIZE);
        let mut valid = Vec::with_capacity(n * COLS);

        for slot in self.slots.iter().filter(|s| s.active) {
            states.extend_from_slice(&slot.game.encode());
            valid.extend_from_slice(&slot.game.get_valid_moves());
        }
        (states, valid)
    }

    /// Get encoded states ONLY for active games whose root node is `!is_expanded`.
    /// Returns `(states_flat, valid_flat, unexpanded_count)`.
    pub fn get_unexpanded_root_states(&self) -> (Vec<f32>, Vec<bool>, usize) {
        let mut states = Vec::new();
        let mut valid = Vec::new();
        let mut count = 0usize;

        for slot in self.slots.iter().filter(|s| s.active) {
            let root = slot.tree.root();
            if !slot.tree.nodes[root].is_expanded {
                states.extend_from_slice(&slot.game.encode());
                valid.extend_from_slice(&slot.game.get_valid_moves());
                count += 1;
            }
        }
        (states, valid, count)
    }

    /// Initialize root nodes with policies (`shape: (num_active, COLS)`).
    pub fn init_roots(&mut self, policies: &[f32]) {
        let num_sims = self.num_simulations;
        let mut active_slots: Vec<&mut SelfPlaySlot> =
            self.slots.iter_mut().filter(|s| s.active).collect();
        active_slots
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, slot)| {
                let policy = &policies[idx * COLS..(idx + 1) * COLS];
                let root = slot.tree.root();
                slot.tree.expand(root, policy);
                if slot.tree.nodes[root].solver_state != SolverState::Ongoing {
                    slot.sims_done = num_sims;
                }
            });
    }

    /// Initialize ONLY unexpanded active roots (`policies` shape: `(num_unexpanded, COLS)`).
    pub fn init_unexpanded_roots(&mut self, policies: &[f32]) {
        let num_sims = self.num_simulations;
        let mut unexpanded_slots: Vec<&mut SelfPlaySlot> = self
            .slots
            .iter_mut()
            .filter(|s| {
                s.active && {
                    let root = s.tree.root();
                    !s.tree.nodes[root].is_expanded
                }
            })
            .collect();

        unexpanded_slots
            .par_iter_mut()
            .enumerate()
            .for_each(|(u_idx, slot)| {
                let policy = &policies[u_idx * COLS..(u_idx + 1) * COLS];
                let root = slot.tree.root();
                slot.tree.expand(root, policy);
                if slot.tree.nodes[root].solver_state != SolverState::Ongoing {
                    slot.sims_done = num_sims;
                }
            });
    }

    /// Legacy compatibility: `policies` has shape `(num_active, COLS)`, expands `!is_expanded` roots.
    pub fn init_new_roots(&mut self, policies: &[f32]) {
        let num_sims = self.num_simulations;
        let mut active_slots: Vec<&mut SelfPlaySlot> =
            self.slots.iter_mut().filter(|s| s.active).collect();
        active_slots
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, slot)| {
                let root = slot.tree.root();
                if !slot.tree.nodes[root].is_expanded {
                    let policy = &policies[idx * COLS..(idx + 1) * COLS];
                    slot.tree.expand(root, policy);
                }
                if slot.tree.nodes[root].solver_state != SolverState::Ongoing {
                    slot.sims_done = num_sims;
                }
            });
    }

    /// Add Dirichlet noise to all active roots.
    pub fn add_noise(&mut self) {
        let alpha = self.dirichlet_alpha;
        let eps = self.dirichlet_epsilon;
        let is_noisy = alpha > 0.0 && eps > 0.0;

        self.slots
            .par_iter_mut()
            .filter(|s| s.active)
            .for_each(|slot| {
                let root = slot.tree.root();
                slot.tree.add_dirichlet_noise(root, alpha, eps);
                slot.root_noisy = is_noisy;
            });
    }

    /// Select up to `min(LEAVES_PER_GAME, num_simulations - sims_done[i])` leaves per game
    /// in parallel across active games using virtual losses + collision-abort.
    pub fn collect_leaves(&mut self) -> (Vec<f32>, Vec<bool>, usize) {
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

        // Parallel leaf selection & encoding per active game slot
        let per_slot_batches: Vec<(Vec<f32>, Vec<bool>)> = self
            .slots
            .par_iter_mut()
            .map(|slot| {
                slot.pending_nodes.clear();
                if !slot.active {
                    return (Vec::new(), Vec::new());
                }
                let root = slot.tree.root();
                if slot.sims_done >= num_sims
                    || slot.tree.nodes[root].solver_state != SolverState::Ongoing
                {
                    slot.sims_done = num_sims;
                    return (Vec::new(), Vec::new());
                }

                let remaining = num_sims - slot.sims_done;
                let budget = LEAVES_PER_GAME.min(remaining);
                let (leaves, terminal_count) = slot.tree.select_multiple_leaves(
                    root,
                    c_puct,
                    c_fpu,
                    slot.root_noisy,
                    budget,
                );

                slot.sims_done += leaves.len() + terminal_count as usize;
                if slot.tree.nodes[root].solver_state != SolverState::Ongoing {
                    slot.sims_done = num_sims;
                }

                if leaves.is_empty() {
                    return (Vec::new(), Vec::new());
                }

                let mut s_buf = Vec::with_capacity(leaves.len() * 3 * BOARD_SIZE);
                let mut v_buf = Vec::with_capacity(leaves.len() * COLS);
                for &leaf in &leaves {
                    s_buf.extend_from_slice(&slot.tree.nodes[leaf].game.encode());
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
        for (s_buf, v_buf) in per_slot_batches {
            states.extend_from_slice(&s_buf);
            valid.extend_from_slice(&v_buf);
        }

        self.pending_count = total_count;
        (states, valid, total_count)
    }

    /// Apply neural network evaluations to pending leaves in parallel across games.
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

    /// Advance active games, decoupling training policy targets from move-selection temperature.
    /// Returns flattened `(states, policies, values, num_examples)` with horizontal mirror augmentation.
    pub fn advance_games_flat(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
        let mut out_states = Vec::new();
        let mut out_policies = Vec::new();
        let mut out_values = Vec::new();
        let mut num_examples = 0usize;
        let mut new_active_count = 0usize;
        let mut rng = rand::thread_rng();

        for slot in &mut self.slots {
            if !slot.active {
                continue;
            }

            let root = slot.tree.root();
            let root_solver = slot.tree.nodes[root].solver_state;
            let move_count = slot.game.move_count as usize;

            // 1. Compute decoupled policy target:
            // ALWAYS normalized visit_counts / total (or uniform over ProvenLoss children if ProvenWin).
            let mut policy_target = [0.0f32; COLS];
            if root_solver == SolverState::ProvenWin {
                let mut win_cols = Vec::with_capacity(COLS);
                for &(col, c_idx) in &slot.tree.nodes[root].children {
                    if slot.tree.nodes[c_idx].solver_state == SolverState::ProvenLoss {
                        win_cols.push(col as usize);
                    }
                }
                let inv = 1.0 / (win_cols.len().max(1) as f32);
                for c in win_cols {
                    policy_target[c] = inv;
                }
            } else {
                let mut counts = slot.tree.get_visit_counts(root);
                if root_solver == SolverState::ProvenDraw {
                    // Never put target mass or sample onto proven losing moves when root is ProvenDraw
                    for &(col, c_idx) in &slot.tree.nodes[root].children {
                        if slot.tree.nodes[c_idx].solver_state == SolverState::ProvenWin {
                            counts[col as usize] = 0.0;
                        }
                    }
                }
                let total: f32 = counts.iter().sum();
                if total > 0.0 {
                    for c in 0..COLS {
                        policy_target[c] = counts[c] / total;
                    }
                } else if root_solver == SolverState::ProvenDraw {
                    let mut draw_cols = Vec::with_capacity(COLS);
                    for &(col, c_idx) in &slot.tree.nodes[root].children {
                        if slot.tree.nodes[c_idx].solver_state == SolverState::ProvenDraw {
                            draw_cols.push(col as usize);
                        }
                    }
                    let inv = 1.0 / (draw_cols.len().max(1) as f32);
                    for c in draw_cols {
                        policy_target[c] = inv;
                    }
                } else {
                    let valid = slot.game.get_valid_moves();
                    let n_valid = valid.iter().filter(|&&v| v).count().max(1) as f32;
                    for c in 0..COLS {
                        if valid[c] {
                            policy_target[c] = 1.0 / n_valid;
                        }
                    }
                }
            }

            // Record decoupled policy target in history
            slot.history.push(HistoryEntry {
                state: slot.game.encode(),
                policy: policy_target,
                player: slot.game.current_player,
            });

            // 2. Select action using `temp_threshold`:
            // - `move_count >= temp_threshold` => deterministic argmax breaking ties leftmost `0..COLS`
            // - `move_count < temp_threshold` => sample proportionally to `policy_target`
            //   (which equals visit counts, or uniform over ProvenLoss children when ProvenWin).
            let action: u8 = if move_count >= self.temp_threshold {
                let mut best_col = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for c in 0..COLS {
                    if policy_target[c] > best_val {
                        best_val = policy_target[c];
                        best_col = c;
                    }
                }
                best_col as u8
            } else {
                let r: f32 = rng.gen();
                let mut cumsum = 0.0f32;
                let mut chosen = None;
                let mut last_nonzero = 0u8;
                for c in 0..COLS {
                    let p = policy_target[c];
                    if p > 0.0 {
                        last_nonzero = c as u8;
                        cumsum += p;
                        if r < cumsum {
                            chosen = Some(c as u8);
                            break;
                        }
                    }
                }
                chosen.unwrap_or(last_nonzero)
            };

            // Reset per-step counters
            slot.sims_done = 0;
            slot.root_noisy = false;
            slot.pending_nodes.clear();

            slot.game.make_move(action);

            if slot.game.is_terminal() {
                slot.active = false;
                let result = slot.game.get_result().unwrap_or(0.0) as f32;
                let current = slot.game.current_player;

                let added = slot.history.len() * 2;
                out_states.reserve(added * 3 * BOARD_SIZE);
                out_policies.reserve(added * COLS);
                out_values.reserve(added);

                for entry in &slot.history {
                    let val = if entry.player == current {
                        result
                    } else {
                        -result
                    };

                    // Original example
                    out_states.extend_from_slice(&entry.state);
                    out_policies.extend_from_slice(&entry.policy);
                    out_values.push(val);

                    // Horizontally mirrored example
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
                    out_states.extend_from_slice(&mirrored_state);
                    out_policies.extend_from_slice(&mirrored_policy);
                    out_values.push(val);
                    num_examples += 2;
                }
                slot.history.clear();
            } else {
                new_active_count += 1;
                if let Some(child) = slot.tree.find_child_action(0, action) {
                    if !slot.tree.rebase(child) {
                        slot.tree = MCTSTree::new(slot.game.clone());
                    }
                } else {
                    slot.tree = MCTSTree::new(slot.game.clone());
                }
            }
        }

        self.active_count = new_active_count;
        (out_states, out_policies, out_values, num_examples)
    }

    #[allow(dead_code)]
    pub fn advance_games(&mut self) -> Vec<Example> {
        let (states, policies, values, n) = self.advance_games_flat();
        let mut examples = Vec::with_capacity(n);
        for i in 0..n {
            let mut state = [0.0f32; 3 * BOARD_SIZE];
            state.copy_from_slice(&states[i * 3 * BOARD_SIZE..(i + 1) * 3 * BOARD_SIZE]);
            let mut policy = [0.0f32; COLS];
            policy.copy_from_slice(&policies[i * COLS..(i + 1) * COLS]);
            examples.push(Example {
                state,
                policy,
                value: values[i],
            });
        }
        examples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_engine() {
        let engine = BatchedSelfPlay::new(10, 100, 1.5, 0.35, 0.25, 10, 0, 0.20);
        assert_eq!(engine.num_active(), 10);
        assert!(!engine.is_done());
        assert!(!engine.is_step_done());
    }

    #[test]
    fn test_opening_ply_distribution_covers_move_0_and_1() {
        let engine = BatchedSelfPlay::new(400, 10, 1.5, 0.35, 0.25, 10, 4, 0.20);
        let mut count_0 = 0;
        let mut count_1 = 0;
        let mut count_ge_2 = 0;
        for slot in &engine.slots {
            assert!(!slot.game.is_terminal());
            match slot.game.move_count {
                0 => count_0 += 1,
                1 => count_1 += 1,
                2..=4 => count_ge_2 += 1,
                other => panic!("Unexpected move_count {}", other),
            }
        }
        assert!(count_0 > 120, "Expected ~50% 0-ply games, got {}", count_0);
        assert!(count_1 > 50, "Expected ~25% 1-ply games, got {}", count_1);
        assert!(count_ge_2 > 50, "Expected ~25% 2..=4-ply games, got {}", count_ge_2);
    }

    #[test]
    fn test_is_step_done_exact_simulations_and_unexpanded_roots() {
        let num_sims = 25; // Not a multiple of LEAVES_PER_GAME (16)!
        let mut engine = BatchedSelfPlay::new(4, num_sims, 1.5, 0.35, 0.25, 0, 0, 0.20);

        let policies = vec![1.0 / COLS as f32; 4 * COLS];
        engine.init_roots(&policies);
        engine.add_noise();

        let mut total_evals = 0usize;
        while !engine.is_step_done() {
            let (_s, _v, count) = engine.collect_leaves();
            if count > 0 {
                total_evals += count;
                let p = vec![1.0 / COLS as f32; count * COLS];
                let v = vec![0.0f32; count];
                engine.apply_evaluations(&p, &v);
            }
        }
        assert!(engine.is_step_done());
        assert_eq!(total_evals, 4 * num_sims);

        // Verify policy target decoupling when temp_threshold == 0 (greedy action, soft policy target!)
        let _ = engine.advance_games();
        for slot in &engine.slots {
            assert!(!slot.history.is_empty());
            let pol = slot.history[0].policy;
            // With 25 uniform visits across 7 columns, soft policy has multiple non-zero entries (NOT one-hot!)
            let nonzero = pol.iter().filter(|&&x| x > 0.01).count();
            assert!(nonzero >= 2, "Expected soft visit distribution, got {:?}", pol);
        }

        // After rebase with 25 sims, chosen children were already expanded!
        let (_u_s, _u_v, u_count) = engine.get_unexpanded_root_states();
        assert_eq!(u_count, 0, "Expected all rebased roots to already be expanded");
    }

    #[test]
    fn test_full_game_loop() {
        let num_sims = 20;
        let mut engine = BatchedSelfPlay::new(3, num_sims, 1.5, 0.35, 0.25, 10, 4, 0.20);
        let mut all_examples = Vec::new();

        let n = engine.num_active();
        let policies = vec![1.0 / COLS as f32; n * COLS];
        engine.init_roots(&policies);

        let mut iterations = 0;
        while !engine.is_done() && iterations < 500 {
            engine.add_noise();
            while !engine.is_step_done() {
                let (_states, _valid, count) = engine.collect_leaves();
                if count > 0 {
                    let p = vec![1.0 / COLS as f32; count * COLS];
                    let v = vec![0.0f32; count];
                    engine.apply_evaluations(&p, &v);
                }
            }

            let examples = engine.advance_games();
            all_examples.extend(examples);

            if engine.num_active() > 0 {
                let (_, _, u_count) = engine.get_unexpanded_root_states();
                if u_count > 0 {
                    let p = vec![1.0 / COLS as f32; u_count * COLS];
                    engine.init_unexpanded_roots(&p);
                }
            }
            iterations += 1;
        }

        assert!(engine.is_done());
        assert!(!all_examples.is_empty());
        assert_eq!(all_examples.len() % 2, 0);
    }
}
