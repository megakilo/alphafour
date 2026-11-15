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

/// History entry recorded during a game (compact `u8` binary planes).
struct HistoryEntry {
    state: [u8; 3 * BOARD_SIZE],
    policy: [f32; COLS],
    player: i8,
}

struct SelfPlaySlot {
    game: ConnectFour,
    tree: MCTSTree,
    history: Vec<HistoryEntry>,
    sims_done: usize,
    root_noisy: bool,
    noise_applied: bool,
    pending_root_init: bool,
    last_batch_leaves: usize,
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

/// Batched self-play engine (supports both step-wise and barrier-free continuous batching).
pub struct BatchedSelfPlay {
    slots: Vec<SelfPlaySlot>,
    initial_games: usize,
    active_count: usize,
    pending_count: usize,

    // Completed examples accumulated during continuous mode
    completed_states_u8: Vec<u8>,
    completed_policies: Vec<f32>,
    completed_values: Vec<f32>,
    completed_examples: usize,

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
            let tree = MCTSTree::new(game);
            slots.push(SelfPlaySlot {
                game,
                tree,
                history: Vec::with_capacity(BOARD_SIZE),
                sims_done: 0,
                root_noisy: false,
                noise_applied: false,
                pending_root_init: false,
                last_batch_leaves: LEAVES_PER_GAME,
                active: true,
                pending_nodes: Vec::with_capacity(LEAVES_PER_GAME),
                batch_offset: 0,
            });
        }

        let est_examples = num_games * 68;
        BatchedSelfPlay {
            slots,
            initial_games: num_games,
            active_count: num_games,
            pending_count: 0,
            completed_states_u8: Vec::with_capacity(est_examples * 3 * BOARD_SIZE),
            completed_policies: Vec::with_capacity(est_examples * COLS),
            completed_values: Vec::with_capacity(est_examples),
            completed_examples: 0,
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
                slot.noise_applied = true;
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

    /// Compute decoupled normalized policy target for `slot`'s current root.
    #[inline]
    fn compute_root_policy_target(slot: &SelfPlaySlot) -> [f32; COLS] {
        let root = slot.tree.root();
        let root_solver = slot.tree.nodes[root].solver_state;
        let mut policy_target = [0.0f32; COLS];

        if root_solver == SolverState::ProvenWin {
            let mut win_cols = [0usize; COLS];
            let mut n_win = 0usize;
            for &(col, c_idx) in &slot.tree.nodes[root].children {
                if slot.tree.nodes[c_idx].solver_state == SolverState::ProvenLoss {
                    win_cols[n_win] = col as usize;
                    n_win += 1;
                }
            }
            let inv = 1.0 / (n_win.max(1) as f32);
            for &c in &win_cols[..n_win] {
                policy_target[c] = inv;
            }
        } else {
            let mut counts = slot.tree.get_visit_counts(root);
            if root_solver == SolverState::ProvenDraw {
                for &(col, c_idx) in &slot.tree.nodes[root].children {
                    if slot.tree.nodes[c_idx].solver_state == SolverState::ProvenWin {
                        counts[col as usize] = 0.0;
                    }
                }
            } else if root_solver == SolverState::Ongoing {
                // Zero proven-losing moves (opponent ProvenWin) when at least one non-losing move was visited
                let has_safe_visits = slot.tree.nodes[root].children.iter().any(|&(_, c_idx)| {
                    let c = &slot.tree.nodes[c_idx];
                    c.solver_state != SolverState::ProvenWin && c.visit_count > 0
                });
                if has_safe_visits {
                    for &(col, c_idx) in &slot.tree.nodes[root].children {
                        if slot.tree.nodes[c_idx].solver_state == SolverState::ProvenWin {
                            counts[col as usize] = 0.0;
                        }
                    }
                }
            }

            let total: f32 = counts.iter().sum();
            if total > 0.0 {
                for c in 0..COLS {
                    policy_target[c] = counts[c] / total;
                }
            } else if root_solver == SolverState::ProvenDraw {
                let mut draw_cols = [0usize; COLS];
                let mut n_draw = 0usize;
                for &(col, c_idx) in &slot.tree.nodes[root].children {
                    if slot.tree.nodes[c_idx].solver_state == SolverState::ProvenDraw {
                        draw_cols[n_draw] = col as usize;
                        n_draw += 1;
                    }
                }
                let inv = 1.0 / (n_draw.max(1) as f32);
                for &c in &draw_cols[..n_draw] {
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
        policy_target
    }

    #[inline]
    fn sample_action_from_policy<R: Rng>(
        policy_target: &[f32; COLS],
        move_count: usize,
        temp_threshold: usize,
        rng: &mut R,
    ) -> u8 {
        if move_count >= temp_threshold {
            let mut best_col = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (c, &p) in policy_target.iter().enumerate() {
                if p > best_val {
                    best_val = p;
                    best_col = c;
                }
            }
            best_col as u8
        } else {
            let r: f32 = rng.gen();
            let mut cumsum = 0.0f32;
            let mut chosen = None;
            let mut last_nonzero = 0u8;
            for (c, &p) in policy_target.iter().enumerate() {
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
        }
    }

    #[inline]
    fn emit_mirrored_u8(
        history: &[HistoryEntry],
        result: f32,
        current_player: i8,
        out_states_u8: &mut Vec<u8>,
        out_policies: &mut Vec<f32>,
        out_values: &mut Vec<f32>,
    ) -> usize {
        let added = history.len() * 2;
        out_states_u8.reserve(added * 3 * BOARD_SIZE);
        out_policies.reserve(added * COLS);
        out_values.reserve(added);

        for entry in history {
            let val = if entry.player == current_player {
                result
            } else {
                -result
            };

            // Original example
            out_states_u8.extend_from_slice(&entry.state);
            out_policies.extend_from_slice(&entry.policy);
            out_values.push(val);

            // Horizontally mirrored example
            let mut mirrored_state = [0u8; 3 * BOARD_SIZE];
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
            out_states_u8.extend_from_slice(&mirrored_state);
            out_policies.extend_from_slice(&mirrored_policy);
            out_values.push(val);
        }
        added
    }

    /// Advance any active slots whose current root search has finished (barrier-free continuous mode).
    /// Returns the number of games that reached a terminal board during this flush.
    pub fn advance_ready_slots(&mut self) -> usize {
        if self.active_count == 0 {
            return 0;
        }
        let mut finished_games = 0usize;
        let mut rng = rand::thread_rng();
        let alpha = self.dirichlet_alpha;
        let eps = self.dirichlet_epsilon;
        let is_noisy = alpha > 0.0 && eps > 0.0;

        for slot in &mut self.slots {
            if !slot.active || !slot.pending_nodes.is_empty() || slot.pending_root_init {
                continue;
            }

            loop {
                let root = slot.tree.root();
                let root_solver = slot.tree.nodes[root].solver_state;
                if !slot.tree.nodes[root].is_expanded && root_solver == SolverState::Ongoing {
                    break;
                }

                if slot.tree.nodes[root].is_expanded && !slot.noise_applied {
                    slot.tree.add_dirichlet_noise(root, alpha, eps);
                    slot.root_noisy = is_noisy;
                    slot.noise_applied = true;
                }

                let tail_straggler_done = self.initial_games >= 64
                    && self.active_count <= 4
                    && slot.last_batch_leaves <= 2
                    && slot.sims_done >= (self.num_simulations * 3) / 4;

                let turn_done = slot.sims_done >= self.num_simulations
                    || slot.tree.nodes[root].solver_state != SolverState::Ongoing
                    || tail_straggler_done;

                if !turn_done {
                    break;
                }

                let policy_target = Self::compute_root_policy_target(slot);
                let move_count = slot.game.move_count as usize;
                slot.history.push(HistoryEntry {
                    state: slot.game.encode_u8(),
                    policy: policy_target,
                    player: slot.game.current_player,
                });

                let action = Self::sample_action_from_policy(
                    &policy_target,
                    move_count,
                    self.temp_threshold,
                    &mut rng,
                );

                slot.sims_done = 0;
                slot.root_noisy = false;
                slot.noise_applied = false;
                slot.pending_root_init = false;
                slot.last_batch_leaves = LEAVES_PER_GAME;
                slot.pending_nodes.clear();

                slot.game.make_move(action);

                if slot.game.is_terminal() {
                    slot.active = false;
                    self.active_count -= 1;
                    finished_games += 1;

                    let result = slot.game.get_result().unwrap_or(0.0) as f32;
                    let current = slot.game.current_player;
                    self.completed_examples += Self::emit_mirrored_u8(
                        &slot.history,
                        result,
                        current,
                        &mut self.completed_states_u8,
                        &mut self.completed_policies,
                        &mut self.completed_values,
                    );
                    slot.history.clear();
                    break;
                } else if let Some(child) = slot.tree.find_child_action(0, action) {
                    if !slot.tree.rebase(child) {
                        slot.tree = MCTSTree::new(slot.game);
                    }
                } else {
                    slot.tree = MCTSTree::new(slot.game);
                }
            }
        }

        finished_games
    }

    /// Barrier-free continuous leaf & unexpanded-root collection encoded as binary `u8` planes.
    /// Automatically advances any slots that finish via terminal/solver hits so `count > 0`
    /// whenever `!self.is_done()`.
    /// Returns `(states_u8, valid_flat, leaf_count, newly_finished_games)`.
    pub fn collect_continuous_u8(&mut self) -> (Vec<u8>, Vec<bool>, usize, usize) {
        if self.pending_count > 0 {
            for slot in &mut self.slots {
                if !slot.pending_root_init {
                    for &node_idx in &slot.pending_nodes {
                        slot.tree.revert_virtual_loss(node_idx);
                    }
                }
                slot.pending_root_init = false;
                slot.pending_nodes.clear();
            }
            self.pending_count = 0;
        }

        let mut total_finished = 0usize;
        let num_sims = self.num_simulations;
        let c_puct = self.c_puct;
        let c_fpu = self.c_fpu;
        let alpha = self.dirichlet_alpha;
        let eps = self.dirichlet_epsilon;
        let is_noisy = alpha > 0.0 && eps > 0.0;

        for _guard in 0..(num_sims + 64) {
            total_finished += self.advance_ready_slots();
            if self.active_count == 0 {
                self.pending_count = 0;
                return (Vec::new(), Vec::new(), 0, total_finished);
            }

            let per_slot: Vec<(Vec<u8>, Vec<bool>)> = self
                .slots
                .par_iter_mut()
                .map(|slot| {
                    slot.pending_nodes.clear();
                    slot.pending_root_init = false;
                    if !slot.active {
                        return (Vec::new(), Vec::new());
                    }

                    let root = slot.tree.root();
                    if slot.tree.nodes[root].solver_state != SolverState::Ongoing {
                        slot.sims_done = num_sims;
                        return (Vec::new(), Vec::new());
                    }

                    // Unexpanded root: evaluate root in the same GPU batch without counting as a simulation
                    if !slot.tree.nodes[root].is_expanded {
                        slot.pending_root_init = true;
                        slot.pending_nodes.push(root);
                        let mut s_buf = Vec::with_capacity(3 * BOARD_SIZE);
                        let mut v_buf = Vec::with_capacity(COLS);
                        s_buf.extend_from_slice(&slot.game.encode_u8());
                        v_buf.extend_from_slice(&slot.game.get_valid_moves());
                        return (s_buf, v_buf);
                    }

                    if !slot.noise_applied {
                        slot.tree.add_dirichlet_noise(root, alpha, eps);
                        slot.root_noisy = is_noisy;
                        slot.noise_applied = true;
                    }

                    if slot.sims_done >= num_sims {
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

            if total_count > 0 {
                let mut states = Vec::with_capacity(total_count * 3 * BOARD_SIZE);
                let mut valid = Vec::with_capacity(total_count * COLS);
                for (s_buf, v_buf) in per_slot {
                    states.extend_from_slice(&s_buf);
                    valid.extend_from_slice(&v_buf);
                }
                self.pending_count = total_count;
                return (states, valid, total_count, total_finished);
            }
        }

        self.pending_count = 0;
        (Vec::new(), Vec::new(), 0, total_finished)
    }

    /// Apply packed `(pending_count, 8)` evaluations (`row[0..7]` = policy, `row[7]` = value)
    /// and immediately advance any slots whose search is now complete.
    /// Returns the number of games that terminated on this step.
    pub fn apply_continuous_packed(&mut self, evals_packed: &[f32]) -> usize {
        let num_sims = self.num_simulations;
        let alpha = self.dirichlet_alpha;
        let eps = self.dirichlet_epsilon;
        let is_noisy = alpha > 0.0 && eps > 0.0;
        let stride = COLS + 1;

        self.slots
            .par_iter_mut()
            .filter(|s| !s.pending_nodes.is_empty())
            .for_each(|slot| {
                let offset = slot.batch_offset;
                if slot.pending_root_init {
                    let row = &evals_packed[offset * stride..(offset + 1) * stride];
                    let policy = &row[..COLS];
                    let root = slot.tree.root();
                    slot.tree.expand(root, policy);
                    slot.tree.add_dirichlet_noise(root, alpha, eps);
                    slot.root_noisy = is_noisy;
                    slot.noise_applied = true;
                    slot.pending_root_init = false;
                    slot.pending_nodes.clear();
                    if slot.tree.nodes[root].solver_state != SolverState::Ongoing {
                        slot.sims_done = num_sims;
                    }
                } else {
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
                }
            });

        self.pending_count = 0;
        self.advance_ready_slots()
    }

    /// Take all accumulated completed examples with `u8` states (zero-copy handoff).
    pub fn take_completed_u8(&mut self) -> (Vec<u8>, Vec<f32>, Vec<f32>, usize) {
        let n = self.completed_examples;
        self.completed_examples = 0;
        (
            std::mem::take(&mut self.completed_states_u8),
            std::mem::take(&mut self.completed_policies),
            std::mem::take(&mut self.completed_values),
            n,
        )
    }

    /// Take all accumulated completed examples converted to `f32` states.
    pub fn take_completed_f32(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
        let (states_u8, policies, values, n) = self.take_completed_u8();
        let states_f32: Vec<f32> = states_u8.into_iter().map(|b| b as f32).collect();
        (states_f32, policies, values, n)
    }

    /// Advance active games, decoupling training policy targets from move-selection temperature.
    /// Returns flattened `(states, policies, values, num_examples)` with horizontal mirror augmentation.
    pub fn advance_games_flat(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
        let mut out_states_u8 = Vec::new();
        let mut out_policies = Vec::new();
        let mut out_values = Vec::new();
        let mut num_examples = 0usize;
        let mut new_active_count = 0usize;
        let mut rng = rand::thread_rng();

        for slot in &mut self.slots {
            if !slot.active {
                continue;
            }

            let move_count = slot.game.move_count as usize;
            let policy_target = Self::compute_root_policy_target(slot);

            slot.history.push(HistoryEntry {
                state: slot.game.encode_u8(),
                policy: policy_target,
                player: slot.game.current_player,
            });

            let action = Self::sample_action_from_policy(
                &policy_target,
                move_count,
                self.temp_threshold,
                &mut rng,
            );

            // Reset per-step counters
            slot.sims_done = 0;
            slot.root_noisy = false;
            slot.noise_applied = false;
            slot.pending_root_init = false;
            slot.last_batch_leaves = LEAVES_PER_GAME;
            slot.pending_nodes.clear();

            slot.game.make_move(action);

            if slot.game.is_terminal() {
                slot.active = false;
                let result = slot.game.get_result().unwrap_or(0.0) as f32;
                let current = slot.game.current_player;
                num_examples += Self::emit_mirrored_u8(
                    &slot.history,
                    result,
                    current,
                    &mut out_states_u8,
                    &mut out_policies,
                    &mut out_values,
                );
                slot.history.clear();
            } else {
                new_active_count += 1;
                if let Some(child) = slot.tree.find_child_action(0, action) {
                    if !slot.tree.rebase(child) {
                        slot.tree = MCTSTree::new(slot.game);
                    }
                } else {
                    slot.tree = MCTSTree::new(slot.game);
                }
            }
        }

        self.active_count = new_active_count;
        let out_states: Vec<f32> = out_states_u8.into_iter().map(|b| b as f32).collect();
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

    #[test]
    fn test_continuous_u8_loop_exact_completion() {
        let num_sims = 20;
        let mut engine = BatchedSelfPlay::new(6, num_sims, 1.5, 0.35, 0.25, 10, 4, 0.20);
        let mut finished_total = 0usize;
        let mut steps = 0usize;

        while !engine.is_done() && steps < 2_000 {
            let (states_u8, valid, count, finished_collect) = engine.collect_continuous_u8();
            finished_total += finished_collect;
            if count > 0 {
                assert_eq!(states_u8.len(), count * 3 * BOARD_SIZE);
                assert_eq!(valid.len(), count * COLS);
                let mut packed = vec![0.0f32; count * (COLS + 1)];
                for k in 0..count {
                    for c in 0..COLS {
                        packed[k * (COLS + 1) + c] = 1.0 / COLS as f32;
                    }
                    packed[k * (COLS + 1) + COLS] = 0.0;
                }
                finished_total += engine.apply_continuous_packed(&packed);
            }
            steps += 1;
        }

        assert!(engine.is_done());
        assert_eq!(finished_total, 6);
        let (states_f32, policies, values, n) = engine.take_completed_f32();
        assert!(n > 0 && n % 2 == 0);
        assert_eq!(states_f32.len(), n * 3 * BOARD_SIZE);
        assert_eq!(policies.len(), n * COLS);
        assert_eq!(values.len(), n);
    }
}
