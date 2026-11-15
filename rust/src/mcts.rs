/// Arena-based MCTS tree for high-performance tree search.
///
/// Nodes are stored in a contiguous Vec (arena allocation) and referenced
/// by index, avoiding heap fragmentation and borrow checker issues.
///
/// Supports:
/// - Virtual losses with collision-abort for duplicate-free parallel leaf selection
/// - Exact MCTS-Solver (ProvenWin / ProvenLoss / ProvenDraw propagation)
/// - Parent-relative First Play Urgency (FPU) reduction

use crate::game::{ConnectFour, COLS};
use rand_distr::{Distribution, Gamma};

/// Exact game-theoretic solver status from `node.game.current_player`'s perspective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverState {
    Ongoing,
    ProvenWin,
    ProvenLoss,
    ProvenDraw,
}

impl SolverState {
    #[inline]
    pub fn from_terminal_result(result: Option<f64>) -> Self {
        match result {
            Some(r) if r < 0.0 => SolverState::ProvenLoss,
            Some(_) => SolverState::ProvenDraw,
            None => SolverState::Ongoing,
        }
    }

    #[inline]
    pub fn proven_value(self) -> Option<f64> {
        match self {
            SolverState::ProvenWin => Some(1.0),
            SolverState::ProvenLoss => Some(-1.0),
            SolverState::ProvenDraw => Some(0.0),
            SolverState::Ongoing => None,
        }
    }
}

/// Inline fixed-capacity child list (`<= COLS`) avoiding per-node heap allocations and drop glue.
#[derive(Clone, Copy, Debug)]
pub struct ChildrenList {
    buf: [(u8, usize); COLS],
    len: u8,
}

impl ChildrenList {
    #[inline]
    pub const fn new() -> Self {
        Self {
            buf: [(0, 0); COLS],
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, item: (u8, usize)) {
        debug_assert!((self.len as usize) < COLS);
        self.buf[self.len as usize] = item;
        self.len += 1;
    }
}

impl Default for ChildrenList {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for ChildrenList {
    type Target = [(u8, usize)];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.buf[..self.len as usize]
    }
}

impl<'a> IntoIterator for &'a ChildrenList {
    type Item = &'a (u8, usize);
    type IntoIter = std::slice::Iter<'a, (u8, usize)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.buf[..self.len as usize].iter()
    }
}

#[derive(Clone, Copy)]
pub struct MCTSNode {
    pub game: ConnectFour,
    pub children: ChildrenList, // inline (action, child_index) pairs
    pub parent: Option<usize>,
    pub visit_count: u32,
    pub total_value: f64,
    pub prior: f32,
    pub is_expanded: bool,
    /// Virtual losses currently applied (for parallel leaf selection).
    pub virtual_losses: u32,
    /// Exact MCTS-Solver state from `game.current_player`'s perspective.
    pub solver_state: SolverState,
}

const _: () = assert!(!std::mem::needs_drop::<MCTSNode>());

pub struct MCTSTree {
    pub nodes: Vec<MCTSNode>,
}

impl MCTSTree {
    /// Create a new tree with the given game state as root.
    pub fn new(game: ConnectFour) -> Self {
        let solver_state = SolverState::from_terminal_result(game.get_result());
        let root = MCTSNode {
            game,
            children: ChildrenList::new(),
            parent: None,
            visit_count: 0,
            total_value: 0.0,
            prior: 0.0,
            is_expanded: false,
            virtual_losses: 0,
            solver_state,
        };
        MCTSTree { nodes: vec![root] }
    }

    #[inline]
    pub fn root(&self) -> usize {
        0
    }

    /// Select a leaf node by traversing from `root` using PUCT + Solver + FPU.
    /// Stops immediately at any proven, unexpanded, or childless node.
    pub fn select_leaf(
        &self,
        root: usize,
        c_puct: f32,
        c_fpu: f32,
        is_noisy_root: bool,
    ) -> usize {
        let mut idx = root;
        loop {
            let node = &self.nodes[idx];
            if node.solver_state != SolverState::Ongoing
                || !node.is_expanded
                || node.children.is_empty()
            {
                return idx;
            }
            let at_noisy_root = is_noisy_root && (idx == root);
            idx = self.best_child(idx, c_puct, c_fpu, at_noisy_root);
        }
    }

    /// Select child with highest PUCT score, respecting virtual losses, solver states, and FPU.
    fn best_child(
        &self,
        parent_idx: usize,
        c_puct: f32,
        c_fpu: f32,
        is_noisy_root: bool,
    ) -> usize {
        let parent = &self.nodes[parent_idx];

        // 1. Immediately pick any child that is ProvenLoss for the opponent (winning move for parent).
        for &(_, child_idx) in &parent.children {
            if self.nodes[child_idx].solver_state == SolverState::ProvenLoss {
                return child_idx;
            }
        }

        // 2. Check if all children are ProvenWin (meaning every move loses for parent).
        let all_proven_win = parent
            .children
            .iter()
            .all(|&(_, child_idx)| self.nodes[child_idx].solver_state == SolverState::ProvenWin);

        // 3. Parent-relative FPU utility (from parent's perspective) for unvisited children.
        let fpu_neg_q = if !is_noisy_root && parent.visit_count > 0 {
            let parent_q = parent.total_value / parent.visit_count as f64;
            let visited_policy: f64 = parent
                .children
                .iter()
                .filter_map(|&(_, c_idx)| {
                    let c = &self.nodes[c_idx];
                    if c.visit_count + c.virtual_losses > 0 {
                        Some(c.prior as f64)
                    } else {
                        None
                    }
                })
                .sum();
            (parent_q - c_fpu as f64 * visited_policy.max(0.0).sqrt()).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let effective_parent_visits = parent.visit_count + parent.virtual_losses;
        let sqrt_parent = (effective_parent_visits.max(1) as f64).sqrt();

        let mut best_idx = parent.children[0].1;
        let mut best_score = f64::NEG_INFINITY;

        for &(_, child_idx) in &parent.children {
            let child = &self.nodes[child_idx];

            // Skip moves that are proven wins for opponent unless all moves lose.
            if !all_proven_win && child.solver_state == SolverState::ProvenWin {
                continue;
            }

            let effective_visits = child.visit_count + child.virtual_losses;
            let neg_q = if effective_visits == 0 {
                fpu_neg_q
            } else {
                // child.total_value is from child's (opponent's) perspective.
                // Adding virtual_losses makes child look better for opponent = worse for parent,
                // discouraging duplicate parallel path selection.
                let child_q =
                    (child.total_value + child.virtual_losses as f64) / effective_visits as f64;
                -child_q
            };

            let exploration =
                c_puct as f64 * child.prior as f64 * sqrt_parent / (1.0 + effective_visits as f64);
            let score = neg_q + exploration;
            if score > best_score {
                best_score = score;
                best_idx = child_idx;
            }
        }
        best_idx
    }

    /// Propagate exact MCTS-Solver states upward starting from `start_idx` through ancestors.
    pub fn propagate_solver_up(&mut self, mut node_opt: Option<usize>) {
        while let Some(p_idx) = node_opt {
            if self.nodes[p_idx].solver_state != SolverState::Ongoing {
                node_opt = self.nodes[p_idx].parent;
                continue;
            }
            if !self.nodes[p_idx].is_expanded || self.nodes[p_idx].children.is_empty() {
                break;
            }

            let mut any_child_loss = false;
            let mut all_children_win = true;
            let mut all_children_proven = true;

            for &(_, c_idx) in &self.nodes[p_idx].children {
                match self.nodes[c_idx].solver_state {
                    SolverState::ProvenLoss => {
                        any_child_loss = true;
                        all_children_win = false;
                        break;
                    }
                    SolverState::ProvenWin => {}
                    SolverState::ProvenDraw => {
                        all_children_win = false;
                    }
                    SolverState::Ongoing => {
                        all_children_win = false;
                        all_children_proven = false;
                    }
                }
            }

            let new_state = if any_child_loss {
                SolverState::ProvenWin
            } else if all_children_win {
                SolverState::ProvenLoss
            } else if all_children_proven {
                SolverState::ProvenDraw
            } else {
                SolverState::Ongoing
            };

            if new_state == SolverState::Ongoing {
                break;
            }
            self.nodes[p_idx].solver_state = new_state;
            node_opt = self.nodes[p_idx].parent;
        }
    }

    /// Apply virtual loss along the path from node to root.
    pub fn apply_virtual_loss(&mut self, node_idx: usize) {
        let mut idx = Some(node_idx);
        while let Some(i) = idx {
            self.nodes[i].virtual_losses += 1;
            idx = self.nodes[i].parent;
        }
    }

    /// Revert virtual loss along the path from node to root.
    pub fn revert_virtual_loss(&mut self, node_idx: usize) {
        let mut idx = Some(node_idx);
        while let Some(i) = idx {
            self.nodes[i].virtual_losses = self.nodes[i].virtual_losses.saturating_sub(1);
            idx = self.nodes[i].parent;
        }
    }

    /// Select up to `max_k` distinct leaves using virtual losses and collision-abort.
    /// Proven/terminal leaves are backpropagated immediately without NN evaluation.
    /// If selection lands on an unexpanded leaf that ALREADY has `virtual_losses > 0`,
    /// aborts the loop immediately so duplicate leaves are never queued in the same batch.
    pub fn select_multiple_leaves(
        &mut self,
        root: usize,
        c_puct: f32,
        c_fpu: f32,
        is_noisy_root: bool,
        max_k: usize,
    ) -> (Vec<usize>, u32) {
        let mut leaves = Vec::with_capacity(max_k);
        let mut terminal_count = 0u32;

        for _ in 0..max_k {
            let leaf = self.select_leaf(root, c_puct, c_fpu, is_noisy_root);

            if let Some(val) = self.nodes[leaf].solver_state.proven_value() {
                self.backpropagate(leaf, val);
                terminal_count += 1;
                if self.nodes[root].solver_state != SolverState::Ongoing {
                    break;
                }
            } else if let Some(val) = self.nodes[leaf].game.get_result() {
                self.nodes[leaf].solver_state = SolverState::from_terminal_result(Some(val));
                let parent = self.nodes[leaf].parent;
                self.propagate_solver_up(parent);
                self.backpropagate(leaf, val);
                terminal_count += 1;
                if self.nodes[root].solver_state != SolverState::Ongoing {
                    break;
                }
            } else {
                // Collision-abort: never queue duplicate leaves in the same batch.
                if self.nodes[leaf].virtual_losses > 0 {
                    break;
                }
                self.apply_virtual_loss(leaf);
                leaves.push(leaf);
            }
        }

        (leaves, terminal_count)
    }

    /// Expand a node: idempotent guard, creates child nodes for valid moves,
    /// sets terminal solver states on children, and propagates proven states upward.
    pub fn expand(&mut self, node_idx: usize, policy: &[f32]) {
        if self.nodes[node_idx].is_expanded {
            return;
        }

        if let Some(res) = self.nodes[node_idx].game.get_result() {
            self.nodes[node_idx].solver_state = SolverState::from_terminal_result(Some(res));
            self.nodes[node_idx].is_expanded = true;
            let parent = self.nodes[node_idx].parent;
            self.propagate_solver_up(parent);
            return;
        }

        let game = self.nodes[node_idx].game;
        let valid = game.get_valid_moves();

        let mut children = ChildrenList::new();
        for col in 0..COLS {
            if valid[col] {
                let mut child_game = game;
                child_game.make_move(col as u8);
                let child_solver = SolverState::from_terminal_result(child_game.get_result());
                let child_idx = self.nodes.len();
                self.nodes.push(MCTSNode {
                    game: child_game,
                    children: ChildrenList::new(),
                    parent: Some(node_idx),
                    visit_count: 0,
                    total_value: 0.0,
                    prior: policy[col],
                    is_expanded: false,
                    virtual_losses: 0,
                    solver_state: child_solver,
                });
                children.push((col as u8, child_idx));
            }
        }
        self.nodes[node_idx].children = children;
        self.nodes[node_idx].is_expanded = true;

        // Propagate any immediate terminal children upward through `node_idx` and ancestors.
        self.propagate_solver_up(Some(node_idx));
    }

    /// Backpropagate value up the tree, negating at each level.
    pub fn backpropagate(&mut self, node_idx: usize, mut value: f64) {
        let mut idx = Some(node_idx);
        while let Some(i) = idx {
            self.nodes[i].visit_count += 1;
            self.nodes[i].total_value += value;
            value = -value;
            idx = self.nodes[i].parent;
        }
    }

    /// Add Dirichlet noise to root's children priors without heap allocations.
    pub fn add_dirichlet_noise(&mut self, node_idx: usize, alpha: f64, epsilon: f64) {
        let children = self.nodes[node_idx].children;
        let num_children = children.len();
        if num_children < 2 || alpha <= 0.0 || epsilon <= 0.0 {
            return;
        }

        if let Ok(gamma) = Gamma::new(alpha, 1.0) {
            let mut rng = rand::thread_rng();
            let mut raw = [0.0f64; COLS];
            let mut sum = 0.0f64;
            for item in raw.iter_mut().take(num_children) {
                let g = gamma.sample(&mut rng);
                *item = g;
                sum += g;
            }
            if sum > 0.0 {
                for (i, &(_, child_idx)) in children.iter().enumerate() {
                    let prior = self.nodes[child_idx].prior as f64;
                    let noise_i = raw[i] / sum;
                    self.nodes[child_idx].prior =
                        ((1.0 - epsilon) * prior + epsilon * noise_i) as f32;
                }
            }
        }
    }

    /// Get visit counts for node's children as a `[f32; COLS]` array.
    pub fn get_visit_counts(&self, node_idx: usize) -> [f32; COLS] {
        let mut counts = [0.0f32; COLS];
        for &(action, child_idx) in &self.nodes[node_idx].children {
            counts[action as usize] = self.nodes[child_idx].visit_count as f32;
        }
        counts
    }

    /// Find the child node index for a given action.
    pub fn find_child_action(&self, node_idx: usize, action: u8) -> Option<usize> {
        self.nodes[node_idx]
            .children
            .iter()
            .find(|&&(a, _)| a == action)
            .map(|&(_, idx)| idx)
    }

    /// Rebase the tree to a child node, making it the new root.
    pub fn rebase(&mut self, child_idx: usize) -> bool {
        if child_idx >= self.nodes.len() {
            return false;
        }

        let cap = (self.nodes.len() / 3).max(64);
        let mut queue: Vec<(usize, Option<usize>)> = Vec::with_capacity(cap);
        let mut new_nodes: Vec<MCTSNode> = Vec::with_capacity(cap);

        queue.push((child_idx, None));
        let mut head = 0usize;

        while head < queue.len() {
            let (old_idx, parent_new) = queue[head];
            let node = &self.nodes[old_idx];

            let mut new_children = ChildrenList::new();
            for &(action, old_child_idx) in &node.children {
                let new_child_idx = queue.len();
                queue.push((old_child_idx, Some(head)));
                new_children.push((action, new_child_idx));
            }

            new_nodes.push(MCTSNode {
                game: node.game,
                children: new_children,
                parent: parent_new,
                visit_count: node.visit_count,
                total_value: node.total_value,
                prior: node.prior,
                is_expanded: node.is_expanded,
                virtual_losses: 0,
                solver_state: node.solver_state,
            });
            head += 1;
        }

        self.nodes = new_nodes;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tree() {
        let game = ConnectFour::new();
        let tree = MCTSTree::new(game);
        assert_eq!(tree.nodes.len(), 1);
        assert!(!tree.nodes[0].is_expanded);
        assert_eq!(tree.nodes[0].visit_count, 0);
        assert_eq!(tree.nodes[0].solver_state, SolverState::Ongoing);
    }

    #[test]
    fn test_expand_root_and_idempotent_guard() {
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [1.0 / 7.0; COLS];
        tree.expand(0, &policy);

        assert!(tree.nodes[0].is_expanded);
        assert_eq!(tree.nodes[0].children.len(), 7);
        assert_eq!(tree.nodes.len(), 8);

        // Second expand call on same node MUST be a no-op!
        let alt_policy = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        tree.expand(0, &alt_policy);
        assert_eq!(tree.nodes.len(), 8);
        assert!((tree.nodes[1].prior - (1.0 / 7.0)).abs() < 1e-6);
    }

    #[test]
    fn test_select_leaf_unexpanded() {
        let game = ConnectFour::new();
        let tree = MCTSTree::new(game);
        assert_eq!(tree.select_leaf(0, 1.5, 0.20, false), 0);
    }

    #[test]
    fn test_select_leaf_after_expand() {
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [1.0 / 7.0; COLS];
        tree.expand(0, &policy);

        let leaf = tree.select_leaf(0, 1.5, 0.20, false);
        assert!((1..=7).contains(&leaf));
    }

    #[test]
    fn test_backpropagate() {
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [1.0 / 7.0; COLS];
        tree.expand(0, &policy);

        let child_idx = tree.nodes[0].children[0].1;
        tree.backpropagate(child_idx, 0.5);

        assert_eq!(tree.nodes[child_idx].visit_count, 1);
        assert!((tree.nodes[child_idx].total_value - 0.5).abs() < 1e-6);
        assert_eq!(tree.nodes[0].visit_count, 1);
        assert!((tree.nodes[0].total_value - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_visit_counts() {
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [1.0 / 7.0; COLS];
        tree.expand(0, &policy);

        let child_idx = tree.nodes[0].children[0].1;
        tree.backpropagate(child_idx, 1.0);
        tree.backpropagate(child_idx, -1.0);

        let counts = tree.get_visit_counts(0);
        let action = tree.nodes[0].children[0].0;
        assert_eq!(counts[action as usize], 2.0);
    }

    #[test]
    fn test_dirichlet_noise() {
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [1.0 / 7.0; COLS];
        tree.expand(0, &policy);

        let before: Vec<f32> = tree.nodes[0]
            .children
            .iter()
            .map(|&(_, idx)| tree.nodes[idx].prior)
            .collect();

        tree.add_dirichlet_noise(0, 0.35, 0.25);

        let after: Vec<f32> = tree.nodes[0]
            .children
            .iter()
            .map(|&(_, idx)| tree.nodes[idx].prior)
            .collect();

        assert_ne!(before, after);
    }

    #[test]
    fn test_virtual_loss_apply_revert() {
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [1.0 / 7.0; COLS];
        tree.expand(0, &policy);

        let child_idx = tree.nodes[0].children[0].1;

        tree.apply_virtual_loss(child_idx);
        assert_eq!(tree.nodes[child_idx].virtual_losses, 1);
        assert_eq!(tree.nodes[0].virtual_losses, 1);

        tree.apply_virtual_loss(child_idx);
        assert_eq!(tree.nodes[child_idx].virtual_losses, 2);
        assert_eq!(tree.nodes[0].virtual_losses, 2);

        tree.revert_virtual_loss(child_idx);
        assert_eq!(tree.nodes[child_idx].virtual_losses, 1);
        assert_eq!(tree.nodes[0].virtual_losses, 1);

        tree.revert_virtual_loss(child_idx);
        assert_eq!(tree.nodes[child_idx].virtual_losses, 0);
        assert_eq!(tree.nodes[0].virtual_losses, 0);
    }

    #[test]
    fn test_select_multiple_leaves_diversity() {
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [1.0 / 7.0; COLS];
        tree.expand(0, &policy);

        for i in 0..7 {
            let child_idx = tree.nodes[0].children[i].1;
            tree.expand(child_idx, &policy);
        }

        let (leaves, _terminal) = tree.select_multiple_leaves(0, 1.5, 0.20, false, 4);
        assert_eq!(leaves.len(), 4);

        let unique: std::collections::HashSet<usize> = leaves.iter().cloned().collect();
        assert_eq!(
            unique.len(),
            4,
            "Expected all 4 selected leaves to be unique, got {:?}",
            leaves
        );

        for &leaf in &leaves {
            tree.revert_virtual_loss(leaf);
        }
        for node in &tree.nodes {
            assert_eq!(node.virtual_losses, 0);
        }
    }

    #[test]
    fn test_collision_abort_prevents_duplicates() {
        // Depth-1 tree has only 7 leaves; requesting k=16 MUST abort at 7 unique leaves!
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [1.0 / 7.0; COLS];
        tree.expand(0, &policy);

        let (leaves, _) = tree.select_multiple_leaves(0, 1.5, 0.20, true, 16);
        assert_eq!(leaves.len(), 7);
        let unique: std::collections::HashSet<usize> = leaves.iter().cloned().collect();
        assert_eq!(unique.len(), 7);
    }

    #[test]
    fn test_mcts_solver_mate_in_1_and_mate_in_2() {
        // Mate-in-1: P1 has 3 in col 0, P2 has 3 in col 1
        let g1 = ConnectFour::from_moves("121212").unwrap();
        let mut tree1 = MCTSTree::new(g1);
        let policy = [1.0 / 7.0; COLS];
        tree1.expand(0, &policy);
        assert_eq!(tree1.nodes[0].solver_state, SolverState::ProvenWin);
        let win_child = tree1.find_child_action(0, 0).unwrap();
        assert_eq!(tree1.nodes[win_child].solver_state, SolverState::ProvenLoss);

        // Mate-in-2 (open-ended 3-in-a-row on bottom: P1 has cols 2,3 (`"34"`), P2 played col 0,0 (`"3141"`)):
        // P1 plays col 1 or col 4 (`2` or `5` 1-indexed) creating open-both-ends 3-in-a-row on bottom row!
        // Regardless of P2's reply, P1 wins on the next move!
        let g2 = ConnectFour::from_moves("3141").unwrap();
        let mut tree2 = MCTSTree::new(g2);
        tree2.expand(0, &policy);
        assert_eq!(tree2.nodes[0].solver_state, SolverState::Ongoing);

        // Expand P1's winning move col 4 (0-indexed: creates P1 stones at cols 2,3,4 with cols 1 & 5 open!)
        let c_win = tree2.find_child_action(0, 4).unwrap();
        tree2.expand(c_win, &policy);
        // Expand all 7 of P2's replies under `c_win`: each gives P1 an immediate win!
        let replies: Vec<usize> = tree2.nodes[c_win]
            .children
            .iter()
            .map(|&(_, idx)| idx)
            .collect();
        for r_idx in replies {
            tree2.expand(r_idx, &policy);
        }
        assert_eq!(tree2.nodes[c_win].solver_state, SolverState::ProvenLoss);
        assert_eq!(tree2.nodes[0].solver_state, SolverState::ProvenWin);
    }

    #[test]
    fn test_fpu_reduction_when_parent_losing() {
        // When parent_q = -0.8 (parent losing) and one child has Q_child = +0.5 (so neg_q = -0.5 for parent),
        // without FPU reduction an unvisited child with tiny prior would have neg_q = 0.0 > -0.5 and be picked;
        // with parent-relative FPU reduction, unvisited neg_q = -0.8 - 0.2*sqrt(p) < -0.8 < -0.5, so MCTS stays focused!
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [0.94, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
        tree.expand(0, &policy);

        let c0 = tree.nodes[0].children[0].1;
        tree.expand(c0, &[1.0 / 7.0; COLS]);
        // Give root parent_q = -0.8, while c0 has child_q = +0.5 (neg_q = -0.5)
        tree.nodes[0].visit_count = 10;
        tree.nodes[0].total_value = -8.0;
        tree.nodes[c0].visit_count = 10;
        tree.nodes[c0].total_value = 5.0;

        let leaf = tree.select_leaf(0, 1.5, 0.20, false);
        // Must continue down c0's subtree rather than jumping to unvisited 0.01-prior root children!
        assert_eq!(tree.nodes[leaf].parent, Some(c0));
    }

    #[test]
    fn test_find_child_action() {
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [1.0 / 7.0; COLS];
        tree.expand(0, &policy);

        for &(action, child_idx) in &tree.nodes[0].children.clone() {
            assert_eq!(tree.find_child_action(0, action), Some(child_idx));
        }
        assert_eq!(tree.find_child_action(0, 100), None);
    }

    #[test]
    fn test_rebase() {
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [1.0 / 7.0; COLS];
        tree.expand(0, &policy);

        let child_idx = tree.nodes[0].children[0].1;
        tree.expand(child_idx, &policy);
        tree.backpropagate(child_idx, 0.5);

        let child_visits = tree.nodes[child_idx].visit_count;
        let child_value = tree.nodes[child_idx].total_value;
        let num_grandchildren = tree.nodes[child_idx].children.len();

        assert!(tree.rebase(child_idx));
        assert_eq!(tree.nodes[0].parent, None);
        assert_eq!(tree.nodes[0].visit_count, child_visits);
        assert!((tree.nodes[0].total_value - child_value).abs() < 1e-6);
        assert!(tree.nodes[0].is_expanded);
        assert_eq!(tree.nodes[0].children.len(), num_grandchildren);
        assert_eq!(tree.nodes.len(), 1 + num_grandchildren);
    }
}
