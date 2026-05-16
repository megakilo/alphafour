/// Arena-based MCTS tree for high-performance tree search.
///
/// Nodes are stored in a contiguous Vec (arena allocation) and referenced
/// by index, avoiding heap fragmentation and borrow checker issues.
///
/// Supports virtual losses for parallel leaf collection: multiple leaves
/// can be selected simultaneously without duplicating paths.

use crate::game::{ConnectFour, COLS};
use rand_distr::{Dirichlet, Distribution};

pub struct MCTSNode {
    pub game: ConnectFour,
    pub children: Vec<(u8, usize)>, // (action, child_index)
    pub parent: Option<usize>,
    pub visit_count: u32,
    pub total_value: f64,
    pub prior: f32,
    pub is_expanded: bool,
    /// Virtual losses currently applied (for parallel leaf selection).
    pub virtual_losses: u32,
}

pub struct MCTSTree {
    pub nodes: Vec<MCTSNode>,
}

impl MCTSTree {
    /// Create a new tree with the given game state as root.
    pub fn new(game: ConnectFour) -> Self {
        let root = MCTSNode {
            game,
            children: Vec::new(),
            parent: None,
            visit_count: 0,
            total_value: 0.0,
            prior: 0.0,
            is_expanded: false,
            virtual_losses: 0,
        };
        MCTSTree { nodes: vec![root] }
    }

    #[inline]
    pub fn root(&self) -> usize {
        0
    }

    /// Select a leaf node by traversing from root using UCB.
    /// Virtual losses are respected during selection.
    pub fn select_leaf(&self, root: usize, c_puct: f32) -> usize {
        let mut idx = root;
        loop {
            let node = &self.nodes[idx];
            if !node.is_expanded || node.children.is_empty() {
                return idx;
            }
            idx = self.best_child(idx, c_puct);
        }
    }

    /// Select child with highest UCB score, accounting for virtual losses.
    fn best_child(&self, parent_idx: usize, c_puct: f32) -> usize {
        let parent = &self.nodes[parent_idx];
        let effective_parent_visits = parent.visit_count + parent.virtual_losses;
        let sqrt_parent = (effective_parent_visits.max(1) as f64).sqrt();

        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;

        for &(_, child_idx) in &parent.children {
            let child = &self.nodes[child_idx];
            let effective_visits = child.visit_count + child.virtual_losses;
            let q = if effective_visits == 0 {
                0.0
            } else {
                // Virtual losses add negative value (pessimistic), discouraging
                // re-selection of in-flight leaves.
                (child.total_value - child.virtual_losses as f64)
                    / effective_visits as f64
            };
            let exploration =
                c_puct as f64 * child.prior as f64 * sqrt_parent / (1.0 + effective_visits as f64);
            // Negate Q because child stores value from its own perspective
            let score = -q + exploration;
            if score > best_score {
                best_score = score;
                best_idx = child_idx;
            }
        }
        best_idx
    }

    /// Apply virtual loss along the path from node to root.
    /// This makes the path look worse, discouraging parallel selection
    /// of the same leaf.
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

    /// Select up to `k` distinct leaves using virtual losses.
    /// Returns a list of leaf node indices.
    /// Terminal leaves are NOT included; they are backpropagated immediately.
    pub fn select_multiple_leaves(
        &mut self,
        root: usize,
        c_puct: f32,
        k: usize,
    ) -> (Vec<usize>, u32) {
        let mut leaves = Vec::with_capacity(k);
        let mut terminal_count = 0u32;

        for _ in 0..k {
            let leaf = self.select_leaf(root, c_puct);

            let result = self.nodes[leaf].game.get_result();
            if let Some(val) = result {
                // Terminal node: backpropagate immediately, no virtual loss needed
                self.backpropagate(leaf, val);
                terminal_count += 1;
            } else {
                // Apply virtual loss to discourage re-selection
                self.apply_virtual_loss(leaf);
                leaves.push(leaf);
            }
        }

        (leaves, terminal_count)
    }

    /// Expand a node: create child nodes for each valid move.
    pub fn expand(&mut self, node_idx: usize, policy: &[f32]) {
        let game = self.nodes[node_idx].game.clone();
        let valid = game.get_valid_moves();

        let mut children = Vec::new();
        for col in 0..COLS {
            if valid[col] {
                let mut child_game = game.clone();
                child_game.make_move(col as u8);
                let child_idx = self.nodes.len();
                self.nodes.push(MCTSNode {
                    game: child_game,
                    children: Vec::new(),
                    parent: Some(node_idx),
                    visit_count: 0,
                    total_value: 0.0,
                    prior: policy[col],
                    is_expanded: false,
                    virtual_losses: 0,
                });
                children.push((col as u8, child_idx));
            }
        }
        self.nodes[node_idx].children = children;
        self.nodes[node_idx].is_expanded = true;
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

    /// Add Dirichlet noise to root's children priors.
    pub fn add_dirichlet_noise(&mut self, node_idx: usize, alpha: f64, epsilon: f64) {
        let num_children = self.nodes[node_idx].children.len();
        if num_children < 2 {
            return;
        }

        let dir = Dirichlet::new(&vec![alpha; num_children]).unwrap();
        let noise: Vec<f64> = dir.sample(&mut rand::thread_rng());

        let children: Vec<(u8, usize)> = self.nodes[node_idx].children.clone();
        for (i, &(_, child_idx)) in children.iter().enumerate() {
            let prior = self.nodes[child_idx].prior as f64;
            self.nodes[child_idx].prior = ((1.0 - epsilon) * prior + epsilon * noise[i]) as f32;
        }
    }

    /// Get visit counts for root's children as a [f32; COLS] array.
    pub fn get_visit_counts(&self, node_idx: usize) -> [f32; COLS] {
        let mut counts = [0.0f32; COLS];
        for &(action, child_idx) in &self.nodes[node_idx].children {
            counts[action as usize] = self.nodes[child_idx].visit_count as f32;
        }
        counts
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
    }

    #[test]
    fn test_expand_root() {
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [1.0 / 7.0; COLS];
        tree.expand(0, &policy);

        assert!(tree.nodes[0].is_expanded);
        assert_eq!(tree.nodes[0].children.len(), 7); // All 7 columns valid
        assert_eq!(tree.nodes.len(), 8); // root + 7 children
    }

    #[test]
    fn test_select_leaf_unexpanded() {
        let game = ConnectFour::new();
        let tree = MCTSTree::new(game);
        assert_eq!(tree.select_leaf(0, 1.5), 0); // Root is leaf
    }

    #[test]
    fn test_select_leaf_after_expand() {
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [1.0 / 7.0; COLS];
        tree.expand(0, &policy);

        // Should select one of the children (all equal prior, 0 visits)
        let leaf = tree.select_leaf(0, 1.5);
        assert!(leaf >= 1 && leaf <= 7);
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
        // Parent gets negated value
        assert_eq!(tree.nodes[0].visit_count, 1);
        assert!((tree.nodes[0].total_value - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_visit_counts() {
        let game = ConnectFour::new();
        let mut tree = MCTSTree::new(game);
        let policy = [1.0 / 7.0; COLS];
        tree.expand(0, &policy);

        // Backprop through first child a few times
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

        // Get priors before noise
        let before: Vec<f32> = tree.nodes[0]
            .children
            .iter()
            .map(|&(_, idx)| tree.nodes[idx].prior)
            .collect();

        tree.add_dirichlet_noise(0, 1.0, 0.25);

        // Priors should have changed
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

        // Apply virtual loss
        tree.apply_virtual_loss(child_idx);
        assert_eq!(tree.nodes[child_idx].virtual_losses, 1);
        assert_eq!(tree.nodes[0].virtual_losses, 1); // parent too

        // Apply again
        tree.apply_virtual_loss(child_idx);
        assert_eq!(tree.nodes[child_idx].virtual_losses, 2);
        assert_eq!(tree.nodes[0].virtual_losses, 2);

        // Revert once
        tree.revert_virtual_loss(child_idx);
        assert_eq!(tree.nodes[child_idx].virtual_losses, 1);
        assert_eq!(tree.nodes[0].virtual_losses, 1);

        // Revert again
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

        // Expand all children so we can go deeper
        for i in 0..7 {
            let child_idx = tree.nodes[0].children[i].1;
            tree.expand(child_idx, &policy);
        }

        // Select 4 leaves — virtual losses should push to different branches
        let (leaves, _terminal) = tree.select_multiple_leaves(0, 1.5, 4);
        assert_eq!(leaves.len(), 4);

        // At least some leaves should be different (virtual losses diversify)
        // With equal priors and no visits, all children look the same initially,
        // but virtual losses should push subsequent selections elsewhere.
        let unique: std::collections::HashSet<usize> = leaves.iter().cloned().collect();
        assert!(unique.len() >= 2, "Expected diverse leaf selection, got {:?}", leaves);

        // Clean up virtual losses
        for &leaf in &leaves {
            tree.revert_virtual_loss(leaf);
        }

        // Virtual losses should be back to 0
        for node in &tree.nodes {
            assert_eq!(node.virtual_losses, 0);
        }
    }
}
