"""Monte Carlo Tree Search for AlphaZero."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from .game import ConnectFour, COLS

if TYPE_CHECKING:
    from .model import AlphaZeroNet


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = (
        "game",
        "parent",
        "action",
        "children",
        "visit_count",
        "total_value",
        "prior",
        "is_expanded",
    )

    def __init__(
        self,
        game: ConnectFour,
        parent: MCTSNode | None = None,
        action: int | None = None,
        prior: float = 0.0,
    ) -> None:
        self.game = game
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Mean action value."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float = 1.5) -> float:
        """Upper Confidence Bound score using PUCT formula.

        Q-value is negated because each child stores value from its own
        current player's perspective (the opponent of the selecting parent).
        The parent wants to maximize its own value = minimize child's value.
        """
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return -self.q_value + exploration

    def select_child(self, c_puct: float = 1.5) -> MCTSNode:
        """Select child with highest UCB score."""
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))

    def expand(self, policy: np.ndarray) -> None:
        """Expand node using policy from neural network."""
        self.is_expanded = True
        valid_moves = self.game.get_valid_moves()

        for col in range(COLS):
            if valid_moves[col]:
                child_game = self.game.copy()
                child_game.make_move(col)
                self.children[col] = MCTSNode(
                    game=child_game,
                    parent=self,
                    action=col,
                    prior=policy[col],
                )

    def backpropagate(self, value: float) -> None:
        """Backpropagate value up the tree.

        Value is from the perspective of the player who just moved
        at this node's PARENT. We negate at each level because
        alternating players.
        """
        node: MCTSNode | None = self
        while node is not None:
            node.visit_count += 1
            # value is from perspective of the player who is about to move
            # at this node. Negate because parent's player is the opponent.
            node.total_value += value
            value = -value
            node = node.parent


class MCTS:
    """Monte Carlo Tree Search using AlphaZero approach."""

    def __init__(
        self,
        model: AlphaZeroNet,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 1.0,
        dirichlet_epsilon: float = 0.25,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def _evaluate(self, game: ConnectFour) -> tuple[np.ndarray, float]:
        """Use neural network to evaluate a position.

        Returns:
            policy: Probability distribution over moves (numpy array, length 7).
            value: Position evaluation from current player's perspective.
        """
        state = torch.from_numpy(game.encode()).to(self.device)
        valid_moves = torch.from_numpy(game.get_valid_moves()).to(self.device)
        policy, value = self.model.predict(state, valid_moves)
        return policy.cpu().numpy(), value.item()

    def search(
        self, game: ConnectFour, add_noise: bool = True
    ) -> np.ndarray:
        """Run MCTS from the given game state.

        Args:
            game: Current game state.
            add_noise: Whether to add Dirichlet noise at root (for training).

        Returns:
            Visit count distribution over actions (normalized).
        """
        root = MCTSNode(game)

        # Evaluate and expand root
        policy, _ = self._evaluate(game)
        root.expand(policy)

        # Add Dirichlet noise at root for exploration
        if add_noise and root.children:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(root.children)
            )
            for i, child in enumerate(root.children.values()):
                child.prior = (
                    (1 - self.dirichlet_epsilon) * child.prior
                    + self.dirichlet_epsilon * noise[i]
                )

        # Run simulations
        for _ in range(self.num_simulations):
            node = root

            # Selection: traverse tree using UCB
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)

            # Check if terminal
            result = node.game.get_result()
            if result is not None:
                # Terminal node: backpropagate actual result
                node.backpropagate(result)
                continue

            # Expansion: evaluate with neural network and expand
            policy, value = self._evaluate(node.game)
            node.expand(policy)

            # Backpropagation
            node.backpropagate(value)

        # Construct action probabilities from visit counts
        visit_counts = np.zeros(COLS, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        return visit_counts

    def get_action_probs(
        self,
        game: ConnectFour,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get action probabilities from MCTS.

        Args:
            game: Current game state.
            temperature: Temperature for action selection.
                1.0 = proportional to visit counts.
                0 = greedy (pick best move).
            add_noise: Whether to add exploration noise.

        Returns:
            action_probs: Probability distribution over actions.
            visit_counts: Raw visit counts.
        """
        visit_counts = self.search(game, add_noise=add_noise)

        if temperature == 0:
            # Greedy: pick the action with most visits
            action_probs = np.zeros(COLS, dtype=np.float32)
            best_action = np.argmax(visit_counts)
            action_probs[best_action] = 1.0
        else:
            # Temperature-scaled
            if visit_counts.sum() == 0:
                # Fallback: uniform over valid moves
                valid = game.get_valid_moves().astype(np.float32)
                action_probs = valid / valid.sum()
            else:
                counts = visit_counts ** (1.0 / temperature)
                action_probs = counts / counts.sum()

        return action_probs, visit_counts

    def get_move_values(self, game: ConnectFour) -> dict[int, float]:
        """Get value estimates for each valid move (for display hints).

        Returns dict mapping column -> estimated win probability (0-100%).
        """
        # Run a full search and build the tree
        root = MCTSNode(game)
        policy, _ = self._evaluate(game)
        root.expand(policy)

        for _ in range(self.num_simulations):
            node = root
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
            result = node.game.get_result()
            if result is not None:
                node.backpropagate(result)
                continue
            p, v = self._evaluate(node.game)
            node.expand(p)
            node.backpropagate(v)

        values = {}
        for action, child in root.children.items():
            # Child Q-value is from child's perspective (opponent).
            # Negate to get value from current player's perspective.
            win_prob = (-child.q_value + 1) / 2 * 100  # Convert [-1,1] to [0,100]%
            values[action] = win_prob

        return values
