"""Monte Carlo Tree Search for AlphaZero with Exact MCTS-Solver, FPU, and Tree Reuse."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from .game import COLS, ConnectFour

if TYPE_CHECKING:
    from .model import AlphaZeroNet

# Exact MCTS-Solver states from `node.game.current_player`'s perspective:
SOLVER_ONGOING = 0
SOLVER_WIN = 1
SOLVER_LOSS = -1
SOLVER_DRAW = 2

DEFAULT_C_FPU = 0.20
DEFAULT_DIRICHLET_ALPHA = 0.35


class MCTSNode:
    """A node in the MCTS search tree."""

    __slots__ = (
        "game",
        "parent",
        "action",
        "children",
        "visit_count",
        "total_value",
        "prior",
        "is_expanded",
        "solver_state",
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
        self.prior = float(prior)
        self.children: dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.is_expanded: bool = False

        res = game.get_result()
        if res is None:
            self.solver_state: int = SOLVER_ONGOING
        elif res < 0.0:
            self.solver_state = SOLVER_LOSS
        elif res > 0.0:
            self.solver_state = SOLVER_WIN
        else:
            self.solver_state = SOLVER_DRAW

    @property
    def q_value(self) -> float:
        """Mean action value from `self.game.current_player`'s perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def _unvisited_fpu_value(
        self,
        c_fpu: float = DEFAULT_C_FPU,
        is_noisy_root: bool = False,
    ) -> float:
        """Compute parent-relative First Play Urgency utility for unvisited children.

        Returns utility in `[-1, 1]` from `self.game.current_player` (parent) perspective.
        """
        if is_noisy_root or self.visit_count == 0:
            return 0.0
        parent_q = self.q_value
        visited_policy = sum(
            c.prior for c in self.children.values() if c.visit_count > 0
        )
        visited_policy = max(0.0, float(visited_policy))
        return max(-1.0, min(1.0, parent_q - c_fpu * math.sqrt(visited_policy)))

    def ucb_score(
        self,
        c_puct: float = 1.5,
        c_fpu: float = DEFAULT_C_FPU,
        is_noisy_root: bool = False,
        unvisited_neg_q: float | None = None,
    ) -> float:
        """Upper Confidence Bound score using PUCT + parent-relative FPU reduction.

        Each child stores `total_value` / `q_value` from its own `current_player`'s
        perspective (the opponent of the selecting parent), so the parent negates
        child Q-value (`neg_q = -self.q_value`) to maximize its own utility.
        """
        parent_visits = self.parent.visit_count if self.parent is not None else 1
        parent_visits = max(1, parent_visits)
        exploration = (
            c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        )

        if self.solver_state == SOLVER_LOSS:
            neg_q = 1.0
        elif self.solver_state == SOLVER_WIN:
            neg_q = -1.0
        elif self.solver_state == SOLVER_DRAW:
            neg_q = 0.0
        elif self.visit_count > 0:
            neg_q = -self.q_value
        elif unvisited_neg_q is not None:
            neg_q = unvisited_neg_q
        elif (
            self.parent is not None
            and not is_noisy_root
            and self.parent.visit_count > 0
        ):
            neg_q = self.parent._unvisited_fpu_value(c_fpu=c_fpu, is_noisy_root=False)
        else:
            neg_q = 0.0

        return neg_q + exploration

    def select_child(
        self,
        c_puct: float = 1.5,
        c_fpu: float = DEFAULT_C_FPU,
        is_noisy_root: bool = False,
    ) -> MCTSNode:
        """Select child with MCTS-Solver priority + PUCT/FPU score."""
        unvisited_neg_q = self._unvisited_fpu_value(
            c_fpu=c_fpu, is_noisy_root=is_noisy_root
        )

        # Immediately select a SOLVER_LOSS child (proven winning move for parent)
        winning_children = [
            c for c in self.children.values() if c.solver_state == SOLVER_LOSS
        ]
        if winning_children:
            if len(winning_children) == 1:
                return winning_children[0]
            return max(
                winning_children,
                key=lambda c: c.ucb_score(
                    c_puct,
                    c_fpu=c_fpu,
                    is_noisy_root=is_noisy_root,
                    unvisited_neg_q=unvisited_neg_q,
                ),
            )

        # Skip SOLVER_WIN children (proven losing moves for parent) unless all are SOLVER_WIN
        candidates = [
            c for c in self.children.values() if c.solver_state != SOLVER_WIN
        ]
        if not candidates:
            candidates = list(self.children.values())

        return max(
            candidates,
            key=lambda c: c.ucb_score(
                c_puct,
                c_fpu=c_fpu,
                is_noisy_root=is_noisy_root,
                unvisited_neg_q=unvisited_neg_q,
            ),
        )

    def _propagate_solver(self) -> None:
        """Propagate exact MCTS-Solver game-theoretic states upward."""
        curr: MCTSNode | None = self
        while curr is not None:
            if curr.solver_state != SOLVER_ONGOING:
                curr = curr.parent
                continue
            if not curr.is_expanded or not curr.children:
                break

            new_state = SOLVER_ONGOING
            all_win = True
            all_solved = True
            for child in curr.children.values():
                cs = child.solver_state
                if cs == SOLVER_LOSS:
                    new_state = SOLVER_WIN
                    break
                if cs != SOLVER_WIN:
                    all_win = False
                if cs == SOLVER_ONGOING:
                    all_solved = False

            if new_state == SOLVER_ONGOING:
                if all_win:
                    new_state = SOLVER_LOSS
                elif all_solved:
                    new_state = SOLVER_DRAW

            if new_state != SOLVER_ONGOING:
                curr.solver_state = new_state
                curr = curr.parent
            else:
                break

    def expand(self, policy: np.ndarray) -> None:
        """Expand node using policy priors from the neural network."""
        if self.is_expanded or self.solver_state != SOLVER_ONGOING:
            return

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
                    prior=float(policy[col]),
                )

        self._propagate_solver()

    def backpropagate(self, value: float) -> None:
        """Backpropagate leaf evaluation up the tree to the root.

        `value` (and `node.total_value` / `node.q_value` stored at each `node`)
        is strictly from the perspective of `node.game.current_player` — the
        player about to move at `node`. Because turns alternate between parent
        and child, `value` is negated at each step up to `node.parent`.
        """
        node: MCTSNode | None = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent


def _games_equal(g1: ConnectFour, g2: ConnectFour) -> bool:
    """Check whether two ConnectFour instances represent the exact same state."""
    return (
        g1._move_count == g2._move_count
        and g1.current_player == g2.current_player
        and np.array_equal(g1.board, g2.board)
        and g1.is_terminal() == g2.is_terminal()
    )


class MCTS:
    """Monte Carlo Tree Search with MCTS-Solver, FPU reduction, and stateful tree reuse."""

    def __init__(
        self,
        model: AlphaZeroNet,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
        dirichlet_epsilon: float = 0.25,
        c_fpu: float = DEFAULT_C_FPU,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.c_fpu = c_fpu
        self.device = device or torch.device("cpu")
        self._root: MCTSNode | None = None
        self._root_noisy: bool = False

    @property
    def root(self) -> MCTSNode | None:
        """Current cached root node (if any)."""
        return self._root

    def reset(self) -> None:
        """Clear the cached search tree (e.g., when starting a new game)."""
        self._root = None
        self._root_noisy = False

    def advance_move(self, col: int) -> None:
        """Advance the cached search tree to the subtree corresponding to `col`."""
        if self._root is not None and col in self._root.children:
            self._root = self._root.children[col]
            self._root.parent = None
            self._root_noisy = False
        else:
            self.reset()

    @torch.no_grad()
    def _evaluate(self, game: ConnectFour) -> tuple[np.ndarray, float]:
        """Use neural network to evaluate a single position.

        Returns:
            policy: Probability distribution over moves (numpy float32 array, shape (7,)).
            value: Position evaluation in [-1, 1] from `game.current_player`'s perspective.
        """
        state = torch.from_numpy(game.encode()).to(self.device)
        valid_moves = torch.from_numpy(game.get_valid_moves()).to(self.device)
        if self.device.type == "cuda":
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            with torch.autocast(device_type="cuda", dtype=dtype):
                policy, value = self.model.predict(state, valid_moves)
        else:
            policy, value = self.model.predict(state, valid_moves)
        return policy.float().cpu().numpy(), float(value.item())

    def _run_search(self, game: ConnectFour, add_noise: bool = False) -> MCTSNode:
        """Prepare/reuse root for `game` and run remaining simulations up to `num_simulations`."""
        if self._root is None or not _games_equal(self._root.game, game):
            self._root = MCTSNode(game.copy())
            self._root_noisy = False
        elif add_noise:
            child_visits = sum(c.visit_count for c in self._root.children.values())
            if (
                self._root_noisy
                and self.num_simulations > 0
                and child_visits >= self.num_simulations
            ):
                self._root = MCTSNode(game.copy())
                self._root_noisy = False
        elif self._root_noisy:
            # Never reuse noisy root priors when a clean noiseless search is requested
            self._root = MCTSNode(game.copy())
            self._root_noisy = False

        root = self._root
        if not root.is_expanded and not root.game.is_terminal():
            policy, _ = self._evaluate(root.game)
            root.expand(policy)

        if (
            add_noise
            and not self._root_noisy
            and self.dirichlet_epsilon > 0.0
            and len(root.children) >= 2
        ):
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children.values()):
                child.prior = float(
                    (1.0 - self.dirichlet_epsilon) * child.prior
                    + self.dirichlet_epsilon * noise[i]
                )
            self._root_noisy = True

        if self.num_simulations == 0 or root.game.is_terminal():
            return root

        existing_sims = sum(c.visit_count for c in root.children.values())
        sims_to_run = max(0, self.num_simulations - existing_sims)

        for _ in range(sims_to_run):
            node = root
            is_root = True

            while node.is_expanded and node.children:
                if not is_root and node.solver_state != SOLVER_ONGOING:
                    break
                node = node.select_child(
                    self.c_puct,
                    c_fpu=self.c_fpu,
                    is_noisy_root=(is_root and self._root_noisy),
                )
                is_root = False

            if node.solver_state != SOLVER_ONGOING:
                if node.solver_state == SOLVER_WIN:
                    proven_val = 1.0
                elif node.solver_state == SOLVER_LOSS:
                    proven_val = -1.0
                else:
                    proven_val = 0.0
                node.backpropagate(proven_val)
                continue

            result = node.game.get_result()
            if result is not None:
                node.solver_state = SOLVER_LOSS if result < 0.0 else SOLVER_DRAW
                if node.parent is not None:
                    node.parent._propagate_solver()
                node.backpropagate(result)
                continue

            policy, value = self._evaluate(node.game)
            node.expand(policy)
            if node.solver_state == SOLVER_WIN:
                bval = 1.0
            elif node.solver_state == SOLVER_LOSS:
                bval = -1.0
            elif node.solver_state == SOLVER_DRAW:
                bval = 0.0
            else:
                bval = value
            node.backpropagate(bval)

        return root

    def _extract_visit_counts(self, root: MCTSNode) -> np.ndarray:
        """Extract root visit counts, respecting exact solver win/loss proofs."""
        visit_counts = np.zeros(COLS, dtype=np.float32)
        if self.num_simulations == 0:
            for action, child in root.children.items():
                visit_counts[action] = child.prior
            return visit_counts

        if root.solver_state == SOLVER_WIN:
            winning_actions = [
                a
                for a, c in root.children.items()
                if c.solver_state == SOLVER_LOSS
            ]
            if winning_actions:
                win_visits = sum(
                    root.children[a].visit_count for a in winning_actions
                )
                if win_visits > 0:
                    for a in winning_actions:
                        visit_counts[a] = float(root.children[a].visit_count)
                else:
                    for a in winning_actions:
                        visit_counts[a] = 1.0
                return visit_counts

        for action, child in root.children.items():
            visit_counts[action] = float(child.visit_count)

        # Zero out proven losing moves (SOLVER_WIN children) when a non-losing alternative exists
        if root.solver_state != SOLVER_LOSS:
            non_losing = visit_counts.copy()
            has_non_losing_child = False
            for action, child in root.children.items():
                if child.solver_state == SOLVER_WIN:
                    non_losing[action] = 0.0
                else:
                    has_non_losing_child = True
            if has_non_losing_child:
                if non_losing.sum() > 0:
                    visit_counts = non_losing
                else:
                    visit_counts = np.zeros(COLS, dtype=np.float32)
                    for action, child in root.children.items():
                        if child.solver_state != SOLVER_WIN:
                            visit_counts[action] = 1.0

        return visit_counts

    def _compute_move_values(self, root: MCTSNode) -> dict[int, float]:
        """Compute column win probabilities (0-100%) from an expanded/searched root."""
        values: dict[int, float] = {}
        if self.num_simulations == 0:
            for action, child in root.children.items():
                res = child.game.get_result()
                if res == -1.0:
                    values[action] = 100.0
                elif res == 0.0:
                    values[action] = 50.0
                elif res == 1.0:
                    values[action] = 0.0
                else:
                    _, child_val = self._evaluate(child.game)
                    values[action] = (-child_val + 1.0) / 2.0 * 100.0
            return values

        for action, child in root.children.items():
            if child.solver_state == SOLVER_LOSS:
                values[action] = 100.0
            elif child.solver_state == SOLVER_WIN:
                values[action] = 0.0
            elif child.solver_state == SOLVER_DRAW:
                values[action] = 50.0
            elif child.visit_count > 0:
                values[action] = (-child.q_value + 1.0) / 2.0 * 100.0
            else:
                res = child.game.get_result()
                if res == -1.0:
                    values[action] = 100.0
                elif res == 0.0:
                    values[action] = 50.0
                elif res == 1.0:
                    values[action] = 0.0
                elif root.visit_count > 0 and root.solver_state == SOLVER_ONGOING:
                    neg_q = root._unvisited_fpu_value(
                        c_fpu=self.c_fpu, is_noisy_root=False
                    )
                    values[action] = (neg_q + 1.0) / 2.0 * 100.0
                else:
                    _, child_val = self._evaluate(child.game)
                    values[action] = (-child_val + 1.0) / 2.0 * 100.0

        return values

    def search(self, game: ConnectFour, add_noise: bool = True) -> np.ndarray:
        """Run MCTS from the given game state and return visit counts over columns."""
        root = self._run_search(game, add_noise=add_noise)
        return self._extract_visit_counts(root)

    def search_with_values(
        self, game: ConnectFour, add_noise: bool = False
    ) -> tuple[np.ndarray, dict[int, float]]:
        """Run/reuse a single MCTS search and return both `(visit_counts, win_probs)`."""
        root = self._run_search(game, add_noise=add_noise)
        visit_counts = self._extract_visit_counts(root)
        move_values = self._compute_move_values(root)
        return visit_counts, move_values

    def get_action_probs(
        self,
        game: ConnectFour,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get temperature-scaled action probabilities and raw visit counts."""
        visit_counts = self.search(game, add_noise=add_noise)

        if temperature == 0:
            action_probs = np.zeros(COLS, dtype=np.float32)
            if visit_counts.sum() > 0:
                best_action = int(np.argmax(visit_counts))
            else:
                valid = game.get_valid_moves()
                best_action = int(np.argmax(valid))
            action_probs[best_action] = 1.0
        else:
            if visit_counts.sum() == 0:
                valid = game.get_valid_moves().astype(np.float32)
                action_probs = valid / valid.sum()
            elif temperature == 1.0:
                action_probs = (visit_counts / visit_counts.sum()).astype(np.float32)
            else:
                mask = visit_counts > 0
                log_counts = np.log(visit_counts[mask].astype(np.float64)) / temperature
                log_counts -= np.max(log_counts)
                scaled = np.exp(log_counts)
                scaled /= scaled.sum()
                action_probs = np.zeros(COLS, dtype=np.float32)
                action_probs[mask] = scaled.astype(np.float32)
                action_probs /= action_probs.sum()

        return action_probs, visit_counts

    def get_move_values(self, game: ConnectFour) -> dict[int, float]:
        """Get estimated win probabilities (0-100%) for each valid column."""
        _, move_values = self.search_with_values(game, add_noise=False)
        return move_values
