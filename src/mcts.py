import math
import numpy as np


class MCTS:
    def __init__(
        self,
        game,
        model,
        num_simulations=50,
        c_puct=1.0,
        dirichlet_alpha=1.4,
        dirichlet_epsilon=0.25,
        sim_batch_size=1,
        fpu_reduction=0.25,
    ):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.sim_batch_size = max(1, sim_batch_size)
        self.fpu_reduction = fpu_reduction

        self.Qsa = {}  # stores Q values for s,a
        self.Nsa = {}  # stores edge visit counts
        self.Ns = {}   # stores state visit counts
        self.Ps = {}   # stores policy priors

        self.Es = {}   # stores terminal outcomes for states
        self.Vs = {}   # stores valid moves for states

    def get_action_prob(self, canonicalBoard, temp=1, last_action=None):
        """
        Runs MCTS simulations from canonicalBoard and returns a policy vector
        proportional to visit counts.
        """
        if self.num_simulations > 0:
            self.search(canonicalBoard, last_action=last_action)
            self._apply_root_dirichlet_noise(canonicalBoard)

            sims_completed = 1
            while sims_completed < self.num_simulations:
                batch_size = min(self.sim_batch_size, self.num_simulations - sims_completed)
                if batch_size == 1 or not hasattr(self.model, "predict_batch"):
                    self.search(canonicalBoard, last_action=last_action)
                else:
                    self._run_batched_simulations(canonicalBoard, batch_size, last_action=last_action)
                sims_completed += batch_size

        s = self.game.string_representation(canonicalBoard)
        counts = np.array([self.Nsa.get((s, a), 0) for a in range(self.game.action_size)], dtype=np.float32)

        valids = self.game.get_valid_moves(canonicalBoard)
        counts = counts * valids

        if temp == 0:
            if counts.sum() == 0:
                probs = np.zeros_like(counts, dtype=np.float64)
                probs[valids] = 1.0 / np.sum(valids)
                return probs.tolist()

            best_actions = np.flatnonzero(counts == counts.max())
            best_action = np.random.choice(best_actions)
            probs = np.zeros_like(counts, dtype=np.float64)
            probs[best_action] = 1.0
            return probs.tolist()

        counts = counts ** (1.0 / temp)
        counts_sum = float(counts.sum())
        if counts_sum == 0:
            probs = np.zeros_like(counts, dtype=np.float64)
            probs[valids] = 1.0 / np.sum(valids)
            return probs.tolist()

        probs = counts.astype(np.float64) / counts_sum
        probs /= probs.sum()
        return probs.tolist()

    def search(self, canonicalBoard, last_action=None):
        """
        Performs one standard MCTS simulation.
        """
        s = self.game.string_representation(canonicalBoard)
        terminal_key = self._terminal_cache_key(s, last_action)

        if terminal_key not in self.Es:
            self.Es[terminal_key] = self.game.get_game_ended(canonicalBoard, 1, last_action)

        if self.Es[terminal_key] != 0:
            return -self.Es[terminal_key]

        if s not in self.Ps:
            _, value = self._expand_leaf(s, canonicalBoard)
            return -value

        a = self._select_action(s)
        next_s = self.game.get_next_state(canonicalBoard, 1, a)
        next_s = self.game.get_canonical_form(next_s, -1)

        value = self.search(next_s, last_action=a)
        self._update_edge(s, a, value)
        return -value

    def _apply_root_dirichlet_noise(self, canonicalBoard):
        if self.dirichlet_epsilon <= 0:
            return

        s = self.game.string_representation(canonicalBoard)
        if s not in self.Ps:
            return

        noise = np.random.dirichlet([self.dirichlet_alpha] * self.game.action_size)
        self.Ps[s] = (1 - self.dirichlet_epsilon) * self.Ps[s] + self.dirichlet_epsilon * noise

        valids = self.Vs.get(s, self.game.get_valid_moves(canonicalBoard))
        self.Ps[s] = self.Ps[s] * valids
        sum_ps = np.sum(self.Ps[s])
        if sum_ps > 0:
            self.Ps[s] /= sum_ps

    def _run_batched_simulations(self, canonicalBoard, batch_size, last_action=None):
        pending = []
        unique_leaves = {}
        leaf_boards = []

        for _ in range(batch_size):
            selection = self._select_leaf(
                canonicalBoard,
                use_virtual_visits=True,
                last_action=last_action,
            )
            pending.append(selection)

            if selection["terminal_value"] is None and selection["state"] not in unique_leaves:
                unique_leaves[selection["state"]] = len(leaf_boards)
                leaf_boards.append(selection["board"])

        predictions = {}
        if leaf_boards:
            policies, values = self.model.predict_batch(leaf_boards)
            for state, index in unique_leaves.items():
                predictions[state] = (policies[index], float(values[index]))

        for selection in pending:
            self._revert_virtual_visits(selection["virtual_path"])

            if selection["terminal_value"] is not None:
                self._backup(selection["path"], -selection["terminal_value"])
                continue

            policy, value = predictions[selection["state"]]
            if selection["state"] not in self.Ps:
                self._expand_leaf(selection["state"], selection["board"], policy, value)
            self._backup(selection["path"], -value)

    def _select_leaf(self, canonicalBoard, use_virtual_visits=False, last_action=None):
        path = []
        virtual_path = []
        board = canonicalBoard
        incoming_action = last_action

        while True:
            s = self.game.string_representation(board)
            terminal_key = self._terminal_cache_key(s, incoming_action)

            if terminal_key not in self.Es:
                self.Es[terminal_key] = self.game.get_game_ended(board, 1, incoming_action)

            if self.Es[terminal_key] != 0:
                return {
                    "path": path,
                    "virtual_path": virtual_path,
                    "board": board,
                    "state": s,
                    "terminal_value": self.Es[terminal_key],
                }

            if s not in self.Ps:
                return {
                    "path": path,
                    "virtual_path": virtual_path,
                    "board": board,
                    "state": s,
                    "terminal_value": None,
                }

            a = self._select_action(s)
            path.append((s, a))

            if use_virtual_visits:
                self._apply_virtual_visit(s, a)
                virtual_path.append((s, a))

            board = self.game.get_next_state(board, 1, a)
            board = self.game.get_canonical_form(board, -1)
            incoming_action = a

    def _expand_leaf(self, state, board, policy=None, value=None):
        if policy is None or value is None:
            policy, value = self.model.predict(board)

        valids = self.game.get_valid_moves(board)
        masked_policy = np.asarray(policy, dtype=np.float32) * valids
        sum_policy = float(masked_policy.sum())
        if sum_policy > 0:
            masked_policy /= sum_policy
        else:
            masked_policy = valids.astype(np.float32)
            masked_policy /= masked_policy.sum()

        self.Ps[state] = masked_policy
        self.Vs[state] = valids
        self.Ns[state] = 0
        return masked_policy, float(value)

    def _select_action(self, state):
        valids = self.Vs[state]
        sqrt_ns = math.sqrt(max(1, self.Ns.get(state, 0)))
        fpu_value = self._get_fpu_value(state)

        best_action = -1
        best_score = -float("inf")

        for action in range(self.game.action_size):
            if not valids[action]:
                continue

            q_value = self.Qsa.get((state, action), fpu_value)
            visits = self.Nsa.get((state, action), 0)
            u_value = q_value + self.c_puct * self.Ps[state][action] * sqrt_ns / (1 + visits)

            if u_value > best_score:
                best_score = u_value
                best_action = action

        if best_action == -1:
            valid_actions = np.flatnonzero(valids)
            return int(np.random.choice(valid_actions))

        return best_action

    def _get_fpu_value(self, state):
        weighted_q = 0.0
        total_visits = 0
        for action in range(self.game.action_size):
            edge = (state, action)
            if edge in self.Qsa:
                visits = self.Nsa[edge]
                weighted_q += visits * self.Qsa[edge]
                total_visits += visits

        if total_visits == 0:
            return 0.0

        return weighted_q / total_visits - self.fpu_reduction

    def _update_edge(self, state, action, value):
        edge = (state, action)
        visits = self.Nsa.get(edge, 0)
        if edge in self.Qsa:
            self.Qsa[edge] = (visits * self.Qsa[edge] + value) / (visits + 1)
        else:
            self.Qsa[edge] = value
        self.Nsa[edge] = visits + 1
        self.Ns[state] = self.Ns.get(state, 0) + 1

    def _backup(self, path, value):
        for state, action in reversed(path):
            self._update_edge(state, action, value)
            value = -value

    def _apply_virtual_visit(self, state, action):
        edge = (state, action)
        self.Ns[state] = self.Ns.get(state, 0) + 1
        self.Nsa[edge] = self.Nsa.get(edge, 0) + 1

    def _revert_virtual_visits(self, virtual_path):
        for state, action in reversed(virtual_path):
            edge = (state, action)
            self.Ns[state] = max(0, self.Ns.get(state, 0) - 1)
            visits = self.Nsa.get(edge, 0) - 1
            if visits <= 0 and edge not in self.Qsa:
                self.Nsa.pop(edge, None)
            else:
                self.Nsa[edge] = max(0, visits)

    def _terminal_cache_key(self, state, last_action):
        return (state, last_action)
