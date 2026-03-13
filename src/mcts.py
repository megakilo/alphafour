import math
import numpy as np

class MCTS:
    def __init__(self, game, model, num_simulations=50, c_puct=1.0, dirichlet_alpha=1.0, dirichlet_epsilon=0.25):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}   # stores #times board s was visited
        self.Ps = {}   # stores initial policy (returned by neural net)

        self.Es = {}   # stores game.get_game_ended ended for board s
        self.Vs = {}   # stores game.get_valid_moves for board s

    def get_action_prob(self, canonicalBoard, temp=1):
        """
        This function performs num_simulations simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.num_simulations):
            self.search(canonicalBoard)
            # After first simulation expands root, add Dirichlet noise to root priors
            if i == 0 and self.dirichlet_epsilon > 0:
                s = self.game.string_representation(canonicalBoard)
                if s in self.Ps:
                    noise = np.random.dirichlet([self.dirichlet_alpha] * self.game.action_size)
                    self.Ps[s] = (1 - self.dirichlet_epsilon) * self.Ps[s] + self.dirichlet_epsilon * noise
                    # Re-mask and renormalize
                    valids = self.Vs.get(s, self.game.get_valid_moves(canonicalBoard))
                    self.Ps[s] = self.Ps[s] * valids
                    sum_ps = np.sum(self.Ps[s])
                    if sum_ps > 0:
                        self.Ps[s] /= sum_ps

        s = self.game.string_representation(canonicalBoard)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.action_size)]
        
        # Safety: mask with valid moves from the actual board
        valids = self.game.get_valid_moves(canonicalBoard)
        counts = [c if valids[a] else 0 for a, c in enumerate(counts)]

        if temp == 0:
            counts = np.array(counts)
            bestAs = np.argwhere(counts == np.max(counts)).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if counts_sum == 0:
            # If all counts are zero (shouldn't happen), return uniform over valids
            print("Warning: MCTS counts_sum is 0. Returning uniform over valids.")
            probs = (valids / np.sum(valids)).tolist()
        else:
            probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, action=None):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound (PUCT).
        """
        s = self.game.string_representation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonicalBoard, 1, action)

        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.model.predict(canonicalBoard)
            valids = self.game.get_valid_moves(canonicalBoard)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best_u = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.action_size):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)  # Q = 0 ?

                if u > cur_best_u:
                    cur_best_u = u
                    best_act = a

        if best_act == -1:
            # Should not happen given terminal check above
            return 0

        a = best_act
        next_s = self.game.get_next_state(canonicalBoard, 1, a)
        next_s = self.game.get_canonical_form(next_s, -1)

        v = self.search(next_s, a)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
