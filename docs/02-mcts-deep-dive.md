# 02. Core Search Algorithm: MCTS, PUCT, FPU & Exact MCTS-Solver

> **Code references in this chapter**:
> - Pure-Python reference implementation (clarity + interactive play): [`src/mcts.py`](../src/mcts.py)
> - Cython arena-allocated implementation (speed + parallel batching): [`src/self_play.py`](../src/self_play.py)

---

## 1. Why Not Plain Minimax (Alpha-Beta) Search?

If you have studied classic Chess engines, you might wonder: *"Once we have a neural network $f_\theta(s)$ that evaluates board positions, why don't we just use standard depth-$d$ Alpha-Beta Minimax search?"*

Alpha-Beta search has two major limitations when paired with a deep neural network:
1. **Uniform depth wastes GPU evaluations**: Alpha-Beta searches *every* reasonable branch to fixed depth $d$ before going to depth $d+1$, evaluating thousands of mediocre sidelines on an expensive neural network. Humans—and MCTS—search **selectively and asymmetrically**: 20 plies deep down a forcing tactical line, while spending only 1 or 2 visits on a passive move.
2. **Brittle single-leaf minimax**: In pure minimax, a single mis-evaluated leaf node (a "hallucination" where an untrained network outputs `+0.95` on a losing position) immediately hijacks the entire minimax value up to the root. **MCTS averages leaf evaluations** across all visits to a subtree ($Q = W / N$), making it naturally resilient to noisy neural network estimates early in training!

At the same time, pure averaging has a weakness of its own in tactical games like Connect Four: when a line *is* a 100%-certain forced win or loss, averaging shouldn't dilute it! As we will see in [Section 6](#6-exact-mcts-solver-game-theoretic-proofs-inside-mcts), `alphafour` gets the best of both worlds by embedding an **Exact MCTS-Solver** directly inside MCTS.

---

## 2. Anatomy of an MCTS Node

Every node in the search tree represents a specific board state `game` and stores statistics accumulated across all simulations that passed through it.

Compare `MCTSNode` in [`src/mcts.py`](../src/mcts.py#L26-L72) and [`src/self_play.py`](../src/self_play.py):

| Field | Math Symbol | Perspective | Meaning |
| :--- | :---: | :--- | :--- |
| `game` | $s$ | `s.current_player` | Exact `ConnectFour` board state at this node |
| `parent` | $\text{parent}(s)$ | Opponent | Pointer (Python) or C-struct arena index `cython.int` (`-1` at root) to parent node |
| `children` | $\{a \mapsto s'\}$ | Opponent | Map / contiguous sibling range (`first_child`, `num_children`) of legal column moves `0..6` |
| `prior` | $P(\text{parent}, a)$ | Parent's move prob | Neural network policy probability assigned to the move $a$ that led to this node |
| `visit_count` | $N(s)$ | — | Number of simulations that have passed through this node |
| `total_value` | $W(s)$ | **`s.current_player`** | Sum of backpropagated leaf evaluations from this node's mover's perspective |
| `q_value` | $Q(s) = \frac{W(s)}{N(s)}$ | **`s.current_player`** | Mean expected outcome in $[-1, +1]$ for the player *about to move at this node* |
| `is_expanded` | — | — | `True` once legal children have been instantiated with neural network priors |
| `solver_state` | — | **`s.current_player`** | Exact game-theoretic proof status: `Ongoing`, `Win (+1)`, `Loss (-1)`, or `Draw (0)` |
| `virtual_losses` | $N_{\text{vl}}(s)$ | — | *(Cython self-play)* In-flight parallel traversals currently evaluating a leaf under this node |

> **Watch the Perspective of `q_value`!**
> In `alphafour`, a child node stores `total_value` and `q_value` from **its own `current_player`'s perspective**—which is the **opponent** of the parent node choosing among its children!
> - If `child.q_value == +0.7`, that means moving to `child` gives the *opponent* an expected score of `+0.7`, which is **`-0.7` for the parent**.
> - Therefore, whenever a parent node evaluates a child's utility, it **negates** the child's Q-value: `neg_q = -child.q_value`.

---

## 3. The Four Phases of One MCTS Simulation

Starting from a root node $s_{\text{root}}$, `alphafour` runs `num_simulations` iterations (e.g., `600` in training, `400` in benchmark evaluation, `200+` in interactive play). Each simulation executes four phases:

```text
   1. SELECTION               2. EVALUATION             3. EXPANSION            4. BACKPROPAGATION
  (Walk down tree            (Query NN or Game         (Create children         (Walk back up to root,
   maximizing PUCT)           rules at leaf s_L)        with priors P)           flipping sign each ply!)

       (Root)                     (Root)                    (Root)                  (Root) N+=1, W+=+v
       /  |  \                    /  |  \                   /  |  \                 /  |  \
     ( ) (*) ( )                ( ) (*) ( )               ( ) (*) ( )             ( ) (*) ( ) N+=1, W+=-v
         / \                        / \                       / \                     / \
       ( ) [s_L]                  ( ) [s_L] ──► f_θ(s_L)    ( ) [s_L]               ( ) [s_L] N+=1, W+=+v
         unexpanded!                  = (p, v)                 / | \                     ▲
                                                             c0 c1 c2                 start with +v
```

### Phase 1: Selection (The PUCT Formula)

Starting at `node = root`, while `node` is already expanded, non-terminal, and not yet game-theoretically proven (`solver_state == Ongoing`), we repeatedly pick the child that maximizes the **PUCT (Predictor + Upper Confidence Bound applied to Trees)** score ([`src/mcts.py`](../src/mcts.py#L92-L130)):

$$
\text{UCB}(s, a) = \underbrace{U_{\text{exploit}}(s, a)}_{\text{How good is move } a\text{?}} + \underbrace{c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\max(1, N(s))}}{1 + N(s')}}_{\text{Exploration bonus}}
$$

where $s'$ is the child node reached by playing column $a$ from parent $s$, $c_{\text{puct}} = 1.5$ is the exploration constant, and the exploitation term (ignoring solver/FPU refinements for a moment) is $-Q(s') = -\frac{W(s')}{N(s')}$.

**Why does this formula work so well?**
- **At the start of search ($N(s') = 0$)**: The denominator $1 + N(s') = 1$ is small, so the **prior $P(s, a)$** dominates. MCTS initially investigates moves that the neural network's intuition considers promising!
- **As a move is visited ($N(s')$ grows)**: Its exploration bonus shrinks like $\frac{1}{1 + N(s')}$. If the move turns out to have a poor value ($-Q(s')$ drops), MCTS naturally switches to alternative moves.
- **As total parent visits $N(s)$ grow**: $\sqrt{N(s)}$ grows slowly, ensuring no legal move with $P(s, a) > 0$ is permanently starved, while **asymptotically concentrating visits** on the move(s) with the highest true action value $-Q(s')$!

### Phase 2 & 3: Leaf Evaluation and Expansion

Once selection stops at a leaf node $s_L$:
1. **Already proven (`solver_state != Ongoing`)**: No neural network call needed! Immediately backpropagate the exact proven value (`+1.0`, `-1.0`, or `0.0`).
2. **Terminal board (`game.get_result() is not None`)**: No neural network call needed! Mark `solver_state` (`ProvenLoss` if `result == -1.0`, `ProvenDraw` if `0.0`), propagate solver proofs upward, and backpropagate `result`.
3. **Unexpanded ongoing board**:
   - Run the neural network on $s_L$ with legal move mask: $(\mathbf{p}, v) = f_\theta(s_L)$.
   - Instantiate a child `MCTSNode` for each legal column $c$ with `prior = p[c]`. (Each child constructor immediately checks `child_game.get_result()`—so if any child is an instant 4-in-a-row win, that child is born with `SOLVER_LOSS`, and `_propagate_solver()` immediately marks $s_L$ as `SOLVER_WIN`!)
   - Choose backpropagated value `bval`: if expansion just proved $s_L$'s exact `solver_state`, use the exact proven value (`±1.0` or `0.0`) instead of the noisy network estimate $v$! Otherwise use $v$.

*(Note: That subtle rule—overriding `value` with `proven_val` on the exact simulation when `expand()` discovers a 1-ply mate—prevents a single inaccurate neural value from polluting ancestor Q-values! See [`src/mcts.py`](../src/mcts.py#L400-L409) and [`src/self_play.py`](../src/self_play.py).)*

### Phase 4: Backpropagation

Walk from `node = s_L` up through `node.parent` to `root` ([`src/mcts.py`](../src/mcts.py#L233-L246)):

```python
def backpropagate(self, value: float) -> None:
    node: MCTSNode | None = self
    while node is not None:
        node.visit_count += 1
        node.total_value += value
        value = -value        # <-- Flip perspective at every ply!
        node = node.parent
```

---

### Worked Numerical Trace (3 Simulations by Hand)

Suppose for clarity there are only **2 legal columns** left (`Col A` and `Col B`), $c_{\text{puct}} = 1.5$, FPU = $0.0$, and no noise.
- **Root Initialization**: Neural network evaluates `root` (Red to move) and gives priors $P(A) = 0.60$, $P(B) = 0.40$. Root has children `A` ($N=0, W=0$) and `B` ($N=0, W=0$).

**Simulation 1**:
1. **Select at Root** ($N(\text{root}) = 0 \implies \max(1, 0) = 1$):
   - $\text{UCB}(A) = 0 + 1.5 \times 0.60 \times \frac{\sqrt{1}}{1 + 0} = \mathbf{0.90}$
   - $\text{UCB}(B) = 0 + 1.5 \times 0.40 \times \frac{\sqrt{1}}{1 + 0} = 0.60$
   - Pick **Child A**!
2. **Evaluate & Expand Child A** (Yellow to move at A): Suppose NN outputs $v(A) = +0.50$ (good for Yellow = bad for Red!).
3. **Backpropagate $+0.50$ starting at Child A**:
   - `Child A`: $N(A) = 1$, $W(A) = +0.50 \implies Q(A) = +0.50$ (from Yellow's view).
   - Flip sign $\to -0.50$.
   - `Root`: $N(\text{root}) = 1$, $W(\text{root}) = -0.50 \implies Q(\text{root}) = -0.50$ (from Red's view).

**Simulation 2**:
1. **Select at Root** ($N(\text{root}) = 1$, using $Q_{\text{unvisited}} = 0$ for simplicity):
   - $\text{UCB}(A) = \underbrace{-0.50}_{-Q(A)} + 1.5 \times 0.60 \times \frac{\sqrt{1}}{1 + 1} = -0.50 + 0.45 = -0.05$
   - $\text{UCB}(B) = \underbrace{0.00}_{\text{unvisited}} + 1.5 \times 0.40 \times \frac{\sqrt{1}}{1 + 0} = \mathbf{+0.60}$
   - Because `A` disappointed Red, MCTS automatically switches to **Child B**!
2. **Evaluate & Expand Child B** (Yellow to move at B): Suppose NN outputs $v(B) = -0.80$ (terrible for Yellow = great for Red!).
3. **Backpropagate $-0.80$ starting at Child B**:
   - `Child B`: $N(B) = 1$, $W(B) = -0.80 \implies Q(B) = -0.80$.
   - Flip sign $\to +0.80$.
   - `Root`: $N(\text{root}) = 2$, $W(\text{root}) = -0.50 + 0.80 = +0.30 \implies Q(\text{root}) = +0.15$.

**Simulation 3**:
1. **Select at Root** ($N(\text{root}) = 2$, $\sqrt{2} \approx 1.414$):
   - $\text{UCB}(A) = -0.50 + 1.5 \times 0.60 \times \frac{1.414}{2} = -0.50 + 0.636 = +0.136$
   - $\text{UCB}(B) = \mathbf{+0.80} + 1.5 \times 0.40 \times \frac{1.414}{2} = +0.80 + 0.424 = \mathbf{+1.224}$
   - Despite having the *lower* initial prior ($0.40 < 0.60$), **Child B** now dominates and gets searched deeper!

*(Notice also at the end of Sim 2: `Root`'s own $Q(\text{root}) = +0.15$ was diluted by the exploratory visit to `A` (`-0.50`), whereas the best move's value `-Q(B)` is **`+0.80`**! Remember this observation—it will be critical in [Chapter 05](./05-evaluation-and-arena.md)!)*

---

## 4. First Play Urgency (FPU) Reduction

Look back at Simulation 2 above: when `Child B` had **zero visits**, what value did we plug in for its exploitation term $U_{\text{exploit}}$? We plugged in **`0.0`**.

In the original 2017 AlphaZero paper, unvisited children always defaulted to $0.0$. Later AlphaZero enhancements (such as Leela Chess Zero and KataGo) discovered a serious pathology with `0.0`:

### The "Losing Position Distraction" Bug
Suppose Red is in a tough defensive position at node $s$:
- Red's expected value is $Q(s) = -0.75$ (Red is losing badly unless Red defends accurately).
- The neural network knows **Column 4** is the *only* real defensive try, assigning $P(s, \text{Col 4}) = 0.94$, and assigning $P = 0.01$ to each of the other 6 columns.
- On the first visit down Column 4, MCTS finds `-Q(Col 4) = -0.55` (still unpleasant, but much better than `-0.95`!).
- What happens if unvisited children have utility **`0.0`**?
  - Column 4 has $\text{UCB} \approx -0.55 + \text{exploration}$.
  - Every garbage $P=0.01$ blunder column has unvisited utility **`0.0`**, which is `+0.55` *higher* than `-0.55`!
  - MCTS thinks *"Ooh, an unvisited move might be a `0.0` draw!"* and wastes precious simulations visiting all 6 blunder columns at every single interior node of the defensive tree, starving the main defensive line of depth!

### Parent-Relative FPU Reduction (`c_fpu = 0.20`)

Instead of assuming an unvisited move has value `0.0` (a draw), **Parent-Relative FPU Reduction** assumes an unvisited move is **slightly worse than the parent node's own current value $Q(s)$**, scaled by how much policy mass we have already explored!

Here is the exact formula implemented in `MCTSNode._unvisited_fpu_value` ([`src/mcts.py`](../src/mcts.py#L74-L90)) and `_best_child` ([`src/self_play.py`](../src/self_play.py)):

$$
U_{\text{FPU}}(s) = \text{clamp}\!\left(Q(s) - c_{\text{fpu}} \sqrt{\sum_{b:\, N(s,b) > 0} P(s, b)},\; -1.0,\; +1.0\right)
$$

with default `DEFAULT_C_FPU = 0.20`:
- Remember $Q(s)$ is already stored from **parent $s$'s mover perspective**!
- In our losing example ($Q(s) = -0.75$, visited Column 4 has $P = 0.94$):

  $$
  U_{\text{FPU}}(s) = -0.75 - 0.20 \sqrt{0.94} \approx -0.944
  $$

- Now `-Q(Col 4) = -0.55` easily beats `-0.944`, so MCTS stays laser-focused on calculating the critical defence instead of chasing phantom `0.0` mirages!
- Conversely, if the best high-prior moves *have not* been visited yet ($\sum_{N>0} P \approx 0$), the penalty $c_{\text{fpu}}\sqrt{\dots} \approx 0$ is tiny so MCTS readily tries high-prior moves.
- **Exception at Noisy Roots (`is_noisy_root = True`)**: During self-play, we deliberately inject Dirichlet noise at the root to encourage trying low-prior surprise moves. Applying pessimistic FPU reduction at the root would fight against root exploration—so when `is_noisy_root` is `True`, `alphafour` sets $U_{\text{FPU}}(s_{\text{root}}) = 0.0$!

---

## 5. Root Exploration: Dirichlet Noise

During self-play training, if MCTS always trusted the neural network's current priors $P(s_{\text{root}}, a)$, the network could fall into a **self-confirming blind spot**: if it initially assigns $P(\text{Col 3}) = 0.001$, MCTS might rarely visit Column 3, so Column 3 never gets discovered even if it is strong.

To guarantee breadth at the root of every self-play turn, AlphaZero mixes the root's priors with random samples from a symmetric **Dirichlet distribution** ([`src/mcts.py`](../src/mcts.py#L347-L359), [`src/self_play.py`](../src/self_play.py)):

$$
\tilde{P}(s_{\text{root}}, a_i) = (1 - \varepsilon) P(s_{\text{root}}, a_i) + \varepsilon \eta_i, \qquad (\eta_1, \dots, \eta_K) \sim \text{Dir}(\alpha, \dots, \alpha)
$$

where $K \le 7$ is the number of legal moves at the root, and `alphafour` uses:
- **$\varepsilon = 0.25$ (`dirichlet_epsilon`)**: 75% learned network prior + 25% random exploration prior.
- **$\alpha = 0.35$ (`DEFAULT_DIRICHLET_ALPHA`)**: Concentration parameter.

> **Why $\alpha = 0.35$?**
> A symmetric $\text{Dir}(\alpha)$ distribution over $K$ actions behaves very differently depending on $\alpha$:
> - If $\alpha \gg 1$, every $\eta_i \approx 1/K$ (flat uniform blur—barely boosts any single move).
> - If $\alpha < 1$, a sample $(\eta_1, \dots, \eta_K)$ is **spiky**: it puts a large spike (e.g., $\eta_i \approx 0.6$) on one or two random columns on each turn!
> DeepMind's rule of thumb is $\alpha \approx \frac{10}{\text{average legal moves}}$ (Go: $\approx 250$ moves $\to \alpha = 0.03$; Chess: $\approx 35$ moves $\to \alpha = 0.3$). With up to 7 legal columns in Connect Four, $\alpha = 0.35$ produces clean spikes that occasionally boost an overlooked column by $+10\%$ to $+20\%$ prior without overwhelming every move simultaneously.
>
> **Important**: Dirichlet noise is enabled **only during self-play training** (`add_noise=True`). During evaluation (`benchmark.py`), Arena matches, and human play (`play.py`), `add_noise=False` so the AI plays at maximum clean strength.

---

## 6. Exact MCTS-Solver: Game-Theoretic Proofs Inside MCTS

Standard MCTS only tracks statistical averages ($N$ and $Q$). In tactical games like Connect Four, that leaves huge value on the table:
1. Suppose at depth 2, MCTS expands all 7 of the opponent's replies to move $a$ and discovers that **every single reply gives us an immediate 4-in-a-row win**. Mathematically, move $a$ is now **100% proven to force a win**—yet standard MCTS would keep spending hundreds of redundant simulations re-visiting that already-solved subtree!
2. Worse, suppose at the root we discover move $b$ allows a forced 2-move mate against us. Standard MCTS only lowers $-Q(b)$ gradually with each visit, and if temperature sampling ($\tau = 1$) is active, a move with even 3 visits out of 600 has a $0.5\%$ chance of blundering a forced mate!

`alphafour` implements **MCTS-Solver** (Winands, Björnsson & Saito, 2008), tracking a 4-state exact game-theoretic status `solver_state` on every node from `node.game.current_player`'s perspective ([`src/mcts.py`](../src/mcts.py#L16-L20), [`src/self_play.py`](../src/self_play.py)):

| Constant | Integer Code | Value (Node Mover's View) | Meaning |
| :--- | :---: | :---: | :--- |
| `SOLVER_ONGOING` | `0` | Unknown | Subtree not yet game-theoretically solved |
| `SOLVER_WIN` | `1` | `+1.0` | Current player at this node has a **proven forced win** |
| `SOLVER_LOSS` | `-1` | `-1.0` | Current player at this node is **proven lost** against best play |
| `SOLVER_DRAW` | `2` | `0.0` | Best play from both sides at this node is a **proven draw** |

### 6.1 Exact Minimax Upward Propagation Rules

Whenever a terminal child is created in `expand()` or reached during search, `_propagate_solver()` / `propagate_solver_up()` walks up parent pointers applying exact minimax logic over **all legal children** of an expanded node $p$:

```text
Parent p (my turn)
 ├── Child c_0 (opponent's turn): ProvenWin (+1 for opp)  ──► Losing move for me
 ├── Child c_1 (opponent's turn): ProvenLoss (-1 for opp) ──► WINNING MOVE FOR ME!
 └── Child c_2 (opponent's turn): Ongoing
```

1. **Proven Win Rule (`∃ child == ProvenLoss`)**:
   If **at least one** child is `ProvenLoss` for the opponent, then playing that move wins by force $\implies$ **`p.solver_state = ProvenWin`**! *(Notice we don't even need the remaining children to be solved!)*
2. **Proven Loss Rule (`∀ children == ProvenWin`)**:
   If $p$ is expanded (all legal moves exist) and **every** legal child is `ProvenWin` for the opponent, then every legal move loses $\implies$ **`p.solver_state = ProvenLoss`**!
3. **Proven Draw Rule (`∀ children != Ongoing` and no `ProvenLoss`, at least one `ProvenDraw`)**:
   If every legal child is solved (`ProvenWin` or `ProvenDraw`) and none is `ProvenLoss`, the best achievable outcome is a draw $\implies$ **`p.solver_state = ProvenDraw`**!

Propagation continues upward (`p = p.parent`) until a parent remains `Ongoing` or we reach the root!

### 6.2 How MCTS-Solver Transforms Search & Move Choice

1. **Selection Priority (`select_child` / `best_child`)**:
   - If any child is `ProvenLoss` (a proven winning move for the parent), **immediately select it**!
   - Filter out any `ProvenWin` children (proven losing moves for the parent) so MCTS **never wastes another simulation** exploring a known losing move (unless *all* moves are `ProvenWin`).
2. **Short-Circuiting Subtrees**:
   - Selection stops immediately upon reaching any solved node (`solver_state != Ongoing`) and backpropagates its exact value (`+1.0`, `-1.0`, or `0.0`) with **zero GPU calls**.
   - In `CythonBatchedSelfPlay` ([`src/self_play.py`](../src/self_play.py)), as soon as `root.solver_state != SOLVER_ONGOING`, that game slot marks `sims_done = num_simulations` and finishes its turn early!
3. **Blunder-Proof Root Visit Extraction**:
   - If `root` is `ProvenWin`, non-winning moves are zeroed out so the agent *only* plays a proven winning column.
   - If `root` is `Ongoing` or `ProvenDraw`, any move proven to lose (`child.solver_state == ProvenWin`) has its selection count zeroed out whenever at least one non-losing move exists.

---

## 7. Action Selection, Temperature & Stateful Tree Reuse

### 7.1 From Visit Counts to Move Probabilities (`temperature`)

After search finishes at `root`, MCTS has an array of root visit counts $N(s_{\text{root}}, a)$ for $a \in \{0, \dots, 6\}$. How do we turn visit counts into an actual move?

Using a **temperature parameter $\tau \ge 0$** ([`src/mcts.py`](../src/mcts.py#L521-L554)):

$$
\pi_\tau(a \mid s_{\text{root}}) = \frac{N(s_{\text{root}}, a)^{1/\tau}}{\sum_{b=0}^{6} N(s_{\text{root}}, b)^{1/\tau}}
$$

- **$\tau = 1.0$ (Proportional exploration — early self-play moves `0..9`)**:
  If Col 4 got 400 visits and Col 3 got 200 visits, we play Col 4 with $66.7\%$ probability and Col 3 with $33.3\%$ probability. This creates rich game variety across 1,000 self-play games!
- **$\tau \to 0$ (Greedy exploitation — `temperature == 0`)**:
  Deterministic `argmax_a N(s_root, a)`. Used after ply 10 in self-play, in all Arena/Benchmark evaluations, and in `play.py`.

> **Why pick `argmax(visit_counts)` instead of `argmax(-child.q_value)`?**
> Visit count $N(s, a)$ is far more robust than raw $Q$: an under-explored child with $N = 1$ might have a lucky single-leaf evaluation of $+0.82$, whereas the true best move with $N = 520$ and $-Q = +0.79$ has survived hundreds of deep adversarial refutation attempts! (And thanks to MCTS-Solver, any proven win/loss overrides raw visits anyway.)

### 7.2 Stateful Tree Reuse (`advance_move` & `rebase`)

Suppose MCTS runs 600 simulations at turn $t$, putting 350 visits into Column 4, and then plays Column 4. Those 350 simulations already built a 350-node search tree rooted at the board state *after* Column 4!

Throwing that subtree away on turn $t+1$ would waste $>50\%$ of the work we just did. Both engines reuse subtrees:
- **Python (`MCTS.advance_move(col)` in [`src/mcts.py`](../src/mcts.py#L292-L299))**: Used interactively in [`play.py`](../play.py#L288-L294) after *both* human moves and AI moves! Detaches `self._root = self._root.children[col]`, sets `parent = None`, and runs only the remaining `num_simulations - existing_sims` needed to top the tree back up to budget (while discarding the tree if `_root_noisy` was true and a clean noiseless search is requested).
- **Cython (`_rebase_tree` in [`src/self_play.py`](../src/self_play.py))**: Performs a fast contiguous BFS compaction of the chosen child's subtree into `scratch_nodes[0..]` and swaps the arena pointers during batched self-play, preserving `visit_count`, `total_value`, `prior`, `is_expanded`, and `solver_state`! *(Note: in self-play, `sims_done` resets to `0` on the new turn so every turn still receives `num_simulations` fresh lookahead simulations on top of the inherited subtree!)*

---

## 8. Summary & Next Steps

1. **PUCT** balances exploitation ($-Q$) and neural-prior-weighted exploration ($c_{\text{puct}} P \frac{\sqrt{N}}{1+n}$), turning neural intuition into selective deep lookahead.
2. **Parent-Relative FPU Reduction (`c_fpu = 0.20`)** prevents losing positions from wasting search budget on low-prior unvisited moves.
3. **Dirichlet Root Noise ($\alpha = 0.35, \varepsilon = 0.25$)** ensures root diversity during self-play without corrupting internal tree nodes.
4. **MCTS-Solver** propagates exact `ProvenWin` / `ProvenLoss` / `ProvenDraw` minimax proofs upward, making tactical mates exact and blunder-proof.

Next, let's examine the "student" side of the partnership: how `AlphaZeroNet` turns a $6 \times 7$ grid into policy priors and Win/Draw/Loss probabilities, and how inference is engineered to run at tens of thousands of positions per second on a GPU.

---

⬅️ **Previous**: **[Chapter 01: AlphaZero from First Principles](./01-alphazero-fundamentals.md)**  
👉 **Next**: **[Chapter 03: Neural Network Architecture, WDL Value Head & Fast Inference](./03-neural-network-and-inference.md)**  
🏠 **Back to Index**: **[AlphaFour Documentation Hub (`docs/README.md`)](./README.md)**
