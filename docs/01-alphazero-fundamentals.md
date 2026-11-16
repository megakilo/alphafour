# 01. AlphaZero from First Principles

> **Code references in this chapter**:
> - Python game rules & tensor encoding: [`src/game.py`](../src/game.py)
> - Cython bitboard game rules & encoding: [`src/self_play.py`](../src/self_play.py)

---

## 1. The Core Question: How Can an AI Learn Without Human Knowledge?

Suppose you want to build a superhuman AI for a board game like Go, Chess, or Connect Four.

Historically, game AIs relied on two human crutches:
1. **Hand-crafted evaluation functions** (like classic Chess engines): Human experts write hundreds of rules (*"rooks on open files = +0.3"*, *"three-in-a-row with an open square = +0.25"*) and use minimax tree search to look ahead.
2. **Supervised learning from human grandmasters** (like the original 2016 AlphaGo): Train a neural network to imitate millions of human expert moves, then refine it with reinforcement learning.

**AlphaZero** (Silver et al., 2017) discards *both*. You give the system **only the rules of the game**—which moves are legal and when the game ends in a win, draw, or loss. Starting from completely random neural network weights, it discovers openings, tactics, sacrifices, and endgame strategy purely by playing against itself (**self-play**).

At first glance, this sounds like pulling yourself up by your own bootstraps:
- If the neural network starts completely random, its move suggestions and board evaluations are garbage.
- If two garbage agents play each other, shouldn't they just generate garbage training data forever?

Understanding **why AlphaZero escapes this trap** is the single most important insight in the entire algorithm.

---

## 2. The Secret Engine: Generalized Policy Iteration via Tree Search

AlphaZero combines two components that compensate for each other's weaknesses:

1. **A Deep Neural Network $f_\theta(s) = (\mathbf{p}, v)$** ("Intuition"):
   Given a board state $s$, a single forward pass immediately outputs:
   - **Policy $\mathbf{p} \in [0, 1]^7$ (with $\sum_{a=0}^6 p_a = 1$)**: A probability distribution over legal moves (*"which columns look promising at a glance?"*).
   - **Value $v \in [-1, +1]$**: The expected game outcome from the current player's perspective (*"am I winning (`+1`), drawing (`0`), or losing (`-1`)?"*).

2. **Monte Carlo Tree Search — MCTS** ("Calculation / Lookahead"):
   Instead of trusting $f_\theta(s)$ blindly and picking a move right away, AlphaZero runs hundreds of lookahead simulations (e.g., `600` simulations per move during training in `alphafour`) to build a local search tree rooted at $s$.

### Why MCTS Improves Even a Weak Network

Imagine the neural network is completely untrained ($v \approx 0$ everywhere, $\mathbf{p} \approx \text{uniform}$ over all 7 columns). Suppose on the board right now:
- Column 4 completes 4-in-a-row immediately (**instant win**).
- Column 1 lets the opponent complete 4-in-a-row on their very next turn (**instant loss**).

Even with a *random* network:
- When MCTS simulates Column 4, it hits the **exact game rules** (`game.get_result()`) and observes a real terminal win! It doesn't ask the neural network what a 4-in-a-row is worth—the game engine tells MCTS with 100% certainty that Column 4 wins.
- When MCTS simulates Column 1 and then checks the opponent's replies, it discovers the opponent has an instant win, so Column 1's value plummets.

Consequently, after 600 simulations, the distribution of **MCTS root visit counts** $\boldsymbol{\pi}$ concentrates heavily on Column 4 and avoids Column 1!

In Reinforcement Learning terms, MCTS is a **Policy Improvement Operator** $\mathcal{I}$:

$$
\boldsymbol{\pi}_t = \mathcal{I}(f_\theta, s_t) \quad \text{is strictly stronger than the raw network policy} \quad \mathbf{p}_\theta(s_t)
$$

AlphaZero then closes the loop (**Generalized Policy Iteration**):

```text
          ┌──────────────────────────────────────────────────────────┐
          │                                                          │
          ▼                                                          │
   Neural Network f_θ(s) = (p, v)                                    │
   ("Fast pattern-recognition intuition")                            │
          │                                                          │
          │ Guides search priors P(s,a) & leaf values V(s_L)         │ Trains θ so that:
          ▼                                                          │   p_θ(s_t) ──► π_t
   MCTS Lookahead (600 sims + Game Rules)                            │   v_θ(s_t) ──► z_t
   ("Slow, deliberate tactical calculation")                         │
          │                                                          │
          │ Produces stronger move distribution π_t                  │
          │ and plays full game to true outcome z ∈ {-1, 0, +1}      │
          ▼                                                          │
   Self-Play Trajectory {(s_1, π_1, z_1), ..., (s_T, π_T, z_T)} ─────┘
```

Let's trace what happens across iterations:
- **Iteration 1**: A random network + 600-sim MCTS discovers 1-ply wins and blocks 1-ply losses. We train the network to predict those MCTS visit distributions $\boldsymbol{\pi}_t$ and game outcomes $z_t$. Now the *raw network* recognizes 1-ply threats in **zero** simulations!
- **Iteration 2**: Because the network's intuition already spots 1-ply tactics at the leaves of the search tree, 600-sim MCTS now effectively sees 2–3 plies deeper, discovering forks and double threats. Training distills *that* deeper search back into the network weights.
- **Iteration 50+**: Search and intuition have ratcheted each other up dozens of times—distilling deep tactical lookahead into fast convolutional weights, which in turn lets search probe subtle opening and endgame lines!

> **Mental Model**: Think of MCTS as the **teacher** (who gets to use a scratchpad and check the rulebook 600 times) and the Neural Network as the **student** (who must answer in a single glance). Every iteration, the student learns to match the teacher's deliberate conclusions—which makes the *teacher* even smarter next iteration because the teacher uses the student's intuition at the leaves of its scratchpad!

---

## 3. Connect Four as an AlphaZero Laboratory

Before looking at code, let's review the exact game environment implemented in [`src/game.py`](../src/game.py) and [`src/self_play.py`](../src/self_play.py).

### Rules & Board Geometry
- **Grid**: `ROWS = 6` rows (`0` = bottom row, `5` = top row) $\times$ `COLS = 7` columns (`0` = leftmost, `6` = rightmost; displayed `1..7` in CLI).
- **Players**: `1` (Player 1 / Red `●`, moves first) and `-1` (Player 2 / Yellow `●`, moves second).
- **Gravity**: Dropping a stone into column $c \in \{0, \dots, 6\}$ places it at `row = heights[c]` (the lowest unoccupied row in column $c$) and increments `heights[c]`. A column is legal iff `heights[c] < 6`.
- **Terminal states**:
  - **Win**: Either player forms `WIN_LENGTH = 4` consecutive stones horizontally, vertically, diagonally (`/`), or anti-diagonally (`\`).
  - **Draw**: All $6 \times 7 = 42$ cells are filled (`move_count == 42`) with no 4-in-a-row.

### Why Connect Four Is Unique for Learning AlphaZero

Connect Four occupies a sweet spot in game complexity:
1. **Non-trivial state space**: There are $4{,}531{,}985{,}219{,}092 \approx 4.53 \times 10^{12}$ legal positions—far too large for tabular Q-learning or naive brute-force search without heavy domain pruning.
2. **Sharp tactics AND deep strategy**: Games feature multi-move double-threat combinations, odd/even row parity ("zugzwang"—forcing the opponent to play under a square you control), and delicate drawing defences.
3. **Trainable on a single GPU in hours**: Unlike Go or Chess, you can run a full 150-iteration AlphaZero training run on one GPU and watch it reach near-perfect play.
4. **Mathematically Solved (Ground-Truth Oracle!)**: Connect Four was solved independently by Victor Allis and James Dow Allen in 1988. From the empty board with best play by both sides:

   | Opening Column (1-indexed) | Col 1 | Col 2 | Col 3 | **Col 4 (Center)** | Col 5 | Col 6 | Col 7 |
   | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
   | **0-indexed action** | `0` | `1` | `2` | **`3`** | `4` | `5` | `6` |
   | **Exact Game-Theoretic Outcome for P1** | **Loss (`-1`)** | **Loss (`-1`)** | **Draw (`0`)** | **Win (`+1`)** | **Draw (`0`)** | **Loss (`-1`)** | **Loss (`-1`)** |

   *(Exact child W/D/L on the empty board: `[-1, -1, 0, 1, 0, -1, -1]`.)*

   Because `alphafour` includes a Pascal-Pons exact bitboard solver ([`src/evaluate.py`](../src/evaluate.py)) and 6,000 exact positions from `gamesolver.org` ([`testdata/`](../testdata)), we don't have to guess whether our AlphaZero agent is playing well—we can measure its **exact move optimality %** and **exact blunder rate %**!

---

## 4. Zero-Sum Alternating Perspective (Crucial Concept!)

If you ever implement AlphaZero yourself, **90% of subtle bugs come from player perspective sign errors**. Let's nail down the exact convention used everywhere in `alphafour`.

### 4.1 One Network for Both Players: Canonical Encoding

We do *not* train one network for Red (`+1`) and a second network for Yellow (`-1`). Instead, we train a **single** network that always views the board from the perspective of **`game.current_player`** (the player whose turn it is to move *right now*).

Look at `ConnectFour.encode()` in [`src/game.py`](../src/game.py#L121-L134) (and `_encode_board_u8` in [`src/self_play.py`](../src/self_play.py)):

```python
def get_canonical_board(self) -> np.ndarray:
    # Multiplying by current_player (+1 or -1) makes:
    #   +1 = MY pieces (whoever is about to move)
    #   -1 = OPPONENT's pieces
    return self.board * self.current_player

def encode(self) -> np.ndarray:
    canonical = self.get_canonical_board()
    planes = np.zeros((3, 6, 7), dtype=np.float32)
    planes[0] = (canonical == 1).astype(np.float32)   # Plane 0: Current player's stones
    planes[1] = (canonical == -1).astype(np.float32)  # Plane 1: Opponent's stones
    if self.current_player == 1:
        planes[2] = 1.0                               # Plane 2: All 1s if P1 to move, all 0s if P2
    return planes
```

> **Why do we need Plane 2 (the turn-indicator plane)?**
> Couldn't the network figure out whose turn it is by counting stones on Plane 0 and Plane 1 (if stone counts are equal, it's Player 1's turn; if Opponent has 1 more stone, it's Player 2's turn)?
> Eventually, yes—especially with global pooling—but local $3\times 3$ convolutions in the first layer only see a $3\times 3$ patch of the board, not the global stone sum! In Connect Four, **first-player vs second-player parity** ("Claimeven" strategy: Player 2 controls 1-indexed even rows `2, 4, 6` — i.e., 0-indexed `row = 1, 3, 5` — while Player 1 controls 1-indexed odd rows `1, 3, 5` — i.e., 0-indexed `row = 0, 2, 4`) is fundamental to endgame evaluation. Providing Plane 2 gives every convolutional filter immediate awareness of turn parity from Layer 1.

### 4.2 Why `game.get_result()` Returns `-1.0` on a Win!

Look closely at `make_move` and `get_result` in [`src/game.py`](../src/game.py#L47-L112):

```python
def make_move(self, col: int) -> ConnectFour:
    row = self._heights[col]
    self.board[row, col] = self.current_player
    self._heights[col] = row + 1
    self.last_move = col
    self._move_count += 1
    self.current_player = -self.current_player  # <-- Turn flips IMMEDIATELY!
    return self

def get_result(self) -> float | None:
    if self.is_win():
        # The player who JUST moved won, so `self.current_player` LOST!
        return -1.0
    if self.is_draw():
        return 0.0
    return None
```

Let's verify why this makes complete mathematical sense:
1. Suppose it is **Player 1's** turn (`current_player = +1`).
2. Player 1 drops their 4th stone in a row into column `3`.
3. `make_move(3)` places `+1` on the board and flips `current_player` to **`-1` (Player 2)**.
4. Now the game is at a terminal state where it would be Player 2's turn, except Player 1 already made 4-in-a-row.
5. From **Player 2's (`current_player`'s) perspective**, what is the value of this board? **`-1.0` (Loss)**!
6. Notice that in Connect Four, you can *never* make a move that causes *yourself* to lose on that exact stone drop, nor can it be your turn when you already have 4-in-a-row on the board. Therefore, at any terminal board state, `get_result()` from `current_player`'s perspective is **always either `-1.0` (the previous mover just beat me) or `0.0` (the 42nd stone filled the board in a draw)**.

### 4.3 Worked Example: Labeling a Self-Play Trajectory

Suppose a tiny game lasts 7 plies (`t = 0, 1, 2, 3, 4, 5, 6`), and **Player 1 wins** on move `t = 6`:

| Ply $t$ | Player to Move at $s_t$ (`entry.player`) | Encoded Board $s_t$ | MCTS Policy Target $\boldsymbol{\pi}_t$ | Terminal State After Game Ends | Value Target $z_t$ Stored for $s_t$ |
| :---: | :---: | :--- | :--- | :--- | :---: |
| `0` | `+1` (Red) | Red's view | `[0.02, ..., 0.65, ...]` | Final `current_player = -1`, `result = -1.0` | **`+1.0`** (Red won!) |
| `1` | `-1` (Yellow) | Yellow's view | `[0.10, ..., 0.40, ...]` | Final `current_player = -1`, `result = -1.0` | **`-1.0`** (Yellow lost!) |
| `2` | `+1` (Red) | Red's view | `[0.05, ..., 0.70, ...]` | Final `current_player = -1`, `result = -1.0` | **`+1.0`** |
| `3` | `-1` (Yellow) | Yellow's view | `[0.15, ..., 0.50, ...]` | Final `current_player = -1`, `result = -1.0` | **`-1.0`** |
| `4` | `+1` (Red) | Red's view | `[0.00, ..., 0.90, ...]` | Final `current_player = -1`, `result = -1.0` | **`+1.0`** |
| `5` | `-1` (Yellow) | Yellow's view | `[0.20, ..., 0.30, ...]` | Final `current_player = -1`, `result = -1.0` | **`-1.0`** |
| `6` | `+1` (Red) | Red's view | `[0.00, 0.00, 0.00, 1.00, ...]` | Red plays winning stone $\to$ `current_player` flips to `-1` | **`+1.0`** |

Look at how clean the value-assignment rule is in [`src/self_play.py`](../src/self_play.py):

```python
# At terminal state: `game.current_player` is the loser (-1), and `result` is -1.0 (or 0.0 for draw)
for encoded, policy, player in histories[i]:
    val = float(result) if player == game.current_player else float(-result)
```

- For Yellow (`player == -1 == game.current_player`): `val = result = -1.0`.
- For Red (`player == +1 != game.current_player`): `val = -result = +1.0`.
- If the game was a Draw (`result == 0.0`): both get `0.0`!

Every single tuple `(s_t, π_t, z_t)` in the replay buffer obeys one universal invariant:

> **Universal Perspective Invariant**:
> Across the entire codebase—board encoding, neural network value output, MCTS node `q_value`, MCTS `SolverState`, and training target $z_t$—every value is measured strictly from the perspective of **the player whose turn it is to move at that state**.

---

## 5. Summary & Next Steps

1. **AlphaZero needs no human data** because MCTS acts as a policy-and-value improvement operator over the neural network, anchored by the exact terminal game rules.
2. **Canonical 3-plane encoding** lets one neural network play both sides by always viewing the board from `current_player`'s perspective, while Plane 2 preserves first/second-player turn parity.
3. **Alternating zero-sum signs** mean a good state for a child node (`+0.8` for the opponent) is a bad state for the parent node (`-0.8` for me).

Now let's open up the black box of the "teacher" itself: how does MCTS decide where to search, how does it balance exploration and exploitation, and how does `alphafour` upgrade standard AlphaZero MCTS with **First Play Urgency (FPU)** and an **Exact MCTS-Solver**?

---

👉 **Next**: **[Chapter 02: Core Search Algorithm — MCTS, PUCT, FPU & Exact MCTS-Solver](./02-mcts-deep-dive.md)**  
🏠 **Back to Index**: **[AlphaFour Documentation Hub (`docs/README.md`)](./README.md)**
