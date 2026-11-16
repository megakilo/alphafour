# 05. Verification, Exact Solver Metrics & Paired-Opening Arena

> **Code references in this chapter**:
> - Dataset evaluation, exact solver caching, Model-Isolated Arena & Anchor Elo: [`src/evaluate.py`](../src/evaluate.py)
> - Standalone benchmark & head-to-head Arena CLI: [`benchmark.py`](../benchmark.py)
> - Ground-truth solved benchmark suites: [`testdata/`](../testdata)

---

## 1. Why Training Loss Does NOT Measure Strength in AlphaZero!

If you come from supervised deep learning, your first instinct during training is to watch `policy_loss` and `value_loss`: *"Loss went down = model got better; loss went up = something broke!"*

**In self-play reinforcement learning, that intuition is dangerously misleading:**
1. **Moving Target Distribution**: In Iteration 2, self-play games are short, tactical 12-move blowouts where one player misses a simple 4-in-a-row. Predicting who wins a lopsided position is *easy* ($\mathcal{L}_{\text{value}}$ drops quickly to `~0.40`).
2. **Stronger Play Creates Harder Positions**: By Iteration 60, both sides defend tenaciously for 36+ moves, walking a razor's edge between subtle draws and deep 14-ply zugzwang wins! Because balanced positions have higher outcome entropy and MCTS distributes visits across multiple subtle tries, **training loss often plateaus or even ticks upward slightly while true playing strength climbs by hundreds of Elo points!**
3. **Self-Play Blind Spots ("Rock-Paper-Scissors" Cycles)**: Without external verification, an agent can beat Iteration $t-1$ in narrow lines while quietly forgetting how to defend early flank attacks or misclassifying all endgames as wins/losses instead of draws.

Because Connect Four is mathematically solved, `alphafour` pairs **two orthogonal verification systems**:
1. **Oracle Dataset Verification (`gamesolver.org` + Exact Bitboard Child Solver)**: Measures absolute game-theoretic truth across 6,000 positions spanning every phase and difficulty of Connect Four.
2. **Competitive Head-to-Head Arena (`Paired-Opening Arena` + `Anchor Elo`)**: Measures practical match play against both historical checkpoints and a fixed non-neural MCTS anchor (`PureMCTSBaseline`).

---

## 2. The `gamesolver.org` Ground-Truth Test Suite (`testdata/`)

`testdata/` contains 6 benchmark files (1,000 positions each = **6,000 positions total**) created by Pascal Pons ([gamesolver.org](http://blog.gamesolver.org/solving-connect-four/02-test-protocol/)).

### File Format & Score Convention
Each line has the format `<moves_str> <exact_score>`, for example:
```text
4453526232122611675776677533351313 -1
```
- **`<moves_str>`**: 1-indexed column sequence (`'1'..'7'`) played from the empty board to reach the test position.
- **`<exact_score>`**: Exact game-theoretic score from **`current_player`'s perspective**:
  - `score > 0`: Current player **forces a win** (`+1` WDL class; in Pascal Pons's convention, $|{\text{score}}| = 22 - k$ where $k$ is the winner's stone count when completing 4-in-a-row under fastest mate—i.e., `1 +` unused stones remaining in the winner's 21-stone reserve *after* mating, so **larger $|{\text{score}}|$ means a faster win**, e.g. `+18` = win on 4th stone, `+1` = win on 21st/last stone).
  - `score == 0`: Best play by both sides ends in a **draw** (`0` WDL class).
  - `score < 0`: Current player **loses against best play** (`-1` WDL class; similarly $-18$ = opponent mates on their 4th stone, $-1$ = opponent mates on their 21st stone).

### The 6 Game-Phase × Difficulty Splits

The 6 files partition positions by **how many stones have already been played** (`nb moves`: Begin $\le 14$, Middle $15..28$, End $> 28$) and **how many plies remain to the end of the game** (`nb remaining`, which determines search depth / difficulty):

| File | Phase & Difficulty | Stones Played | Remaining Plies | True Wins (`+1`) | True Draws (`0`) | True Losses (`-1`) | Key Capability Tested |
| :--- | :--- | :---: | :---: | ---: | ---: | ---: | :--- |
| **`Test_L3_R1`** | End-Easy | `29..41` | `< 14` | 327 | **432** | 241 | Endgame calculation & **Draw calibration** |
| **`Test_L2_R1`** | Middle-Easy | `15..28` | `< 14` | 548 | 12 | 440 | Midgame tactical combinations & mates |
| **`Test_L2_R2`** | Middle-Medium | `15..28` | `14..27` | 447 | 170 | 383 | Deep midgame threats & positional parity |
| **`Test_L1_R1`** | Begin-Easy | `0..14` | `< 14` | 723 | **0** | 277 | Early tactical traps & instant refutations |
| **`Test_L1_R2`** | Begin-Medium | `0..14` | `14..27` | 552 | 22 | 426 | Strategic opening-to-midgame transitions |
| **`Test_L1_R3`** | Begin-Hard | `0..14` | `>= 28` | 550 | 197 | 253 | **Hardest**: whole-game 30+ ply opening intuition |

> **Why does `Test_L1_R1` have exactly `0` draws (and report `draw_recall = N/A`)?**
> In Connect Four, a draw *only* happens when all **42** squares fill up! In `Test_L1_R1`, `moves <= 14` and `remaining < 14`, so the game terminates in at most $14 + 13 = 27 < 42$ stones—meaning every single position in `Test_L1_R1` is a decisive tactical win or loss before the board can fill! Conversely, `Test_L3_R1` is `43.2%` draws, making it the ultimate test of whether our 3-logit WDL value head truly understands drawn endgames.

---

## 3. Dataset Verification Metrics (`evaluate_dataset`)

When `main.py` or `uv run benchmark.py` runs `evaluate_dataset` ([`src/evaluate.py`](../src/evaluate.py#L680-L777)), it evaluates **both `0`-simulation raw network intuition** and **`400`-simulation MCTS search** across the datasets:

### 3.1 Opening Move Sanity Check (`evaluate_opening_move`)
Tests the empty board (`""`):
- Checks whether the chosen move is **Column 4** (`0`-indexed `3`, displayed as `column 4`—the *only* winning opening move for Player 1!).
- Reports **Center policy %** (at `0` sims) or **Center visit %** (at `>0` sims). By Iteration 2–3 of training, a healthy run switches to `✅ Opening Move: column 4` and steadily concentrates $>75\%$ of root visits on center.

### 3.2 3-Class WDL Accuracy & Per-Class Recall (`Win`, `Draw`, `Loss`)
Compares predicted outcome $\hat{y}_i \in \{-1, 0, +1\}$ against `true_wdl = sign(true_score)`:
- **At `0` simulations (Raw Network)**: Uses the WDL head's most probable class directly (`argmax([P(Loss), P(Draw), P(Win)]) - 1` in [`src/evaluate.py`](../src/evaluate.py#L642-L645)).
- **At `>0` simulations (MCTS Search)**: Classifies the MCTS best-move value $\hat{v}_i \in [-1, +1]$ using symmetric $\pm \frac{1}{3}$ thresholds ([`src/evaluate.py`](../src/evaluate.py#L670-L675)):

  $$
  \hat{y}_i = \begin{cases}
  +1 \text{ (Win)} & \text{if } \hat{v}_i > +\frac{1}{3} \\
  0 \text{ (Draw)} & \text{if } -\frac{1}{3} \le \hat{v}_i \le +\frac{1}{3} \\
  -1 \text{ (Loss)} & \text{if } \hat{v}_i < -\frac{1}{3}
  \end{cases}
  $$

  Notice $\pm \frac{1}{3}$ divides the `[-1, +1]` interval (length `2.0`) into three equal-width buckets of width $\frac{2}{3}$ each!

> **Subtle Engineering Lesson: Why MCTS Must Use `-best_child.q_value`, NEVER `root.q_value`!**
> Suppose at a root position, Column 4 is a winning move (`-child_4.q_value = +0.92`, visited 340 times), while Columns 1..7 include 6 losing moves (`-child_k.q_value = -0.90`, visited 10 times each during PUCT exploration).
> What is `root.q_value`? It is the **visit-weighted average of ALL simulations through the root**, including those 60 exploratory visits to losing moves (`(340*0.92 - 60*0.90)/400 = +0.64`)—and in narrower positions, exploration dilutes `root.q_value` below `+0.33`, misclassifying wins as draws!
> Game-theoretic value of a state under best play is the value of its **best move**, not the average of good and bad moves! `CythonBatchedMCTS` ([`src/self_play.py`](../src/self_play.py)) and `_run_python_batched_mcts` ([`src/evaluate.py`](../src/evaluate.py)) extract **undiluted `-best_child.q_value`** (or the exact MCTS-Solver proof value `±1.0` / `0.0`).

### 3.3 Exact Move Optimality % & Blunder % (`get_dataset_child_wdl`)
Notice that `gamesolver.org` lines only give the score of the *root* position—they don't list which of the 7 columns are optimal! (Often 2 or 3 columns all win or all hold a draw.)

To grade *move selection* with 100% mathematical precision, `get_dataset_child_wdl` ([`src/evaluate.py`](../src/evaluate.py#L278-L382)) loads SHA256-cached `.npz` matrices from `testdata/.solver_cache/` (or runs the exact bounded-budget Pascal-Pons bitboard solver `_bb_solve_wdl` for tactical and endgame positions) to determine **all 7 child columns** of every benchmark position as an `int8[N, 7]` matrix:
- `-2`: Illegal move (column is already full)
- `-1`: Playing this column loses against best reply
- `0`: Playing this column leads to a draw with best play
- `+1`: Playing this column forces a win!

*(Precomputed SHA256 `.npz` child-WDL matrices in `testdata/.solver_cache/*.npz` allow every benchmark or training run to load all 42,000 exact child outcomes in `< 5 ms`, even for `--limit` subsets via `_lookup_moves_in_dir_cache`; if an uncached early-game split exceeds `_bb_solve_wdl`'s node budget, `get_dataset_child_wdl` returns `None` and reports `N/A` gracefully.)*

Let $W_{i, c} \in \{-2, -1, 0, +1\}$ denote `child_wdl[i, c]`, let $a_i$ denote the agent's chosen column `best_actions[i]`, and let $w^*_i = \max_{c} W_{i, c}$ be the best achievable outcome on position $i$:
1. **Optimal Move % (`optimal_move_pct`)**:

   $$
   \text{Optimal Move \%} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}\big[W_{i, a_i} = w^*_i\big] \times 100\%
   $$

   Did the agent pick *any* move that preserves the game-theoretic best outcome (Win $\to$ Win, Draw $\to$ Draw, Loss $\to$ Loss)?
2. **Blunder % (`blunder_pct`)**:
   Evaluated strictly over **non-already-lost positions** ($w^*_i \ge 0$, because when $w^*_i = -1$ every legal move loses so a "blunder" is impossible):

   $$
   \text{Blunder \%} = \frac{\sum_{i:\, w^*_i \ge 0} \mathbf{1}\big[W_{i, a_i} < w^*_i\big]}{\sum_{i=1}^N \mathbf{1}[w^*_i \ge 0]} \times 100\%
   $$

   Measures how often the agent throws away a Win into a Draw/Loss, or throws away a Draw into a Loss!

### 3.4 Value MSE (`value_mse`)
Let $y_i = \operatorname{sign}(\text{score}_i) \in \{-1, 0, +1\}$ denote `np.sign(true_scores[i])`:

$$
\text{Value MSE} = \frac{1}{N} \sum_{i=1}^N \big(\hat{v}_i - y_i\big)^2
$$

Tracks continuous calibration of $\hat{v}_i \in [-1, +1]$ against $\{-1.0, 0.0, +1.0\}$.

---

## 4. Competitive Evaluation: Model-Isolated Paired-Opening Arena & Anchor Elo

Dataset metrics test single-move decisions on fixed boards. To test whether the agent can **string together 40 moves of coherent pressure and convert advantages into actual wins**, `alphafour` runs head-to-head Arena matches (`play_batched_arena` in [`src/evaluate.py`](../src/evaluate.py#L1031-L1192)).

Building a fair, fast batched Connect Four Arena requires solving three classic pitfalls:

### Pitfall 1: Cross-Model Tree Contamination $\implies$ Strict Model Isolation
When batching 40 Arena games simultaneously on a GPU, it is tempting to evaluate leaves across all active games using whichever model happens to be handy, or to share MCTS trees between turns of Player 1 and Player 2. **Never do that!** Model A's move choice must depend *100% on Model A's own neural evaluations* at its root and internal MCTS nodes, and *0% on Model B's weights*.

At every ply of `play_batched_arena` ([`src/evaluate.py`](../src/evaluate.py#L1109-L1147)):
1. Active games are partitioned into **`m1_turn_indices`** (boards where `model1` is to move) and **`m2_turn_indices`** (boards where `model2` is to move).
2. A fresh batched MCTS search runs `sims_m1` simulations on `m1_turn_indices` querying **only `model1`**.
3. A separate batched MCTS search runs `sims_m2` simulations on `m2_turn_indices` querying **only `model2`**.

### Pitfall 2: Deterministic Collapse & First-Player Bias $\implies$ Duplicate Paired Openings
In Connect Four:
1. If two deterministic agents (`temperature = 0.0`, no noise) start from the empty board, all 40 games play the **exact same move sequence** 40 times!
2. Furthermore, Connect Four is a **First-Player Win**—so if Model 1 plays Red more often or gets easier openings as Red, winrate reflects color luck rather than skill!

`generate_paired_openings` ([`src/evaluate.py`](../src/evaluate.py#L945-L999)) solves both problems cleanly:
1. Enumerates 2-to-4-ply opening sequences (`min_plies=2, max_plies=4`), shuffles deterministically (`seed=42`), and filters each candidate through a **6-ply bitboard tactical minimax check** (`_bb_shallow_tactical_score(..., depth=6) == 0`) so neither player starts in an accidental forced tactical win or loss!
2. For `num_games = 40`, selects `20` balanced openings and plays **each opening twice**:
   - **Game $k$ (`0..19`)**: Opening $k$ with **`model1` as Player 1 (Red)** and `model2` as Player 2 (Yellow).
   - **Game $k + 20$ (`20..39`)**: The **exact same Opening $k$** with **`model2` as Player 1 (Red)** and `model1` as Player 2 (Yellow)!
3. Both sides play at full deterministic tournament strength (`temperature = 0.0`, `add_noise = False`). To come out ahead on a pair, a model must either win with *both* colors or win as Red while holding a draw/win as Yellow!

### Pitfall 3: Relative Drift $\implies$ Fixed Anchor Elo (`PureMCTSBaseline`)
`main.py` runs **two** Arena matches every `--eval-interval` (default `5`) iterations:
1. **Relative Arena (`Current vs Iter-(t - 10)`)**: Measures whether the newest checkpoint beats itself from 10 iterations ago. Great for spotting regressions, *but* relative winrate stays around 55–70% as both `t` and `t-10` get stronger together!
2. **Absolute Anchor Arena (`Current vs PureMCTSBaseline`)**: Pits the current checkpoint (`400` sims) against **`PureMCTSBaseline`** ([`src/evaluate.py`](../src/evaluate.py#L1195-L1308)) running `200` MCTS-Solver simulations.

What is `PureMCTSBaseline`? It is a **fast, non-neural, zero-learned-parameter MCTS player** whose `predict()` method combines:
- A uniform policy prior over legal columns, and
- A vectorized GPU/CPU heuristic leaf evaluator scanning all **69 four-in-a-row windows** ($24$ horizontal + $21$ vertical + $12$ `/` diagonal + $12$ `\` anti-diagonal) for unblocked 3-in-a-rows (`my_3s`, `opp_3s`), 2-in-a-rows (`my_2s`, `opp_2s`), and center-column stones (`my_center`, `opp_center`):

  $$
  V_{\text{anchor}}(s) = \tanh\!\Big(0.25 \cdot (n_{\text{my},3} - n_{\text{opp},3}) + 0.04 \cdot (n_{\text{my},2} - n_{\text{opp},2}) + 0.06 \cdot (n_{\text{my,ctr}} - n_{\text{opp,ctr}})\Big)
  $$

  paired with the exact same MCTS-Solver tree search!

Because `PureMCTSBaseline` never changes, pinning its rating at **`0 Elo`** gives an **absolute Elo ruler** across the entire training run (`compute_elo_diff` in [`src/evaluate.py`](../src/evaluate.py#L1311-L1317)):

$$
\Delta \text{Elo} = 400 \log_{10}\!\left(\frac{s}{1 - s}\right), \qquad s = \text{clamp}\!\left(\frac{W + 0.5 D}{N_{\text{games}}},\; 10^{-3},\; 1 - 10^{-3}\right)
$$

- $s = 0.50 \implies \mathbf{0\text{ Elo}}$ (even match)
- $s = 0.76 \implies \mathbf{+200\text{ Elo}}$
- $s = 0.91 \implies \mathbf{+400\text{ Elo}}$
- $s = 0.99 \implies \mathbf{+800\text{ Elo}}$

---

## 5. Practical Guide to Reading `benchmark.py` Output

When you run:
```bash
uv run benchmark.py --arena-anchor
```
you get a complete diagnostic table across all 6,000 positions at both `0` sims and `400` sims, plus Anchor Elo:

| What You Observe | What It Tells You About Your Model |
| :--- | :--- |
| **`0-sim` WDL Acc & Opt Move % rising steadily** | Search knowledge is successfully distilling into raw convolutional intuition (`AlphaZeroNet` weights). |
| **Big jump from `0-sim` $\to$ `400-sim` (e.g. `+15%` Opt Move %)** | MCTS + MCTS-Solver is actively amplifying the network via tactical lookahead. |
| **`Test_L3_R1` Draw Recall % rising from `0%` $\to$ `80%+`** | The 3-logit WDL head has calibrated draw probabilities in locked endgames rather than over-predicting wins/losses. |
| **`Test_L1_R1` / `Test_L2_R1` Blunder % `< 2%` at `400-sim`** | Tactical 1-to-13-ply threat detection and MCTS-Solver refutations are rock solid. |
| **`Test_L1_R3` Opt Move % is lowest of the 6 splits** | Expected! `Test_L1_R3` requires 28–42 plies of whole-game foresight—even strong AlphaZero nets improve here last. |

---

## 6. Summary & Next Steps

1. **Never rely on self-play loss alone**: Use `gamesolver.org` oracle splits + Paired-Opening Arena to measure true strength.
2. **Undiluted `-best_child.q_value`**, **SHA256-cached exact child WDL (`Optimal Move %`, `Blunder %`)**, and **3-class WDL recall** pinpoint exact strengths and weaknesses across openings, midgames, and drawn endgames.
3. **Model-Isolated Paired-Opening Arena** + **`PureMCTSBaseline` Anchor Elo** provide clean, color-balanced match ratings.

There is one major piece of the puzzle left: **how can a single machine run ~12,000,000 MCTS simulations per iteration (`1,000` games $\times \sim 20$ moves $\times 600$ sims/move) in seconds while remaining 100% valid Pure Python?** Let's look under the hood of the **Cython Engine (`src/self_play.py`) & Bitboard Solver**!

---

⬅️ **Previous**: **[Chapter 04: Self-Play Data Generation, Replay Buffer & Training Pipeline](./04-self-play-and-training.md)**  
👉 **Next**: **[Chapter 06: High-Performance Cython Engine & Bitboard Solver](./06-cython-engine-and-architecture.md)**  
🏠 **Back to Index**: **[AlphaFour Documentation Hub (`docs/README.md`)](./README.md)**
