# 06. High-Performance Rust Engine (`alphafour_engine`) & Exact Bitboard Solver

> **Code references in this chapter**:
> - Safe PyO3 Python bindings & zero-copy NumPy transfers: [`rust/src/lib.rs`](../rust/src/lib.rs)
> - 49-bit Connect Four bitboard engine: [`rust/src/game.rs`](../rust/src/game.rs)
> - Inline `Copy` arena MCTS-Solver, virtual losses & collision-abort: [`rust/src/mcts.rs`](../rust/src/mcts.rs)
> - Barrier-free continuous Rayon self-play engine: [`rust/src/batch_self_play.rs`](../rust/src/batch_self_play.rs)
> - Parallel batched MCTS evaluation engine: [`rust/src/batch_mcts.rs`](../rust/src/batch_mcts.rs)
> - Exact Pascal-Pons bitboard alpha-beta solver & lock-free TT: [`rust/src/solver.rs`](../rust/src/solver.rs)

---

## 1. Why Does AlphaZero Need a Systems-Level Search Engine?

Suppose you put `AlphaZeroNet` on a fast GPU, where evaluating a batch of 1,024 board positions takes `< 1 ms`. You might expect training to fly!

Yet if you run MCTS in pure Python, **your GPU will sit 95% idle** waiting for the CPU! Look at the CPU arithmetic for just **one** training iteration (`1,000` games $\times \sim 20$ moves/game $\times 600$ simulations/move = **`12,000,000` MCTS simulations**):
- Every simulation walks 5–20 tree levels computing PUCT/FPU scores, copies board states on expansion, checks 4-in-a-row win conditions across 7 children, and backpropagates values up parent pointers.
- In pure Python, 12 million object allocations, dictionary lookups, NumPy array copies, and Global Interpreter Lock (GIL) acquisitions take minutes per iteration.

`alphafour` solves this with a clean **two-tier architecture**:
1. **Python + PyTorch (`src/`)**: Owns neural network definition, GPU mixed-precision/compiled inference, VRAM ring-buffer training, DDP collectives, CLI ergonomics, **plus a 100%-feature-complete pure-Python fallback** for every MCTS, self-play, and tactical solver routine (making the algorithm easy to read, debug, and cross-test!).
2. **Rust + PyO3 + Rayon (`rust/src/`)**: Compiles into the native `alphafour_engine` Python extension (`opt-level = 3`, `lto = true`, **zero `unsafe` blocks**), releasing Python's GIL (`py.allow_threads`) and parallelizing bitboard MCTS and exact alpha-beta solving across all CPU cores via [Rayon](https://github.com/rayon-rs/rayon).

---

## 2. The 49-Bit Pascal-Pons Bitboard (`rust/src/game.rs`)

Instead of storing a $6 \times 7$ array of bytes on the heap, `ConnectFour` in [`rust/src/game.rs`](../rust/src/game.rs#L21-L31) packs each player's stones into a **single 64-bit CPU register (`u64`)** (`bb_p1` and `bb_p2`).

### 2.1 Column-Major Layout with Top Sentinel Bits

Each of the 7 columns uses **`H1 = ROWS + 1 = 7` bits**: 6 bits for the playable rows (`row = 0` bottom .. `row = 5` top) plus **1 permanently-zero sentinel bit** at the top of each column (`row = 6`), using $7 \times 7 = 49$ bits of a `u64`:

```text
  Sentinel row 6 (ALWAYS 0!):   [ 6]  [13]  [20]  [27]  [34]  [41]  [48]
                                ────  ────  ────  ────  ────  ────  ────
  Top playable row 5:            5    12    19    26    33    40    47
               row 4:            4    11    18    25    32    39    46
               row 3:            3    10    17    24    31    38    45
               row 2:            2     9    16    23    30    37    44
               row 1:            1     8    15    22    29    36    43
  Bottom       row 0:            0     7    14    21    28    35    42
                               Col 0 Col 1 Col 2 Col 3 Col 4 Col 5 Col 6
```

### 2.2 Why the Sentinel Bit Is Pure Magic: $O(1)$ Win Detection

Look at the bit-index stride between adjacent squares in all four winning directions:
- **Vertical (`|`)**: `+1` (moving up 1 row within the same column)
- **Horizontal (`—`)**: `+7` (`H1`: moving 1 column right on the same row)
- **Diagonal (`/`)**: `+8` (`H1 + 1`: moving 1 column right and 1 row up)
- **Anti-Diagonal (`\`)**: `+6` (`H1 - 1`: moving 1 column right and 1 row down)

*Wait*: Without the sentinel bits at `[6, 13, 20, 27, 34, 41, 48]`, shifting vertically by `+1` from the top of Column 0 (`bit 5`) would wrap directly into the bottom of Column 1 (`bit 6` if packed tightly)—creating fake vertical/diagonal 4-in-a-rows across column boundaries!

Because every sentinel bit is **always `0`**, any vertical or diagonal run that tries to cross the top/bottom of a column hits a `0` sentinel bit and breaks automatically! Checking the *entire board* for a 4-in-a-row takes just **two shifts and two bitwise ANDs per direction** (`ConnectFour::has_won` in [`rust/src/game.rs`](../rust/src/game.rs#L138-L149)):

```rust
pub(crate) fn has_won(bb: u64) -> bool {
    const DIRECTIONS: [u32; 4] = [1, 7, 8, 6]; // vertical, horizontal, /, \
    for &dir in &DIRECTIONS {
        let m = bb & (bb >> dir);       // Adjacent pairs of stones
        if m & (m >> (2 * dir)) != 0 {  // Two adjacent pairs = 4-in-a-row!
            return true;
        }
    }
    false
}
```

### 2.3 Gravity Arithmetic & Unique 49-Bit Position Key

In [`rust/src/solver.rs`](../rust/src/solver.rs#L47-L49), let:
- `position`: `u64` bitboard of the current player's stones.
- `mask = bb_p1 | bb_p2`: `u64` bitboard of *all* stones on the board.
- `BOTTOM_MASK`: `1` at the bottom cell of every column (`bits 0, 7, 14, 21, 28, 35, 42`).

Because stones in each column form a contiguous run of `1`s starting at row 0 (e.g., `0b000111` for 3 stones), **adding `1` to the bottom of a column carries through all existing stones and lands on the exact next empty row** (`0b000111 + 0b000001 = 0b001000`)!
- **All legal next stone positions in parallel**: `possible = (mask + BOTTOM_MASK) & BOARD_MASK` (if a column has 6 stones `0b0111111`, adding `1` lands on the sentinel bit `0b1000000`, which `& BOARD_MASK` masks out!).
- **Unique 49-bit Transposition Key (`key = position + mask`)**: Adding `mask` to `position` propagates a carry inside each column to `row = height[c]`, uniquely encoding both the column height AND every stone's owner below it in **49 bits**!

---

## 3. Inline Zero-Drop Arena MCTS Trees (`rust/src/mcts.rs`)

In languages like Python or naive C++/Rust, each MCTS node is a separate heap allocation containing a heap-allocated `Vec<Box<Node>>` of children. Creating and freeing 1,000 trees $\times$ 10,000 nodes = **10,000,000 `malloc`/`free` calls per iteration**!

Look at `MCTSNode` and `MCTSTree` in [`rust/src/mcts.rs`](../rust/src/mcts.rs#L45-L113):
1. **Contiguous Arena (`Vec<MCTSNode>`)**: Every node in a game's search tree lives contiguously inside one flat `Vec<MCTSNode>`, referenced by integer `usize` index (`root = 0`).
2. **Inline Stack-Sized `ChildrenList`**: Because Connect Four has at most `COLS = 7` legal moves, `ChildrenList` stores `buf: [(u8, usize); 7]` and `len: u8` **inline inside `MCTSNode`**—zero per-node `Vec` heap allocations!
3. **Compile-Time Zero-Drop Guarantee**:
   ```rust
   #[derive(Clone, Copy)]
   pub struct MCTSNode { ... }

   const _: () = assert!(!std::mem::needs_drop::<MCTSNode>());
   ```
   Because `ConnectFour`, `ChildrenList`, and `MCTSNode` are all plain `Copy` structs with no heap pointers, dropping an entire 20,000-node MCTS tree doesn't walk 20,000 nodes calling destructors—it frees **one** contiguous buffer in $O(1)$ time!

---

## 4. Parallel Batched MCTS: Virtual Losses & Collision-Abort

### 4.1 The Batching Dilemma
MCTS wants to evaluate **1 leaf**, update Q-values, and use that updated Q-value to guide the very next simulation. Meanwhile, a GPU achieves peak throughput when evaluating **1,000+ leaves at once**!

`alphafour` batches across **two dimensions simultaneously**:
1. **Inter-game batching**: Running `N = 1,000` self-play games concurrently across Rayon threads.
2. **Intra-game batching (`LEAVES_PER_GAME = 16`)**: Selecting up to **16 leaves per game tree** in a single batch!

### 4.2 Virtual Losses (`apply_virtual_loss` / `revert_virtual_loss`)
Wait—if PUCT is deterministic, won't selecting 16 leaves from the same tree without running the GPU in between just pick the **exact same leaf 16 times**?

Yes—unless we temporarily penalize in-flight paths using **Virtual Losses** ([`rust/src/mcts.rs`](../rust/src/mcts.rs#L218-L231)):
- Whenever leaf selection picks an unexpanded leaf `L`, `apply_virtual_loss(L)` increments `virtual_losses += 1` on `L` and every ancestor up to `root`.
- During subsequent selections in the *same* batch, `best_child` treats each virtual loss as if **that trajectory had temporarily lost the game (`+1.0` for the child/opponent = `-1.0` for the parent)**:

  $$
  \tilde{N}(s') = N(s') + N_{\text{vl}}(s'), \qquad -\tilde{Q}(s') = -\frac{W(s') + N_{\text{vl}}(s')}{N(s') + N_{\text{vl}}(s')}
  $$

  while reducing the exploration term $\frac{\sqrt{N(s) + N_{\text{vl}}(s)}}{1 + N(s') + N_{\text{vl}}(s')}$ and counting children with $\tilde{N}(s') > 0$ toward FPU's `visited_policy` sum ([`rust/src/mcts.rs`](../rust/src/mcts.rs#L186-L198)).
- This temporary pessimism deflects the 2nd, 3rd, ..., 16th traversals down **different promising branches** of the tree!
- As soon as the GPU returns true `(policy, value)` evaluations, `revert_virtual_loss(L)` subtracts `1` along the path before `backpropagate(L, bval)` records the real statistics—leaving zero permanent trace!

### 4.3 Collision-Abort: Guaranteeing Zero Duplicate Leaves
What happens at the very beginning of a turn when the root has only **7 children** (`depth = 1`), and we ask for `LEAVES_PER_GAME = 16` leaves? Even with virtual losses, after selecting all 7 children, an 8th selection *must* land on one of those 7 unexpanded children again!

Evaluating the exact same board twice in the same GPU batch wastes GPU compute, and expanding the same node twice would overwrite children! Look at how `select_multiple_leaves` ([`rust/src/mcts.rs`](../rust/src/mcts.rs#L343-L350)) handles this cleanly:

```rust
// Collision-abort: never queue duplicate leaves in the same batch!
if self.nodes[leaf].virtual_losses > 0 {
    break;
}
self.apply_virtual_loss(leaf);
leaves.push(leaf);
```

And as a second line of defense, `MCTSTree::expand` starts with an idempotent guard (`if self.nodes[node_idx].is_expanded { return; }`). Together, every leaf sent to the GPU within a tree is **100% unique**!

---

## 5. Barrier-Free Continuous Self-Play (`collect_continuous_u8`)

### Why Lockstep Ply Barriers Waste Throughput
In a traditional batched self-play loop ("Step-Wise Mode"), all 1,000 games march in lockstep ply by ply:
1. Expand roots for all 1,000 games $\to$ add noise.
2. Loop `collect_leaves` $\leftrightarrow$ GPU until **every** game finishes all 600 simulations for Move $t$.
3. Advance all 1,000 games to Move $t+1$ $\to$ repeat.

Why is a ply barrier inefficient?
- Game A might hit an `MCTS-Solver` forced win after 32 simulations!
- Game B might have a wide tree that collects 16 leaves per batch and finishes 600 simulations in $600 / 16 = 38$ batches.
- Game C might be in a forced narrow corridor where collision-abort yields only 2 leaves per batch, needing $150+$ batches!
Under a lockstep barrier, **999 games sit completely idle waiting for Game C**—shrinking the GPU batch size from `12,000` down to `2` at the end of *every single move*!

### How Barrier-Free Continuous Mode Works ([`rust/src/batch_self_play.rs`](../rust/src/batch_self_play.rs#L605-L877))
`collect_continuous_u8()` + `apply_continuous_packed()` eliminates ply barriers completely:

```text
Game Slot 0: [Move 4: batch..batch..DONE!]──►advance──►[Move 5: rebase..batch..batch..]
Game Slot 1: [Move 2: PROVEN WIN!]──►advance──►[Move 3: root_init]──►[Move 3: batch...]
Game Slot 2: [Move 7: batch..batch..DONE!]──►advance──►[TERMINAL! emit 2× examples ✓]
                     │                                        │
                     └────────── All packed together ─────────┘
                            into ONE continuous GPU batch:
                       uint8[K, 3, 6, 7] ──► GPU ──► float32[K, 8]
```

1. **Asynchronous Per-Slot Turn Advancement (`advance_ready_slots`)**: As soon as *any* individual slot reaches `sims_done >= num_simulations` (or proves its root via `MCTS-Solver`), it immediately records its smooth policy target, samples its move, updates its board, emits mirrored examples if terminal, or `rebase`s its MCTS tree onto the next ply—**without waiting for any other game**!
2. **Unified Root & Leaf Batching (`pending_root_init`)**: If a newly advanced game reaches an unexpanded root (e.g., a low-visit surprise move wasn't expanded yet), its root evaluation is packed right alongside all other games' MCTS leaf evaluations in the **exact same `uint8[K, 3, 6, 7]` GPU call**!
3. **End-of-Iteration Straggler Tail Guard**: When $\ge 64$ games started and $\le 4$ active tail games remain at the very end of the entire 1,000-game run, any tail slot that has already completed $\ge 75\%$ of `num_simulations` and is yielding $\le 2$ leaves/batch (`tail_straggler_done`) finishes cleanly rather than dragging out tiny 1-leaf GPU calls.

---

## 6. Exact Bitboard Alpha-Beta Solver (`rust/src/solver.rs`)

To compute ground-truth **Optimal Move %** and **Blunder %** across 6,000 benchmark positions (`42,000` child positions, some from move 4 with 38 plies remaining!), [`rust/src/solver.rs`](../rust/src/solver.rs) implements a full **Pascal-Pons-style Connect Four W/D/L solver**.

Here are the 6 algorithmic techniques that allow it to solve Connect Four positions orders of magnitude faster than naive minimax:

### 6.1 Null-Window W/D/L Search (`can_achieve_budget`)
Because we only care whether the game-theoretic outcome $V \in \{-1, 0, +1\}$ is Win (`+1`), Draw (`0`), or Loss (`-1`), we never search a wide $[\alpha, \beta]$ window! Instead, `solve_bitboard_wdl_budget` runs at most **two boolean zero-window tests** `can_achieve(beta)`:
1. **Test `beta = 1` ("Can I force a Win, $V \ge 1$?")**: If `true` $\implies$ return **`+1`**!
2. **Test `beta = 0` ("Can I force at least a Draw, $V \ge 0$?")**: If `true` $\implies$ return **`0`**, else return **`-1`**!

### 6.2 Bitwise Threat Map & Non-Losing Move Pruning (`compute_winning_position`)
Using parallel bit shifts in all 4 directions ([`rust/src/solver.rs`](../rust/src/solver.rs#L208-L239)), `compute_winning_position(pos, mask)` computes in ~20 CPU instructions a 49-bit bitmap of **every empty cell on the entire board that would immediately complete a 4-in-a-row**!
From `possible` (legal moves right now), `winning_p` (my winning squares), and `opponent_win` (opponent's winning squares), the solver prunes massively *before* recursing:
1. **Immediate Win**: `(possible & winning_p) != 0` $\implies$ return `true` (`+1`) immediately!
2. **Forced Block & Double-Threat Detection**: Let `forced = possible & opponent_win` (playable squares where the opponent threatens 4-in-a-row next turn).
   - If `forced` has **$\ge 2$ bits set** (`(forced & (forced - 1)) != 0`), the opponent has two simultaneous immediate threats—we can only block one, so we **lose immediately** (`false`)!
   - If `forced` has **1 bit set**, we *must* play that exact column (`candidate_mask = forced`), pruning the other 6 columns!
3. **Never Play Directly Beneath an Opponent Threat!**:
   Look at `non_losing = candidate_mask & !(opponent_win >> 1)` ([`rust/src/solver.rs`](../rust/src/solver.rs#L384)):
   `opponent_win >> 1` shifts every opponent winning square **down by 1 row**! Any playable move in `opponent_win >> 1` would support the opponent's winning square on the very next turn ("giving away the game"), so those suicide moves are eliminated in a single bitwise instruction!

### 6.3 Threat-Creation Move Ordering & Shallow Tactical Pre-Check
- **Move Ordering**: Alpha-beta pruning is fastest when the best move is searched first! At interior nodes (`work > 6`), candidate columns are sorted by **how many new winning threats `compute_winning_position(pos | move_bit, ...).count_ones()` the move creates**, breaking ties by center-outward priority `[3, 2, 4, 1, 5, 0, 6]` (since center columns participate in the most 4-in-a-row windows!).
- **Shallow Tactical Pre-Check (`tactical_can_win` / `tactical_is_forced_loss`)**: Before plunging 35 plies deep down a quiet column, a microsecond 8-to-9-ply forced-threat search checks if there is a short tactical mate right away!

### 6.4 Lock-Free 128 MiB `AtomicU64` TwoBig1 Transposition Table
In Connect Four, playing columns `4, 3, 5` reaches the exact same board as `5, 3, 4` (**transposition**). Furthermore, early positions (`moves <= 14`) are canonicalized with their horizontal mirror `min(k, mirror_key(k))` (`tt_key` in [`rust/src/solver.rs`](../rust/src/solver.rs#L189-L197)).

To share cached bounds across all Rayon threads **with zero mutex locks and zero data tearing**, `TranspositionTable` ([`rust/src/solver.rs`](../rust/src/solver.rs#L59-L169)) packs an entire entry into a single 64-bit `AtomicU64`:

```text
63    60 59        54   53   52    51 50    49 48                                 0
┌───────┬────────────┬──────┬────────┬────────┬───────────────────────────────────┐
│ unused│ work (6b)  │valid │upper+1 │lower+1 │     full 49-bit key (pos + mask)  │
└───────┴────────────┴──────┴────────┴────────┴───────────────────────────────────┘
```

- **Zero Hash Collisions (Full Key Stored!)**: Because all 49 bits of `pos + mask` fit inside bits `0..48` alongside `lower` (2 bits), `upper` (2 bits), `valid` (1 bit), and `work` depth (6 bits), a TT lookup `if (raw & VALID_BIT) != 0 && (raw & KEY_MASK) == key` can **never** suffer a false key collision!
- **TwoBig1 Replacement Policy**: Each of the $2^{23} = 8{,}388{,}608$ buckets (`128 MiB` total) holds **two** `AtomicU64` slots:
  1. `depth_pref`: Replaced only by searches with equal or greater remaining `work` (preserving expensive deep opening/midgame subtrees from being overwritten by millions of cheap endgame leaves!).
  2. `always`: Always-replace slot capturing local transpositions in the current subtree.
- **Batch Deduplication & Shallow-First Scheduling (`solve_batch_children_wdl`)**: Deduplicates identical/mirrored child boards across the entire 6,000-position batch and sorts unique tasks in ascending `moves` order with `.with_max_len(1)` Rayon work-stealing so all CPU cores stay busy until the deepest position finishes!

---

## 7. Where to Go From Here

Congratulations—you now have a complete mental and engineering map of **AlphaFour**, from the foundational theory of Generalized Policy Iteration all the way down to `AtomicU64` bitfields and Triton-compiled `NHWC` Tensor Core kernels!

### Suggested Hands-On Experiments to Deepen Your Understanding
1. **Watch Intuition vs Search Live**:
   Run `uv run play.py --hints --simulations 0` (pure neural intuition, zero search!) and compare how it plays against `uv run play.py --hints --simulations 800`. Notice where raw intuition already spots traps and where 800 simulations find deep multi-move wins!
2. **Inspect Checkpoint Evolution**:
   Run `uv run benchmark.py --checkpoint checkpoints/checkpoint_0005.pt --no-compile` vs a later checkpoint (e.g., `checkpoint_0070.pt`) or pit them head-to-head:
   ```bash
   uv run benchmark.py --checkpoint checkpoints/checkpoint_0070.pt --arena-vs checkpoints/checkpoint_0010.pt
   ```
3. **Read the Test Suites (`tests/`)**:
   The unit tests in [`tests/test_selfplay_mcts.py`](../tests/test_selfplay_mcts.py), [`tests/test_model_trainer.py`](../tests/test_model_trainer.py), [`tests/test_evaluate.py`](../tests/test_evaluate.py), [`tests/test_rust_mcts.py`](../tests/test_rust_mcts.py), and [`tests/test_distributed.py`](../tests/test_distributed.py) construct small, isolated positions (like mate-in-1s, mate-in-2 forks, losing-parent FPU traps, and duplicate openings) that you can step through in a debugger!

---

⬅️ **Previous**: **[Chapter 05: Verification, Exact Solver Metrics & Paired-Opening Arena](./05-evaluation-and-arena.md)**  
👉 **Next**: **[Chapter 07: Practical Guide, Hyperparameter Intuition & Reference](./07-practical-guide-and-reference.md)**  
🏠 **Back to Index**: **[AlphaFour Documentation Hub (`docs/README.md`)](./README.md)**
