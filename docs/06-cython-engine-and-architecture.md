# 06. High-Performance Cython Engine (`src/self_play.py`) & Bitboard Solver

> **Code references in this chapter**:
> - Cython Pure Python Mode batched MCTS-Solver self-play engine: [`src/self_play.py`](../src/self_play.py)
> - Cython OpenMP build configuration: [`setup_cython.py`](../setup_cython.py)
> - Pascal-Pons bitboard W/D/L negamax solver & `.npz` cache: [`src/evaluate.py`](../src/evaluate.py)

---

## 1. Why Does AlphaZero Need a Systems-Level Search Engine?

Suppose you put `AlphaZeroNet` on a fast GPU, where evaluating a batch of 1,024 board positions takes `< 1 ms`. You might expect training to fly!

Yet if you run MCTS using standard Python objects, **your GPU will sit 95% idle** waiting for the CPU! Look at the CPU arithmetic for just **one** training iteration (`1,000` games $\times \sim 20$ moves/game $\times 600$ simulations/move = **`12,000,000` MCTS simulations**):
- Every simulation walks 5–20 tree levels computing PUCT/FPU scores, copies board states on expansion, checks 4-in-a-row win conditions across 7 children, and backpropagates values up parent pointers.
- In standard Python, 12 million object allocations, dictionary lookups, NumPy array copies, and Global Interpreter Lock (GIL) acquisitions take minutes per iteration.

`alphafour` solves this using **Cython Pure Python Mode** in [`src/self_play.py`](../src/self_play.py):
1. **Single Valid `.py` Source File**: Uses `import cython`, `@cython.cfunc`, `@cython.cclass`, `cython.struct`, and `cython.nogil` + `cython.parallel.prange` — zero `.pyx` or `.pxd` files! When uncompiled (`is_cython_compiled() == False`), `src/self_play.py` executes cleanly as 100%-valid Pure Python; when compiled via `setup_cython.py`, it compiles to a single `-O3 -march=native -fopenmp` C translation unit (`is_cython_compiled() == True`).
2. **PyTorch GPU Inference & Training (`src/model.py`, `src/trainer.py`)**: Pairs `CythonBatchedSelfPlay` (and `CythonBatchedMCTS` for evaluation/Arena search) with `AlphaZeroNet.predict_packed` (`uint8[B, 3, 6, 7]` H2D + `float32[B, 8]` D2H), `channels_last` BF16 mixed precision, and dynamic `torch.compile`.

---

## 2. The 49-Bit Pascal-Pons Bitboard (`src/self_play.py`, `src/evaluate.py`)

Instead of storing a $6 \times 7$ array of bytes on the heap for every MCTS node, [`src/self_play.py`](../src/self_play.py) packs each player's stones into a **single 64-bit unsigned integer (`cython.ulonglong`)** (`bb_p1` and `bb_p2`).

### 2.1 Column-Major Layout with Top Sentinel Bits

Each of the 7 columns uses **`H1 = ROWS + 1 = 7` bits**: 6 bits for the playable rows (`row = 0` bottom .. `row = 5` top) plus **1 permanently-zero sentinel bit** at the top of each column (`row = 6`), using $7 \times 7 = 49$ bits of a 64-bit register:

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

### 2.2 Why the Sentinel Bit Enables $O(1)$ Branchless Win Detection

Look at the bit-index stride between adjacent squares in all four winning directions:
- **Vertical (`|`)**: `+1` (moving up 1 row within the same column)
- **Horizontal (`—`)**: `+7` (`H1`: moving 1 column right on the same row)
- **Diagonal (`/`)**: `+8` (`H1 + 1`: moving 1 column right and 1 row up)
- **Anti-Diagonal (`\`)**: `+6` (`H1 - 1`: moving 1 column right and 1 row down)

Because every sentinel bit at `[6, 13, 20, 27, 34, 41, 48]` is **always `0`**, any vertical or diagonal run that tries to cross a column boundary hits a `0` sentinel bit and breaks automatically! Checking the *entire board* for a 4-in-a-row takes just shifts and bitwise ANDs (`_has_won` in [`src/self_play.py`](../src/self_play.py)):

```python
@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _has_won(bb: cython.ulonglong) -> cython.bint:
    m1: cython.ulonglong = bb & (bb >> 1)
    m2: cython.ulonglong = bb & (bb >> 7)
    m3: cython.ulonglong = bb & (bb >> 8)
    m4: cython.ulonglong = bb & (bb >> 6)
    return (
        (m1 & (m1 >> 2))
        | (m2 & (m2 >> 14))
        | (m3 & (m3 >> 16))
        | (m4 & (m4 >> 12))
    ) != 0
```

---

## 3. Flat C-Struct MCTS Arenas & Double-Buffered Rebasing

In naive Python/C++, each MCTS node is a separate heap allocation containing a dynamically allocated array of children. Creating and freeing 1,000 trees $\times$ 10,000 nodes = **10,000,000 `malloc`/`free` calls per iteration**!

Look at `MCTSNode` and `SelfPlaySlot` in [`src/self_play.py`](../src/self_play.py):
1. **Flat Contiguous Arena (`cython.pointer[MCTSNode]`)**: Every node in a game slot's search tree lives contiguously inside a preallocated C `MCTSNode*` buffer, referenced by 32-bit `cython.int` indices (`root = 0`).
2. **Contiguous Sibling Ranges (`first_child` + `num_children`)**: Expansion allocates all legal children (`<= 7`) contiguously in one bump of `nodes_len`—zero per-node heap allocations!
3. **Double-Buffered Pointer-Swap Subtree Rebasing**: Each `SelfPlaySlot` owns both `nodes` and `scratch_nodes` buffers. When advancing a move, BFS compaction copies the chosen child's subtree into `scratch_nodes[0..]` and swaps the `nodes` $\leftrightarrow$ `scratch_nodes` pointers in $O(\text{subtree})$ with zero `malloc`/`free`!

---

## 4. Parallel Batched MCTS: Virtual Losses & Collision-Abort

`CythonBatchedSelfPlay` batches across **two dimensions simultaneously** under `with cython.nogil:` and OpenMP `prange`:
1. **Inter-game batching**: Running `N = 1,000` self-play slots concurrently across OpenMP threads.
2. **Intra-game batching (`LEAVES_PER_GAME = 16`)**: Selecting up to **16 leaves per game tree** in a single GPU batch!

### Virtual Losses & Collision-Abort
- Whenever leaf selection picks an unexpanded leaf `L`, `_apply_virtual_loss` increments `virtual_losses += 1` along `L` up to `root`.
- Subsequent selections in the same batch treat each virtual loss as a temporary loss (`+1.0` for the child = `-1.0` for the parent):

  $$
  \tilde{N}(s') = N(s') + N_{\text{vl}}(s'), \qquad -\tilde{Q}(s') = -\frac{W(s') + N_{\text{vl}}(s')}{N(s') + N_{\text{vl}}(s')}
  $$

  deflecting traversals `2..16` down distinct promising branches!
- **Collision-Abort**: If traversal reaches a leaf that already has `virtual_losses > 0`, selection breaks immediately so **zero duplicate leaves** are ever queued in the same GPU batch.
- Once the GPU returns packed `(policy, value)` results, virtual losses are reverted before `_backpropagate` records the true value (or exact proven MCTS-Solver outcome).

---

## 5. Barrier-Free Continuous Self-Play (`collect_continuous_u8`)

`collect_continuous_u8()` + `apply_continuous_packed()` eliminates lockstep ply barriers completely:

```text
Game Slot 0: [Move 4: batch..batch..DONE!]──►advance──►[Move 5: rebase..batch..batch..]
Game Slot 1: [Move 2: PROVEN WIN!]──►advance──►[Move 3: root_init]──►[Move 3: batch...]
Game Slot 2: [Move 7: batch..batch..DONE!]──►advance──►[TERMINAL! emit 2× examples ✓]
                     │                                        │
                     └────────── All packed together ─────────┘
                            into ONE continuous GPU batch:
                       uint8[K, 3, 6, 7] ──► GPU ──► float32[K, 8]
```

1. **Asynchronous Per-Slot Turn Advancement (`_advance_ready_slots_c`)**: As soon as *any* slot reaches `sims_done >= num_simulations` (or proves its root via `MCTS-Solver`), it records its visit-count policy target, samples its move, emits 2× horizontally mirrored examples if terminal, or rebases its tree onto the next ply—without waiting for any other game!
2. **Unified Root & Leaf Batching (`pending_root_init`)**: Unexpanded roots are evaluated inside the exact same `uint8[K, 3, 6, 7]` GPU batch as interior MCTS leaves.
3. **In-Engine PRNG & Dirichlet Noise**: Each `SelfPlaySlot` carries its own 64-bit `XorShift64*` state plus Marsaglia-Tsang Gamma sampler (`_rng_gamma`), generating root Dirichlet noise and stochastic openings entirely inside `nogil` C!

---

## 6. Bitboard W/D/L Solver & Ground-Truth Cache (`src/evaluate.py`)

To grade **Optimal Move %** and **Blunder %** across benchmark positions (`get_dataset_child_wdl` in [`src/evaluate.py`](../src/evaluate.py)):
1. **SHA256 `.npz` Disk Cache (`testdata/.solver_cache/`)**: Precomputed exact `int8[N, 7]` child W/D/L matrices reload in `< 5 ms` (including `--limit` subsets via `_lookup_moves_in_dir_cache`).
2. **Bounded-Budget Pascal-Pons Bitboard Negamax (`_bb_solve_wdl`)**: Uses 49-bit `pos + mask` transposition keys, immediate 4-in-a-row win detection, simultaneous double-threat loss detection, forced-block pruning, and center-first ordering `(3, 2, 4, 1, 5, 0, 6)` to solve tactical and endgame positions directly in Python when uncached.

---

## 7. Where to Go From Here

### Suggested Hands-On Experiments
1. **Benchmark Self-Play Throughput**:
   ```bash
   uv run python benchmark_selfplay.py
   ```
   Compares pure Cython CPU engine speed, end-to-end GPU Eager vs `torch.compile` self-play throughput, and compiled `.so` vs uncompiled `.py` execution!
2. **Watch Intuition vs Search Live**:
   Run `uv run play.py --hints --simulations 0` vs `uv run play.py --hints --simulations 800`.
3. **Read the Test Suites (`tests/`)**:
   Explore [`tests/test_self_play.py`](../tests/test_self_play.py), [`tests/test_selfplay_mcts.py`](../tests/test_selfplay_mcts.py), [`tests/test_model_trainer.py`](../tests/test_model_trainer.py), [`tests/test_evaluate.py`](../tests/test_evaluate.py), and [`tests/test_distributed.py`](../tests/test_distributed.py).

---

⬅️ **Previous**: **[Chapter 05: Verification, Exact Solver Metrics & Paired-Opening Arena](./05-evaluation-and-arena.md)**  
👉 **Next**: **[Chapter 07: Practical Guide, Hyperparameter Intuition & Reference](./07-practical-guide-and-reference.md)**  
🏠 **Back to Index**: **[AlphaFour Documentation Hub (`docs/README.md`)](./README.md)**
