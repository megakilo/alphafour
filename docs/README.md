# AlphaFour Documentation & AlphaZero Learning Guide

Welcome to the **AlphaFour** documentation!

**AlphaFour** is a complete, from-scratch implementation of DeepMind's **AlphaZero** algorithm applied to **Connect Four**, combining a PyTorch Squeeze-and-Excitation ResNet (with a 3-logit Win/Draw/Loss value head) and an OpenMP-parallelized **Cython Pure Python Mode** MCTS-Solver self-play engine (`src/self_play.py`).

This documentation serves **two purposes simultaneously**:
1. **An Educational Textbook on AlphaZero**: Written for readers who know general programming and basic deep learning, but want to deeply understand *how and why* AlphaZero works—covering the core algorithm, Monte Carlo Tree Search (MCTS), neural network design, self-play data generation, training, inference, and rigorous evaluation.
2. **A Comprehensive Engineering Reference for AlphaFour**: Every concept, equation, and design trade-off is tied directly to the exact classes, functions, constants, and tensor shapes in this repository (`src/`).

---

## Recommended Reading Path

If you are new to AlphaZero, read the chapters in numerical order—each chapter builds directly on the concepts introduced in the previous one:

| Chapter | Title | What You Will Learn | Primary Code Files |
| :--- | :--- | :--- | :--- |
| **[01](./01-alphazero-fundamentals.md)** | **AlphaZero from First Principles** | Generalized Policy Iteration without human data; how neural intuition and tree search bootstrap each other; zero-sum alternating-turn perspective (and the #1 sign-flip bug); why Connect Four is an ideal laboratory. | [`src/game.py`](../src/game.py), [`src/self_play.py`](../src/self_play.py) |
| **[02](./02-mcts-deep-dive.md)** | **Core Search Algorithm: MCTS, PUCT, FPU & MCTS-Solver** | Step-by-step numerical trace of Selection, Evaluation, Expansion, and Backpropagation; the PUCT equation; First Play Urgency (FPU) reduction; Dirichlet root noise; exact **MCTS-Solver** game-theoretic proof propagation. | [`src/mcts.py`](../src/mcts.py), [`src/self_play.py`](../src/self_play.py) |
| **[03](./03-neural-network-and-inference.md)** | **Neural Network Architecture, WDL Value Head & Inference** | 3-plane canonical board encoding; SE-ResNet tower (`3.19M` params); masked policy head; why a 16-channel **3-logit Win/Draw/Loss (WDL)** head beats scalar `tanh`; 0-sim vs MCTS inference; stateful tree reuse in play; GPU optimizations (`channels_last`, BF16 AMP, `torch.compile`, packed I/O). | [`src/model.py`](../src/model.py), [`play.py`](../play.py) |
| **[04](./04-self-play-and-training.md)** | **Self-Play, Replay Buffer & Training Pipeline** | Outer iteration loop; stochastic 0–4 ply opening diversification; decoupling smooth visit-count policy targets from move-selection temperature; horizontal mirror augmentation; GPU `uint8` ring buffer with analytical $O(B)$ 2:1 recency sampling; WDL + policy cross-entropy losses; AdamW, warmup+cosine LR, checkpointing, and multi-GPU DDP. | [`main.py`](../main.py), [`src/self_play.py`](../src/self_play.py), [`src/trainer.py`](../src/trainer.py), [`src/distributed.py`](../src/distributed.py), [`src/utils.py`](../src/utils.py) |
| **[05](./05-evaluation-and-arena.md)** | **Evaluation, Exact Solver Verification & Arena** | Why training loss does *not* measure playing strength; `gamesolver.org` test splits; WDL accuracy & per-class recall; why MCTS evaluation must use undiluted `-best_child.q_value`; exact child-WDL move optimality % & blunder %; model-isolated duplicate Paired-Opening Arena; Anchor Elo vs `PureMCTSBaseline`. | [`src/evaluate.py`](../src/evaluate.py), [`benchmark.py`](../benchmark.py), [`testdata/`](../testdata) |
| **[06](./06-cython-engine-and-architecture.md)** | **High-Performance Cython Engine & Bitboard Solver** | Cython Pure Python Mode (`nogil` + OpenMP `prange`); 49-bit Pascal-Pons bitboards & $O(1)$ win detection; flat C-struct arena MCTS trees; parallel leaf batching via virtual losses + collision-abort; barrier-free continuous self-play; bitboard WDL solver & `.npz` cache. | [`src/self_play.py`](../src/self_play.py), [`setup_cython.py`](../setup_cython.py), [`src/evaluate.py`](../src/evaluate.py) |
| **[07](./07-practical-guide-and-reference.md)** | **Practical Guide, Hyperparameter Intuition & Reference** | What happens when every hyperparameter is too low vs too high; line-by-line walkthrough of training logs; CLI recipes, performance checklist, and API reference tables. | [`main.py`](../main.py), [`benchmark.py`](../benchmark.py), [`play.py`](../play.py) |

---

## System Architecture at a Glance

AlphaZero is a closed-loop self-improvement engine. Every training iteration in [`main.py`](../main.py) cycles through four stages:

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         1. SELF-PLAY DATA GENERATION                             │
│  CythonBatchedSelfPlay (OpenMP prange)   ◄──►  AlphaZeroNet.predict_packed (GPU) │
│   • 1,000 concurrent games, 600 MCTS-Solver simulations/move                     │
│   • Stochastic 0–4 ply openings + root Dirichlet noise (α=0.35, ε=0.25)          │
│   • Barrier-free continuous u8[K,3,6,7] ──► f32[K,8] GPU leaf batching           │
│   • Emits (state, soft_visit_policy, game_outcome) × 2 (horizontal mirror)       │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ ~35,000–45,000 augmented positions / iter
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         2. EXPERIENCE REPLAY BUFFER                              │
│  ReplayBuffer (300,000 capacity, ~45.2 MiB contiguous uint8 ring buffer in VRAM) │
│   • Retains ~7–9 iterations of recent self-play history                          │
│   • O(B) analytical inverse-CDF 2:1 linear recency sampling (newest = 2× weight) │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ Mini-batches of B = 256 (or 256 × GPUs)
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         3. NEURAL NETWORK TRAINING                               │
│  Trainer (BF16 Mixed Precision + channels_last + torch.compile + Fused AdamW)    │
│   • Policy Loss: Masked soft cross-entropy vs MCTS visit distribution π          │
│   • Value Loss:  3-class cross-entropy on WDL logits [Loss, Draw, Win] vs z      │
│   • Linear warmup + single-cycle cosine decay (1e-3 ──► 1e-5), grad clip 1.0     │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ Updated weights θ_{t+1}
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         4. VERIFICATION & CHECKPOINTING                          │
│  Every iteration:   0-sim opening check + 6,000-position gamesolver.org WDL eval │
│  Every 5 iters:     400-sim MCTS dataset eval (Optimality %, Blunder %, Recall)  │
│                     + Paired-Opening Anchor Arena (vs PureMCTSBaseline → Elo)    │
│                     + Relative Arena (Current vs Iter t-10)                      │
│  Atomic disk save:  checkpoints/checkpoint_NNNN.pt + checkpoints/replay_buffer.pt│
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Repository Map

```text
alphafour/
├── main.py                      # CLI entrypoint: self-play training loop (single-GPU & multi-GPU DDP)
├── play.py                      # CLI entrypoint: interactive console Human vs AI with hints & tree reuse
├── benchmark.py                 # CLI entrypoint: exact gamesolver.org verification & Arena Elo matches
├── benchmark_selfplay.py        # CLI entrypoint: Cython self-play engine throughput benchmark
├── setup_cython.py              # Build script compiling src/self_play.py with Cython + OpenMP
├── docs/                        # Educational & architectural documentation (you are here)
├── src/                         # Python & Cython Pure Python Mode package
│   ├── game.py                  # ConnectFour rules, canonical perspective, 3×6×7 encoding, ANSI display
│   ├── model.py                 # AlphaZeroNet (10× SE-ResBlock, 32-ch Policy Head, 16-ch WDL Value Head)
│   ├── mcts.py                  # Pure-Python MCTS-Solver, PUCT, FPU reduction, and stateful tree reuse
│   ├── self_play.py             # Cython Pure Python Mode batched MCTS-Solver self-play engine
│   ├── trainer.py               # Contiguous GPU ReplayBuffer, WDL+Policy losses, AMP/DDP Trainer, LR schedule
│   ├── evaluate.py              # Solver-cached dataset metrics, isolated Paired-Opening Arena, PureMCTSBaseline
│   ├── distributed.py           # Multi-GPU DDP setup, OpenMP thread partitioning, rank sharding & all_gather
│   └── utils.py                 # Device auto-detection, atomic checkpoint/buffer I/O, compiler log filters
├── testdata/                    # 6,000 exact solved positions from gamesolver.org (Test_L1_R1 .. Test_L3_R1)
└── tests/                       # Comprehensive MCTS, Cython self-play, solver, trainer & DDP test suites
```

---

## Quick-Start Cheat Sheet

### 1. Build & Install
Requires Python 3.12+, a C compiler, and [`uv`](https://docs.astral.sh/uv/):
```bash
uv sync
uv run python setup_cython.py build_ext --inplace
```
`setup_cython.py` compiles `src/self_play.py` into a native C + OpenMP extension in-place (`is_cython_compiled() == True`).

### 2. Run the Test Suites
```bash
uv run pytest
```

### 3. Train a Model from Scratch (or Resume)
```bash
# Train with defaults (150 iterations, auto-resumes from checkpoints/ if present)
uv run main.py

# Start a fresh run on a single GPU
uv run main.py --no-resume --gpus 1
```

### 4. Benchmark & Play
```bash
# Verify latest checkpoint on 6,000 exact positions (0-sim raw net + 400-sim MCTS + Anchor Elo)
uv run benchmark.py --arena-anchor

# Play against your trained AI in the terminal with real-time win-probability hints
uv run play.py --hints --simulations 800
```

---

Ready to dive in? Start with **[Chapter 01: AlphaZero from First Principles](./01-alphazero-fundamentals.md)**.
