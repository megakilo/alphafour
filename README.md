# alphafour

An AlphaZero-style AI for Connect Four trained entirely through self-play on a single GPU (or Apple Silicon MPS / CPU). Combines a Squeeze-and-Excitation ResNet with a 3-logit Win/Draw/Loss (WDL) value head and a Rayon-parallelized Rust MCTS-Solver engine (`alphafour_engine`).

## Setup

Requires Python 3.12+, a Rust toolchain (`cargo`), and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

`uv sync` compiles and installs the `alphafour_engine` PyO3 extension in release mode alongside the Python dependencies. A pure-Python fallback is also included for every MCTS, self-play, and tactical solver path.

## Training

```bash
# Train with defaults (auto-uses all visible CUDA GPUs via DDP; resumes automatically if checkpoints/ exists)
uv run main.py

# Force single-GPU execution on a multi-GPU machine
uv run main.py --gpus 1

# Or launch explicitly with torchrun
uv run torchrun --nproc_per_node=2 main.py

# Custom training run
uv run main.py --iterations 50 --games-per-iteration 500 --training-simulations 800 --eval-simulations 400

# Start fresh from scratch
uv run main.py --no-resume
```

### Training Options (`main.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | `150` | Number of self-play + training iterations |
| `--games-per-iteration` | `1000` | Self-play games generated per iteration |
| `--training-simulations` | `600` | MCTS simulations per move during self-play |
| `--eval-simulations` | `400` | MCTS simulations per move for periodic dataset, Arena, and Anchor evaluation |
| `--eval-interval` | `5` | Run MCTS dataset eval, Paired-Opening Arena, and Anchor Elo every N iterations |
| `--arena-games` | `40` | Number of duplicate paired-opening games per Arena / Anchor match |
| `--epochs` | `2` | Training epochs per iteration (`4` during the first 20 iterations) |
| `--batches-per-epoch` | auto (`<=500`) | Mini-batches per epoch (auto-scaled to buffer size, capped at 500) |
| `--batch-size` | `256` | Per-GPU training mini-batch size (`global_batch = batch_size × GPUs`) |
| `--lr` | `0.001` | Peak learning rate (linear warmup + single-cycle cosine decay to `1e-5`) |
| `--gpus` | auto | Number of CUDA GPUs for DDP (`None`/`0` = auto-detect; `1` = single GPU) |
| `--compile` / `--no-compile` | `--compile` | Enable `torch.compile` + `channels_last` on CUDA for self-play, eval, and training |
| `--res-blocks` | `10` | Number of Squeeze-and-Excitation residual blocks |
| `--filters` | `128` | Number of convolutional filters in the residual tower |
| `--checkpoint-dir` | `checkpoints` | Directory for model checkpoints and replay buffer state |
| `--no-resume` | — | Start fresh instead of resuming from the latest checkpoint |
| `--buffer-capacity` | `300000` | Contiguous ring-buffer capacity (positions) |

Checkpoints are written atomically to `checkpoints/checkpoint_NNNN.pt` (lightweight model, optimizer, and LR scheduler state) while the replay buffer is persisted separately to an overwritten `checkpoints/replay_buffer.pt`.

## Playing

```bash
# Human plays first (Red ●)
uv run play.py

# Computer plays first
uv run play.py --computer-first

# Display column win-probability hints with tree reuse across turns
uv run play.py --hints --simulations 800

# Play against a specific checkpoint
uv run play.py --checkpoint checkpoints/checkpoint_0100.pt
```

### Play Options (`play.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--computer-first` | — | Let the AI make the opening move |
| `--hints` | — | Display estimated win % above each valid column |
| `--checkpoint` | latest | Path to a specific checkpoint file |
| `--checkpoint-dir` | `checkpoints` | Directory searched when `--checkpoint` is omitted |
| `--simulations` | `200` | MCTS simulations per move |
| `--res-blocks` | `10` | Residual blocks in model |
| `--filters` | `128` | Convolutional filters in model |
| `--cpu` | — | Force CPU inference even when a GPU is available |

## Benchmarking & Verification

Evaluate checkpoints against the [gamesolver.org test suite](http://blog.gamesolver.org/solving-connect-four/02-test-protocol/) (6,000 exact positions across 6 game-phase/difficulty splits) as well as paired-opening Arena matches:

```bash
# Evaluate latest checkpoint with both 0-sim raw network and 400-sim MCTS
uv run benchmark.py

# Evaluate a specific checkpoint and measure Anchor Elo vs PureMCTSBaseline
uv run benchmark.py --checkpoint checkpoints/checkpoint_0150.pt --eval-simulations 400 --arena-anchor

# Evaluate MCTS only on a specific test split or position limit
uv run benchmark.py --no-both --eval-simulations 200 --test-files testdata/Test_L3_R1 --limit 200

# Head-to-head duplicate paired-opening Arena between two checkpoints
uv run benchmark.py --checkpoint checkpoints/checkpoint_0150.pt --arena-vs checkpoints/checkpoint_0100.pt
```

### Benchmark Options (`benchmark.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | latest | Path to checkpoint file to evaluate |
| `--checkpoint-dir` | `checkpoints` | Directory searched when `--checkpoint` is omitted |
| `--eval-simulations` | `400` | MCTS simulations per move for search evaluation |
| `--both` / `--no-both` | `--both` | Run both `0`-sim raw network intuition and `--eval-simulations` MCTS |
| `--testdata-dir` | `testdata` | Directory containing `Test_L*_R*` files |
| `--test-files` | all `Test_*` | Explicit dataset file path(s) to evaluate |
| `--limit` | `None` | Cap positions evaluated per dataset file |
| `--arena-anchor` | — | Run Paired-Opening Arena against `PureMCTSBaseline` and report Anchor Elo |
| `--arena-vs` | `None` | Opponent checkpoint path for head-to-head Paired-Opening Arena |
| `--arena-games` | `40` | Number of duplicate paired-opening games in Arena evaluation |
| `--res-blocks` | `10` | Residual blocks in model |
| `--filters` | `128` | Convolutional filters in model |
| `--compile` / `--no-compile` | `--compile` | Enable `torch.compile` + `channels_last` on CUDA |

### Verification Metrics

- **WDL Accuracy & Per-Class Win / Draw / Loss Recall**: Evaluates 3-class outcome classification against `sign(true_score) ∈ {-1, 0, +1}` using `argmax(wdl_probs)` for raw network predictions (`0` sims) and symmetric `±(1/3)` thresholds on undiluted best-child Q-values (`-best_child.q_value`) for MCTS search (`>0` sims).
- **Move Optimality % & Blunder %**: Uses the exact 49-bit Pascal-Pons bitboard alpha-beta solver (`solve_batch_children_wdl`, cached by SHA256 in `testdata/.solver_cache/`) to compute exact W/D/L outcomes for all 7 child columns of every test position. A chosen move is **optimal** iff its child W/D/L matches the best legal move (`sign(true_score)`), and a **blunder** iff it strictly degrades the game-theoretic outcome on a non-already-lost position.
- **Value MSE**: Mean squared error of predicted values in `[-1, +1]` against exact game-theoretic W/D/L targets `{-1.0, 0.0, +1.0}`.
- **Model-Isolated Paired-Opening Arena & Anchor Elo**: Active Arena games are partitioned by root mover at every ply so each model evaluates 100% of its own MCTS search tree without cross-model leaf contamination. Balanced 2–4 ply tactical-filtered openings are played twice (swapping colors) with `temperature = 0.0`, alongside absolute Elo tracking against `PureMCTSBaseline`.

## Architecture

```
Input (3×6×7) ─► Conv 3×3 (128) + BN + ReLU ─► 10 × SE-ResBlock (128)
                                                      │
                   ┌──────────────────────────────────┴──────────────────────────────────┐
                   ▼                                                                     ▼
    Policy Head: Conv 1×1 (32) + BN + ReLU                               Value Head: Conv 1×1 (16) + BN + ReLU
          └─► Linear(32×6×7 → 7)                                               └─► Linear(16×6×7 → 256) + ReLU
          └─► Masked Softmax over legal cols                                   └─► Linear(256 → 3) [Loss, Draw, Win]
                                                                               └─► V = P(Win) - P(Loss) ∈ [-1, +1]
```

- **Rust Engine (`alphafour_engine`)**: Parallelized across CPU cores with Rayon (`RustBatchedSelfPlay`, `RustBatchedMCTS`, and exact bitboard solver), featuring barrier-free continuous self-play batching (`uint8` states + packed `float32[B, 8]` policy/value outputs), inline `Copy` MCTS arenas (`!needs_drop`), collision-aborted virtual-loss leaf batching, exact **MCTS-Solver** (`ProvenWin` / `ProvenLoss` / `ProvenDraw` propagation), **First Play Urgency (FPU)** reduction (`c_fpu = 0.20`), `dirichlet_alpha = 0.35`, stochastic 0–4 ply opening diversification, and normalized visit-count policy targets decoupled from move selection temperature (`TEMP_THRESHOLD = 10`).
- **Single-GPU Training & Inference**: `channels_last` (`NHWC`) memory layout + dynamic `torch.compile` + `bfloat16` mixed precision on CUDA across self-play, evaluation, and training; preallocated `uint8` ring-buffer `ReplayBuffer` (`300K` states in ~47 MB) with O(B) analytical inverse-CDF 2:1 linear recency sampling; decoupled `AdamW` weight decay (excluding BatchNorm and bias parameters); horizontal board-mirroring data augmentation; and linear warmup + single-cycle cosine LR schedule.

## Project Structure

```
├── main.py                  # Self-play training entry point
├── play.py                  # Interactive Human vs AI console game
├── benchmark.py             # Dataset & Arena verification CLI
├── rust/                    # PyO3 + Rayon Rust engine (alphafour_engine)
│   └── src/                 # Bitboard game, MCTS-Solver, batched self-play & solver
├── src/
│   ├── game.py              # Python Connect Four board & tensor encoding
│   ├── model.py             # SE-ResNet with 32-ch policy & 16-ch 3-logit WDL value head
│   ├── mcts.py              # Python MCTS-Solver, FPU reduction & stateful tree reuse
│   ├── self_play.py         # Batched self-play orchestration
│   ├── trainer.py           # Contiguous GPU ring buffer & BF16 WDL trainer
│   ├── evaluate.py          # Solver-cached dataset metrics, isolated Arena & Anchor Elo
│   └── utils.py             # Device selection, checkpoint & replay-buffer I/O
├── testdata/                # gamesolver.org benchmark positions (Test_L*_R*)
└── tests/                   # Rust & Python unit test suites
```

## Test Data

- Source: [gamesolver.org](http://blog.gamesolver.org/solving-connect-four/02-test-protocol/)

| Test Set (1,000 cases each) | Test Set Name | Moves Played (`nb moves`) | Remaining Moves (`nb remaining`) |
|-----------------------------|---------------|---------------------------|----------------------------------|
| `Test_L3_R1`                | End-Easy      | `28 < moves`              | `remaining < 14`                 |
| `Test_L2_R1`                | Middle-Easy   | `14 < moves <= 28`        | `remaining < 14`                 |
| `Test_L2_R2`                | Middle-Medium | `14 < moves <= 28`        | `14 <= remaining < 28`           |
| `Test_L1_R1`                | Begin-Easy    | `moves <= 14`             | `remaining < 14`                 |
| `Test_L1_R2`                | Begin-Medium  | `moves <= 14`             | `14 <= remaining < 28`           |
| `Test_L1_R3`                | Begin-Hard    | `moves <= 14`             | `28 <= remaining`                |
