# alphafour

An AlphaZero-style AI for Connect Four, trained entirely through self-play. Uses a ResNet neural network (10 residual blocks, 128 filters, ~3M parameters) guided by Monte Carlo Tree Search (MCTS).

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Training

```bash
# Full training run (~2-4 hours on Mac M-series for strong play)
uv run main.py

# Custom configuration
uv run main.py --iterations 50 --games-per-iteration 200 --simulations 400

# Resume from last checkpoint (default behavior)
uv run main.py

# Start fresh
uv run main.py --no-resume
```

### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | 100 | Number of training iterations |
| `--games-per-iteration` | 100 | Self-play games per iteration |
| `--simulations` | 200 | MCTS simulations per move |
| `--epochs` | 10 | Training epochs per iteration |
| `--batch-size` | 256 | Mini-batch size |
| `--lr` | 0.001 | Learning rate (cosine annealed) |
| `--workers` | CPU count - 1 | Parallel self-play workers |
| `--res-blocks` | 10 | Residual blocks in network |
| `--filters` | 128 | Convolutional filters |
| `--checkpoint-dir` | `checkpoints/` | Checkpoint directory |
| `--no-resume` | — | Start fresh training |

Checkpoints are saved to `checkpoints/checkpoint_NNNN.pt` after each iteration and include model weights, optimizer state, LR scheduler state, and the full replay buffer — so training can be stopped and resumed at any time.

## Playing

```bash
# Human goes first (default)
uv run play.py

# Computer goes first
uv run play.py --computer-first

# Show win% hints for each column
uv run play.py --hints

# Stronger AI (more MCTS simulations = more thinking time)
uv run play.py --simulations 1600

# Use a specific checkpoint
uv run play.py --checkpoint checkpoints/checkpoint_0050.pt
```

The board is drawn in the console with colored pieces (Red ● / Yellow ●) and the last move highlighted. Enter a column number (1-7) to play, or `q` to quit.

## Benchmarking

Evaluate checkpoint quality against the [gamesolver.org test suite](http://blog.gamesolver.org/solving-connect-four/02-test-protocol/) (6000 positions across 6 difficulty levels):

```bash
# Run all test files against latest checkpoint
uv run benchmark.py

# Specific test file with verbose per-position output
uv run benchmark.py --test-files testdata/Test_L3_R1 -v

# Quick check: first 100 positions, stronger search
uv run benchmark.py --limit 100 --simulations 800

# Benchmark a specific checkpoint
uv run benchmark.py --checkpoint checkpoints/checkpoint_0050.pt
```

Reports **sign accuracy** (win/draw/loss classification) since the value head outputs a continuous score rather than exact move counts. Results are broken down per test file and by outcome category.

## Architecture

```
Input (3×6×7) → Conv Block → 10 Residual Blocks → ┬→ Policy Head → 7 move probabilities
                                                    └→ Value Head  → position evaluation [-1, 1]
```

- **Backend**: PyTorch with Metal (MPS) on Mac, CUDA on GPU, or CPU fallback
- **Self-play**: Parallelized across CPU cores via `multiprocessing.Pool`
- **Training**: Adam optimizer, cosine LR schedule, gradient clipping, 200K replay buffer
- **Data augmentation**: Horizontal board mirroring (doubles training data)

## Project Structure

```
├── main.py              # Training entry point
├── play.py              # Human vs AI game
├── benchmark.py         # Checkpoint evaluation against test suite
├── checkpoints/         # Saved model checkpoints
├── testdata/            # gamesolver.org test positions (Test_L*_R*)
└── src/
    ├── game.py          # Connect Four game engine
    ├── model.py         # AlphaZero neural network
    ├── mcts.py          # Monte Carlo Tree Search
    ├── self_play.py     # Parallel self-play generation
    ├── trainer.py       # Training loop & replay buffer
    └── utils.py         # Device detection, checkpoint I/O
```
