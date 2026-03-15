# alphafour
AlphaZero based Connect Four model

### Train Model
`uv run main.py`

> If there are existing checkpoint files in the output directory, it will load the last checkpoint and continue the training.
> Checkpoints are treated as trusted local artifacts. The loader uses PyTorch's safer weights-only mode when possible and falls back to full checkpoint loading only for trusted files.

Useful training flags:
- `--no-resume` to ignore existing checkpoints
- `--res-blocks` / `--hidden-channels` to scale model size
- `--replay-buffer-iters` / `--train-samples` to control replay usage
- `--weight-decay` to tune `AdamW`
- `--sim-batch-size` / `--fpu-reduction` to tune MCTS throughput and exploration
- `--self-play-device cpu|auto|cuda|mps` to control worker inference placement

### Compare Model Strength
`uv run eval.py --cp1 temp/checkpoint_1.pth.tar --cp2 temp/checkpoint_2.pth.tar`

### Human vs Model
`uv run play.py --cp temp/checkpoint_1.pth.tar`
