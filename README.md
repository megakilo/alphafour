# alphafour
AlphaZero based Connect Four model

### Train Model
`uv run main.py`

> If there are existing checkpoint files in the output directory, it will load the last checkpoint and continue the training.

### Compare Model Strength
`uv run eval.py --cp1 temp/checkpoint_1.pth.tar --cp2 temp/checkpoint_2.pth.tar`

### Human vs Model
`uv run play.py --cp temp/checkpoint_1.pth.tar`