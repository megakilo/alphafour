# 07. Practical Guide, Hyperparameter Intuition & Reference

> **Code references in this chapter**:
> - Training CLI: [`main.py`](../main.py)
> - Benchmark & Arena CLI: [`benchmark.py`](../benchmark.py)
> - Self-play Throughput Benchmark CLI: [`benchmark_selfplay.py`](../benchmark_selfplay.py)
> - Interactive Play CLI: [`play.py`](../play.py)
> - Cython OpenMP build script: [`setup_cython.py`](../setup_cython.py)

---

## 1. Hyperparameter Intuition: What Happens If Too Low vs. Too High?

One of the hardest parts of learning AlphaZero is knowing *why* each hyperparameter is set the way it is—and what symptoms appear if you change it. Every value below matches the exact defaults in `alphafour`:

### 1.1 MCTS & Exploration Hyperparameters

| Parameter | Default | Location | If Set Too Low | If Set Too High |
| :--- | :---: | :--- | :--- | :--- |
| **`training_simulations`** | `600` | `main.py` | Search looks too shallow; MCTS "teacher" fails to spot multi-move tactics, slowing down or capping peak Elo. | Linearly increases self-play wall-clock time per iteration with diminishing per-move policy improvement beyond ~800–1,200 sims in Connect Four. |
| **`c_puct`** | `1.5` | `src/mcts.py`, `src/self_play.py` | Over-exploits the first move with a decent Q-value; misses better alternative columns whose initial network prior was modest. | Spreads simulations too thinly across weak moves instead of searching deep enough along the critical tactical line. |
| **`c_fpu`** | `0.20` | `src/mcts.py`, `src/self_play.py` | (`0.0`): In losing positions, unvisited moves look artificially better than the best defence, distracting search onto garbage moves. | (`> 0.6`): Unvisited moves are penalized so harshly that MCTS locks onto the very first child visited and ignores viable sibling moves. |
| **`dirichlet_alpha`** | `0.35` | `src/mcts.py`, `src/self_play.py` | (`< 0.05`): Puts almost 100% of the noise spike onto a single random column. | (`> 2.0`): Noise approaches a flat `[1/7, ..., 1/7]` uniform blur that barely boosts any single overlooked move. |
| **`dirichlet_epsilon`** | `0.25` | `src/self_play.py` | (`0.0`): Root relies strictly on existing priors; risks tunnel vision on narrow openings. | (`> 0.50`): Root priors become mostly random noise, polluting the visit-count policy target $\boldsymbol{\pi}_t$. |
| **`LEAVES_PER_GAME`** | `16` | `src/self_play.py` | Small GPU batch sizes $\implies$ lower GPU utilization and more Python↔C round-trips. | Excessive virtual-loss distortion: selecting 64+ leaves at once from a small Connect Four tree forces traversals down unrealistic blunders. |

### 1.2 Self-Play & Replay Buffer Hyperparameters

| Parameter | Default | Location | If Set Too Low | If Set Too High |
| :--- | :---: | :--- | :--- | :--- |
| **`TEMP_THRESHOLD`** | `10` | `src/self_play.py` | (`0`–`2`): Games become repetitive; early self-play lacks diversity across midgame structures. | (`> 24`): Stochastic move sampling in sharp endgames causes random blunders that corrupt the entire game's outcome label $z_t$! |
| **`RANDOM_OPENING_PLIES`** | `4` | `src/self_play.py` | (`0`): Nearly every game plays `Col 4`, leaving the model fragile when facing off-beat human openings (`Col 1`, `Col 2`). | (`> 8`): Many games start in already-decided lopsided positions unrelated to realistic play. |
| **`buffer_capacity`** | `300,000` | `main.py`, `src/trainer.py` | (`< 50,000`): Catastrophic forgetting / rock-paper-scissors policy oscillations between consecutive iterations. | (`> 1,500,000`): Retains stale, low-quality games from 30+ iterations ago, slowing value convergence. |
| **`games_per_iteration`** | `1000` | `main.py` | Too few fresh games between weight updates increases variance in Arena & buffer turnover. | Delays weight updates unnecessarily (network plays thousands of games with stale weights before learning). |

### 1.3 Network & Optimization Hyperparameters

| Parameter | Default | Location | If Set Too Low | If Set Too High |
| :--- | :---: | :--- | :--- | :--- |
| **`res_blocks` / `filters`** | `10` / `128` | `src/model.py` | (`< 4` blocks / `32` filters): Underfits deep positional parity (`Test_L1_R3`). | (`> 20` blocks / `256` filters): Slower MCTS leaf inference for marginal Connect Four gain. |
| **`epochs`** | `4` $\to$ `2` | `main.py` | (`1` early): Slow initial feature bootstrapping. | (`> 6` on steady buffer): Overfits current replay buffer, spiking validation Value MSE and degrading generalization! |
| **`lr`** | `0.001` | `main.py`, `src/trainer.py` | Very slow policy/value distillation over 150 iterations. | (`> 0.005`): Unstable BatchNorm statistics and noisy policy swings. |

---

## 2. Annotated Walkthrough of Training Console Output

When you run `uv run main.py`, each iteration prints structured telemetry. Here is how to read every line:

```text
━━━ Iteration 25/150 (lr=0.000956) ━━━
  🎲 Self-play: 1000 games, 600 sims/move ...
     Generated 41280 examples in 11.4s (buffer: 300000)
```
- **`41280 examples` from `1000 games`**: Remember horizontal mirror augmentation emits $2\times$ examples per move—so games averaged $41{,}280 / (2 \times 1{,}000) \approx \mathbf{20.6}$ recorded plies! Early iterations average ~15–17 plies (quick tactical wins); mature iterations lengthen toward 22–28 plies as defense improves.
- **`buffer: 300000`**: Ring buffer is full and now evicting the oldest ~41k positions per iteration.

```text
  🧠 Training: 2 epochs × 500 batches ...
     Loss: policy=1.2410, value=0.3812, total=1.6222 (14.2s)
```
- **`policy=1.2410`**: Soft cross-entropy against MCTS visit distributions $\boldsymbol{\pi}_t$. Because $\boldsymbol{\pi}_t$ has non-zero entropy across legal columns ($\ln 7 \approx 1.946$ for uniform), `policy_loss` typically starts around `~1.88` and settles around `0.95–1.30` (it will *not* go to `0.0`!).
- **`value=0.3812`**: 3-class WDL cross-entropy ($\ln 3 \approx 1.099$ for uniform `[1/3, 1/3, 1/3]`). Drops from `~0.85` down to `~0.35–0.50`.

```text
  📊 Evaluation (0-sim raw network) ...
     ✅ Opening Move (0-sim): played column 4 (Center policy: 74.2%)
     Dataset Evaluation (0-sim):
       - Test_L1_R1: wdl_acc= 86.4%  draw_recall=   N/A  mse=0.241  opt_move= 89.2%
       ...
       - Test_L3_R1: wdl_acc= 79.1%  draw_recall= 76.4%  mse=0.298  opt_move= 88.5%
       * OVERALL   : wdl_acc= 74.8%  draw_recall= 68.2%  mse=0.352  opt_move= 83.7%
```
- **Every iteration** runs this fast `0`-simulation pass (~0.2s total!) so you can monitor raw network intuition (`wdl_acc`, `draw_recall`, `mse`, and solver-verified `opt_move`) without slowing down training.

```text
  🔍 MCTS Evaluation (400 sims/move) ...
     ✅ Opening Move (400-sim): played column 4 (Center visits: 88.5%)
     Dataset Evaluation (400-sim MCTS):
       ...
       * OVERALL   : wdl_acc= 88.9%  draw_recall= 82.1%  mse=0.184  opt_move= 94.3%
  ⚓ Anchor Arena: Current vs PureMCTSBaseline (40 paired games, 400 vs 200 sims, temp=0) ...
     Result: Current 37 - 1 Anchor (Draws: 2) | Winrate: 95.0% | Anchor Elo: +511
  ⚔️ Arena: Current vs Iter-15 (40 paired games, 400 sims, temp=0) ...
     Result: Current 24 - 10 Iter-15 (Draws: 6)
     Breakdown (Current): 15 wins as P1, 9 wins as P2
     Breakdown (Iter-15): 7 wins as P1, 3 wins as P2
     Current Winrate: 67.5% (ΔElo: +127)
```
- **Every `--eval-interval` (`5`) iterations**, full 400-sim MCTS dataset verification + Anchor Arena + Relative Arena confirm both game-theoretic move optimality and competitive Elo gains!

---

## 3. Common CLI Recipes

### 3.1 Quick 3-Iteration Smoke Test (Verify Everything Works in ~1 Minute)
```bash
uv run main.py \
  --no-resume \
  --iterations 3 \
  --games-per-iteration 64 \
  --training-simulations 100 \
  --eval-simulations 50 \
  --eval-interval 1 \
  --arena-games 8 \
  --checkpoint-dir checkpoints_smoke
```

### 3.2 Production Training (Single GPU or Multi-GPU DDP)
```bash
# Auto-uses all visible CUDA GPUs (or MPS / CPU) and auto-resumes from checkpoints/
uv run main.py

# Extend a completed 150-iteration run to 200 iterations (horizon-safe LR rebuild!)
uv run main.py --iterations 200

# Force single-GPU execution on a multi-GPU workstation
uv run main.py --gpus 1

# Explicit torchrun launch across 4 GPUs (partition CPU OpenMP threads across ranks)
OMP_NUM_THREADS=$(( $(nproc) / 4 )) uv run torchrun --standalone --nproc_per_node=4 main.py
```

### 3.3 Deep Checkpoint Verification (`benchmark.py`)
```bash
# Full 6,000-position report (both 0-sim raw network and 400-sim MCTS) + Anchor Elo
uv run benchmark.py --arena-anchor

# Fast subset check on End-Easy and Begin-Hard (first 250 positions each, MCTS only)
uv run benchmark.py --no-both --eval-simulations 200 \
  --test-files testdata/Test_L3_R1 testdata/Test_L1_R3 --limit 250

# Duplicate Paired-Opening tournament between two checkpoints (100 games = 50 pairs)
uv run benchmark.py \
  --checkpoint checkpoints/checkpoint_0070.pt \
  --arena-vs checkpoints/checkpoint_0030.pt \
  --arena-games 100 \
  --eval-simulations 400
```

### 3.4 Playing Against Your Model (`play.py`)
```bash
# Human plays first (Red ●) with column Win% hints and 800 MCTS simulations/move
uv run play.py --hints --simulations 800

# Let the AI open first (Red ●) — try holding a draw as Yellow!
uv run play.py --computer-first --hints --simulations 800
```

---

## 4. Troubleshooting & Performance Checklist

1. **Self-play feels slow on CPU / Cython extension isn't compiled**:
   Check whether `src.self_play` is running compiled native C + OpenMP:
   ```bash
   uv run python -c "import src.self_play as sp; print('is_cython_compiled =', sp.is_cython_compiled())"
   ```
   If `False`, run `uv run python setup_cython.py build_ext --inplace` to compile `src/self_play.py` in-place.
2. **First iteration pauses for 10–25 seconds on CUDA**:
   That one-time pause is TorchInductor compiling the dynamic `NHWC` BF16 Triton kernels (`--compile` is enabled by default). Pass `--no-compile` during quick 1-iteration debugging runs if you want instant startup.
3. **Solver ground-truth cache (`testdata/.solver_cache/`)**:
   `get_dataset_child_wdl` caches SHA256 `.npz` child W/D/L matrices in `testdata/.solver_cache/` so dataset optimality and blunder evaluation loads from disk in `< 5 ms`.

---

## 5. Module & API Quick Reference

### Python & Cython Package (`src/`)

| Symbol | Signature / Input → Output | Purpose |
| :--- | :--- | :--- |
| `ConnectFour` (`src.game`) | `.encode() -> float32[3, 6, 7]`<br>`.get_valid_moves() -> bool[7]`<br>`.get_result() -> {-1.0, 0.0, None}` | Canonical Python Connect Four board (`current_player ∈ {+1, -1}`). |
| `AlphaZeroNet` (`src.model`) | `forward(x: [B,3,6,7]) -> (logits[B,7], wdl[B,3])`<br>`predict_wdl(s, mask) -> (pol, wdl_probs, val)`<br>`predict_packed(s_u8, mask) -> float32[B, 8]` | SE-ResNet with 32-ch policy head and 16-ch `[Loss, Draw, Win]` value head (`val = P(Win) - P(Loss)`). |
| `MCTS` (`src.mcts`) | `.search_with_values(game, add_noise)`<br>`.get_action_probs(game, temperature, add_noise)`<br>`.advance_move(col)` / `.reset()` | Single-game Python MCTS-Solver with FPU reduction and subtree reuse across turns. |
| `CythonBatchedSelfPlay` (`src.self_play`) | `.collect_continuous_u8()`<br>`.apply_continuous_packed(packed)`<br>`.take_completed_u8()` / `.take_completed_f32()` | OpenMP-parallel Cython Pure Python Mode batched MCTS-Solver self-play engine. |
| `CythonBatchedMCTS` (`src.self_play`) | `.collect_leaves_u8()`<br>`.apply_evaluations_packed(packed)`<br>`.get_results() -> (visits, best_actions, best_values)` | OpenMP-parallel Cython Pure Python Mode noiseless batched MCTS-Solver engine for dataset evaluation & Arena. |
| `run_self_play` (`src.self_play`) | `(model, num_games, num_simulations, ..., return_arrays=True, return_uint8=True)` | Runs batched self-play + $2\times$ mirror augmentation, returning `(states, policies, values)`. |
| `ReplayBuffer` (`src.trainer`) | `.add_batch(s, p, v)`<br>`.sample_tensors(B, device) -> (s, p, v)` | Preallocated `uint8` circular buffer with $O(B)$ analytical 2:1 recency sampling. |
| `Trainer` (`src.trainer`) | `.train_epoch(replay_buffer, num_batches)`<br>`.step_scheduler()` / `.restore_scheduler(...)` | BF16/FP16 AMP + DDP + `torch.compile` trainer with AdamW and Warmup+Cosine LR. |
| `evaluate_dataset` (`src.evaluate`) | `(model, device, lines, num_simulations, child_wdl)` | Computes WDL accuracy, Win/Draw/Loss recall, Optimal Move %, Blunder %, and Value MSE. |
| `play_batched_arena` (`src.evaluate`) | `(model1, model2, device, num_games, num_simulations, ...)` | Strictly model-isolated duplicate Paired-Opening head-to-head Arena. |

---

⬅️ **Previous**: **[Chapter 06: High-Performance Cython Engine & Bitboard Solver](./06-cython-engine-and-architecture.md)**  
🏠 **Back to Index**: **[AlphaFour Documentation Hub (`docs/README.md`)](./README.md)**
