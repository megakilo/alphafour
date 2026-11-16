# 04. Self-Play Data Generation, Replay Buffer & Training Pipeline

> **Code references in this chapter**:
> - Outer training loop & CLI flags: [`main.py`](../main.py)
> - Self-play orchestration & symmetry augmentation: [`src/self_play.py`](../src/self_play.py)
> - Contiguous ring buffer, recency sampler, WDL trainer & LR schedule: [`src/trainer.py`](../src/trainer.py)
> - Multi-GPU Distributed Data Parallel (DDP): [`src/distributed.py`](../src/distributed.py)
> - Atomic checkpoint & replay buffer persistence: [`src/utils.py`](../src/utils.py)

---

## 1. The Outer Training Loop (`main.py`)

In supervised learning, your dataset is static: you load ImageNet once and loop over epochs. In **AlphaZero**, the training loop is **online and non-stationary**—the model generates its own dataset, trains on recent history, and immediately uses its upgraded weights to generate higher-quality games in the next iteration!

Here is what happens inside a single iteration (`iteration = 0 .. 149`) of [`main.py`](../main.py#L422-L723):

```text
Iteration t (1 .. 150)
 ├── 1. Self-Play Phase (`model.eval()`)
 │       • Play 1,000 games (`600` MCTS-Solver sims/move)
 │       • Emit horizontally mirrored pairs (~35k–45k positions)
 │       • Write batch into circular `ReplayBuffer` (capacity 300,000)
 │
 ├── 2. Training Phase (`model.train()`)
 │       • Epochs: `4` during warmup iterations 1..20, then `2` thereafter
 │       • Batches/epoch: `min(500, len(buffer) // 256)` (scaled per GPU under DDP)
 │       • Sample 2:1 recency-weighted mini-batches (`B = 256`) directly in VRAM
 │       • Update `AlphaZeroNet` via BF16 AMP + Fused AdamW + grad clipping (1.0)
 │       • Step Warmup+Cosine LR scheduler once per iteration
 │
 ├── 3. Evaluation Phase (`model.eval()`)
 │       • Every iteration: 0-sim opening check + 6,000-pos `gamesolver.org` WDL eval
 │       • Every 5 iterations: 400-sim MCTS dataset eval + Anchor Arena + Relative Arena
 │
 └── 4. Atomic Persistence
         • Save `checkpoints/checkpoint_NNNN.pt` (weights + optimizer + scheduler)
         • Overwrite `checkpoints/replay_buffer.pt` (compact chronological uint8 state)
```

---

## 2. Four Self-Play Subtleties That Make or Break AlphaZero

Generating good self-play data is a tug-of-war between **Exploration (Diversity)** and **Exploitation (Quality)**:
- If self-play is **too deterministic**, all 1,000 games play the exact same 2 or 3 main lines. The network memorizes a narrow tunnel and blunders immediately when a human plays an unusual column!
- If self-play is **too random**, players make silly blunders on move 25, turning a won opening (`+1`) into a loss (`-1`) and corrupting the value labels $z_t$ of every earlier move in that game!

`alphafour` balances both needs using four complementary mechanisms in [`src/self_play.py`](../src/self_play.py):

### 2.1 Stochastic Opening Diversification (`RANDOM_OPENING_PLIES = 4`)
Even with Dirichlet root noise and temperature $\tau = 1$, once the network learns that **Column 4** is the unique winning opening move for Player 1, center visits exceed 80–90%—meaning edge openings (`Col 1`, `Col 2`, `Col 6`, `Col 7`) rarely occur naturally.

On the other hand, if *every* game started with 4 random moves, the network would **never train on the empty board (Move 0) or 1-stone board (Move 1)**!

Look at how `_sample_opening_into_slot` ([`src/self_play.py`](../src/self_play.py)) samples each game's starting position:
- **50% of games (`0` random plies)**: Start from the **empty board** so Moves 0 and 1 get heavy direct training.
- **25% of games (`1` random ply)**: Start after 1 uniformly random legal column, guaranteeing equal exposure to all 7 first moves (`1..7`).
- **25% of games (`2..=4` random plies)**: Start from a diverse 2-, 3-, or 4-stone board (rolling back/resampling if a random move ever hits an immediate terminal state, though impossible in $\le 4$ plies).

### 2.2 Temperature Schedule (`TEMP_THRESHOLD = 10`)
- **Opening/Early Midgame (`move_count < 10`)**: Sample the played action proportionally to root visit counts ($\tau = 1.0$) so games branch naturally across reasonable continuations.
- **Tactical Midgame/Endgame (`move_count >= 10`)**: Play **greedily** (`argmax`, $\tau = 0.0$) so games are decided by each side's best play! This keeps late-game outcomes $z \in \{-1, 0, +1\}$ clean and accurate.

### 2.3 Decoupling the Policy Target $\boldsymbol{\pi}_t$ from Move-Selection Temperature!
> **Common Implementation Pitfall**: Many beginner AlphaZero implementations record the *post-temperature* action distribution `action_probs` as the training target $\boldsymbol{\pi}_t$.
> Think about what happens after ply 10 when $\tau = 0$: `action_probs` becomes a **hard one-hot vector** `[0, 0, 1, 0, 0, 0, 0]`!
> Suppose Column 3 had `305` visits and Column 4 had `295` visits because both moves are nearly equally strong. Training against one-hot `[0, 0, 1, 0, 0, 0, 0]` forces the network to crush Column 4's probability toward `0.0`—throwing away rich dark knowledge from 600 MCTS simulations!

In `alphafour` (`_advance_single_slot` in [`src/self_play.py`](../src/self_play.py)), **the training policy target $\boldsymbol{\pi}_t$ is completely decoupled from move selection**:
1. If `root` is `SOLVER_WIN`, $\boldsymbol{\pi}_t$ is **uniform over all proven-winning columns** (`child == SOLVER_LOSS`).
2. Otherwise, any proven-losing moves (`child == ProvenWin`) are zeroed out into $N_{\text{safe}}$ (when a safe visited move exists), and $\boldsymbol{\pi}_t$ is **always the smooth normalized visit distribution**:

   $$
   \pi_t(a) = \frac{N_{\text{safe}}(s_t, a)}{\sum_{b=0}^6 N_{\text{safe}}(s_t, b)}
   $$

   even on ply 30 when the actual move played on the board is chosen greedily via `argmax`! *(In the pure-Python fallback `_play_batched_python`, `policy_target` normalizes raw `visit_counts` directly while `selection_counts` applies the `SOLVER_WIN` / `SOLVER_LOSS` zeroing before sampling the move.)*

### 2.4 Free $2\times$ Data via Horizontal Mirror Symmetry
Connect Four rules are invariant under left-right reflection (`col c` $\leftrightarrow$ `col 6 - c`). Neither player parity nor gravity changes when you flip the board horizontally!

Therefore, for **every** recorded state $(s_t, \boldsymbol{\pi}_t, z_t)$ in a finished game, `_emit_slot_examples_c` and `augment_arrays` ([`src/self_play.py`](../src/self_play.py)) emit **two** examples:
1. Original: `(s_t, π_t, z_t)`
2. Mirrored: `(s_t[:, :, ::-1], π_t[::-1], z_t)`

This doubles sample efficiency with **zero** extra MCTS simulations and enforces symmetric left-right play early in training! (A typical 1,000-game iteration averaging ~20 plies/game yields ~40,000 augmented training examples.)

---

## 3. Contiguous Ring-Buffer Replay Memory & Recency Sampling

### 3.1 Why Use a Sliding-Window Ring Buffer (`capacity = 300,000`)?
- **If you only train on the latest iteration (~40k positions)**: Adjacent moves from the same 1,000 games are heavily correlated; training multiple epochs over just one iteration causes the network to overfit to recent quirks and forget defensive patterns from 3 iterations ago (**policy oscillation / catastrophic forgetting**).
- **If you keep every game ever played since Iteration 1**: Iteration 1's games were played by a clueless random network! Training Iteration 100 on blunders from Iteration 1 drags value predictions down toward mediocre play.

A circular **Ring Buffer of 300,000 examples** ([`ReplayBuffer` in `src/trainer.py`](../src/trainer.py#L22-L245)) holds roughly **7 to 9 iterations** of history—wide enough to prevent oscillation, narrow enough to continuously evict outdated self-play data.

### 3.2 Zero-Overhead VRAM Storage (`45.20 MiB` Total!)
A naive Python `collections.deque` of 300,000 `(float32[3,6,7], float32[7], float)` tuples consumes **hundreds of megabytes** in Python object headers and stalls every training step assembling CPU NumPy arrays and copying them over PCIe.

Because board planes `0, 1, 2` only ever take binary values `{0, 1}`, `ReplayBuffer` preallocates three contiguous PyTorch tensors directly on the target device (`cuda` or `cpu`):
- `states_buf`: `uint8[300_000, 3, 6, 7]` = `37,800,000` bytes
- `policies_buf`: `float32[300_000, 7]` = `8,400,000` bytes
- `values_buf`: `float32[300_000]` = `1,200,000` bytes
- **Total footprint**: **`47,400,000` bytes (`45.20 MiB`)**—comfortably resident in GPU VRAM, so `sample_tensors()` indexes batches on-GPU in microseconds with **zero host-to-device PCIe traffic**!

### 3.3 Analytical $O(B)$ Inverse-CDF Recency Sampling
Even within the 300,000-example window, the newest positions (generated by Iteration $t$) are higher quality than the oldest positions still in the buffer (generated by Iteration $t - 7$).

`alphafour` samples mini-batches with a **2:1 linear recency bias**: the newest example in the buffer is **twice as likely** to be sampled as the oldest example!

Instead of maintaining a $300{,}000$-element weight array for `torch.multinomial` (which costs $O(N)$ prefix sums), `sample_tensors` ([`src/trainer.py`](../src/trainer.py#L161-L187)) solves the continuous Cumulative Distribution Function (CDF) analytically in $O(B)$ time:

1. Let $x \in [0, 1]$ represent chronological age from oldest ($x = 0$) to newest ($x = 1$).
2. Define unnormalized linear weight $w(x) = 1 + x$ (so $w(0) = 1$ and $w(1) = 2$).
3. Normalize over $[0, 1]$: since $\int_0^1 (1 + x)\,dx = 1 + \frac{1}{2} = \frac{3}{2}$, the probability density function is:

   $$
   p(x) = \frac{2}{3}(1 + x)
   $$

4. Integrate to obtain the CDF $F(x) \in [0, 1]$:

   $$
   F(x) = \int_0^x \frac{2}{3}(1 + t)\,dt = \frac{2x + x^2}{3}
   $$

5. Invert $F(x) = u$ for a uniform random variable $u \sim \text{Uniform}(0, 1)$ by completing the square ($x^2 + 2x + 1 = 1 + 3u$):

   $$
   x = \sqrt{1 + 3u} - 1
   $$

In PyTorch, sampling `B = 256` recency-weighted indices from a circular buffer of size `n` takes just **four vectorized lines**:
```python
u = torch.rand(actual_size, device=self.buf_device, dtype=torch.float32)
rank = ((torch.sqrt(1.0 + 3.0 * u) - 1.0) * n).long().clamp_(0, n - 1)
oldest_idx = (self._write_pos - n) % self.capacity
idx = (oldest_idx + rank) % self.capacity
```

---

## 4. Loss Functions, Optimizer & Learning Rate Schedule (`Trainer`)

### 4.1 Multi-Task Loss Function (`_run_batch_step`)
Given a sampled mini-batch `(states_t, target_policies_t, target_values_t)` of size $B = 256$, `AlphaZeroNet` outputs raw `policy_logits` $\in \mathbb{R}^{B \times 7}$ (masked with `-1e4` on full columns) and `wdl_logits` $\in \mathbb{R}^{B \times 3}$ ([`src/trainer.py`](../src/trainer.py#L502-L532)):

1. **Policy Loss (Soft Cross-Entropy)**:

   $$
   \mathcal{L}_{\text{policy}} = -\frac{1}{B} \sum_{i=1}^B \sum_{a=0}^6 \pi_{i,a} \log \hat{p}_{i,a}
   $$

   Minimizing cross-entropy pushes the network's predicted move probabilities $\hat{\mathbf{p}}_i$ to match MCTS's search distribution $\boldsymbol{\pi}_i$. *(Note: Because $\boldsymbol{\pi}_i$ is usually soft rather than one-hot, the theoretical minimum of $\mathcal{L}_{\text{policy}}$ is the entropy $H(\boldsymbol{\pi}) > 0$, not $0.0$!)*

2. **WDL Value Loss (3-Class Cross-Entropy)**:
   Each scalar outcome $z_i \in [-1, +1]$ is mapped to a 3-element WDL target $\mathbf{y}_i = \big(y_{i,L},\, y_{i,D},\, y_{i,W}\big)$:

   $$
   \mathbf{y}_i = \big(\max(0, -z_i),\; 1 - |z_i|,\; \max(0, z_i)\big)
   $$

   - On terminal self-play outcomes $z_i \in \{-1, 0, +1\}$, this produces **exact one-hot targets**:
     - Loss ($z = -1$) $\to$ `[1, 0, 0]`
     - Draw ($z = 0$) $\to$ `[0, 1, 0]`
     - Win ($z = +1$) $\to$ `[0, 0, 1]`
   - On any intermediate target $z_i \in (-1, +1)$, it interpolates smoothly while preserving $y_W - y_L = z_i$ and $y_L + y_D + y_W = 1$!

   $$
   \mathcal{L}_{\text{value}} = -\frac{1}{B} \sum_{i=1}^B \sum_{c \in \{L, D, W\}} y_{i,c} \log \hat{q}_{i,c}
   $$

3. **Total Loss**:

   $$
   \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}}
   $$

### 4.2 Optimizer, Parameter Groups & Gradient Alignment
`Trainer.__init__` ([`src/trainer.py`](../src/trainer.py#L293-L366)) configures optimization carefully:
- **Decoupled `AdamW` (`lr = 1e-3`, `weight_decay = 1e-4`, `fused = True` on CUDA)**:
  - **2D+ parameters** (`Conv2d` and `Linear` weights, `p.ndim >= 2`): `weight_decay = 1e-4` ($L_2$ regularization prevents overfitting to transient replay-buffer distributions).
  - **1D parameters** (`BatchNorm2d` $\gamma, \beta$ and `Linear` biases, `p.ndim < 2`): **`weight_decay = 0.0`**. *(Why? Applying weight decay to BatchNorm scale $\gamma$ shrinks channel activations only for the next BatchNorm layer to re-normalize them—creating an artificial effective-learning-rate drift across layers!)*
- **Gradient Clipping (`max_norm = 1.0`)**: Caps total $L_2$ gradient norm at `1.0` before stepping, preventing rare tactical outlier batches from destabilizing BatchNorm running statistics.
- **NHWC Stride Alignment (`_align_grad_layouts` & `_align_optimizer_state`)**: CUDA's `fused=True` AdamW kernel requires `param`, `param.grad`, `exp_avg`, and `exp_avg_sq` to share identical memory strides. Because `channels_last` (`NHWC`) convolutions and checkpoint resumes across CPU/GPU can occasionally produce `NCHW`-strided gradients or optimizer states, `alphafour` explicitly aligns tensor strides before stepping!

### 4.3 Epoch Schedule & Warmup + Cosine Learning Rate Schedule
- **Adaptive Epochs & Batches ([`main.py`](../main.py#L502-L509))**:
  - During the first **20 iterations** (`iteration < 20`), `main.py` runs **4 epochs** per iteration to rapidly bootstrap basic rules, threat detection, and center control from early data; from iteration 21 onward, it runs **2 epochs** (`--epochs 2`) per iteration to avoid overfitting the replay buffer.
  - Mini-batches per epoch auto-scale with buffer fill: `min(500, max(1, len(replay_buffer) // batch_size))`—so Iteration 1 (~40k examples) runs ~156 batches instead of over-sampling a tiny buffer 500 times!
- **Monotonic Warmup + Single-Cycle Cosine LR ([`src/trainer.py`](../src/trainer.py#L368-L402))**:
  1. **`LinearLR` Warmup**: Over `warmup_iters = min(5, max(1, total_iterations // 20))` iterations (5 iters for a 150-iter run), ramps LR linearly from `0.2 * lr` (`2e-4`) up to `1.0 * lr` (`1e-3`).
  2. **`CosineAnnealingLR` Decay**: Smoothly decays LR from `1e-3` down to `eta_min = 1e-5` over the remaining `145` iterations—with **no cosine restarts** (which cause sudden LR spikes that wreck trained MCTS priors) and a hard clamp at `eta_min` if training continues past `total_iterations`.
  3. **Horizon-Safe Resume (`restore_scheduler`)**: If you train for 50 iterations (`--iterations 50`) and later resume with `--iterations 150`, `Trainer.restore_scheduler` detects that `T_max` changed, rebuilds the 150-iteration schedule cleanly, and fast-forwards to step 50 while preserving all AdamW momentum/variance buffers!

---

## 5. Atomic Checkpointing & Multi-GPU Distributed Training

### 5.1 Split Checkpoint & Replay Buffer Persistence ([`src/utils.py`](../src/utils.py#L96-L134))
Saving a 45 MB replay buffer *inside* every `checkpoint_0001.pt .. checkpoint_0150.pt` would waste ~7 GB of disk space and slow down Arena checkpoint loading. Instead:
- **`checkpoints/checkpoint_NNNN.pt` (~36 MB each)**: Stores clean unwrapped `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, and `iteration`. Kept for every iteration so you can benchmark historical strength progression or run relative Arena matches (`Iter t` vs `Iter t-10`).
- **`checkpoints/replay_buffer.pt` (~45 MB single file)**: Overwritten atomically each iteration.
- **Crash-Safe Atomic Writes**: Every save writes to `<filename>.tmp` first and calls POSIX atomic `Path.replace()`, so `Ctrl+C` or power loss mid-save never corrupts your latest checkpoint!

### 5.2 Multi-GPU Scaling via DDP ([`src/distributed.py`](../src/distributed.py))
If your machine has multiple CUDA GPUs, `uv run main.py` automatically spawns one worker per GPU via `torch.multiprocessing.spawn` (or attaches directly under `torchrun --nproc_per_node=K main.py`):
1. **CPU Thread Partitioning (`configure_omp_threads`)**: Divides CPU cores evenly across ranks (`OMP_NUM_THREADS = cpu_count // world_size`) for Cython OpenMP `prange`, preventing $K$ ranks from oversubscribing $K \times$ CPU threads!
2. **Sharded Self-Play + Deterministic `all_gather`**: Each GPU generates `1000 // world_size` self-play games independently using distinct rank RNG seeds (`seed_rank`). `gather_self_play_arrays` then pads and `all_gather`s the compact `uint8` states, policies, and values across NCCL so **every rank's `ReplayBuffer` contains the exact same chronological history**.
3. **DDP Training + BatchNorm Sync**: `DistributedDataParallel` averages gradients across GPUs on each mini-batch (`global_batch = 256 * world_size`, while scaling `effective_batches` down by `world_size` so total updates per epoch stay consistent), and broadcasts rank 0's BatchNorm running stats at the end of every epoch so self-play inference remains bit-identical across ranks!

---

## 6. Summary & Next Steps

1. **Self-play** combines 50%/25%/25% stochastic opening plies, early proportional temperature ($\tau=1$ for `<10` plies) transitioning to greedy play ($\tau=0$ for $\ge 10$ plies), **decoupled smooth visit-count policy targets**, and $2\times$ horizontal mirror augmentation.
2. **The `300,000`-entry VRAM Ring Buffer** stores binary boards as `uint8` in `45.2 MiB` and draws mini-batches using an exact $O(B)$ analytical inverse-CDF 2:1 recency formula (`sqrt(1 + 3u) - 1`).
3. **Training** minimizes masked policy soft cross-entropy + 3-class WDL value cross-entropy using fused `AdamW` (no weight decay on 1D params), gradient clipping, and a horizon-safe warmup + cosine LR schedule.

Now: how do we *know* our training run is actually getting stronger—and not just memorizing self-play quirks? Let's explore **Evaluation, Exact Solver Verification & the Paired-Opening Arena**!

---

⬅️ **Previous**: **[Chapter 03: Neural Network Architecture, WDL Value Head & Fast Inference](./03-neural-network-and-inference.md)**  
👉 **Next**: **[Chapter 05: Verification, Exact Solver Metrics & Paired-Opening Arena](./05-evaluation-and-arena.md)**  
🏠 **Back to Index**: **[AlphaFour Documentation Hub (`docs/README.md`)](./README.md)**
