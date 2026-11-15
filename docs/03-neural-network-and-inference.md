# 03. Neural Network Architecture, WDL Value Head & Fast Inference

> **Code references in this chapter**:
> - Neural network architecture & compiled/packed GPU inference: [`src/model.py`](../src/model.py)
> - Interactive human-vs-AI console client & win-hint inference: [`play.py`](../play.py)
> - Compiler configuration & checkpoint state-dict cleaning: [`src/utils.py`](../src/utils.py)

---

## 1. What Role Does the Neural Network Play?

In AlphaZero, the neural network $f_\theta$ is queried **hundreds of thousands of times per training iteration** (1,000 games $\times \sim 20$ moves $\times 600$ simulations $\approx$ millions of leaf evaluations, grouped into GPU batches).

Its design must strike a delicate balance:
- **Expressive enough** to recognize spatial Connect Four patterns—open 3-in-a-row threats, diagonal forks, odd/even column parity ("Claimeven"), and subtle locked draws across all 42 cells.
- **Compact & fast enough** that batched GPU inference takes fractions of a millisecond per batch, keeping MCTS throughput high on a single consumer GPU.

`AlphaZeroNet` ([`src/model.py`](../src/model.py#L66-L121)) achieves this with a **10-block, 128-filter Squeeze-and-Excitation Residual Network (SE-ResNet)** totaling **3,187,626 parameters**, feeding a **32-channel Policy Head** and a **16-channel 3-logit Win/Draw/Loss (WDL) Value Head**.

---

## 2. End-to-End Architecture & Parameter Breakdown

```text
Input Board Tensor x: (B, 3, 6, 7)   [uint8 {0,1} or float32 {0.0, 1.0}]
 │
 ▼  Stem (conv_block — 3,712 params)
┌───────────────────────────────────────────────────────────┐
│ Conv2d(3 → 128, kernel=3×3, padding=1, bias=False)        │
│ BatchNorm2d(128) ──► ReLU                                 │
└─────────────────────────────┬─────────────────────────────┘
                              │ (B, 128, 6, 7)
                              ▼
┌───────────────────────────────────────────────────────────┐
│ Residual Tower: 10 × ResBlock(128)  (2,995,200 params)    │
│                                                           │
│   x ──► Conv3×3(128→128) ──► BN ──► ReLU                  │
│     │   ──► Conv3×3(128→128) ──► BN                       │
│     │   ──► SEBlock(128, reduction=8):                    │
│     │         AvgPool(6×7→1×1) → FC(128→16) → ReLU        │
│     │         → FC(16→128) → Sigmoid → channel scale      │
│     └────── (+) residual skip connection ──► ReLU         │
└──────────────┬──────────────────────────────┬─────────────┘
               │ (B, 128, 6, 7)               │ (B, 128, 6, 7)
               ▼                              ▼
┌──────────────────────────────┐ ┌─────────────────────────────────────────┐
│ Policy Head (13,575 params)  │ │ WDL Value Head (175,139 params)         │
│                              │ │                                         │
│ Conv1×1(128 → 32, no bias)   │ │ Conv1×1(128 → 16, no bias)              │
│ BatchNorm2d(32) ──► ReLU     │ │ BatchNorm2d(16) ──► ReLU                │
│ Flatten ──► (B, 1344)        │ │ Flatten ──► (B, 672)                    │
│ Linear(1344 → 7)             │ │ Linear(672 → 256) ──► ReLU              │
│                              │ │ Linear(256 → 3)                         │
└──────────────┬───────────────┘ └────────────────────┬────────────────────┘
               │ policy_logits (B, 7)                 │ wdl_logits (B, 3)
               ▼                                      ▼
   Mask illegal cols (-1e9)               Softmax ──► [P(Loss), P(Draw), P(Win)]
   Softmax ──► Policy p (B, 7)            Expected Value V = P(Win) - P(Loss) ∈ [-1, +1]
```

### Exact Parameter Table (Default `--res-blocks 10 --filters 128`)

| Submodule | Layer Details | Output Shape | Parameter Count |
| :--- | :--- | :---: | ---: |
| **`conv_block`** | `Conv3x3(3→128)` (`3,456`) + `BN(128)` (`256`) | `(B, 128, 6, 7)` | `3,712` |
| **1 × `ResBlock`** | `2 × [Conv3x3(128→128) + BN(128)]` (`295,424`) + `SEBlock(128→16→128)` (`4,096`) | `(B, 128, 6, 7)` | `299,520` |
| **`res_blocks` (×10)** | `10 × ResBlock(128)` | `(B, 128, 6, 7)` | `2,995,200` |
| **`policy_head`** | `Conv1x1(128→32)` (`4,096`) + `BN(32)` (`64`) + `Linear(1344→7)` (`9,415`) | `(B, 7)` | `13,575` |
| **`value_head`** | `Conv1x1(128→16)` (`2,048`) + `BN(16)` (`32`) + `Linear(672→256)` (`172,288`) + `Linear(256→3)` (`771`) | `(B, 3)` | `175,139` |
| **Total** | | | **`3,187,626`** |

*(Notice `bias=False` on every `Conv2d` immediately followed by `BatchNorm2d`, since BatchNorm subtracts the channel mean anyway—making a conv bias redundant!)*

---

## 3. Why Squeeze-and-Excitation (SE) Residual Blocks?

### 3.1 Preserving Spatial Geometry (`6 × 7` Throughout)
Unlike image classifiers (ResNet-50 on ImageNet) that downsample spatial dimensions with pooling or stride-2 convolutions, board-game networks **never downsample**! Every layer in the stem and 10-block tower keeps the exact `(6, 7)` board grid (`padding=1` on $3 \times 3$ kernels) because shifting a stone by 1 square completely changes whether a 4-in-a-row exists.

### 3.2 Why Add `SEBlock` Channel Attention?
A single $3 \times 3$ convolution only sees a $3 \times 3$ local neighborhood. Even after several layers, purely local convolutions take effort to aggregate **whole-board properties**, such as:
- *"How full is the board overall (opening vs endgame)?"*
- *"Does Player 1 have an odd threat on the left side AND an even threat on the right side?"*

`SEBlock` ([`src/model.py`](../src/model.py#L20-L43), introduced to AlphaZero networks by Leela Chess Zero) adds lightweight global context to every residual block for just `4,096` parameters (`< 1.4%` overhead per block):
1. **Squeeze**: Global average pool across the entire $6 \times 7$ board: `scale = x.mean(dim=(2, 3))` $\to$ `(B, 128)`.
2. **Excite**: Two-layer bottleneck MLP `Linear(128 → 16, bias=False) → ReLU → Linear(16 → 128, bias=False) → Sigmoid` $\to$ `(B, 128, 1, 1)` gating weights in $(0, 1)$.
3. **Rescale**: Multiply `x * scale`, selectively amplifying tactical or endgame feature channels conditioned on the global board state before adding the residual skip connection!

---

## 4. Deep Dive: Why a 3-Logit WDL Value Head Beats Scalar `tanh`

In the original 2017 AlphaZero paper, the value head compressed the tower through a **1-channel `1×1` convolution** down to a single scalar $v = \tanh(\cdot) \in (-1, +1)$ trained with Mean Squared Error: $\mathcal{L}_v = (v - z)^2$.

`alphafour` upgrades the value head in **two critical ways** ([`src/model.py`](../src/model.py#L107-L117)):

### Upgrade 1: 16-Channel Spatial Bottleneck (`16 × 6 × 7 = 672` vs `1 × 6 × 7 = 42`)
Crushing 128 rich feature channels into **1 channel** (`42` numbers) before the fully connected layer creates a severe information bottleneck: the network cannot simultaneously encode horizontal, vertical, and both diagonal threat maps plus column heights in a single float per square. Keeping `value_channels = 16` (`672` spatial features into `Linear(672 → 256)`) preserves multi-threat spatial geometry.

### Upgrade 2: 3-Logit Win / Draw / Loss (`[Loss, Draw, Win]`) Classification
Consider two completely different Connect Four positions:
- **Position X (Razor-sharp tactical chaos)**: White wins in 50% of continuations and Black wins in 50% of continuations; draws never happen. Expected value = $0.5(+1) + 0.5(-1) = \mathbf{0.0}$.
- **Position Y (Locked endgame fortress)**: Every remaining column leads to a 42-stone full-board draw with 100% certainty. Expected value = $\mathbf{0.0}$.

With a scalar `tanh` + MSE head:
1. **Ambiguity**: Both Position X and Position Y produce the exact same scalar `0.0`—the network cannot express *draw probability* separately from *win/loss uncertainty*!
2. **Vanishing gradients near $\pm 1$**: Once $\tanh(x) \approx 0.98$, $\tanh'(x) = 1 - \tanh^2(x) \approx 0.04$, slowing down gradient updates when correcting blunders in seemingly won positions.

With `alphafour`'s **3-logit WDL value head**:
- The network outputs logits $\mathbf{u} = (u_L, u_D, u_W) \in \mathbb{R}^3$, converted via softmax into a calibrated 3-class distribution:

  $$
  \mathbf{q} = \big(P(\text{Loss}),\; P(\text{Draw}),\; P(\text{Win})\big) = \text{Softmax}(u_L, u_D, u_W)
  $$

- Now Position X is represented as **`[0.50, 0.00, 0.50]`**, while Position Y is represented as **`[0.00, 1.00, 0.00]`**! In evaluation ([Chapter 05](./05-evaluation-and-arena.md)), `argmax(wdl_probs)` directly classifies true draws with high recall instead of guessing an arbitrary scalar cutoff!
- And whenever MCTS needs a single scalar utility $V \in [-1, +1]$ to backpropagate, we compute the exact expected game score (`wdl_logits_to_value` in [`src/model.py`](../src/model.py#L224-L233)):

  $$
  V(s) = (-1)\cdot P(\text{Loss}) + 0\cdot P(\text{Draw}) + (+1)\cdot P(\text{Win}) = P(\text{Win}) - P(\text{Loss}) \in [-1, +1]
  $$

---

## 5. Policy Masking & Numerical Stability

What happens if column 3 already has 6 stones (the column is full)? The neural network's `Linear(1344 → 7)` still outputs a raw logit for column 3. If we applied `Softmax` directly, illegal columns would receive non-zero probability!

Instead, we **mask illegal columns before softmax**:
- **During inference (`src/model.py`, lines 175 & 187)**:
  ```python
  masked_logits = policy_logits.float().masked_fill(~valid_moves, -1e9)
  policy = F.softmax(masked_logits, dim=-1)
  ```
  Casting to `float32` first and filling illegal slots with `-1e9` makes $\exp(-10^9) = 0.0$ cleanly while avoiding `-inf` (which would produce `NaN` if a terminal board with *all* `valid_moves == False` were ever passed through!).
- **During mixed-precision training (`src/trainer.py`, lines 514–520)**:
  ```python
  occupied = states_t[:, 0] + states_t[:, 1]
  valid_moves = occupied.sum(dim=1) < ROWS
  policy_logits = policy_logits.masked_fill(~valid_moves, -1e4)
  log_probs = F.log_softmax(policy_logits.float(), dim=1)
  policy_loss = -torch.sum(target_policies_t * log_probs, dim=1).mean()
  ```
  **Why `-1e4` instead of `-1e9` or `-inf` here?**
  1. Under `float16` autocast (fallback on older GPUs without `bfloat16`), the maximum finite FP16 number is `65,504`—so `-1e9` overflows to `-inf`!
  2. In `target_policies_t * log_probs`, illegal columns have `target_policies_t == 0.0`. In IEEE-754 floating point, `0.0 * (-inf) == NaN`! Using `-1e4` stays well inside FP16 range (`exp(-10000) == 0` to machine precision, `0.0 * (-10000) == 0.0`), guaranteeing NaN-free gradients!

---

## 6. Inference Modes: From Zero-Sim Intuition to Interactive Play

Once a checkpoint is trained, `alphafour` uses `AlphaZeroNet` for inference in three distinct regimes:

### Mode 1: `0`-Simulation Raw Network Intuition (`predict` / `predict_wdl`)
- **Single forward pass, zero tree search** (~microseconds per position in batch).
- `policy, wdl_probs, value = model.predict_wdl(state, valid_moves)`
- Move choice: `argmax(policy)` over legal columns; outcome classification: `argmax(wdl_probs) - 1`.
- Measures **pure distilled intuition**—how much tactical and strategic understanding has been baked directly into the convolutional weights $\theta$ without any lookahead!

### Mode 2: Interactive Console Play with Hints & Tree Reuse (`play.py`)
When you run `uv run play.py --hints --simulations 800` ([`play.py`](../play.py)):
1. Loads the latest checkpoint cleanly via `_clean_state_dict` (`save_checkpoint` in [`src/utils.py`](../src/utils.py#L63-L120) unwraps `torch.compile` and DDP wrappers on save, and `_clean_state_dict` strips any `_orig_mod.` prefix on load).
2. Wraps the model in stateful `MCTS(model, num_simulations=800, device=device)`.
3. On the **human's turn** when `--hints` is enabled ([`play.py`](../play.py#L283-L285)), `display_hints` calls `mcts.get_move_values(game)` $\to$ `mcts.search_with_values(game, add_noise=False)`. Inside `MCTS._compute_move_values` ([`src/mcts.py`](../src/mcts.py#L462-L505)), each legal column's search evaluation is converted into a human-friendly **Win % from `0%` to `100%`**:

   $$
   \text{Win\%}(a) = \frac{-Q(\text{child}_a) + 1.0}{2.0} \times 100\%
   $$

   - Proven win (`SOLVER_LOSS` child): **`100%`** (green)
   - Proven draw (`SOLVER_DRAW` child): **`50%`** (yellow)
   - Proven loss (`SOLVER_WIN` child): **`0%`** (red)
4. On the **AI's turn** ([`play.py`](../play.py#L150-L180), [`play.py`](../play.py#L291-L294)), `get_ai_move` calls `mcts.get_action_probs(game, temperature=0, add_noise=False)` (and also supports `show_hints=True` via `search_with_values`), playing the column with the highest MCTS visit count.
5. After *every* move (human or AI), `mcts.advance_move(col)` reuses the searched child subtree for the next turn!

### Mode 3: High-Throughput Fused Batched Inference (`predict_packed`)
During Rust batched self-play and dataset evaluation, Python↔Rust boundary overhead and PCIe transfers can easily rival the GPU compute time of a 3.2M-parameter network! `AlphaZeroNet.predict_packed` ([`src/model.py`](../src/model.py#L193-L222)) eliminates every redundant copy:
- **Compact `uint8` Host-to-Device (H2D) Input**: Accepts `uint8[B, 3, 6, 7]` (`0` or `1` bytes—**4× smaller** PCIe transfer than `float32`) and converts `x = x.float()` directly on the GPU!
- **Single Packed `float32[B, 8]` Device-to-Host (D2H) Output**: Concatenates `[policy_0..6, value]` on GPU into one contiguous `(B, 8)` tensor so only **one** D2H copy and **one** PyO3 slice borrow (`evals_packed`) occur per MCTS step instead of two separate arrays!

---

## 7. Engineering Fast GPU Inference (`optimize_for_device`)

Look inside `AlphaZeroNet.optimize_for_device` ([`src/model.py`](../src/model.py#L123-L148))—it contains four subtle PyTorch performance optimizations tailored specifically to MCTS workloads:

1. **Disabling `cudnn.benchmark` (`torch.backends.cudnn.benchmark = False`)**:
   - In standard vision training, batch size $B$ is fixed (e.g., always `256`), so `cudnn.benchmark = True` benchmarks kernel algorithms once and speeds up training.
   - In MCTS leaf batching, **every single step has a different dynamic batch size $B$** (e.g., `B = 1412`, then `1389`, then `1205`, ..., down to `17` as games finish!). Enabling `cudnn.benchmark` would trigger hundreds of multi-second cuDNN autotuning stalls—making self-play $5\times$ slower!
2. **`torch.channels_last` (`NHWC`) Memory Format**:
   - Modern NVIDIA Tensor Cores execute 2D convolutions natively in `NHWC` (`channels_last`) layout rather than PyTorch's default `NCHW`. Converting both model weights and 4D inputs to `torch.channels_last` avoids internal layout transposes around every `Conv2d`.
3. **`bfloat16` Automatic Mixed Precision (`torch.autocast`) + TF32 Matmuls**:
   - Uses `torch.bfloat16` on Ampere+ GPUs (same 8-bit exponent range as `float32`, so it never overflows!) with `float16` fallback on older GPUs, while casting final policy/WDL logits back to `float32` before softmax so probabilities stay exact.
4. **Dynamic `torch.compile(..., dynamic=True)` with `B == 1` Guard**:
   - Compiles `_forward_packed_fn` and `_forward_wdl_fn` with `dynamic=True` so TorchInductor fuses BatchNorm, ReLU, SE-Block pointwise ops, masking, softmax, and concatenation into single Triton GPU kernels that work across arbitrary batch sizes $B \ge 2$ **without recompiling**!
   - Why `and states.shape[0] > 1` ([`src/model.py`](../src/model.py#L213))? In PyTorch's symbolic shape system, `s0 == 1` triggers a separate static shape specialization compile in TorchDynamo. Routing rare `B == 1` tail leaves through eager mode avoids a multi-second compiler stall for tiny 1-leaf batches!
   - Also notice `self._use_compiled_infer` is stored as a plain `bool` while `_COMPILED_PACKED_FN` lives at module scope—keeping `AlphaZeroNet` instances 100% picklable for `torch.multiprocessing.spawn`!

---

## 8. Summary & Next Steps

1. **`AlphaZeroNet` (`3.19M` params)** preserves the full $6 \times 7$ grid across 10 SE-ResBlocks, combining local $3 \times 3$ pattern detectors with global Squeeze-and-Excitation channel attention.
2. **The 16-channel, 3-logit WDL Value Head** explicitly models `[P(Loss), P(Draw), P(Win)]`, cleanly separating true draws from sharp 50/50 positions while outputting $V = P(\text{Win}) - P(\text{Loss})$ for MCTS.
3. **`predict_packed` + `uint8` H2D + `NHWC` + BF16 + dynamic `torch.compile`** turns thousands of dynamic-sized MCTS leaf batches into fused Tensor Core kernel calls.

Now that we understand both MCTS ([Chapter 02](./02-mcts-deep-dive.md)) and the Neural Network ([Chapter 03](./03-neural-network-and-inference.md)), let's connect them into the full **Self-Play & Training Pipeline**!

---

⬅️ **Previous**: **[Chapter 02: Core Search Algorithm — MCTS, PUCT, FPU & Exact MCTS-Solver](./02-mcts-deep-dive.md)**  
👉 **Next**: **[Chapter 04: Self-Play Data Generation, Replay Buffer & Training Pipeline](./04-self-play-and-training.md)**  
🏠 **Back to Index**: **[AlphaFour Documentation Hub (`docs/README.md`)](./README.md)**
