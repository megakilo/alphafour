"""Unit and single-GPU benchmark tests for AlphaZeroNet, ReplayBuffer, Trainer, and utils."""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.game import COLS, ROWS, ConnectFour
from src.model import AlphaZeroNet
from src.trainer import ReplayBuffer, Trainer
from src.utils import (
    _unwrap_model,
    get_device,
    get_latest_checkpoint,
    load_checkpoint,
    load_replay_buffer,
    save_checkpoint,
    save_model_for_play,
    save_replay_buffer,
)


def _make_random_examples(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic valid Connect Four states, legal policies, and WDL values."""
    states = np.zeros((n, 3, ROWS, COLS), dtype=np.float32)
    # Populate bottom rows with sparse pieces so columns stay mostly non-full,
    # and strictly zero policy probability on any full column.
    p0 = rng.random((n, ROWS, COLS)) < 0.22
    p1 = (rng.random((n, ROWS, COLS)) < 0.22) & (~p0)
    states[:, 0] = p0.astype(np.float32)
    states[:, 1] = p1.astype(np.float32)
    states[:, 2] = (rng.integers(0, 2, size=(n, 1, 1))).astype(np.float32)

    heights = (states[:, 0] + states[:, 1]).sum(axis=1)  # (n, COLS)
    valid = heights < ROWS
    # Guarantee at least column 3 is open on every synthetic board
    full_all = ~valid.any(axis=1)
    if np.any(full_all):
        states[full_all, 0, ROWS - 1, 3] = 0.0
        states[full_all, 1, ROWS - 1, 3] = 0.0
        valid[full_all, 3] = True

    raw_pol = (rng.random((n, COLS)).astype(np.float32) + 0.05) * valid.astype(
        np.float32
    )
    policies = raw_pol / raw_pol.sum(axis=1, keepdims=True)
    values = rng.choice(np.array([-1.0, 0.0, 1.0], dtype=np.float32), size=(n,))
    return states, policies, values


# ─────────────────────────────────────────────────────────────────────────────
# 1. AlphaZeroNet Architecture, Shapes & WDL Calibration
# ─────────────────────────────────────────────────────────────────────────────


def test_alphazeronet_architecture_and_shapes() -> None:
    model = AlphaZeroNet(num_res_blocks=2, num_filters=64, value_channels=16)
    model.eval()

    # Check 16-channel spatial value head + 3-logit WDL output without Tanh bottleneck
    val_layers = list(model.value_head.children())
    assert isinstance(val_layers[0], nn.Conv2d)
    assert val_layers[0].in_channels == 64
    assert val_layers[0].out_channels == 16
    assert isinstance(val_layers[1], nn.BatchNorm2d)
    assert val_layers[1].num_features == 16
    assert isinstance(val_layers[4], nn.Linear)
    assert val_layers[4].in_features == 16 * ROWS * COLS
    assert val_layers[4].out_features == 256
    assert isinstance(val_layers[6], nn.Linear)
    assert val_layers[6].in_features == 256
    assert val_layers[6].out_features == 3
    assert not any(isinstance(m, nn.Tanh) for m in val_layers)

    # Forward shapes: (B, 7) and (B, 3)
    batch_x = torch.randn(5, 3, ROWS, COLS)
    pol_logits, wdl_logits = model(batch_x)
    assert pol_logits.shape == (5, COLS)
    assert wdl_logits.shape == (5, 3)

    # Single 3D predict & predict_wdl shapes
    single_x = torch.randn(3, ROWS, COLS)
    valid_1d = torch.tensor([True, False, True, True, False, True, True])
    pol_1d, val_0d = model.predict(single_x, valid_1d)
    assert pol_1d.shape == (COLS,)
    assert val_0d.shape == ()
    assert float(pol_1d[1]) == 0.0 and float(pol_1d[4]) == 0.0
    assert math.isclose(float(pol_1d.sum()), 1.0, rel_tol=1e-5)
    assert -1.0 <= float(val_0d) <= 1.0

    pol_w1, wdl_w1, val_w1 = model.predict_wdl(single_x, valid_1d)
    assert pol_w1.shape == (COLS,)
    assert wdl_w1.shape == (3,)
    assert val_w1.shape == ()
    assert math.isclose(float(wdl_w1.sum()), 1.0, rel_tol=1e-5)
    assert math.isclose(float(val_w1), float(wdl_w1[2] - wdl_w1[0]), abs_tol=1e-6)

    # Batched 4D predict & predict_wdl shapes (including B=1 and B=8)
    for b in (1, 8):
        bx = torch.randn(b, 3, ROWS, COLS)
        bv = torch.ones((b, COLS), dtype=torch.bool)
        bv[:, 0] = False
        pol_b, val_b = model.predict(bx, bv)
        assert pol_b.shape == (b, COLS)
        assert val_b.shape == (b,)
        assert torch.all(pol_b[:, 0] == 0.0)
        assert torch.allclose(pol_b.sum(dim=-1), torch.ones(b), atol=1e-5)

        pol_wb, wdl_wb, val_wb = model.predict_wdl(bx, bv)
        assert pol_wb.shape == (b, COLS)
        assert wdl_wb.shape == (b, 3)
        assert val_wb.shape == (b,)
        assert torch.allclose(wdl_wb.sum(dim=-1), torch.ones(b), atol=1e-5)
        assert torch.allclose(val_wb, wdl_wb[:, 2] - wdl_wb[:, 0], atol=1e-6)


def test_wdl_value_sign_calibration_and_fp16_bf16_masking() -> None:
    # Exact analytical checks on wdl_logits_to_value ([Loss (-1), Draw (0), Win (+1)])
    logits = torch.tensor(
        [
            [30.0, 0.0, -30.0],  # Pure Loss -> -1.0
            [-30.0, 30.0, -30.0],  # Pure Draw -> 0.0
            [-30.0, 0.0, 30.0],  # Pure Win -> +1.0
            [4.0, 1.0, 4.0],  # Equal Win/Loss -> 0.0
            [math.log(0.2), math.log(0.3), math.log(0.5)],  # 0.5 - 0.2 = +0.3
        ],
        dtype=torch.float32,
    )
    vals = AlphaZeroNet.wdl_logits_to_value(logits)
    expected = torch.tensor([-1.0, 0.0, 1.0, 0.0, 0.3], dtype=torch.float32)
    assert torch.allclose(vals, expected, atol=1e-5)

    # Also test float16 and bfloat16 inputs to wdl_logits_to_value
    for dt in (torch.float16, torch.bfloat16):
        vals_half = AlphaZeroNet.wdl_logits_to_value(logits.to(dt))
        assert torch.isfinite(vals_half).all()
        assert torch.allclose(vals_half.float(), expected, atol=1e-2)

    # Test safe valid_moves masking in FP16 and BF16 (no overflow, numpy conversion works)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16).to(device)
    model.eval()
    states = torch.randn(4, 3, ROWS, COLS, device=device)
    valid = torch.tensor(
        [
            [False, False, False, True, False, False, False],
            [True, False, True, False, True, False, True],
            [False, False, False, False, False, False, True],
            [False, False, False, False, False, False, False],  # Degenerate all-false safety
        ],
        device=device,
    )

    amp_dtypes = [torch.float16, torch.bfloat16] if device.type == "cuda" else [torch.bfloat16]
    for amp_dt in amp_dtypes:
        with torch.autocast(device_type=device.type, dtype=amp_dt):
            pol, wdl, val = model.predict_wdl(states, valid)
            assert torch.isfinite(pol).all()
            assert torch.isfinite(wdl).all()
            assert torch.isfinite(val).all()
            assert math.isclose(float(pol[0, 3]), 1.0, abs_tol=1e-5)
            assert float(pol[1, 1]) == 0.0
            # Verify .cpu().numpy() never fails with BFloat16 TypeError
            _ = pol.cpu().numpy()
            _ = wdl.cpu().numpy()
            _ = val.cpu().numpy()


def test_wdl_training_convergence_and_calibration() -> None:
    """Verify Trainer WDL loss learns Loss (-1), Draw (0), Win (+1), and soft targets (+0.5)."""
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16).to(device)
    trainer = Trainer(model=model, device=device, lr=3e-3, batch_size=32, total_iterations=20)

    # Construct 4 distinct board patterns with targets -1.0, 0.0, +1.0, +0.5
    g_loss = ConnectFour().make_move(0)
    g_draw = ConnectFour().make_move(2).make_move(3)
    g_win = ConnectFour().make_move(3)
    g_soft = ConnectFour().make_move(4).make_move(4).make_move(2)

    patterns = [
        (g_loss.encode(), -1.0, 0),
        (g_draw.encode(), 0.0, 3),
        (g_win.encode(), 1.0, 3),
        (g_soft.encode(), 0.5, 2),
    ]

    buf = ReplayBuffer(capacity=256, device=device)
    states_list, pols_list, vals_list = [], [], []
    for _ in range(32):
        for st, v, best_col in patterns:
            pol = np.zeros(COLS, dtype=np.float32)
            pol[best_col] = 1.0
            states_list.append(st)
            pols_list.append(pol)
            vals_list.append(v)

    buf.add_batch(
        np.stack(states_list),
        np.stack(pols_list),
        np.array(vals_list, dtype=np.float32),
    )

    for _ in range(25):
        losses = trainer.train_epoch(buf, num_batches=8)
        assert math.isfinite(losses["total_loss"])

    model.eval()
    eval_states = torch.from_numpy(np.stack([p[0] for p in patterns])).to(device)
    pol_pred, wdl_pred, val_pred = model.predict_wdl(eval_states)

    # Loss state -> class 0 dominant, value near -1
    assert int(torch.argmax(wdl_pred[0])) == 0
    assert float(val_pred[0]) < -0.85
    # Draw state -> class 1 dominant, value near 0
    assert int(torch.argmax(wdl_pred[1])) == 1
    assert abs(float(val_pred[1])) < 0.15
    # Win state -> class 2 dominant, value near +1
    assert int(torch.argmax(wdl_pred[2])) == 2
    assert float(val_pred[2]) > 0.85
    # Soft +0.5 state -> P(Win) - P(Loss) calibrated around +0.5
    assert 0.30 < float(val_pred[3]) < 0.70


# ─────────────────────────────────────────────────────────────────────────────
# 2. ReplayBuffer Ring-Buffer, Inverse-CDF 2:1 Sampling & State Serialization
# ─────────────────────────────────────────────────────────────────────────────


def test_replay_buffer_ring_wraparound_and_formats() -> None:
    # Verify 300K uint8 memory footprint (~47.4 MB total)
    big_buf = ReplayBuffer(capacity=300_000, device="cpu")
    total_bytes = (
        big_buf.states_buf.nbytes
        + big_buf.policies_buf.nbytes
        + big_buf.values_buf.nbytes
    )
    assert big_buf.states_buf.dtype == torch.uint8
    assert total_bytes == 300_000 * (3 * ROWS * COLS * 1 + COLS * 4 + 4)  # 47,400,000 bytes

    # Test wrap-around chronological ordering on capacity=10
    buf = ReplayBuffer(capacity=10, device="cpu")
    for batch_start, count in ((0, 6), (6, 7)):  # Total 13 items -> survivors: 3..12
        states = np.zeros((count, 3, ROWS, COLS), dtype=np.float32)
        policies = np.zeros((count, COLS), dtype=np.float32)
        values = np.arange(batch_start, batch_start + count, dtype=np.float32)
        for k in range(count):
            idx_val = batch_start + k
            r, c = divmod(idx_val, COLS)
            states[k, 0, r % ROWS, c] = 1.0
            policies[k, idx_val % COLS] = 1.0
        buf.add_batch(states, policies, values)

    assert len(buf) == 10
    state_dict = buf.get_state()
    assert state_dict["states"].dtype == torch.uint8
    assert state_dict["values"].tolist() == [float(x) for x in range(3, 13)]
    for rank, idx_val in enumerate(range(3, 13)):
        r, c = divmod(idx_val, COLS)
        assert int(state_dict["states"][rank, 0, r % ROWS, c]) == 1
        assert float(state_dict["policies"][rank, idx_val % COLS]) == 1.0

    # Test batch larger than capacity (add 15 items into capacity 10 -> keeps newest 10)
    rng = np.random.default_rng(123)
    st15, pol15, _ = _make_random_examples(15, rng)
    val15 = np.arange(100, 115, dtype=np.float32)
    buf.add((st15, pol15, val15))
    assert len(buf) == 10
    assert buf.get_state()["values"].tolist() == [float(x) for x in range(105, 115)]

    # Test legacy list-of-tuples add() and load_state()
    legacy_examples = [
        (st15[i], pol15[i], float(i)) for i in range(5)
    ]
    buf2 = ReplayBuffer(capacity=20, device="cpu")
    buf2.add(legacy_examples)
    assert len(buf2) == 5

    # Compact storage check: get_state() on 5/20 capacity must NOT retain 20-item backing storage
    compact = buf2.get_state()
    assert compact["states"].untyped_storage().nbytes() == 5 * 3 * ROWS * COLS

    buf3 = ReplayBuffer(capacity=20, device="cpu")
    buf3.load_state(legacy_examples)
    assert len(buf3) == 5
    assert buf3.get_state()["values"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    # Sample wrapper returns float32 numpy arrays
    s_np, p_np, v_np = buf3.sample(4)
    assert s_np.shape == (4, 3, ROWS, COLS) and s_np.dtype == np.float32
    assert p_np.shape == (4, COLS) and p_np.dtype == np.float32
    assert v_np.shape == (4,) and v_np.dtype == np.float32


def test_replay_buffer_inverse_cdf_2_to_1_recency_ratio() -> None:
    """Statistically verify O(B) analytical inverse-CDF has linear 2:1 newest:oldest density."""
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = 1_000
    # Use capacity=1_000 and force wrap-around first (write 400 dummy, then 1_000 ranked 0..999)
    buf = ReplayBuffer(capacity=n, device=device)
    dummy_s = np.zeros((400, 3, ROWS, COLS), dtype=np.float32)
    dummy_p = np.full((400, COLS), 1.0 / COLS, dtype=np.float32)
    dummy_v = np.full((400,), -999.0, dtype=np.float32)
    buf.add_batch(dummy_s, dummy_p, dummy_v)

    s = np.zeros((n, 3, ROWS, COLS), dtype=np.float32)
    p = np.full((n, COLS), 1.0 / COLS, dtype=np.float32)
    ranks = np.arange(n, dtype=np.float32)  # 0 = oldest, 999 = newest
    buf.add_batch(s, p, ranks)

    # Sample 1,000,000 draws in batches of 500
    counts = torch.zeros(n, dtype=torch.float64)
    for _ in range(2_000):
        _, _, sampled_ranks = buf.sample_tensors(500, device=device)
        counts += torch.bincount(sampled_ranks.long().cpu(), minlength=n).double()

    # Never sampled overwritten dummy items
    assert counts.sum().item() == 1_000_000.0

    # Fit linear density across 10 deciles: theoretical share of decile d in 0..9 is
    # integral_{d/10}^{(d+1)/10} (2/3)(1 + x) dx = (2/3)*(0.1 + (2d + 1)*0.005)
    decile_shares = counts.view(10, 100).sum(dim=1).numpy() / 1_000_000.0
    x_centers = (np.arange(10) + 0.5) / 10.0
    expected_shares = (2.0 / 3.0) * 0.1 * (1.0 + x_centers)
    np.testing.assert_allclose(decile_shares, expected_shares, atol=0.003)

    # Fit line w(x) = a + b*x to empirical PDF and check endpoint ratio w(1) / w(0) ≈ 2.0
    slope, intercept = np.polyfit(x_centers, decile_shares * 10.0, 1)
    endpoint_ratio = (intercept + slope) / intercept
    assert 1.95 <= endpoint_ratio <= 2.05


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trainer Weight Decay Groups & Monotonic Warmup+Cosine LR Schedule
# ─────────────────────────────────────────────────────────────────────────────


def test_trainer_weight_decay_groups_and_lr_schedule_no_spike() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    trainer = Trainer(
        model=model,
        device=device,
        lr=0.001,
        weight_decay=1e-4,
        batch_size=64,
        total_iterations=150,
    )

    # Check parameter groups: 2D+ conv/linear weights get 1e-4, 1D BN/bias get 0.0
    assert len(trainer.optimizer.param_groups) == 2
    decay_group, no_decay_group = trainer.optimizer.param_groups
    assert decay_group["weight_decay"] == 1e-4
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay_group["params"])
    assert all(p.ndim < 2 for p in no_decay_group["params"])

    grouped_numel = sum(
        p.numel() for g in trainer.optimizer.param_groups for p in g["params"]
    )
    total_numel = sum(p.numel() for p in model.parameters())
    assert grouped_numel == total_numel

    if device.type == "cuda":
        assert trainer.optimizer.defaults.get("fused") is True

    # Record LR across 165 iterations (including iter 140/141/150 and post-150 clamping)
    lrs: list[float] = []
    for _ in range(165):
        lrs.append(trainer.optimizer.param_groups[0]["lr"])
        trainer.optimizer.step()
        trainer.step_scheduler()

    # Warmup (steps 0..5): starts at 0.2 * 0.001 = 0.0002 and rises strictly to 0.001 at index 5
    assert math.isclose(lrs[0], 0.0002, rel_tol=1e-6)
    for i in range(5):
        assert lrs[i + 1] > lrs[i], f"Expected warmup increase at step {i}: {lrs[i]} -> {lrs[i+1]}"
    assert math.isclose(lrs[5], 0.001, rel_tol=1e-6)

    # Cosine decay (steps 5..150): strictly monotonically decreasing all the way to 1e-5
    for i in range(5, 150):
        assert lrs[i + 1] < lrs[i], f"Unexpected LR increase at iter {i+1}: {lrs[i]} -> {lrs[i+1]}"

    # Specifically verify ZERO warm-restart spike around iter 140/141/150!
    assert lrs[140] < 5e-5
    assert lrs[141] < lrs[140]
    assert math.isclose(lrs[150], 1e-5, rel_tol=1e-5)

    # Clamped safely past total_iterations (steps 150..164 stay flat at eta_min = 1e-5)
    for i in range(150, 165):
        assert math.isclose(lrs[i], 1e-5, rel_tol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Checkpoints, Clean Compiled State Dicts & Separate Replay Buffer I/O
# ─────────────────────────────────────────────────────────────────────────────


class _DummyCompiledWrapper(nn.Module):
    """Mimics torch.compile OptimizedModule with `_orig_mod` attribute and prefixed keys."""

    def __init__(self, mod: nn.Module) -> None:
        super().__init__()
        self._orig_mod = mod

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)


def test_checkpoint_and_replay_buffer_io(tmp_path: Path) -> None:
    device = torch.device("cpu")
    model = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    trainer = Trainer(model=model, device=device, lr=1e-3, total_iterations=20)
    trainer.step_scheduler()

    # 1. Default save_checkpoint omits 'replay_buffer' for lightweight checkpoints
    ckpt_path = tmp_path / "checkpoint_0001.pt"
    save_checkpoint(
        model=model,
        optimizer=trainer.optimizer,
        iteration=1,
        path=ckpt_path,
        scheduler=trainer.scheduler,
    )
    assert ckpt_path.exists()
    assert not ckpt_path.with_suffix(".tmp").exists()

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "replay_buffer" not in raw_ckpt
    assert raw_ckpt["iteration"] == 1
    assert not any(k.startswith("_orig_mod.") for k in raw_ckpt["model_state_dict"])

    # 2. Wrapped / compiled model saves clean state_dict and loads into fresh model
    wrapped = _DummyCompiledWrapper(model)
    assert _unwrap_model(wrapped) is model
    ckpt_path_2 = tmp_path / "checkpoint_0002.pt"
    save_checkpoint(wrapped, trainer.optimizer, 2, ckpt_path_2, scheduler=trainer.scheduler)
    assert get_latest_checkpoint(tmp_path) == ckpt_path_2

    model_restored = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    trainer_restored = Trainer(model=model_restored, device=device, lr=1e-3, total_iterations=20)
    loaded = load_checkpoint(
        ckpt_path_2,
        _DummyCompiledWrapper(model_restored),
        optimizer=trainer_restored.optimizer,
        device=device,
        scheduler=trainer_restored.scheduler,
    )
    assert loaded["iteration"] == 2
    for k, v in model.state_dict().items():
        assert torch.equal(v, model_restored.state_dict()[k])

    # Also verify loading a checkpoint whose state_dict already had '_orig_mod.' prefixes
    prefixed_sd = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    legacy_prefixed_path = tmp_path / "prefixed.pt"
    torch.save({"iteration": 3, "model_state_dict": prefixed_sd}, legacy_prefixed_path)
    model_from_prefixed = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    load_checkpoint(legacy_prefixed_path, model_from_prefixed, device=device)
    for k, v in model.state_dict().items():
        assert torch.equal(v, model_from_prefixed.state_dict()[k])

    # Verify save_model_for_play strips _orig_mod. cleanly
    play_path = tmp_path / "play_weights.pt"
    save_model_for_play(wrapped, play_path)
    play_sd = torch.load(play_path, map_location="cpu", weights_only=False)
    assert not any(k.startswith("_orig_mod.") for k in play_sd)

    # 3. Separate save_replay_buffer / load_replay_buffer atomic round-trip
    rng = np.random.default_rng(99)
    st, pol, val = _make_random_examples(120, rng)
    rb = ReplayBuffer(capacity=100, device="cpu")  # Forces wrap-around (120 > 100)
    rb.add_batch(st, pol, val)

    rb_path = tmp_path / "replay_buffer.pt"
    save_replay_buffer(rb.get_state(), rb_path)
    assert rb_path.exists()
    assert not rb_path.with_suffix(".tmp").exists()

    rb_loaded_state = load_replay_buffer(rb_path)
    rb2 = ReplayBuffer(capacity=100, device="cpu")
    rb2.load_state(rb_loaded_state)
    assert len(rb2) == 100
    s1, s2 = rb.get_state(), rb2.get_state()
    assert torch.equal(s1["states"], s2["states"])
    assert torch.equal(s1["policies"], s2["policies"])
    assert torch.equal(s1["values"], s2["values"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Single-GPU BF16 Training, Full-Column Masking Safety & Speed Benchmark
# ─────────────────────────────────────────────────────────────────────────────


def test_full_column_masking_no_nan_in_trainer() -> None:
    """Ensure full columns (valid_moves=False) never cause FP16/BF16 overflow or 0 * -inf NaN."""
    device = get_device()
    model = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    trainer = Trainer(model=model, device=device, lr=1e-3, batch_size=16, total_iterations=10)

    # Create boards where columns 0, 1, 2 are completely full (6 pieces)
    states = np.zeros((32, 3, ROWS, COLS), dtype=np.float32)
    states[:, 0, :3, :3] = 1.0
    states[:, 1, 3:, :3] = 1.0  # Height = 6 in cols 0, 1, 2 -> invalid!
    policies = np.zeros((32, COLS), dtype=np.float32)
    policies[:, 3:] = 0.25  # Probability only on legal cols 3..6
    values = np.tile(np.array([-1.0, 0.0, 1.0, 0.5], dtype=np.float32), 8)

    buf = ReplayBuffer(capacity=64, device=device)
    buf.add_batch(states, policies, values)

    losses = trainer.train_epoch(buf, num_batches=4)
    assert math.isfinite(losses["policy_loss"]) and losses["policy_loss"] > 0.0
    assert math.isfinite(losses["value_loss"]) and losses["value_loss"] > 0.0
    assert math.isfinite(losses["total_loss"])


def test_compile_fallback_safety() -> None:
    """If a compiled wrapper fails lazily during train_epoch, Trainer falls back cleanly to eager."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    trainer = Trainer(model=model, device=device, batch_size=16, use_compile=False)

    def _broken_compiled_fn(_x: torch.Tensor):
        raise RuntimeError("Simulated Triton backend failure")

    trainer.use_compile = True
    trainer._train_model = _broken_compiled_fn  # type: ignore[assignment]

    rng = np.random.default_rng(5)
    buf = ReplayBuffer(capacity=64, device=device)
    buf.add_batch(*_make_random_examples(32, rng))

    losses = trainer.train_epoch(buf, num_batches=2)
    assert trainer.use_compile is False
    assert trainer._train_model is trainer.model
    assert math.isfinite(losses["total_loss"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Single-GPU CUDA benchmark")
def test_single_gpu_bf16_training_speed_benchmark() -> None:
    """Benchmark full 10x128 AlphaZeroNet BF16 training + GPU uint8 ReplayBuffer throughput."""
    device = torch.device("cuda")
    rng = np.random.default_rng(42)

    buf = ReplayBuffer(capacity=300_000, device=device)
    st, pol, val = _make_random_examples(50_000, rng)
    buf.add_batch(st, pol, val)
    assert buf.states_buf.device.type == "cuda"

    model = AlphaZeroNet(num_res_blocks=10, num_filters=128, value_channels=16)
    trainer = Trainer(
        model=model,
        device=device,
        lr=1e-3,
        batch_size=256,
        total_iterations=150,
        use_compile=False,
    )
    if torch.cuda.is_bf16_supported():
        assert trainer._amp_dtype == torch.bfloat16
        assert trainer._use_scaler is False

    # Warmup GPU kernels / cuDNN autotuner
    warmup_losses = trainer.train_epoch(buf, num_batches=20)
    assert math.isfinite(warmup_losses["total_loss"])
    torch.cuda.synchronize()

    # Timed benchmark: 200 batches of 256 (51,200 positions)
    num_batches = 200
    t0 = time.perf_counter()
    losses = trainer.train_epoch(buf, num_batches=num_batches)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    batches_per_sec = num_batches / elapsed
    samples_per_sec = (num_batches * 256) / elapsed
    print(
        f"\n[GPU Benchmark] 10x128 WDL BF16 eager: {num_batches}x256 in {elapsed:.3f}s "
        f"({batches_per_sec:.1f} batches/s, {samples_per_sec:,.0f} pos/s), "
        f"loss={losses['total_loss']:.4f}"
    )

    assert math.isfinite(losses["policy_loss"])
    assert math.isfinite(losses["value_loss"])
    assert losses["total_loss"] < 10.0
    assert batches_per_sec > 10.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Single-GPU CUDA compile test")
def test_cuda_use_compile_and_checkpoint_clean_keys(tmp_path: Path) -> None:
    """Verify Trainer(use_compile=True) runs on CUDA and saves clean unprefixed state_dict."""
    device = torch.device("cuda")
    rng = np.random.default_rng(77)
    buf = ReplayBuffer(capacity=512, device=device)
    buf.add_batch(*_make_random_examples(256, rng))

    model = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    trainer = Trainer(
        model=model,
        device=device,
        lr=1e-3,
        batch_size=64,
        total_iterations=20,
        use_compile=True,
    )
    losses = trainer.train_epoch(buf, num_batches=3)
    assert math.isfinite(losses["total_loss"])

    ckpt_path = tmp_path / "checkpoint_0001.pt"
    save_checkpoint(
        model=trainer._train_model,
        optimizer=trainer.optimizer,
        iteration=1,
        path=ckpt_path,
        scheduler=trainer.scheduler,
    )
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert not any(k.startswith("_orig_mod.") for k in raw["model_state_dict"])

    fresh = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    load_checkpoint(ckpt_path, fresh, device=torch.device("cpu"))
    for k, v in model.cpu().state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Checkpoint Resume LR Scheduler Safety (Horizon Changes & Legacy Checkpoints)
# ─────────────────────────────────────────────────────────────────────────────


def test_scheduler_resume_extended_horizon_no_spike(tmp_path: Path) -> None:
    """Resuming a 50-iter checkpoint with total_iterations=100 must never explode LR."""
    device = torch.device("cpu")
    rng = np.random.default_rng(123)
    buf = ReplayBuffer(capacity=128, device=device)
    buf.add_batch(*_make_random_examples(64, rng))

    model_50 = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    trainer_50 = Trainer(
        model=model_50, device=device, lr=1e-3, batch_size=32, total_iterations=50
    )
    trainer_50.train_epoch(buf, num_batches=1)  # Populate real AdamW momentum buffers
    for _ in range(50):
        trainer_50.step_scheduler()
    assert math.isclose(trainer_50.optimizer.param_groups[0]["lr"], 1e-5, rel_tol=1e-6)

    ckpt_50 = tmp_path / "checkpoint_0050.pt"
    save_checkpoint(
        model=model_50,
        optimizer=trainer_50.optimizer,
        iteration=50,
        path=ckpt_50,
        scheduler=trainer_50.scheduler,
    )

    # Reference uninterrupted 100-iter schedule
    ref_model = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    ref_trainer = Trainer(model=ref_model, device=device, lr=1e-3, total_iterations=100)
    ref_lrs: list[float] = []
    for _ in range(110):
        ref_lrs.append(ref_trainer.optimizer.param_groups[0]["lr"])
        ref_trainer.step_scheduler()

    # Path A: main.py pattern — load_checkpoint + trainer.restore_scheduler(...)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model_100 = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
        trainer_100 = Trainer(
            model=model_100, device=device, lr=1e-3, batch_size=32, total_iterations=100
        )
        ckpt = load_checkpoint(ckpt_50, model_100, trainer_100.optimizer, device=device)
        trainer_100.restore_scheduler(
            ckpt["iteration"], ckpt.get("scheduler_state_dict")
        )

        # AdamW state must remain intact after scheduler rebuild
        assert len(trainer_100.optimizer.state) > 0

        resumed_lrs: list[float] = []
        for _ in range(50, 110):
            lr_now = trainer_100.optimizer.param_groups[0]["lr"]
            assert math.isclose(
                lr_now, trainer_100.scheduler.get_last_lr()[0], rel_tol=1e-9
            )
            resumed_lrs.append(lr_now)
            trainer_100.step_scheduler()

    assert len(caught_warnings) == 0, [str(w.message) for w in caught_warnings]

    # Every LR from iter 50..109 must lie strictly in [1e-5, 1e-3] and match reference 100-iter curve
    for offset, lr_val in enumerate(resumed_lrs):
        step_idx = 50 + offset
        assert 1e-5 - 1e-12 <= lr_val <= 1e-3 + 1e-12, (
            f"LR out of bounds at step {step_idx}: {lr_val}"
        )
        assert math.isclose(lr_val, ref_lrs[step_idx], rel_tol=1e-6), (
            f"Mismatch vs 100-iter schedule at step {step_idx}: {lr_val} vs {ref_lrs[step_idx]}"
        )

    # Strictly monotonically decreasing across cosine steps 50..100, then clamped at 1e-5
    for idx in range(50):
        assert resumed_lrs[idx + 1] < resumed_lrs[idx], (
            f"Expected monotonic decay at step {50 + idx}: {resumed_lrs[idx]} -> {resumed_lrs[idx + 1]}"
        )
    for idx in range(50, len(resumed_lrs)):
        assert math.isclose(resumed_lrs[idx], 1e-5, rel_tol=1e-6)

    # Path B: load_checkpoint(..., scheduler=trainer.scheduler) direct helper path
    model_100_b = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    trainer_100_b = Trainer(
        model=model_100_b, device=device, lr=1e-3, total_iterations=100
    )
    load_checkpoint(
        ckpt_50,
        model_100_b,
        optimizer=trainer_100_b.optimizer,
        device=device,
        scheduler=trainer_100_b.scheduler,
    )
    for step_idx in range(50, 105):
        lr_b = trainer_100_b.optimizer.param_groups[0]["lr"]
        assert 1e-5 - 1e-12 <= lr_b <= 1e-3 + 1e-12
        assert math.isclose(lr_b, ref_lrs[step_idx], rel_tol=1e-6)
        trainer_100_b.step_scheduler()


def test_scheduler_resume_shortened_horizon_clamped_at_eta_min(tmp_path: Path) -> None:
    """Resuming a 100-iter checkpoint (at iter 80 or 100) with total_iterations=50 clamps at eta_min."""
    device = torch.device("cpu")
    model_100 = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    trainer_100 = Trainer(model=model_100, device=device, lr=1e-3, total_iterations=100)

    for _ in range(80):
        trainer_100.step_scheduler()
    ckpt_80 = tmp_path / "checkpoint_0080.pt"
    save_checkpoint(
        model=model_100,
        optimizer=trainer_100.optimizer,
        iteration=80,
        path=ckpt_80,
        scheduler=trainer_100.scheduler,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model_50 = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
        trainer_50 = Trainer(model=model_50, device=device, lr=1e-3, total_iterations=50)
        ckpt = load_checkpoint(ckpt_80, model_50, trainer_50.optimizer, device=device)
        trainer_50.restore_scheduler(
            ckpt["iteration"], ckpt.get("scheduler_state_dict")
        )

        for _ in range(10):
            for pg in trainer_50.optimizer.param_groups:
                assert math.isclose(pg["lr"], 1e-5, rel_tol=1e-6)
            assert math.isclose(
                trainer_50.scheduler.get_last_lr()[0], 1e-5, rel_tol=1e-6
            )
            trainer_50.step_scheduler()

    assert len(caught_warnings) == 0, [str(w.message) for w in caught_warnings]


def test_scheduler_resume_legacy_checkpoint_without_scheduler_state(
    tmp_path: Path,
) -> None:
    """Resuming a legacy checkpoint without scheduler_state_dict fast-forwards to iteration."""
    device = torch.device("cpu")
    model_legacy = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    trainer_legacy = Trainer(
        model=model_legacy, device=device, lr=1e-3, total_iterations=100
    )

    # Save legacy checkpoint at iteration=25 WITHOUT passing scheduler=...
    legacy_ckpt_path = tmp_path / "checkpoint_0025.pt"
    save_checkpoint(
        model=model_legacy,
        optimizer=trainer_legacy.optimizer,
        iteration=25,
        path=legacy_ckpt_path,
        scheduler=None,
    )
    raw = torch.load(legacy_ckpt_path, map_location="cpu", weights_only=False)
    assert "scheduler_state_dict" not in raw

    # Reference 100-iter schedule at steps 25..105
    ref_model = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    ref_trainer = Trainer(model=ref_model, device=device, lr=1e-3, total_iterations=100)
    ref_lrs: list[float] = []
    for _ in range(105):
        ref_lrs.append(ref_trainer.optimizer.param_groups[0]["lr"])
        ref_trainer.step_scheduler()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        resumed_model = AlphaZeroNet(
            num_res_blocks=2, num_filters=32, value_channels=16
        )
        resumed_trainer = Trainer(
            model=resumed_model, device=device, lr=1e-3, total_iterations=100
        )
        ckpt = load_checkpoint(
            legacy_ckpt_path, resumed_model, resumed_trainer.optimizer, device=device
        )
        resumed_trainer.restore_scheduler(
            ckpt["iteration"], ckpt.get("scheduler_state_dict")
        )

        # Must NOT reset to iter-0 warmup (0.0002); must equal step 25 (~0.000895)
        lr_at_25 = resumed_trainer.optimizer.param_groups[0]["lr"]
        assert lr_at_25 > 0.0008
        assert math.isclose(lr_at_25, ref_lrs[25], rel_tol=1e-6)

        for step_idx in range(25, 105):
            cur_lr = resumed_trainer.optimizer.param_groups[0]["lr"]
            assert 1e-5 - 1e-12 <= cur_lr <= 1e-3 + 1e-12
            assert math.isclose(cur_lr, ref_lrs[step_idx], rel_tol=1e-6)
            resumed_trainer.step_scheduler()

    assert len(caught_warnings) == 0, [str(w.message) for w in caught_warnings]


def test_predict_packed_uint8_and_dynamic_compile_equivalence() -> None:
    """Verify optimize_for_device + predict_packed(uint8) matches eager float32 across dynamic batch sizes."""
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16).to(device)
    ref.eval()

    opt = AlphaZeroNet(num_res_blocks=2, num_filters=32, value_channels=16)
    opt.load_state_dict(ref.state_dict())
    opt.optimize_for_device(device, use_compile=(device.type == "cuda"))
    opt.eval()

    # state_dict keys must remain identical
    assert list(opt.state_dict().keys()) == list(ref.state_dict().keys())

    for bsz in (1, 13, 128):
        states_u8 = torch.randint(0, 2, (bsz, 3, ROWS, COLS), dtype=torch.uint8, device=device)
        valid = torch.ones((bsz, COLS), dtype=torch.bool, device=device)
        valid[:, 0] = False

        ref_pol, ref_val = ref.predict(states_u8.float(), valid)
        packed = opt.predict_packed(states_u8, valid)
        assert packed.shape == (bsz, COLS + 1)
        assert packed.dtype == torch.float32
        assert torch.all(packed[:, 0] == 0.0)
        assert torch.allclose(ref_pol, packed[:, :COLS], atol=2e-2)
        assert torch.allclose(ref_val, packed[:, COLS], atol=2e-2)
