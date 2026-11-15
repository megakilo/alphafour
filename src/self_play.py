"""Self-play data generation with batched MCTS.

Uses the Rust engine (alphafour_engine) for high-performance parallel tree search,
with single-GPU PyTorch BF16/FP16 inference, plus full-parity Python fallback.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import numpy as np
import torch
from tqdm import tqdm

from .game import COLS, ROWS, ConnectFour
from .mcts import (
    DEFAULT_C_FPU,
    DEFAULT_DIRICHLET_ALPHA,
    SOLVER_DRAW,
    SOLVER_LOSS,
    SOLVER_ONGOING,
    SOLVER_WIN,
    MCTSNode,
)
from .model import AlphaZeroNet

# Temperature schedule: sample proportionally for first N plies, then play greedily
TEMP_THRESHOLD = 10
RANDOM_OPENING_PLIES = 4

try:
    from alphafour_engine import RustBatchedSelfPlay

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

SelfPlayExample = tuple[np.ndarray, np.ndarray, float]
SelfPlayArrays = tuple[np.ndarray, np.ndarray, np.ndarray]


@contextlib.contextmanager
def _inference_context(device: torch.device) -> Iterator[None]:
    """Run model inference under no_grad + BF16 (or FP16 fallback) autocast on CUDA."""
    if device.type == "cuda":
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        with torch.no_grad():
            yield


def _predict_packed_numpy(
    model: AlphaZeroNet,
    device: torch.device,
    states_np: np.ndarray,
    valid_np: np.ndarray,
) -> np.ndarray:
    """Run batched GPU inference and return contiguous `float32[K, 8]` (`[policy_0..6, value]`)."""
    states_t = torch.from_numpy(states_np).to(device, non_blocking=True)
    valid_t = torch.from_numpy(valid_np).to(device, non_blocking=True)
    if hasattr(model, "predict_packed"):
        packed_t = model.predict_packed(states_t, valid_t)
        return np.ascontiguousarray(packed_t.cpu().numpy(), dtype=np.float32)

    with _inference_context(device):
        states_f = states_t.float() if states_t.dtype == torch.uint8 else states_t
        policies, values = model.predict(states_f, valid_t)
        packed_t = torch.cat(
            (policies.float(), values.float().reshape(-1, 1)), dim=-1
        )
    return np.ascontiguousarray(packed_t.cpu().numpy(), dtype=np.float32)


def _empty_arrays(uint8_states: bool = False) -> SelfPlayArrays:
    state_dtype = np.uint8 if uint8_states else np.float32
    return (
        np.empty((0, 3, ROWS, COLS), dtype=state_dtype),
        np.empty((0, COLS), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )


def _arrays_to_examples(
    states_arr: np.ndarray,
    policies_arr: np.ndarray,
    values_arr: np.ndarray,
) -> list[SelfPlayExample]:
    return [
        (states_arr[i], policies_arr[i], float(values_arr[i]))
        for i in range(len(states_arr))
    ]


def _sample_opening_plies(random_opening_plies: int) -> int:
    """Sample per-game opening plies: 50% 0 plies, 25% 1 ply, 25% uniform 2..=K."""
    if random_opening_plies <= 0:
        return 0
    r = float(np.random.random())
    if r < 0.50:
        return 0
    if r < 0.75 or random_opening_plies == 1:
        return 1
    return int(np.random.randint(2, random_opening_plies + 1))


def _create_opening_game(random_opening_plies: int) -> ConnectFour:
    """Create a ConnectFour board with stochastic random opening plies, rolling back terminal hits."""
    target_plies = _sample_opening_plies(random_opening_plies)
    if target_plies == 0:
        return ConnectFour()

    last_safe = ConnectFour()
    for _ in range(16):
        game = ConnectFour()
        ok = True
        for _ in range(target_plies):
            valid = game.get_valid_move_indices()
            if not valid:
                ok = False
                break
            prev = game.copy()
            col = valid[int(np.random.randint(0, len(valid)))]
            game.make_move(col)
            if game.is_terminal():
                game = prev
                ok = False
                break
        last_safe = game
        if ok and not game.is_terminal():
            return game

    return last_safe


def play_batched_games(
    model: AlphaZeroNet,
    num_games: int,
    num_simulations: int,
    c_puct: float = 1.5,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    dirichlet_epsilon: float = 0.25,
    temp_threshold: int = TEMP_THRESHOLD,
    random_opening_plies: int = RANDOM_OPENING_PLIES,
    c_fpu: float = DEFAULT_C_FPU,
    return_arrays: bool = False,
    return_uint8: bool = False,
) -> list[SelfPlayExample] | SelfPlayArrays:
    """Play multiple self-play games simultaneously using Batched MCTS.

    Uses the Rust engine if available, otherwise falls back to Python.
    Note: Rust `advance_games()` already emits horizontally mirrored pairs;
    `_play_batched_python` emits un-augmented examples (augmented in `run_self_play`).
    """
    if RUST_AVAILABLE:
        return _play_batched_rust(
            model=model,
            num_games=num_games,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            temp_threshold=temp_threshold,
            random_opening_plies=random_opening_plies,
            c_fpu=c_fpu,
            return_arrays=return_arrays,
            return_uint8=return_uint8,
        )
    out = _play_batched_python(
        model=model,
        num_games=num_games,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        temp_threshold=temp_threshold,
        random_opening_plies=random_opening_plies,
        c_fpu=c_fpu,
        return_arrays=return_arrays,
    )
    if return_arrays and return_uint8:
        st, pol, val = out  # type: ignore[misc]
        return np.ascontiguousarray(st > 0.5, dtype=np.uint8), pol, val
    return out


def _play_batched_rust(
    model: AlphaZeroNet,
    num_games: int,
    num_simulations: int,
    c_puct: float = 1.5,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    dirichlet_epsilon: float = 0.25,
    temp_threshold: int = TEMP_THRESHOLD,
    random_opening_plies: int = RANDOM_OPENING_PLIES,
    c_fpu: float = DEFAULT_C_FPU,
    return_arrays: bool = False,
    return_uint8: bool = False,
) -> list[SelfPlayExample] | SelfPlayArrays:
    """High-performance self-play using the Rust MCTS engine."""
    if num_games <= 0:
        return _empty_arrays(uint8_states=return_uint8) if return_arrays else []

    device = next(model.parameters()).device
    model.eval()

    try:
        engine = RustBatchedSelfPlay(
            num_games,
            num_simulations,
            c_puct,
            dirichlet_alpha,
            dirichlet_epsilon,
            temp_threshold,
            random_opening_plies,
            c_fpu,
        )
    except TypeError:
        engine = RustBatchedSelfPlay(
            num_games,
            num_simulations,
            c_puct,
            dirichlet_alpha,
            dirichlet_epsilon,
            temp_threshold,
            random_opening_plies,
        )

    # Fast path: barrier-free continuous self-play with compact uint8 H2D & single packed D2H transfer
    if hasattr(engine, "collect_continuous_u8") and hasattr(
        engine, "apply_continuous_packed"
    ):
        pbar = tqdm(total=num_games, desc="    Self-play", unit="game")
        while not engine.is_done():
            leaf_states_u8, leaf_valid, count, fin_collect = (
                engine.collect_continuous_u8()
            )
            if fin_collect > 0:
                pbar.update(fin_collect)
            if count > 0:
                packed_np = _predict_packed_numpy(
                    model, device, leaf_states_u8, leaf_valid
                )
                fin_apply = engine.apply_continuous_packed(packed_np)
                if fin_apply > 0:
                    pbar.update(fin_apply)
            elif engine.is_done():
                break
        pbar.close()

        if return_arrays:
            if return_uint8:
                s_u8, pol_arr, val_arr = engine.take_completed_u8()
                return (
                    np.ascontiguousarray(s_u8, dtype=np.uint8),
                    np.ascontiguousarray(pol_arr, dtype=np.float32),
                    np.ascontiguousarray(val_arr, dtype=np.float32),
                )
            s_f32, pol_arr, val_arr = engine.take_completed_f32()
            return (
                np.ascontiguousarray(s_f32, dtype=np.float32),
                np.ascontiguousarray(pol_arr, dtype=np.float32),
                np.ascontiguousarray(val_arr, dtype=np.float32),
            )

        s_f32, pol_arr, val_arr = engine.take_completed_f32()
        return _arrays_to_examples(
            np.ascontiguousarray(s_f32, dtype=np.float32),
            np.ascontiguousarray(pol_arr, dtype=np.float32),
            np.ascontiguousarray(val_arr, dtype=np.float32),
        )

    has_unexpanded_api = hasattr(engine, "get_unexpanded_root_states") and hasattr(
        engine, "init_unexpanded_roots"
    )
    has_step_done = hasattr(engine, "is_step_done")

    # Initial root expansion
    if engine.num_active() > 0:
        if has_unexpanded_api:
            states, valid_moves, u_count = engine.get_unexpanded_root_states()
            if u_count > 0:
                with _inference_context(device):
                    states_t = torch.from_numpy(states).to(device)
                    valid_t = torch.from_numpy(valid_moves).to(device)
                    policies, _ = model.predict(states_t, valid_t)
                engine.init_unexpanded_roots(
                    np.ascontiguousarray(policies.float().cpu().numpy(), dtype=np.float32)
                )
        else:
            states, valid_moves = engine.get_root_states()
            with _inference_context(device):
                states_t = torch.from_numpy(states).to(device)
                valid_t = torch.from_numpy(valid_moves).to(device)
                policies, _ = model.predict(states_t, valid_t)
            engine.init_roots(
                np.ascontiguousarray(policies.float().cpu().numpy(), dtype=np.float32)
            )

    all_states: list[np.ndarray] = []
    all_policies: list[np.ndarray] = []
    all_values: list[np.ndarray] = []

    pbar = tqdm(total=num_games, desc="    Self-play", unit="game")

    while not engine.is_done():
        prev_active = engine.num_active()

        # Add Dirichlet exploration noise to active roots
        engine.add_noise()

        # Run simulations until every active game completes all `num_simulations`
        if has_step_done:
            max_step_iters = max(1, num_simulations * 2 + 4)
            step_iters = 0
            zero_leaf_streak = 0
            while not engine.is_step_done() and step_iters < max_step_iters:
                step_iters += 1
                leaf_states, leaf_valid, count = engine.collect_leaves()
                if count > 0:
                    zero_leaf_streak = 0
                    with _inference_context(device):
                        states_t = torch.from_numpy(leaf_states).to(device)
                        valid_t = torch.from_numpy(leaf_valid).to(device)
                        policies, values = model.predict(states_t, valid_t)
                    engine.apply_evaluations(
                        np.ascontiguousarray(
                            policies.float().cpu().numpy(), dtype=np.float32
                        ),
                        np.ascontiguousarray(
                            values.float().cpu().numpy(), dtype=np.float32
                        ),
                    )
                elif engine.is_step_done():
                    break
                else:
                    zero_leaf_streak += 1
                    if zero_leaf_streak > num_simulations + 2:
                        break
        else:
            leaves_per_game = 16
            num_batches = (num_simulations + leaves_per_game - 1) // leaves_per_game
            for _ in range(num_batches):
                leaf_states, leaf_valid, count = engine.collect_leaves()
                if count > 0:
                    with _inference_context(device):
                        states_t = torch.from_numpy(leaf_states).to(device)
                        valid_t = torch.from_numpy(leaf_valid).to(device)
                        policies, values = model.predict(states_t, valid_t)
                    engine.apply_evaluations(
                        np.ascontiguousarray(
                            policies.float().cpu().numpy(), dtype=np.float32
                        ),
                        np.ascontiguousarray(
                            values.float().cpu().numpy(), dtype=np.float32
                        ),
                    )

        # Advance games and collect completed training examples
        ex_states, ex_policies, ex_values = engine.advance_games()
        if len(ex_states) > 0:
            all_states.append(ex_states)
            all_policies.append(ex_policies)
            all_values.append(ex_values)

        finished = prev_active - engine.num_active()
        if finished > 0:
            pbar.update(finished)

        # Expand ONLY unexpanded roots for still-active games (skips GPU forward when count == 0)
        if engine.num_active() > 0:
            if has_unexpanded_api:
                states, valid_moves, u_count = engine.get_unexpanded_root_states()
                if u_count > 0:
                    with _inference_context(device):
                        states_t = torch.from_numpy(states).to(device)
                        valid_t = torch.from_numpy(valid_moves).to(device)
                        policies, _ = model.predict(states_t, valid_t)
                    engine.init_unexpanded_roots(
                        np.ascontiguousarray(
                            policies.float().cpu().numpy(), dtype=np.float32
                        )
                    )
            else:
                states, valid_moves = engine.get_root_states()
                with _inference_context(device):
                    states_t = torch.from_numpy(states).to(device)
                    valid_t = torch.from_numpy(valid_moves).to(device)
                    policies, _ = model.predict(states_t, valid_t)
                engine.init_new_roots(
                    np.ascontiguousarray(policies.float().cpu().numpy(), dtype=np.float32)
                )

    pbar.close()

    if not all_states:
        return _empty_arrays(uint8_states=return_uint8) if return_arrays else []

    states_arr = np.ascontiguousarray(np.concatenate(all_states, axis=0), dtype=np.float32)
    policies_arr = np.ascontiguousarray(
        np.concatenate(all_policies, axis=0), dtype=np.float32
    )
    values_arr = np.ascontiguousarray(np.concatenate(all_values, axis=0), dtype=np.float32)

    if return_arrays:
        if return_uint8:
            return (
                np.ascontiguousarray(states_arr > 0.5, dtype=np.uint8),
                policies_arr,
                values_arr,
            )
        return states_arr, policies_arr, values_arr
    return _arrays_to_examples(states_arr, policies_arr, values_arr)


def _play_batched_python(
    model: AlphaZeroNet,
    num_games: int,
    num_simulations: int,
    c_puct: float = 1.5,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    dirichlet_epsilon: float = 0.25,
    temp_threshold: int = TEMP_THRESHOLD,
    random_opening_plies: int = RANDOM_OPENING_PLIES,
    c_fpu: float = DEFAULT_C_FPU,
    return_arrays: bool = False,
) -> list[SelfPlayExample] | SelfPlayArrays:
    """Full-parity pure-Python batched self-play fallback."""
    if num_games <= 0:
        return _empty_arrays() if return_arrays else []

    device = next(model.parameters()).device
    model.eval()

    games = [_create_opening_game(random_opening_plies) for _ in range(num_games)]
    histories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [
        [] for _ in range(num_games)
    ]
    roots = [MCTSNode(g.copy()) for g in games]
    active_indices = [i for i in range(num_games) if not games[i].is_terminal()]

    # Initial evaluate and expand roots
    needs_init = [
        i
        for i in active_indices
        if not roots[i].is_expanded and roots[i].solver_state == SOLVER_ONGOING
    ]
    if needs_init:
        with _inference_context(device):
            states_t = torch.from_numpy(
                np.stack([games[i].encode() for i in needs_init])
            ).to(device)
            valid_t = torch.from_numpy(
                np.stack([games[i].get_valid_moves() for i in needs_init])
            ).to(device)
            policies_t, _ = model.predict(states_t, valid_t)
            init_policies = policies_t.float().cpu().numpy()

        for idx, i in enumerate(needs_init):
            roots[i].expand(init_policies[idx])

    out_states: list[np.ndarray] = []
    out_policies: list[np.ndarray] = []
    out_values: list[float] = []

    pbar = tqdm(total=num_games, desc="    Self-play", unit="game")
    if len(active_indices) < num_games:
        pbar.update(num_games - len(active_indices))

    while active_indices:
        # Add Dirichlet noise to active roots
        if dirichlet_epsilon > 0.0:
            for i in active_indices:
                root = roots[i]
                if len(root.children) >= 2:
                    noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
                    for idx, child in enumerate(root.children.values()):
                        child.prior = float(
                            (1.0 - dirichlet_epsilon) * child.prior
                            + dirichlet_epsilon * noise[idx]
                        )

        # MCTS simulations with exact MCTS-Solver + FPU reduction
        for _ in range(num_simulations):
            leaves_to_eval: list[np.ndarray] = []
            eval_nodes: list[MCTSNode] = []

            for i in active_indices:
                root = roots[i]
                node = root
                is_root = True
                while node.is_expanded and node.children:
                    if not is_root and node.solver_state != SOLVER_ONGOING:
                        break
                    node = node.select_child(
                        c_puct,
                        c_fpu=c_fpu,
                        is_noisy_root=(is_root and dirichlet_epsilon > 0.0),
                    )
                    is_root = False

                if node.solver_state != SOLVER_ONGOING:
                    if node.solver_state == SOLVER_WIN:
                        proven_val = 1.0
                    elif node.solver_state == SOLVER_LOSS:
                        proven_val = -1.0
                    else:
                        proven_val = 0.0
                    node.backpropagate(proven_val)
                else:
                    result = node.game.get_result()
                    if result is not None:
                        node.solver_state = (
                            SOLVER_LOSS if result < 0.0 else SOLVER_DRAW
                        )
                        if node.parent is not None:
                            node.parent._propagate_solver()
                        node.backpropagate(result)
                    else:
                        leaves_to_eval.append(node.game.encode())
                        eval_nodes.append(node)

            if leaves_to_eval:
                with _inference_context(device):
                    states_t = torch.from_numpy(np.stack(leaves_to_eval)).to(device)
                    val_moves_t = torch.from_numpy(
                        np.stack([n.game.get_valid_moves() for n in eval_nodes])
                    ).to(device)
                    policies_t, values_t = model.predict(states_t, val_moves_t)
                    leaf_policies = policies_t.float().cpu().numpy()
                    leaf_values = values_t.float().cpu().numpy()

                for idx, node in enumerate(eval_nodes):
                    node.expand(leaf_policies[idx])
                    if node.solver_state == SOLVER_WIN:
                        bval = 1.0
                    elif node.solver_state == SOLVER_LOSS:
                        bval = -1.0
                    elif node.solver_state == SOLVER_DRAW:
                        bval = 0.0
                    else:
                        bval = float(leaf_values[idx])
                    node.backpropagate(bval)

        # Record smooth policy target & sample temperature-controlled move
        new_active: list[int] = []
        for i in active_indices:
            game = games[i]
            root = roots[i]

            visit_counts = np.zeros(COLS, dtype=np.float32)
            for act, child in root.children.items():
                visit_counts[act] = float(child.visit_count)

            winning_cols = [
                act
                for act, child in root.children.items()
                if child.solver_state == SOLVER_LOSS
            ]

            # 1. Policy target: ALWAYS normalized visit counts (or uniform over proven wins)
            if root.solver_state == SOLVER_WIN and winning_cols:
                policy_target = np.zeros(COLS, dtype=np.float32)
                policy_target[winning_cols] = 1.0 / len(winning_cols)

                selection_counts = np.zeros(COLS, dtype=np.float32)
                win_visit_sum = float(visit_counts[winning_cols].sum())
                if win_visit_sum > 0.0:
                    selection_counts[winning_cols] = visit_counts[winning_cols]
                else:
                    selection_counts[winning_cols] = 1.0
            else:
                total_visits = float(visit_counts.sum())
                if total_visits > 0.0:
                    policy_target = (visit_counts / total_visits).astype(np.float32)
                elif root.children:
                    priors = np.zeros(COLS, dtype=np.float32)
                    for act, child in root.children.items():
                        priors[act] = child.prior
                    if priors.sum() > 0.0:
                        policy_target = (priors / priors.sum()).astype(np.float32)
                    else:
                        valid = game.get_valid_moves().astype(np.float32)
                        policy_target = valid / valid.sum()
                else:
                    valid = game.get_valid_moves().astype(np.float32)
                    policy_target = valid / valid.sum()

                # Avoid proven losing moves (SOLVER_WIN children) when non-losing moves exist
                selection_counts = visit_counts.copy()
                if root.solver_state != SOLVER_LOSS:
                    for act, child in root.children.items():
                        if child.solver_state == SOLVER_WIN:
                            selection_counts[act] = 0.0
                    if selection_counts.sum() == 0.0:
                        for act, child in root.children.items():
                            if child.solver_state != SOLVER_WIN:
                                selection_counts[act] = 1.0
                if selection_counts.sum() == 0.0:
                    selection_counts = policy_target.copy()

            histories[i].append((game.encode(), policy_target, game.current_player))

            # 2. Action selection: greedy at >= temp_threshold, proportional before
            if game._move_count >= temp_threshold:
                action = int(np.argmax(selection_counts))
            else:
                probs = selection_counts.astype(np.float64)
                probs /= probs.sum()
                action = int(np.random.choice(COLS, p=probs))

            game.make_move(action)

            if game.is_terminal():
                result = game.get_result()
                if result is None:
                    result = 0.0

                for encoded, policy, player in histories[i]:
                    val = float(result) if player == game.current_player else float(-result)
                    out_states.append(encoded)
                    out_policies.append(policy)
                    out_values.append(val)

                histories[i].clear()
                pbar.update(1)
            else:
                new_active.append(i)
                if action in root.children:
                    new_root = root.children[action]
                    new_root.parent = None
                    roots[i] = new_root
                else:
                    roots[i] = MCTSNode(game.copy())

        active_indices = new_active

        # Evaluate ONLY unexpanded new roots (skipping already-expanded rebased subtrees)
        needs_expand = [
            i
            for i in active_indices
            if not roots[i].is_expanded and roots[i].solver_state == SOLVER_ONGOING
        ]
        if needs_expand:
            with _inference_context(device):
                states_t = torch.from_numpy(
                    np.stack([games[i].encode() for i in needs_expand])
                ).to(device)
                val_moves_t = torch.from_numpy(
                    np.stack([games[i].get_valid_moves() for i in needs_expand])
                ).to(device)
                policies_t, _ = model.predict(states_t, val_moves_t)
                new_policies = policies_t.float().cpu().numpy()

            for idx, i in enumerate(needs_expand):
                roots[i].expand(new_policies[idx])

    pbar.close()

    if not out_states:
        return _empty_arrays() if return_arrays else []

    if return_arrays:
        states_arr = np.ascontiguousarray(np.stack(out_states, axis=0), dtype=np.float32)
        policies_arr = np.ascontiguousarray(
            np.stack(out_policies, axis=0), dtype=np.float32
        )
        values_arr = np.ascontiguousarray(out_values, dtype=np.float32)
        return states_arr, policies_arr, values_arr

    return list(zip(out_states, out_policies, out_values))


def augment_examples(
    examples: list[SelfPlayExample],
) -> list[SelfPlayExample]:
    """Augment training examples by horizontal mirroring."""
    augmented = list(examples)
    for state, policy, value in examples:
        mirrored_state = state[:, :, ::-1].copy()
        mirrored_policy = policy[::-1].copy()
        augmented.append((mirrored_state, mirrored_policy, value))
    return augmented


def augment_arrays(
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
) -> SelfPlayArrays:
    """Augment contiguous numpy training arrays by horizontal mirroring without Python loops."""
    if len(states) == 0:
        return _empty_arrays()

    mirrored_states = states[:, :, :, ::-1]
    mirrored_policies = policies[:, ::-1]
    state_dtype = np.uint8 if states.dtype == np.uint8 else np.float32
    aug_states = np.ascontiguousarray(
        np.concatenate([states, mirrored_states], axis=0), dtype=state_dtype
    )
    aug_policies = np.ascontiguousarray(
        np.concatenate([policies, mirrored_policies], axis=0), dtype=np.float32
    )
    aug_values = np.ascontiguousarray(
        np.concatenate([values, values], axis=0), dtype=np.float32
    )
    return aug_states, aug_policies, aug_values


def run_self_play(
    model: AlphaZeroNet,
    num_games: int,
    num_simulations: int,
    c_puct: float = 1.5,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    dirichlet_epsilon: float = 0.25,
    temp_threshold: int = TEMP_THRESHOLD,
    random_opening_plies: int = RANDOM_OPENING_PLIES,
    c_fpu: float = DEFAULT_C_FPU,
    return_arrays: bool = False,
    return_uint8: bool = False,
) -> list[SelfPlayExample] | SelfPlayArrays:
    """Run self-play games using Batched MCTS with horizontal mirror augmentation.

    When `return_arrays=True`, returns `(states_arr, policies_arr, values_arr)` as
    contiguous numpy arrays `(N, 3, 6, 7), (N, 7), (N,)` (`uint8` states when
    `return_uint8=True`, else `float32`) with zero Python per-example tuple loop.
    """
    extra_kwargs = {"return_uint8": True} if return_uint8 else {}
    result = play_batched_games(
        model=model,
        num_games=num_games,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        temp_threshold=temp_threshold,
        random_opening_plies=random_opening_plies,
        c_fpu=c_fpu,
        return_arrays=return_arrays,
        **extra_kwargs,
    )

    # Rust engine already emits augmented pairs directly
    if RUST_AVAILABLE:
        return result

    if return_arrays:
        states_arr, policies_arr, values_arr = result  # type: ignore[misc]
        return augment_arrays(states_arr, policies_arr, values_arr)

    return augment_examples(result)  # type: ignore[arg-type]
