"""Self-play data generation with batched MCTS.

Uses the Rust engine (alphafour_engine) for high-performance tree search,
with PyTorch for neural network inference on GPU.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .model import AlphaZeroNet

# Temperature schedule: use τ=1 for first N moves, then τ→0
TEMP_THRESHOLD = 30  # After this many moves, switch to greedy
RANDOM_OPENING_PLIES = 2  # Random moves before MCTS (not recorded)

try:
    from alphafour_engine import RustBatchedSelfPlay

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def play_batched_games(
    model: AlphaZeroNet,
    num_games: int,
    num_simulations: int,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 1.0,
    dirichlet_epsilon: float = 0.25,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play multiple games simultaneously using Batched MCTS.

    Uses Rust engine if available, otherwise falls back to Python.
    """
    if RUST_AVAILABLE:
        return _play_batched_rust(
            model, num_games, num_simulations, c_puct,
            dirichlet_alpha, dirichlet_epsilon,
        )
    else:
        return _play_batched_python(
            model, num_games, num_simulations, c_puct,
            dirichlet_alpha, dirichlet_epsilon,
        )


def _play_batched_rust(
    model: AlphaZeroNet,
    num_games: int,
    num_simulations: int,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 1.0,
    dirichlet_epsilon: float = 0.25,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """High-performance self-play using Rust MCTS engine."""
    device = next(model.parameters()).device
    model.eval()

    engine = RustBatchedSelfPlay(
        num_games, num_simulations, c_puct,
        dirichlet_alpha, dirichlet_epsilon, TEMP_THRESHOLD,
        RANDOM_OPENING_PLIES,
    )

    # Initial root expansion
    states, valid_moves = engine.get_root_states()
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        states_t = torch.from_numpy(states).to(device)
        valid_t = torch.from_numpy(valid_moves).to(device)
        policies, _ = model.predict(states_t, valid_t)
    engine.init_roots(policies.float().cpu().numpy())

    all_states = []
    all_policies = []
    all_values = []

    pbar = tqdm(total=num_games, desc="    Self-play", unit="game")

    while not engine.is_done():
        prev_active = engine.num_active()

        # Add Dirichlet noise
        engine.add_noise()

        # Run all simulations for this move
        for _ in range(num_simulations):
            leaf_states, leaf_valid, count = engine.collect_leaves()
            if count > 0:
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    states_t = torch.from_numpy(leaf_states).to(device)
                    valid_t = torch.from_numpy(leaf_valid).to(device)
                    policies, values = model.predict(states_t, valid_t)
                engine.apply_evaluations(
                    policies.float().cpu().numpy(),
                    values.float().cpu().numpy(),
                )

        # Advance games, collect finished examples
        ex_states, ex_policies, ex_values = engine.advance_games()
        if len(ex_states) > 0:
            all_states.append(ex_states)
            all_policies.append(ex_policies)
            all_values.append(ex_values)

        # Update progress
        finished = prev_active - engine.num_active()
        if finished > 0:
            pbar.update(finished)

        # Expand new roots for still-active games
        n_active = engine.num_active()
        if n_active > 0:
            states, valid_moves = engine.get_root_states()
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                states_t = torch.from_numpy(states).to(device)
                valid_t = torch.from_numpy(valid_moves).to(device)
                policies, _ = model.predict(states_t, valid_t)
            engine.init_new_roots(policies.float().cpu().numpy())

    pbar.close()

    # Convert numpy arrays to list of tuples for replay buffer
    if not all_states:
        return []

    states_arr = np.concatenate(all_states, axis=0)
    policies_arr = np.concatenate(all_policies, axis=0)
    values_arr = np.concatenate(all_values, axis=0)

    examples = []
    for i in range(len(states_arr)):
        examples.append((states_arr[i], policies_arr[i], float(values_arr[i])))

    return examples


def _play_batched_python(
    model: AlphaZeroNet,
    num_games: int,
    num_simulations: int,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 1.0,
    dirichlet_epsilon: float = 0.25,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Fallback: pure Python batched self-play (original implementation)."""
    from .game import ConnectFour, COLS
    from .mcts import MCTSNode

    device = next(model.parameters()).device
    model.eval()

    games = [ConnectFour() for _ in range(num_games)]
    histories = [[] for _ in range(num_games)]
    roots = [MCTSNode(g) for g in games]
    active_indices = list(range(num_games))

    # Initial evaluate and expand roots
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        states = torch.from_numpy(np.array([g.encode() for g in games])).to(device)
        valid_moves = torch.from_numpy(
            np.array([g.get_valid_moves() for g in games])
        ).to(device)
        policies, values = model.predict(states, valid_moves)
        policies = policies.cpu().numpy()

    for i, root in enumerate(roots):
        root.expand(policies[i])

    all_examples: list[tuple[np.ndarray, np.ndarray, float]] = []

    pbar = tqdm(total=num_games, desc="    Self-play", unit="game")

    while active_indices:
        # Add Dirichlet noise to roots
        for i in active_indices:
            root = roots[i]
            if root.children:
                noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
                for idx, child in enumerate(root.children.values()):
                    child.prior = (
                        1 - dirichlet_epsilon
                    ) * child.prior + dirichlet_epsilon * noise[idx]

        # MCTS Simulations
        for _ in range(num_simulations):
            leaves_to_eval = []
            eval_indices = []

            for i in active_indices:
                node = roots[i]
                while node.is_expanded and node.children:
                    node = node.select_child(c_puct)

                result = node.game.get_result()
                if result is not None:
                    node.backpropagate(result)
                else:
                    leaves_to_eval.append(node.game.encode())
                    eval_indices.append((i, node))

            if leaves_to_eval:
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    states_t = torch.from_numpy(np.stack(leaves_to_eval)).to(device)
                    val_moves_t = torch.from_numpy(
                        np.array(
                            [node.game.get_valid_moves() for _, node in eval_indices]
                        )
                    ).to(device)
                    policies, values = model.predict(states_t, val_moves_t)
                    policies = policies.cpu().numpy()
                    values = values.cpu().numpy()

                for idx, (i, node) in enumerate(eval_indices):
                    node.expand(policies[idx])
                    node.backpropagate(values[idx].item())

        # Sample actions for all active games
        new_active = []
        for i in active_indices:
            game = games[i]
            root = roots[i]

            visit_counts = np.zeros(COLS, dtype=np.float32)
            for action, child in root.children.items():
                visit_counts[action] = child.visit_count

            temperature = 1.0 if game._move_count < TEMP_THRESHOLD else 0.0

            if temperature == 0:
                action_probs = np.zeros(COLS, dtype=np.float32)
                best_action = np.argmax(visit_counts)
                action_probs[best_action] = 1.0
            else:
                if visit_counts.sum() == 0:
                    valid = game.get_valid_moves().astype(np.float32)
                    action_probs = valid / valid.sum()
                else:
                    mask = visit_counts > 0
                    log_counts = np.full(COLS, -np.inf)
                    log_counts[mask] = (1.0 / temperature) * np.log(visit_counts[mask])
                    max_log = np.max(log_counts[mask])
                    counts = np.zeros(COLS, dtype=np.float64)
                    counts[mask] = np.exp(log_counts[mask] - max_log)
                    action_probs = counts / counts.sum()

            histories[i].append((game.encode(), action_probs, game.current_player))

            action = np.random.choice(COLS, p=action_probs)
            game.make_move(action)

            if game.is_terminal():
                result = game.get_result()
                if result is None:
                    result = 0.0

                for encoded, policy, player in histories[i]:
                    if player == game.current_player:
                        val = result
                    else:
                        val = -result
                    all_examples.append((encoded, policy, val))

                pbar.update(1)
            else:
                new_active.append(i)
                if action in roots[i].children:
                    roots[i] = roots[i].children[action]
                    roots[i].parent = None
                else:
                    roots[i] = MCTSNode(game)

        active_indices = new_active

        # Initial evaluate for new roots (skip already-expanded subtree roots)
        needs_expand = [i for i in active_indices if not roots[i].is_expanded]
        if needs_expand:
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                states_t = torch.from_numpy(
                    np.array([games[i].encode() for i in needs_expand])
                ).to(device)
                val_moves_t = torch.from_numpy(
                    np.array([games[i].get_valid_moves() for i in needs_expand])
                ).to(device)
                policies, values = model.predict(states_t, val_moves_t)
                policies = policies.cpu().numpy()

            for idx, i in enumerate(needs_expand):
                roots[i].expand(policies[idx])

    pbar.close()
    return all_examples


def augment_examples(
    examples: list[tuple[np.ndarray, np.ndarray, float]],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Augment training examples by horizontal mirroring."""
    augmented = list(examples)
    for state, policy, value in examples:
        mirrored_state = state[:, :, ::-1].copy()
        mirrored_policy = policy[::-1].copy()
        augmented.append((mirrored_state, mirrored_policy, value))
    return augmented


def run_self_play(
    model: AlphaZeroNet,
    num_games: int,
    num_simulations: int,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 1.0,
    dirichlet_epsilon: float = 0.25,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Run self-play games using Batched MCTS.

    If Rust engine is available, augmentation is done in Rust.
    Otherwise, Python augmentation is applied.
    """
    examples = play_batched_games(
        model=model,
        num_games=num_games,
        num_simulations=num_simulations,
    )

    # Rust engine already includes augmentation
    if RUST_AVAILABLE:
        return examples
    else:
        return augment_examples(examples)
