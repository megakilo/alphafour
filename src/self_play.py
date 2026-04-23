"""Self-play data generation with multiprocessing support."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .game import ConnectFour, COLS
from .mcts import MCTS
from .model import AlphaZeroNet


# Temperature schedule: use τ=1 for first N moves, then τ→0
TEMP_THRESHOLD = 15  # After this many moves, switch to greedy


def play_single_game(
    model: AlphaZeroNet,
    num_simulations: int,
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play a single self-play game and return training examples.

    Returns:
        List of (encoded_state, policy, value) tuples.
        Value is from the perspective of the current player at that state.
    """
    mcts = MCTS(model=model, num_simulations=num_simulations, device=device)
    game = ConnectFour()
    history: list[tuple[np.ndarray, np.ndarray, int]] = []

    while not game.is_terminal():
        # Temperature schedule
        temperature = 1.0 if game._move_count < TEMP_THRESHOLD else 0.0

        # Get MCTS policy
        action_probs, _ = mcts.get_action_probs(
            game, temperature=temperature, add_noise=True
        )

        # Record state and policy
        encoded = game.encode()
        history.append((encoded, action_probs, game.current_player))

        # Sample action
        action = np.random.choice(COLS, p=action_probs)
        game.make_move(action)

    # Determine game result
    result = game.get_result()
    if result is None:
        result = 0.0

    # The result is from the perspective of the *current* player (who didn't
    # get to move because the game is over). If the last player won, result = -1.
    # We need to assign values from each position's player perspective.
    # result is from current_player's view. Last mover = -current_player.
    # If result == -1, last mover won.

    examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for encoded, policy, player in history:
        # Value from this position's player perspective
        if player == game.current_player:
            value = result
        else:
            value = -result
        examples.append((encoded, policy, value))

    return examples


def _worker_init(model_path: str, num_res_blocks: int, num_filters: int) -> None:
    """Initialize worker process with model."""
    # Prevent CPU thread contention across multiple worker processes
    torch.set_num_threads(1)

    global _worker_model
    _worker_model = AlphaZeroNet(
        num_res_blocks=num_res_blocks, num_filters=num_filters
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    _worker_model.load_state_dict(state_dict)
    _worker_model.eval()


def _worker_play_game(args: tuple[int, int]) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Worker function to play a single game."""
    num_simulations, game_idx = args
    global _worker_model
    return play_single_game(
        model=_worker_model,
        num_simulations=num_simulations,
        device=torch.device("cpu"),
    )


def augment_examples(
    examples: list[tuple[np.ndarray, np.ndarray, float]],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Augment training examples by horizontal mirroring."""
    augmented = list(examples)
    for state, policy, value in examples:
        # Mirror state: flip along column axis
        mirrored_state = state[:, :, ::-1].copy()
        # Mirror policy: reverse column order
        mirrored_policy = policy[::-1].copy()
        augmented.append((mirrored_state, mirrored_policy, value))
    return augmented


def run_self_play(
    model: AlphaZeroNet,
    num_games: int,
    num_simulations: int,
    num_workers: int | None = None,
    model_path: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Run self-play games to generate training data.

    Args:
        model: The neural network model (used if single-process).
        num_games: Number of games to play.
        num_simulations: MCTS simulations per move.
        num_workers: Number of parallel workers (default: CPU count).
        model_path: Path to saved model weights (required for multiprocessing).

    Returns:
        List of (state, policy, value) training examples.
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    all_examples: list[tuple[np.ndarray, np.ndarray, float]] = []

    if num_workers <= 1 or model_path is None:
        # Single-process mode — move model to CPU for inference
        original_device = next(model.parameters()).device
        model_cpu = model.cpu()
        model_cpu.eval()
        for i in tqdm(range(num_games), desc="    Self-play", unit="game"):
            examples = play_single_game(
                model=model_cpu,
                num_simulations=num_simulations,
                device=torch.device("cpu"),
            )
            examples = augment_examples(examples)
            all_examples.extend(examples)
        model.to(original_device)  # Move back
    else:
        # Multi-process mode
        # Save model weights temporarily if not already saved
        temp_path = model_path
        args_list = [(num_simulations, i) for i in range(num_games)]

        # Determine chunk size for better load balancing
        chunk_size = max(1, num_games // (num_workers * 4))

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(
                temp_path,
                model.num_res_blocks,
                model.num_filters,
            ),
        ) as pool:
            results = list(tqdm(
                pool.imap_unordered(_worker_play_game, args_list),
                total=num_games,
                desc="    Self-play",
                unit="game",
            ))

        for examples in results:
            examples = augment_examples(examples)
            all_examples.extend(examples)

    return all_examples
