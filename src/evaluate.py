"""Evaluation metrics and dataset testing for AlphaZero."""

from __future__ import annotations

import numpy as np
import torch

from .game import ConnectFour, COLS
from .mcts import MCTS, MCTSNode
from .model import AlphaZeroNet


def evaluate_opening_move(
    model: AlphaZeroNet, device: torch.device, num_simulations: int = 0
) -> tuple[int, float]:
    """Evaluate the opening move for Connect Four.

    Returns:
        (best_move_column, center_column_visit_pct)
    """
    game = ConnectFour()

    if num_simulations > 0:
        mcts = MCTS(model=model, num_simulations=num_simulations, device=device)
        # Run MCTS search
        visit_counts = mcts.search(game, add_noise=False)
        best_move = int(np.argmax(visit_counts))
        center_pct = float(visit_counts[3] / visit_counts.sum()) * 100
    else:
        model.eval()
        state_t = torch.from_numpy(np.array([game.encode()])).to(device)
        val_moves_t = torch.from_numpy(np.array([game.get_valid_moves()])).to(device)
        with torch.no_grad():
            policies, _ = model.predict(state_t, val_moves_t)
            policies = policies[0].cpu().numpy()

        best_move = int(np.argmax(policies))
        center_pct = float(policies[3] / policies.sum()) * 100

    return best_move, center_pct


def evaluate_dataset(
    model: AlphaZeroNet,
    device: torch.device,
    lines: list[str],
    num_simulations: int = 0,
) -> float:
    """Evaluate model value accuracy against a single dataset.

    lines should be a list of strings following the protocol:
    [Moves string] [Score]
    where score > 0 is a win, score < 0 is a loss, and score == 0 is a draw.

    If num_simulations > 0, evaluates the dataset using Batched MCTS.
    Otherwise, evaluates using raw network predictions.

    Returns:
        Accuracy percentage.
    """
    model.eval()

    correct = 0
    total = 0

    batch_states = []
    batch_games = []
    batch_scores = []
    batch_size = 1024

    def evaluate_batch_raw(states: list[np.ndarray], scores: list[int]) -> None:
        nonlocal correct, total
        if not states:
            return

        states_t = torch.from_numpy(np.stack(states)).to(device)
        with torch.no_grad():
            _, values = model.predict(states_t)
            values = values.cpu().numpy()

        for pred_val, true_score in zip(values, scores):
            if true_score > 0 and pred_val > 0.05:
                correct += 1
            elif true_score < 0 and pred_val < -0.05:
                correct += 1
            elif true_score == 0 and -0.05 <= pred_val <= 0.05:
                correct += 1
        total += len(scores)

    def evaluate_batch_mcts(games: list[ConnectFour], scores: list[int]) -> None:
        nonlocal correct, total
        if not games:
            return

        roots = [MCTSNode(g) for g in games]

        # Initial evaluate for roots
        states_t = torch.from_numpy(np.array([g.encode() for g in games])).to(device)
        val_moves_t = torch.from_numpy(
            np.array([g.get_valid_moves() for g in games])
        ).to(device)

        with torch.no_grad():
            policies, _ = model.predict(states_t, val_moves_t)
            policies = policies.cpu().numpy()

        for i, root in enumerate(roots):
            root.expand(policies[i])

        for _ in range(num_simulations):
            leaves_to_eval = []
            eval_indices = []

            for i, root in enumerate(roots):
                node = root
                while node.is_expanded and node.children:
                    node = node.select_child(1.5)

                result = node.game.get_result()
                if result is not None:
                    node.backpropagate(result)
                else:
                    leaves_to_eval.append(node.game.encode())
                    eval_indices.append((i, node))

            if leaves_to_eval:
                with torch.no_grad():
                    states_t = torch.from_numpy(np.stack(leaves_to_eval)).to(device)
                    val_moves_t = torch.from_numpy(
                        np.array([n.game.get_valid_moves() for _, n in eval_indices])
                    ).to(device)
                    pols, vals = model.predict(states_t, val_moves_t)
                    pols = pols.cpu().numpy()
                    vals = vals.cpu().numpy()

                for idx, (i, node) in enumerate(eval_indices):
                    node.expand(pols[idx])
                    node.backpropagate(vals[idx].item())

        for i, root in enumerate(roots):
            pred_val = root.q_value
            true_score = scores[i]

            if true_score > 0 and pred_val > 0.05:
                correct += 1
            elif true_score < 0 and pred_val < -0.05:
                correct += 1
            elif true_score == 0 and -0.05 <= pred_val <= 0.05:
                correct += 1

        total += len(scores)

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        moves_str, score_str = parts
        score = int(score_str)

        game = ConnectFour()
        for move_char in moves_str:
            col = int(move_char) - 1
            game.make_move(col)

        if num_simulations > 0:
            batch_games.append(game)
        else:
            batch_states.append(game.encode())

        batch_scores.append(score)

        if num_simulations > 0 and len(batch_games) >= batch_size:
            evaluate_batch_mcts(batch_games, batch_scores)
            batch_games = []
            batch_scores = []
        elif num_simulations == 0 and len(batch_states) >= batch_size:
            evaluate_batch_raw(batch_states, batch_scores)
            batch_states = []
            batch_scores = []

    if num_simulations > 0 and batch_games:
        evaluate_batch_mcts(batch_games, batch_scores)
    elif num_simulations == 0 and batch_states:
        evaluate_batch_raw(batch_states, batch_scores)

    if total > 0:
        return (correct / total) * 100
    return 0.0


def play_batched_arena(
    model1: AlphaZeroNet,
    model2: AlphaZeroNet,
    device: torch.device,
    num_games: int = 40,
    num_simulations: int = 200,
) -> tuple[int, int, int, int, int]:
    """Play games between two models using Batched MCTS.

    Model 1 plays as Player 1 for the first half of games,
    and as Player 2 for the second half.

    Returns:
        (m1_wins_p1, m1_wins_p2, m2_wins_p1, m2_wins_p2, draws)
    """
    model1.eval()
    model2.eval()

    m1_is_p1 = [True] * (num_games // 2) + [False] * (num_games - num_games // 2)
    games = [ConnectFour() for _ in range(num_games)]

    roots = [MCTSNode(g) for g in games]
    active_indices = list(range(num_games))

    # Initial evaluate for roots
    states_t = torch.from_numpy(np.array([g.encode() for g in games])).to(device)
    val_moves_t = torch.from_numpy(np.array([g.get_valid_moves() for g in games])).to(
        device
    )

    with torch.no_grad():
        p1, _ = model1.predict(states_t, val_moves_t)
        p2, _ = model2.predict(states_t, val_moves_t)

    p1 = p1.cpu().numpy()
    p2 = p2.cpu().numpy()

    for i, root in enumerate(roots):
        current_model = 1 if (m1_is_p1[i] == (root.game.current_player == 1)) else 2
        pol = p1[i] if current_model == 1 else p2[i]
        root.expand(pol)

    model1_wins_p1 = 0
    model1_wins_p2 = 0
    model2_wins_p1 = 0
    model2_wins_p2 = 0
    draws = 0

    while active_indices:
        for _ in range(num_simulations):
            leaves_to_eval_m1 = []
            eval_indices_m1 = []
            leaves_to_eval_m2 = []
            eval_indices_m2 = []

            for i in active_indices:
                node = roots[i]
                while node.is_expanded and node.children:
                    node = node.select_child(1.5)

                result = node.game.get_result()
                if result is not None:
                    node.backpropagate(result)
                else:
                    # Determine which model's turn it is at this leaf
                    is_m1 = m1_is_p1[i] == (node.game.current_player == 1)
                    if is_m1:
                        leaves_to_eval_m1.append(node.game.encode())
                        eval_indices_m1.append((i, node))
                    else:
                        leaves_to_eval_m2.append(node.game.encode())
                        eval_indices_m2.append((i, node))

            if leaves_to_eval_m1:
                with torch.no_grad():
                    states_t = torch.from_numpy(np.stack(leaves_to_eval_m1)).to(device)
                    val_moves_t = torch.from_numpy(
                        np.array([n.game.get_valid_moves() for _, n in eval_indices_m1])
                    ).to(device)
                    policies, values = model1.predict(states_t, val_moves_t)
                    policies = policies.cpu().numpy()
                    values = values.cpu().numpy()
                for idx, (i, node) in enumerate(eval_indices_m1):
                    node.expand(policies[idx])
                    node.backpropagate(values[idx].item())

            if leaves_to_eval_m2:
                with torch.no_grad():
                    states_t = torch.from_numpy(np.stack(leaves_to_eval_m2)).to(device)
                    val_moves_t = torch.from_numpy(
                        np.array([n.game.get_valid_moves() for _, n in eval_indices_m2])
                    ).to(device)
                    policies, values = model2.predict(states_t, val_moves_t)
                    policies = policies.cpu().numpy()
                    values = values.cpu().numpy()
                for idx, (i, node) in enumerate(eval_indices_m2):
                    node.expand(policies[idx])
                    node.backpropagate(values[idx].item())

        new_active = []
        for i in active_indices:
            game = games[i]
            root = roots[i]

            visit_counts = np.zeros(COLS, dtype=np.float32)
            for action, child in root.children.items():
                visit_counts[action] = child.visit_count

            # Greedy play for arena
            action = int(np.argmax(visit_counts))
            game.make_move(action)

            if game.is_terminal():
                res = game.get_result()
                if res == 0.0:
                    draws += 1
                else:
                    # res is -1.0 meaning the person who just moved won.
                    winner_player = -game.current_player
                    if m1_is_p1[i]:
                        if winner_player == 1:
                            model1_wins_p1 += 1
                        else:
                            model2_wins_p2 += 1
                    else:
                        if winner_player == 1:
                            model2_wins_p1 += 1
                        else:
                            model1_wins_p2 += 1
            else:
                new_active.append(i)
                roots[i] = MCTSNode(game)

        active_indices = new_active

        # Initial evaluate for new roots
        if active_indices:
            states_t = torch.from_numpy(
                np.array([games[i].encode() for i in active_indices])
            ).to(device)
            val_moves_t = torch.from_numpy(
                np.array([games[i].get_valid_moves() for i in active_indices])
            ).to(device)

            with torch.no_grad():
                p1, _ = model1.predict(states_t, val_moves_t)
                p2, _ = model2.predict(states_t, val_moves_t)
            p1 = p1.cpu().numpy()
            p2 = p2.cpu().numpy()

            for idx, i in enumerate(active_indices):
                current_model = (
                    1 if (m1_is_p1[i] == (games[i].current_player == 1)) else 2
                )
                pol = p1[idx] if current_model == 1 else p2[idx]
                roots[i].expand(pol)

    return model1_wins_p1, model1_wins_p2, model2_wins_p1, model2_wins_p2, draws
