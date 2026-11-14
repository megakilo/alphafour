"""Evaluation metrics, exact solver caching, and Arena testing for AlphaZero."""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .game import COLS, ROWS, ConnectFour
from .mcts import MCTSNode

if TYPE_CHECKING:
    from .model import AlphaZeroNet

try:
    import alphafour_engine as _rust_engine

    RustConnectFour = getattr(_rust_engine, "RustConnectFour", None)
    RustBatchedMCTS = getattr(_rust_engine, "RustBatchedMCTS", None)
    solve_position_wdl = getattr(_rust_engine, "solve_position_wdl", None)
    solve_children_wdl = getattr(_rust_engine, "solve_children_wdl", None)
    solve_batch_children_wdl = getattr(_rust_engine, "solve_batch_children_wdl", None)
    RUST_MCTS_AVAILABLE = RustConnectFour is not None and RustBatchedMCTS is not None
    RUST_SOLVER_AVAILABLE = (
        solve_batch_children_wdl is not None or solve_children_wdl is not None
    )
except ImportError:
    _rust_engine = None
    RustConnectFour = None
    RustBatchedMCTS = None
    solve_position_wdl = None
    solve_children_wdl = None
    solve_batch_children_wdl = None
    RUST_MCTS_AVAILABLE = False
    RUST_SOLVER_AVAILABLE = False

DEFAULT_SOLVER_CACHE_DIR = Path("testdata/.solver_cache")
_MEMORY_SOLVER_CACHE: dict[str, np.ndarray | None] = {}
_CENTER_FIRST_ORDER = (3, 2, 4, 1, 5, 0, 6)


def _autocast_ctx(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    """Return bfloat16/float16 autocast context on CUDA, or nullcontext otherwise."""
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


# ── Bitboard W/D/L Solver Fallback & Shallow Tactical Filter ────────────────


class _SolverBudgetExceeded(Exception):
    """Raised when the pure-Python fallback solver exceeds its node budget."""


def _bb_has_won(bb: int) -> bool:
    """Return True iff 49-bit Pascal-Pons bitboard `bb` contains a 4-in-a-row."""
    m = bb & (bb >> 1)
    if m & (m >> 2):
        return True
    m = bb & (bb >> 7)
    if m & (m >> 14):
        return True
    m = bb & (bb >> 8)
    if m & (m >> 16):
        return True
    m = bb & (bb >> 6)
    if m & (m >> 12):
        return True
    return False


def _bb_from_moves(moves_str: str) -> tuple[int, int, int]:
    """Parse 1-indexed move string into `(pos, mask, move_count)` bitboards."""
    pos = 0
    mask = 0
    count = 0
    for ch in moves_str:
        if ch < "1" or ch > "7":
            continue
        col = ord(ch) - ord("1")
        top_bit = 1 << (col * 7 + 5)
        if mask & top_bit:
            raise ValueError(f"Invalid move in full column {col + 1}: {moves_str!r}")
        new_mask = mask | (mask + (1 << (col * 7)))
        pos ^= mask
        mask = new_mask
        count += 1
    return pos, mask, count


def _bb_solve_wdl(
    pos: int,
    mask: int,
    moves: int,
    tt: dict[int, int],
    budget: list[int],
) -> int:
    """Exact Connect Four W/D/L negamax (`-1`, `0`, `+1`) with node budget."""
    budget[0] -= 1
    if budget[0] < 0:
        raise _SolverBudgetExceeded

    if moves >= ROWS * COLS:
        return 0

    # Immediate 4-in-a-row check for current player
    for c in _CENTER_FIRST_ORDER:
        if (mask & (1 << (c * 7 + 5))) == 0:
            stone = (mask + (1 << (c * 7))) & ~mask
            if _bb_has_won(pos | stone):
                return 1

    key = pos + mask
    cached = tt.get(key)
    if cached is not None:
        return cached

    opp = pos ^ mask
    forced_col = -1
    for c in _CENTER_FIRST_ORDER:
        if (mask & (1 << (c * 7 + 5))) == 0:
            stone = (mask + (1 << (c * 7))) & ~mask
            if _bb_has_won(opp | stone):
                if forced_col != -1:
                    # Opponent has two simultaneous immediate threats -> loss
                    tt[key] = -1
                    return -1
                forced_col = c

    cols = (forced_col,) if forced_col != -1 else _CENTER_FIRST_ORDER
    best = -1
    for c in cols:
        if (mask & (1 << (c * 7 + 5))) == 0:
            new_mask = mask | (mask + (1 << (c * 7)))
            val = -_bb_solve_wdl(opp, new_mask, moves + 1, tt, budget)
            if val > best:
                best = val
                if best == 1:
                    break

    tt[key] = best
    return best


def _python_solve_children_wdl(
    moves_str: str,
    max_nodes: int = 25_000,
) -> np.ndarray | None:
    """Pure-Python exact child W/D/L solver for tactical / endgame positions.

    Returns `int8[7]` (`-2` illegal, `-1/0/+1` from root mover's view), or
    `None` if the position is too deep for the Python node budget.
    """
    pos, mask, mc = _bb_from_moves(moves_str)
    opp = pos ^ mask
    result = np.full(COLS, -2, dtype=np.int8)

    if _bb_has_won(opp) or mc >= ROWS * COLS:
        return result

    tt: dict[int, int] = {}
    budget = [max_nodes]

    try:
        for c in range(COLS):
            if mask & (1 << (c * 7 + 5)):
                result[c] = -2
                continue
            stone = (mask + (1 << (c * 7))) & ~mask
            if _bb_has_won(pos | stone):
                result[c] = 1
            elif mc + 1 >= ROWS * COLS:
                result[c] = 0
            else:
                new_mask = mask | stone
                result[c] = -_bb_solve_wdl(opp, new_mask, mc + 1, tt, budget)
    except _SolverBudgetExceeded:
        return None

    return result


def _bb_shallow_tactical_score(pos: int, mask: int, moves: int, depth: int) -> int:
    """Depth-limited tactical minimax returning `+1` (forced win), `-1` (forced loss), or `0`."""
    if moves >= ROWS * COLS:
        return 0
    for c in _CENTER_FIRST_ORDER:
        if (mask & (1 << (c * 7 + 5))) == 0:
            stone = (mask + (1 << (c * 7))) & ~mask
            if _bb_has_won(pos | stone):
                return 1
    if depth <= 1:
        return 0

    opp = pos ^ mask
    forced_col = -1
    for c in _CENTER_FIRST_ORDER:
        if (mask & (1 << (c * 7 + 5))) == 0:
            stone = (mask + (1 << (c * 7))) & ~mask
            if _bb_has_won(opp | stone):
                if forced_col != -1:
                    return -1
                forced_col = c

    cols = (forced_col,) if forced_col != -1 else _CENTER_FIRST_ORDER
    best = -1
    for c in cols:
        if (mask & (1 << (c * 7 + 5))) == 0:
            new_mask = mask | (mask + (1 << (c * 7)))
            val = -_bb_shallow_tactical_score(opp, new_mask, moves + 1, depth - 1)
            if val > best:
                best = val
                if best == 1:
                    break
    return best


def _parse_dataset_lines(lines: list[str]) -> tuple[list[str], np.ndarray]:
    """Parse `<moves_str> <score>` lines into `(moves_list, true_scores)`."""
    moves_list: list[str] = []
    scores_list: list[int] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        moves_str, score_str = parts
        moves_list.append(moves_str)
        scores_list.append(int(score_str))
    return moves_list, np.asarray(scores_list, dtype=np.int32)


def _save_solver_cache_npz(cache_file: Path, child_wdl: np.ndarray) -> None:
    """Atomically write `child_wdl` into `.npz` cache with self-ignoring `.gitignore`."""
    cache_dir = cache_file.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    gitignore_path = cache_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("*\n", encoding="utf-8")

    tmp_file = cache_dir / f".{cache_file.name}.{os.getpid()}.tmp"
    with open(tmp_file, "wb") as f:
        np.savez(f, child_wdl=child_wdl)
    tmp_file.replace(cache_file)


def get_dataset_child_wdl(
    lines: list[str],
    cache_dir: str | Path = DEFAULT_SOLVER_CACHE_DIR,
    use_cache: bool = True,
    python_max_nodes: int = 25_000,
) -> np.ndarray | None:
    """Compute or load cached exact child W/D/L outcomes `int8[N, 7]` for dataset `lines`.

    Entries are from the root mover's perspective:
      `-2`: illegal (full column)
      `-1`: root mover loses with best play
       `0`: draw with best play
      `+1`: root mover wins with best play

    Uses `alphafour_engine.solve_batch_children_wdl` when available and caches
    results by SHA256 hash of the moves list in `testdata/.solver_cache/*.npz`
    so subsequent calls load in <5ms. Falls back to an exact bounded-budget
    pure-Python bitboard solver for tactical / endgame unit-test positions,
    returning `None` only when neither cache nor Rust solver is available on
    deep early-game positions.
    """
    moves_list, _ = _parse_dataset_lines(lines)
    n = len(moves_list)
    if n == 0:
        return np.zeros((0, COLS), dtype=np.int8)

    cache_path = Path(cache_dir)
    cache_key = hashlib.sha256(
        ("child_wdl_v1\n" + "\n".join(moves_list)).encode("utf-8")
    ).hexdigest()
    mem_key = f"{cache_key}:rust={int(bool(RUST_SOLVER_AVAILABLE))}:b={python_max_nodes}"
    cache_file = cache_path / f"{cache_key}.npz"

    if use_cache:
        if mem_key in _MEMORY_SOLVER_CACHE:
            mem_hit = _MEMORY_SOLVER_CACHE[mem_key]
            if mem_hit is None or mem_hit.shape == (n, COLS):
                return mem_hit

        if cache_file.exists():
            try:
                with np.load(cache_file) as data:
                    arr = np.asarray(data["child_wdl"], dtype=np.int8)
                if arr.shape == (n, COLS):
                    _MEMORY_SOLVER_CACHE[mem_key] = arr
                    return arr
            except Exception:
                pass

    child_wdl: np.ndarray | None = None
    solved_by_rust = False

    if RUST_SOLVER_AVAILABLE and solve_batch_children_wdl is not None:
        child_wdl = np.asarray(solve_batch_children_wdl(moves_list), dtype=np.int8)
        solved_by_rust = True
    elif RUST_SOLVER_AVAILABLE and solve_children_wdl is not None:
        child_wdl = np.asarray(
            [solve_children_wdl(m) for m in moves_list], dtype=np.int8
        )
        solved_by_rust = True
    else:
        rows = np.empty((n, COLS), dtype=np.int8)
        total_budget_limit = max(python_max_nodes * 4, 250_000)
        nodes_spent = 0
        all_solved = True
        for i, m in enumerate(moves_list):
            rem = total_budget_limit - nodes_spent
            if rem <= 0:
                all_solved = False
                break
            pos, mask, mc = _bb_from_moves(m)
            # Positions with <20 stones either resolve tactically in <2k nodes or need full Rust solver
            pos_cap = python_max_nodes if mc >= 20 else min(python_max_nodes, 2_000)
            per_pos_budget = min(pos_cap, rem)
            opp = pos ^ mask
            row = np.full(COLS, -2, dtype=np.int8)
            if not (_bb_has_won(opp) or mc >= ROWS * COLS):
                tt: dict[int, int] = {}
                budget = [per_pos_budget]
                try:
                    for c in range(COLS):
                        if mask & (1 << (c * 7 + 5)):
                            continue
                        stone = (mask + (1 << (c * 7))) & ~mask
                        if _bb_has_won(pos | stone):
                            row[c] = 1
                        elif mc + 1 >= ROWS * COLS:
                            row[c] = 0
                        else:
                            row[c] = -_bb_solve_wdl(
                                opp, mask | stone, mc + 1, tt, budget
                            )
                except _SolverBudgetExceeded:
                    all_solved = False
                    break
                nodes_spent += per_pos_budget - budget[0]
            rows[i] = row
        if all_solved:
            child_wdl = rows

    if use_cache:
        _MEMORY_SOLVER_CACHE[mem_key] = child_wdl
        # Persist exact result to disk when solved by Rust or when a custom cache_dir is supplied
        if child_wdl is not None and (
            solved_by_rust or cache_path != DEFAULT_SOLVER_CACHE_DIR
        ):
            try:
                _save_solver_cache_npz(cache_file, child_wdl)
            except OSError:
                pass

    return child_wdl


# ── Batched MCTS Helpers (Rust + Python Fallback) ───────────────────────────


def _rust_game_from_moves(moves_str: str) -> Any:
    """Create a `RustConnectFour` from a 1-indexed move string."""
    if hasattr(RustConnectFour, "from_moves"):
        return RustConnectFour.from_moves(moves_str)
    g = RustConnectFour()
    for ch in moves_str:
        if "1" <= ch <= "7":
            g.make_move(ord(ch) - ord("1"))
    return g


def _python_game_from_moves(moves_str: str) -> ConnectFour:
    """Create a Python `ConnectFour` from a 1-indexed move string."""
    g = ConnectFour()
    for ch in moves_str:
        if "1" <= ch <= "7":
            g.make_move(ord(ch) - ord("1"))
    return g


def _run_rust_batched_mcts(
    model: Any,
    device: torch.device,
    rust_games: list[Any],
    num_simulations: int,
    c_puct: float = 1.5,
    c_fpu: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run `RustBatchedMCTS` exclusively with `model`, returning `(visits, best_actions, best_values)`."""
    n = len(rust_games)
    if n == 0:
        return (
            np.zeros((0, COLS), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    try:
        engine = RustBatchedMCTS(rust_games, num_simulations, c_puct, c_fpu)
    except TypeError:
        engine = RustBatchedMCTS(rust_games, num_simulations)

    root_states, root_valid = engine.get_root_states()
    if len(root_states) > 0:
        with torch.no_grad(), _autocast_ctx(device):
            states_t = torch.from_numpy(root_states).to(device)
            valid_t = torch.from_numpy(root_valid).to(device)
            pols_t, _ = model.predict(states_t, valid_t)
        pols_np = np.ascontiguousarray(pols_t.float().cpu().numpy(), dtype=np.float32)
        engine.init_roots(pols_np)
    else:
        engine.init_roots(np.zeros((0, COLS), dtype=np.float32))

    zero_leaf_steps = 0
    max_zero_steps = max(8, num_simulations + 4)
    while not engine.is_done():
        leaf_states, leaf_valid, count = engine.collect_leaves()
        if count > 0:
            zero_leaf_steps = 0
            with torch.no_grad(), _autocast_ctx(device):
                states_t = torch.from_numpy(leaf_states).to(device)
                valid_t = torch.from_numpy(leaf_valid).to(device)
                pols_t, vals_t = model.predict(states_t, valid_t)
            pols_np = np.ascontiguousarray(
                pols_t.float().cpu().numpy(), dtype=np.float32
            )
            vals_np = np.ascontiguousarray(
                vals_t.float().cpu().numpy().reshape(-1), dtype=np.float32
            )
            engine.apply_evaluations(pols_np, vals_np)
        else:
            zero_leaf_steps += 1
            if zero_leaf_steps > max_zero_steps:
                break

    visit_counts, best_actions, best_values, _ = engine.get_results()
    return (
        np.asarray(visit_counts, dtype=np.float32),
        np.asarray(best_actions, dtype=np.int64),
        np.asarray(best_values, dtype=np.float32),
    )


def _run_python_batched_mcts(
    model: Any,
    device: torch.device,
    games: list[ConnectFour],
    num_simulations: int,
    c_puct: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run pure-Python batched MCTS exclusively with `model`.

    Extracts undiluted `best_values = -best_child.q_value` (or exact proven
    solver outcome) from the root mover's perspective — never diluted `root.q_value`.
    """
    n = len(games)
    if n == 0:
        return (
            np.zeros((0, COLS), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    roots = [MCTSNode(g) for g in games]
    non_term_indices = [i for i, g in enumerate(games) if not g.is_terminal()]

    if non_term_indices:
        states_np = np.stack([games[i].encode() for i in non_term_indices])
        valid_np = np.stack([games[i].get_valid_moves() for i in non_term_indices])
        with torch.no_grad(), _autocast_ctx(device):
            states_t = torch.from_numpy(states_np).to(device)
            valid_t = torch.from_numpy(valid_np).to(device)
            pols_t, _ = model.predict(states_t, valid_t)
        pols_np = pols_t.float().cpu().numpy()
        for idx, game_idx in enumerate(non_term_indices):
            roots[game_idx].expand(pols_np[idx])

    for _ in range(num_simulations):
        leaves_to_eval: list[np.ndarray] = []
        valid_to_eval: list[np.ndarray] = []
        eval_nodes: list[MCTSNode] = []

        for i in non_term_indices:
            root = roots[i]
            root_solver = getattr(root, "solver_state", 0)
            if root_solver != 0:
                continue

            node = root
            while (
                node.is_expanded
                and node.children
                and getattr(node, "solver_state", 0) == 0
            ):
                node = node.select_child(c_puct)

            node_solver = getattr(node, "solver_state", 0)
            if node_solver != 0:
                val = 1.0 if node_solver == 1 else (-1.0 if node_solver == -1 else 0.0)
                node.backpropagate(val)
                continue

            res = node.game.get_result()
            if res is not None:
                node.backpropagate(res)
            else:
                leaves_to_eval.append(node.game.encode())
                valid_to_eval.append(node.game.get_valid_moves())
                eval_nodes.append(node)

        if leaves_to_eval:
            with torch.no_grad(), _autocast_ctx(device):
                states_t = torch.from_numpy(np.stack(leaves_to_eval)).to(device)
                valid_t = torch.from_numpy(np.stack(valid_to_eval)).to(device)
                pols_t, vals_t = model.predict(states_t, valid_t)
            pols_np = pols_t.float().cpu().numpy()
            vals_np = vals_t.float().cpu().numpy().reshape(-1)

            for idx, node in enumerate(eval_nodes):
                node.expand(pols_np[idx])
                node.backpropagate(float(vals_np[idx]))

    visit_counts = np.zeros((n, COLS), dtype=np.float32)
    best_actions = np.zeros((n,), dtype=np.int64)
    best_values = np.zeros((n,), dtype=np.float32)

    for i, root in enumerate(roots):
        res = root.game.get_result()
        if res is not None or not root.children:
            best_values[i] = float(res) if res is not None else 0.0
            continue

        for col, child in root.children.items():
            visit_counts[i, col] = float(child.visit_count)

        # Prefer proven winning moves (child is SOLVER_LOSS == -1 or terminal -1.0 for opponent)
        winning_cols = [
            col
            for col, ch in root.children.items()
            if getattr(ch, "solver_state", 0) == -1 or ch.game.get_result() == -1.0
        ]
        if winning_cols:
            best_col = max(
                winning_cols,
                key=lambda c: (root.children[c].visit_count, root.children[c].prior, -c),
            )
            best_actions[i] = best_col
            best_values[i] = 1.0
            if visit_counts[i].sum() == 0:
                visit_counts[i, best_col] = 1.0
            continue

        # Avoid proven losing moves (child is SOLVER_WIN == +1 for opponent) unless all lose
        non_losing_cols = [
            col
            for col, ch in root.children.items()
            if getattr(ch, "solver_state", 0) != 1
        ]
        candidates = non_losing_cols if non_losing_cols else list(root.children.keys())
        best_col = max(
            candidates,
            key=lambda c: (root.children[c].visit_count, root.children[c].prior, -c),
        )
        best_child = root.children[best_col]
        best_actions[i] = best_col

        ch_solver = getattr(best_child, "solver_state", 0)
        ch_res = best_child.game.get_result()
        if ch_solver == -1 or ch_res == -1.0:
            best_values[i] = 1.0
        elif ch_solver == 1:
            best_values[i] = -1.0
        elif ch_solver == 2 or ch_res == 0.0:
            best_values[i] = 0.0
        elif best_child.visit_count > 0:
            best_values[i] = -float(best_child.q_value)
        else:
            best_values[i] = 0.0

    return visit_counts, best_actions, best_values


# ── Public Evaluation Functions ─────────────────────────────────────────────


def evaluate_opening_move(
    model: AlphaZeroNet, device: torch.device, num_simulations: int = 0
) -> tuple[int, float]:
    """Evaluate the opening move on an empty Connect Four board.

    Returns:
        `(best_move_column, center_column_pct)` where `center_column_pct` is
        center policy % at `0` simulations or center visit % at `>0` simulations.
    """
    model.eval()

    if num_simulations > 0:
        if RUST_MCTS_AVAILABLE and RustConnectFour is not None:
            vc, best_actions, _ = _run_rust_batched_mcts(
                model, device, [RustConnectFour()], num_simulations
            )
        else:
            vc, best_actions, _ = _run_python_batched_mcts(
                model, device, [ConnectFour()], num_simulations
            )
        counts = vc[0]
        best_move = int(best_actions[0])
        total = float(counts.sum())
        center_pct = float(counts[3] / total) * 100.0 if total > 0 else 0.0
        return best_move, center_pct

    game = ConnectFour()
    state_t = torch.from_numpy(np.expand_dims(game.encode(), axis=0)).to(device)
    val_moves_t = torch.from_numpy(np.expand_dims(game.get_valid_moves(), axis=0)).to(
        device
    )
    with torch.no_grad(), _autocast_ctx(device):
        policies_t, _ = model.predict(state_t, val_moves_t)
    policies = policies_t[0].float().cpu().numpy()

    best_move = int(np.argmax(policies))
    total = float(policies.sum())
    center_pct = float(policies[3] / total) * 100.0 if total > 0 else 0.0
    return best_move, center_pct


def evaluate_dataset(
    model: AlphaZeroNet,
    device: torch.device,
    lines: list[str],
    num_simulations: int = 0,
    child_wdl: np.ndarray | None = None,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Evaluate model policy and value quality against a gamesolver.org dataset.

    Args:
        model: Neural network (or compatible predictor) to evaluate.
        device: Torch inference device.
        lines: Strings of format `"<moves_str> <score>"`.
        num_simulations: `0` for raw network evaluation, `>0` for batched MCTS.
        child_wdl: Precomputed `int8[N, 7]` exact child W/D/L array; if `None`,
            computed/loaded via `get_dataset_child_wdl(lines)`.
        batch_size: Positions evaluated per batch.

    Returns:
        Dict with:
            - `wdl_accuracy` (float): 3-class W/D/L accuracy (%).
            - `accuracy` (float): Alias for `wdl_accuracy`.
            - `win_recall` (float): Recall (%) on true Win positions (`nan` if 0).
            - `draw_recall` (float): Recall (%) on true Draw positions (`nan` if 0).
            - `loss_recall` (float): Recall (%) on true Loss positions (`nan` if 0).
            - `value_mse` (float): MSE of predicted values vs `sign(true_score)`.
            - `optimal_move_pct` (float | None): % choosing a W/D/L-optimal move.
            - `blunder_pct` (float | None): % blundering on non-already-lost positions.
    """
    model.eval()
    moves_list, true_scores = _parse_dataset_lines(lines)
    n = len(moves_list)

    if n == 0:
        return {
            "wdl_accuracy": 0.0,
            "accuracy": 0.0,
            "win_recall": float("nan"),
            "draw_recall": float("nan"),
            "loss_recall": float("nan"),
            "value_mse": 0.0,
            "optimal_move_pct": None,
            "blunder_pct": None,
        }

    if child_wdl is None:
        child_wdl = get_dataset_child_wdl(lines)

    pred_values = np.empty(n, dtype=np.float32)
    best_actions = np.empty(n, dtype=np.int64)
    raw_wdl_classes: np.ndarray | None = None

    use_predict_wdl = num_simulations == 0 and hasattr(model, "predict_wdl")
    if use_predict_wdl:
        raw_wdl_classes = np.empty(n, dtype=np.int8)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk_moves = moves_list[start:end]

        if num_simulations == 0:
            py_games = [_python_game_from_moves(m) for m in chunk_moves]
            states_np = np.stack([g.encode() for g in py_games])
            valid_np = np.stack([g.get_valid_moves() for g in py_games])

            states_t = torch.from_numpy(states_np).to(device)
            valid_t = torch.from_numpy(valid_np).to(device)

            with torch.no_grad(), _autocast_ctx(device):
                if use_predict_wdl:
                    pols_t, wdl_probs_t, vals_t = model.predict_wdl(states_t, valid_t)
                    wdl_probs_np = wdl_probs_t.float().cpu().numpy()
                    assert raw_wdl_classes is not None
                    # wdl_probs ordered [P(Loss), P(Draw), P(Win)] => classes {-1, 0, +1}
                    raw_wdl_classes[start:end] = (
                        np.argmax(wdl_probs_np, axis=1).astype(np.int8) - 1
                    )
                else:
                    pols_t, vals_t = model.predict(states_t, valid_t)

            pols_np = pols_t.float().cpu().numpy()
            vals_np = vals_t.float().cpu().numpy().reshape(-1)
            masked_pols = np.where(valid_np, pols_np, -np.inf)

            best_actions[start:end] = np.argmax(masked_pols, axis=1)
            pred_values[start:end] = vals_np
        else:
            if RUST_MCTS_AVAILABLE and RustConnectFour is not None:
                r_games = [_rust_game_from_moves(m) for m in chunk_moves]
                _, chunk_actions, chunk_values = _run_rust_batched_mcts(
                    model, device, r_games, num_simulations
                )
            else:
                py_games = [_python_game_from_moves(m) for m in chunk_moves]
                _, chunk_actions, chunk_values = _run_python_batched_mcts(
                    model, device, py_games, num_simulations
                )
            best_actions[start:end] = chunk_actions
            pred_values[start:end] = chunk_values

    true_wdl = np.sign(true_scores).astype(np.int8)

    if raw_wdl_classes is not None:
        pred_wdl = raw_wdl_classes
    else:
        threshold = 1.0 / 3.0
        pred_wdl = np.where(
            pred_values > threshold,
            1,
            np.where(pred_values < -threshold, -1, 0),
        ).astype(np.int8)

    wdl_accuracy = float(np.mean(pred_wdl == true_wdl) * 100.0)

    def _class_recall(target_cls: int) -> float:
        mask = true_wdl == target_cls
        if not np.any(mask):
            return float("nan")
        return float(np.mean(pred_wdl[mask] == target_cls) * 100.0)

    win_recall = _class_recall(1)
    draw_recall = _class_recall(0)
    loss_recall = _class_recall(-1)

    value_mse = float(
        np.mean((pred_values.astype(np.float64) - true_wdl.astype(np.float64)) ** 2)
    )

    optimal_move_pct: float | None = None
    blunder_pct: float | None = None
    if child_wdl is not None and child_wdl.shape == (n, COLS):
        max_valid_wdl = child_wdl.max(axis=1)
        chosen_wdl = child_wdl[np.arange(n), best_actions]
        optimal_move_pct = float(np.mean(chosen_wdl == max_valid_wdl) * 100.0)
        non_lost = max_valid_wdl >= 0
        if np.any(non_lost):
            blunder_pct = float(
                np.mean(chosen_wdl[non_lost] < max_valid_wdl[non_lost]) * 100.0
            )
        else:
            blunder_pct = float("nan")

    return {
        "wdl_accuracy": wdl_accuracy,
        "accuracy": wdl_accuracy,
        "win_recall": win_recall,
        "draw_recall": draw_recall,
        "loss_recall": loss_recall,
        "value_mse": value_mse,
        "optimal_move_pct": optimal_move_pct,
        "blunder_pct": blunder_pct,
    }


# ── Paired Openings, Model-Isolated Arena, and Anchor Elo ───────────────────


def generate_paired_openings(
    num_pairs: int,
    seed: int = 42,
    min_plies: int = 2,
    max_plies: int = 4,
) -> list[str]:
    """Generate `num_pairs` diverse, balanced 2–4 ply non-terminal openings.

    Candidates are deterministically sampled and filtered via a 6-ply bitboard
    tactical check so neither side starts in a blundered forced tactical loss.
    """
    if num_pairs <= 0:
        return []

    rng = random.Random(seed)
    candidates: list[str] = []

    def _enumerate_plies(prefix: str, target_len: int) -> None:
        if len(prefix) == target_len:
            candidates.append(prefix)
            return
        for c in range(1, COLS + 1):
            _enumerate_plies(prefix + str(c), target_len)

    for length in range(min_plies, max_plies + 1):
        _enumerate_plies("", length)

    rng.shuffle(candidates)

    balanced: list[str] = []
    fallback_non_terminal: list[str] = []

    for moves_str in candidates:
        try:
            pos, mask, mc = _bb_from_moves(moves_str)
        except ValueError:
            continue
        opp = pos ^ mask
        if _bb_has_won(pos) or _bb_has_won(opp) or mc >= ROWS * COLS:
            continue

        fallback_non_terminal.append(moves_str)
        # Filter out openings where either side has walked into a shallow forced loss/win
        tactical = _bb_shallow_tactical_score(pos, mask, mc, depth=6)
        if tactical == 0:
            balanced.append(moves_str)
            if len(balanced) >= num_pairs:
                break

    if len(balanced) >= num_pairs:
        return balanced[:num_pairs]

    # Cycle if num_pairs exceeds the total balanced 2-4 ply pool
    pool = balanced if balanced else fallback_non_terminal
    return [pool[i % len(pool)] for i in range(num_pairs)]


def _select_arena_actions_for_partition(
    model: Any,
    device: torch.device,
    partition_games: list[Any],
    num_simulations: int,
    use_rust: bool,
) -> np.ndarray:
    """Run MCTS search for `partition_games` using ONLY `model` and return `visit_counts[K, 7]`."""
    if not partition_games:
        return np.zeros((0, COLS), dtype=np.float32)

    if num_simulations <= 0:
        if use_rust:
            states_np = np.stack([g.encode() for g in partition_games])
            valid_np = np.stack([g.get_valid_moves() for g in partition_games])
        else:
            states_np = np.stack([g.encode() for g in partition_games])
            valid_np = np.stack([g.get_valid_moves() for g in partition_games])
        with torch.no_grad(), _autocast_ctx(device):
            states_t = torch.from_numpy(states_np).to(device)
            valid_t = torch.from_numpy(valid_np).to(device)
            pols_t, _ = model.predict(states_t, valid_t)
        pols_np = pols_t.float().cpu().numpy()
        return np.where(valid_np, pols_np, 0.0).astype(np.float32)

    if use_rust:
        vc, _, _ = _run_rust_batched_mcts(
            model, device, partition_games, num_simulations
        )
        return vc
    vc, _, _ = _run_python_batched_mcts(model, device, partition_games, num_simulations)
    return vc


def play_batched_arena(
    model1: Any,
    model2: Any,
    device: torch.device,
    num_games: int = 40,
    num_simulations: int = 800,
    temperature_moves: int = 6,
    temperature: float = 0.0,
    paired_openings: bool = True,
    seed: int = 42,
    openings: list[str] | None = None,
    num_simulations_m2: int | None = None,
) -> tuple[int, int, int, int, int]:
    """Play head-to-head Arena games between `model1` and `model2` with strict model isolation.

    At every ply, active games are partitioned into `model1`-to-move and
    `model2`-to-move subsets. Each model evaluates 100% of its own root and
    internal MCTS search-tree leaves without any opponent-model contamination.

    When `paired_openings=True` (default), `num_games // 2` balanced 2–4 ply
    openings are played twice each (once with `model1` as Player 1 and once
    with `model1` as Player 2) using deterministic `temperature=0.0` by default.

    Returns:
        `(m1_wins_p1, m1_wins_p2, m2_wins_p1, m2_wins_p2, draws)`
    """
    if hasattr(model1, "eval"):
        model1.eval()
    if hasattr(model2, "eval"):
        model2.eval()

    if num_games <= 0:
        return 0, 0, 0, 0, 0

    sims_m1 = num_simulations
    sims_m2 = num_simulations if num_simulations_m2 is None else num_simulations_m2
    use_rust = RUST_MCTS_AVAILABLE and RustConnectFour is not None

    num_pairs = num_games // 2
    m1_is_p1: list[bool] = []
    games: list[Any] = []
    move_counts: list[int] = []

    if paired_openings and num_pairs > 0:
        pair_list = (
            list(openings[:num_pairs])
            if openings is not None and len(openings) >= num_pairs
            else generate_paired_openings(num_pairs, seed=seed)
        )
        # First num_pairs: model1 is P1; next num_pairs: model1 is P2 on the exact same openings
        for moves_str in pair_list:
            g1 = (
                _rust_game_from_moves(moves_str)
                if use_rust
                else _python_game_from_moves(moves_str)
            )
            games.append(g1)
            m1_is_p1.append(True)
            move_counts.append(0)
        for moves_str in pair_list:
            g2 = (
                _rust_game_from_moves(moves_str)
                if use_rust
                else _python_game_from_moves(moves_str)
            )
            games.append(g2)
            m1_is_p1.append(False)
            move_counts.append(0)
        if num_games % 2 == 1:
            g_extra = RustConnectFour() if use_rust else ConnectFour()
            games.append(g_extra)
            m1_is_p1.append(True)
            move_counts.append(0)
    else:
        first_half = (num_games + 1) // 2
        m1_is_p1 = [True] * first_half + [False] * (num_games - first_half)
        games = [
            RustConnectFour() if use_rust else ConnectFour() for _ in range(num_games)
        ]
        move_counts = [0] * num_games

    rng = np.random.default_rng(seed)
    active_indices = [i for i in range(num_games) if not games[i].is_terminal()]

    model1_wins_p1 = 0
    model1_wins_p2 = 0
    model2_wins_p1 = 0
    model2_wins_p2 = 0
    draws = 0

    sort_divisor = num_pairs if (paired_openings and num_pairs > 0) else num_games

    while active_indices:
        m1_turn_indices = [
            i
            for i in active_indices
            if m1_is_p1[i] == (int(games[i].current_player) == 1)
        ]
        m2_turn_indices = [
            i
            for i in active_indices
            if m1_is_p1[i] != (int(games[i].current_player) == 1)
        ]

        # Align batch order by opening pair index so twin games see identical batch layouts
        if paired_openings and num_pairs > 0:
            m1_turn_indices.sort(key=lambda idx: (idx % sort_divisor, idx))
            m2_turn_indices.sort(key=lambda idx: (idx % sort_divisor, idx))

        visits_by_game: dict[int, np.ndarray] = {}

        if m1_turn_indices:
            vc_m1 = _select_arena_actions_for_partition(
                model1,
                device,
                [games[i] for i in m1_turn_indices],
                sims_m1,
                use_rust=use_rust,
            )
            for pos_idx, game_idx in enumerate(m1_turn_indices):
                visits_by_game[game_idx] = vc_m1[pos_idx]

        if m2_turn_indices:
            vc_m2 = _select_arena_actions_for_partition(
                model2,
                device,
                [games[i] for i in m2_turn_indices],
                sims_m2,
                use_rust=use_rust,
            )
            for pos_idx, game_idx in enumerate(m2_turn_indices):
                visits_by_game[game_idx] = vc_m2[pos_idx]

        new_active: list[int] = []
        for i in active_indices:
            game = games[i]
            visit_counts = visits_by_game[i]
            valid = np.asarray(game.get_valid_moves(), dtype=bool)

            if move_counts[i] < temperature_moves and temperature > 0.0:
                masked = np.where(valid, visit_counts, 0.0).astype(np.float64)
                total = masked.sum()
                if total > 0:
                    probs = masked ** (1.0 / temperature)
                    probs /= probs.sum()
                else:
                    probs = valid.astype(np.float64) / valid.sum()
                action = int(rng.choice(COLS, p=probs))
            else:
                masked = np.where(valid, visit_counts, -np.inf)
                action = int(np.argmax(masked))

            game.make_move(action)
            move_counts[i] += 1

            if game.is_terminal():
                res = game.get_result()
                if res == 0.0:
                    draws += 1
                else:
                    # res == -1.0 means the player who just moved won
                    winner_player = -int(game.current_player)
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

        active_indices = new_active

    return model1_wins_p1, model1_wins_p2, model2_wins_p1, model2_wins_p2, draws


class PureMCTSBaseline:
    """Fast non-neural Connect Four baseline opponent for absolute Anchor Elo tracking.

    Combines a uniform valid-move prior with a vectorized tactical/positional
    leaf evaluator (counting open 3-in-a-row threats, open 2-in-a-row lines,
    and center-column control across all 69 four-in-a-row windows) so MCTS +
    MCTS-Solver plays crisp, deterministic tactical Connect Four without any
    neural network weights.
    """

    def __init__(self, use_heuristic: bool = True) -> None:
        self.use_heuristic = use_heuristic

    def eval(self) -> PureMCTSBaseline:
        return self

    def predict(
        self,
        state: torch.Tensor,
        valid_moves: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single = state.dim() == 3
        if single:
            state = state.unsqueeze(0)
            if valid_moves is not None:
                valid_moves = valid_moves.unsqueeze(0)

        bsz = state.shape[0]
        dev = state.device
        state_f = state.float()

        if valid_moves is None:
            occupied = state_f[:, 0] + state_f[:, 1]
            valid_moves = occupied.sum(dim=1) < ROWS

        valid_f = valid_moves.to(dtype=torch.float32)
        denom = valid_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        policy = valid_f / denom

        if not self.use_heuristic:
            values = torch.zeros(bsz, dtype=torch.float32, device=dev)
        else:
            me = state_f[:, 0]  # (B, 6, 7)
            opp = state_f[:, 1]  # (B, 6, 7)

            # Horizontal windows (B, 6, 4)
            mh = me[:, :, 0:4] + me[:, :, 1:5] + me[:, :, 2:6] + me[:, :, 3:7]
            oh = opp[:, :, 0:4] + opp[:, :, 1:5] + opp[:, :, 2:6] + opp[:, :, 3:7]
            # Vertical windows (B, 3, 7)
            mv = me[:, 0:3, :] + me[:, 1:4, :] + me[:, 2:5, :] + me[:, 3:6, :]
            ov = opp[:, 0:3, :] + opp[:, 1:4, :] + opp[:, 2:5, :] + opp[:, 3:6, :]
            # Diagonal / windows (B, 3, 4)
            md1 = (
                me[:, 0:3, 0:4]
                + me[:, 1:4, 1:5]
                + me[:, 2:5, 2:6]
                + me[:, 3:6, 3:7]
            )
            od1 = (
                opp[:, 0:3, 0:4]
                + opp[:, 1:4, 1:5]
                + opp[:, 2:5, 2:6]
                + opp[:, 3:6, 3:7]
            )
            # Diagonal \ windows (B, 3, 4)
            md2 = (
                me[:, 3:6, 0:4]
                + me[:, 2:5, 1:5]
                + me[:, 1:4, 2:6]
                + me[:, 0:3, 3:7]
            )
            od2 = (
                opp[:, 3:6, 0:4]
                + opp[:, 2:5, 1:5]
                + opp[:, 1:4, 2:6]
                + opp[:, 0:3, 3:7]
            )

            me_threats3 = (
                ((mh == 3) & (oh == 0)).sum(dim=(1, 2))
                + ((mv == 3) & (ov == 0)).sum(dim=(1, 2))
                + ((md1 == 3) & (od1 == 0)).sum(dim=(1, 2))
                + ((md2 == 3) & (od2 == 0)).sum(dim=(1, 2))
            ).float()
            opp_threats3 = (
                ((oh == 3) & (mh == 0)).sum(dim=(1, 2))
                + ((ov == 3) & (mv == 0)).sum(dim=(1, 2))
                + ((od1 == 3) & (md1 == 0)).sum(dim=(1, 2))
                + ((od2 == 3) & (md2 == 0)).sum(dim=(1, 2))
            ).float()
            me_lines2 = (
                ((mh == 2) & (oh == 0)).sum(dim=(1, 2))
                + ((mv == 2) & (ov == 0)).sum(dim=(1, 2))
                + ((md1 == 2) & (od1 == 0)).sum(dim=(1, 2))
                + ((md2 == 2) & (od2 == 0)).sum(dim=(1, 2))
            ).float()
            opp_lines2 = (
                ((oh == 2) & (mh == 0)).sum(dim=(1, 2))
                + ((ov == 2) & (mv == 0)).sum(dim=(1, 2))
                + ((od1 == 2) & (md1 == 0)).sum(dim=(1, 2))
                + ((od2 == 2) & (md2 == 0)).sum(dim=(1, 2))
            ).float()

            center_diff = (me[:, :, 3] - opp[:, :, 3]).sum(dim=1)
            raw_score = (
                0.25 * (me_threats3 - opp_threats3)
                + 0.04 * (me_lines2 - opp_lines2)
                + 0.06 * center_diff
            )
            values = torch.tanh(raw_score)

        if single:
            return policy.squeeze(0), values.squeeze(0)
        return policy, values


def compute_elo_diff(win_rate: float, eps: float = 1e-3) -> float:
    """Compute Elo rating difference `400 * log10(s / (1 - s))` from score `s ∈ [0, 1]`.

    Clamps `win_rate` to `[eps, 1 - eps]` so `0.0` and `1.0` return safe finite bounds.
    """
    s = min(max(float(win_rate), eps), 1.0 - eps)
    return 400.0 * math.log10(s / (1.0 - s))


def play_anchor_arena(
    model: Any,
    device: torch.device,
    num_games: int = 40,
    num_simulations: int = 400,
    anchor_simulations: int = 200,
    paired_openings: bool = True,
    seed: int = 42,
) -> dict[str, float | int]:
    """Evaluate `model` against `PureMCTSBaseline` on duplicate paired openings.

    Returns dict with win/loss/draw counts, fractional `win_rate`, `win_rate_pct`,
    and `elo_diff` relative to the fixed non-neural anchor (`0` Elo).
    """
    anchor = PureMCTSBaseline(use_heuristic=True)
    m1_w_p1, m1_w_p2, m2_w_p1, m2_w_p2, draws = play_batched_arena(
        model1=model,
        model2=anchor,
        device=device,
        num_games=num_games,
        num_simulations=num_simulations,
        temperature=0.0,
        paired_openings=paired_openings,
        seed=seed,
        num_simulations_m2=anchor_simulations,
    )
    model_wins = m1_w_p1 + m1_w_p2
    anchor_wins = m2_w_p1 + m2_w_p2
    total = model_wins + anchor_wins + draws
    win_rate = (model_wins + 0.5 * draws) / total if total > 0 else 0.5
    elo_diff = compute_elo_diff(win_rate)

    return {
        "model_wins": model_wins,
        "model_wins_p1": m1_w_p1,
        "model_wins_p2": m1_w_p2,
        "anchor_wins": anchor_wins,
        "anchor_wins_p1": m2_w_p1,
        "anchor_wins_p2": m2_w_p2,
        "draws": draws,
        "win_rate": win_rate,
        "win_rate_pct": win_rate * 100.0,
        "elo_diff": elo_diff,
    }
