"""High-performance AlphaZero batched self-play engine in Cython Pure Python Mode.

Implements bitboard ConnectFour, a flat C-struct MCTS node arena with double-buffered
pointer-swap subtree rebasing, exact MCTS-Solver (win/loss/draw propagation),
virtual-loss duplicate-free multi-leaf batching, parent-relative First Play Urgency (FPU),
SplitMix64 + Marsaglia-Tsang Gamma/Dirichlet root exploration noise, stochastic
opening plies, temperature-controlled move selection with decoupled visit-count policy
targets, and horizontal mirror pair augmentation — all in one C translation unit.

Remains 100% valid, importable Pure Python when uncompiled (`is_cython_compiled() == False`),
while compiling to native `nogil` C + OpenMP via `setup_cython.py`.
"""

import cython
from cython.cimports.libc import math as cmath
from cython.parallel import prange
import numpy as np

if cython.compiled:
    from cython.cimports.libc import stdlib, string

# Board & solver constants (matching src/game.py, src/mcts.py, and rust/src/*)
ROWS: int = 6
COLS: int = 7
WIN_LENGTH: int = 4
BOARD_SIZE: int = 42
H1: int = 7
PLANE_FLOATS: int = 126  # 3 * 6 * 7

SOLVER_ONGOING: int = 0
SOLVER_WIN: int = 1
SOLVER_LOSS: int = -1
SOLVER_DRAW: int = 2

LEAVES_PER_GAME: int = 16


def is_cython_compiled() -> bool:
    """Return True if this module was compiled to native C by Cython."""
    return bool(cython.compiled)


# ── Flat C-struct layouts (zero per-node / per-move Python objects) ──────────

HistoryEntry = cython.struct(
    state=cython.uchar[126],
    policy=cython.float[7],
    player=cython.schar,
)

MCTSNode = cython.struct(
    bb_p1=cython.ulonglong,
    bb_p2=cython.ulonglong,
    total_value=cython.double,
    prior=cython.float,
    heights_packed=cython.uint,
    parent=cython.int,
    first_child=cython.int,
    visit_count=cython.int,
    virtual_losses=cython.int,
    current_player=cython.schar,
    last_move=cython.schar,
    move_count=cython.uchar,
    num_children=cython.uchar,
    is_expanded=cython.uchar,
    solver_state=cython.schar,
)

SelfPlaySlot = cython.struct(
    game_bb_p1=cython.ulonglong,
    game_bb_p2=cython.ulonglong,
    rng_state=cython.ulonglong,
    game_heights_packed=cython.uint,
    game_current_player=cython.schar,
    game_last_move=cython.schar,
    game_move_count=cython.uchar,
    active=cython.uchar,
    root_noisy=cython.uchar,
    noise_applied=cython.uchar,
    pending_root_init=cython.uchar,
    sims_done=cython.int,
    last_batch_leaves=cython.int,
    batch_offset=cython.int,
    n_pending=cython.int,
    pending_nodes=cython.int[16],
    history_len=cython.int,
    history=HistoryEntry[42],
    nodes=cython.pointer[MCTSNode],
    nodes_len=cython.int,
    nodes_cap=cython.int,
    scratch_nodes=cython.pointer[MCTSNode],
    scratch_cap=cython.int,
    rebase_queue=cython.pointer[cython.int],
    rebase_queue_cap=cython.int,
)


AdvanceResult = cython.struct(
    finished=cython.int,
    active_count=cython.int,
    completed_examples=cython.int,
)

if not cython.compiled:

    def _new_py_node() -> object:
        """Create a default-initialized MCTSNode struct instance for pure-Python mode."""
        return MCTSNode(
            bb_p1=0,
            bb_p2=0,
            total_value=0.0,
            prior=0.0,
            heights_packed=0,
            parent=-1,
            first_child=-1,
            visit_count=0,
            virtual_losses=0,
            current_player=1,
            last_move=-1,
            move_count=0,
            num_children=0,
            is_expanded=0,
            solver_state=0,
        )

    def _new_py_slot(init_node_cap: int, seed: int) -> object:
        """Create a default-initialized SelfPlaySlot struct instance for pure-Python mode."""
        return SelfPlaySlot(
            game_bb_p1=0,
            game_bb_p2=0,
            rng_state=seed,
            game_heights_packed=0,
            game_current_player=1,
            game_last_move=-1,
            game_move_count=0,
            active=1,
            root_noisy=0,
            noise_applied=0,
            pending_root_init=0,
            sims_done=0,
            last_batch_leaves=16,
            batch_offset=0,
            n_pending=0,
            pending_nodes=[0] * 16,
            history_len=0,
            history=[
                HistoryEntry(state=[0] * 126, policy=[0.0] * 7, player=1)
                for _ in range(42)
            ],
            nodes=[_new_py_node() for _ in range(init_node_cap)],
            nodes_len=0,
            nodes_cap=init_node_cap,
            scratch_nodes=[_new_py_node() for _ in range(init_node_cap)],
            scratch_cap=init_node_cap,
            rebase_queue=[0] * (2 * init_node_cap),
            rebase_queue_cap=2 * init_node_cap,
        )


# ── Inline nogil PRNG & Gamma/Dirichlet samplers ─────────────────────────────


@cython.cfunc
@cython.inline
@cython.cdivision(True)
@cython.exceptval(check=False)
@cython.nogil
def _rng_next_u64(slot: cython.pointer[SelfPlaySlot]) -> cython.ulonglong:
    """64-bit XorShift64* PRNG step (exact 64-bit wraparound in both C and pure Python)."""
    x: cython.ulonglong = slot[0].rng_state
    if cython.compiled:
        x = x ^ (x >> 12)
        x = x ^ (x << 25)
        x = x ^ (x >> 27)
        slot[0].rng_state = x
        return x * cython.cast(cython.ulonglong, 2685821657736338717)
    else:
        x = (x ^ (x >> 12)) & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x << 25)) & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 27)) & 0xFFFFFFFFFFFFFFFF
        slot[0].rng_state = x
        return (x * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF


@cython.cfunc
@cython.inline
@cython.cdivision(True)
@cython.exceptval(check=False)
@cython.nogil
def _rng_uniform_f64(slot: cython.pointer[SelfPlaySlot]) -> cython.double:
    """Uniform float64 in [0.0, 1.0)."""
    u: cython.ulonglong = _rng_next_u64(slot) >> 11
    return cython.cast(cython.double, u) * (1.0 / 9007199254740992.0)


@cython.cfunc
@cython.inline
@cython.cdivision(True)
@cython.exceptval(check=False)
@cython.nogil
def _rng_uniform_open_f64(slot: cython.pointer[SelfPlaySlot]) -> cython.double:
    """Uniform float64 strictly inside (0.0, 1.0)."""
    u: cython.ulonglong = _rng_next_u64(slot) >> 12
    return (cython.cast(cython.double, u) + 0.5) * (1.0 / 4503599627370496.0)


@cython.cfunc
@cython.inline
@cython.cdivision(True)
@cython.exceptval(check=False)
@cython.nogil
def _rng_range(slot: cython.pointer[SelfPlaySlot], upper: cython.int) -> cython.int:
    """Uniform integer in 0..upper-1 (assumes upper >= 1)."""
    if upper <= 1:
        return 0
    return cython.cast(
        cython.int,
        _rng_next_u64(slot) % cython.cast(cython.ulonglong, upper),
    )


@cython.cfunc
@cython.cdivision(True)
@cython.exceptval(check=False)
@cython.nogil
def _rng_gamma(
    slot: cython.pointer[SelfPlaySlot], alpha: cython.double
) -> cython.double:
    """Exact Marsaglia-Tsang Gamma(alpha, 1.0) sampler with alpha < 1 boost."""
    boost: cython.double = 1.0
    a: cython.double = alpha
    u_boost: cython.double
    if a < 1.0:
        u_boost = _rng_uniform_open_f64(slot)
        boost = cmath.pow(u_boost, 1.0 / a)
        a = a + 1.0

    d: cython.double = a - (1.0 / 3.0)
    c: cython.double = 1.0 / cmath.sqrt(9.0 * d)
    u1: cython.double
    u2: cython.double
    s: cython.double
    z: cython.double
    v: cython.double
    u: cython.double
    z2: cython.double

    while True:
        while True:
            u1 = 2.0 * _rng_uniform_f64(slot) - 1.0
            u2 = 2.0 * _rng_uniform_f64(slot) - 1.0
            s = u1 * u1 + u2 * u2
            if 0.0 < s < 1.0:
                break
        z = u1 * cmath.sqrt(-2.0 * cmath.log(s) / s)
        v = 1.0 + c * z
        if v <= 0.0:
            continue
        v = v * v * v
        u = _rng_uniform_open_f64(slot)
        z2 = z * z
        if u < 1.0 - 0.0331 * (z2 * z2):
            return d * v * boost
        if cmath.log(u) < 0.5 * z2 + d * (1.0 - v + cmath.log(v)):
            return d * v * boost


# ── Inline nogil Bitboard ConnectFour primitives ─────────────────────────────


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _col_height(heights_packed: cython.uint, col: cython.int) -> cython.int:
    return cython.cast(cython.int, (heights_packed >> (col * 4)) & 0xF)


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _has_won(bb: cython.ulonglong) -> cython.bint:
    """Branchless O(1) bitboard 4-in-a-row check across all 4 directions."""
    m1: cython.ulonglong = bb & (bb >> 1)
    m2: cython.ulonglong = bb & (bb >> 7)
    m3: cython.ulonglong = bb & (bb >> 8)
    m4: cython.ulonglong = bb & (bb >> 6)
    return (
        (m1 & (m1 >> 2))
        | (m2 & (m2 >> 14))
        | (m3 & (m3 >> 16))
        | (m4 & (m4 >> 12))
    ) != 0


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _is_win_state(
    bb_p1: cython.ulonglong,
    bb_p2: cython.ulonglong,
    current_player: cython.schar,
    last_move: cython.schar,
    move_count: cython.uchar,
) -> cython.bint:
    if last_move < 0 or move_count < 7:
        return False
    if current_player == -1:
        return _has_won(bb_p1)
    return _has_won(bb_p2)


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _terminal_solver_state(
    bb_p1: cython.ulonglong,
    bb_p2: cython.ulonglong,
    current_player: cython.schar,
    last_move: cython.schar,
    move_count: cython.uchar,
) -> cython.schar:
    """Exact solver state if board is immediately terminal, else SOLVER_ONGOING (0)."""
    if _is_win_state(bb_p1, bb_p2, current_player, last_move, move_count):
        return -1  # SOLVER_LOSS: player who just moved won, so current_player lost
    if move_count >= 42:
        return 2  # SOLVER_DRAW
    return 0  # SOLVER_ONGOING


@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.exceptval(check=False)
@cython.nogil
def _encode_board_u8_to_slice(
    bb_p1: cython.ulonglong,
    bb_p2: cython.ulonglong,
    heights_packed: cython.uint,
    current_player: cython.schar,
    out_flat: cython.uchar[::1],
    base_offset: cython.int,
) -> cython.void:
    """Encode board as compact uint8 [3, 6, 7] (126 bytes) directly into `out_flat`."""
    bb_cur: cython.ulonglong
    bb_opp: cython.ulonglong
    if current_player == 1:
        bb_cur = bb_p1
        bb_opp = bb_p2
    else:
        bb_cur = bb_p2
        bb_opp = bb_p1

    k: cython.int
    p2_val: cython.uchar = 1 if current_player == 1 else 0
    p_dst: cython.p_uchar
    if cython.compiled:
        p_dst = cython.address(out_flat[base_offset])
        string.memset(p_dst, 0, 84)
        string.memset(p_dst + 84, p2_val, 42)
    else:
        for k in range(84):
            out_flat[base_offset + k] = 0
        for k in range(84, 126):
            out_flat[base_offset + k] = p2_val

    c: cython.int
    r: cython.int
    h: cython.int
    flat_idx: cython.int
    cur_col: cython.ulonglong
    opp_col: cython.ulonglong

    for c in range(7):
        h = cython.cast(cython.int, (heights_packed >> (c * 4)) & 0xF)
        if h > 0:
            cur_col = bb_cur >> (c * 7)
            opp_col = bb_opp >> (c * 7)
            for r in range(h):
                flat_idx = r * 7 + c
                out_flat[base_offset + flat_idx] = cython.cast(
                    cython.uchar, (cur_col >> r) & 1
                )
                out_flat[base_offset + 42 + flat_idx] = cython.cast(
                    cython.uchar, (opp_col >> r) & 1
                )


@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.exceptval(check=False)
@cython.nogil
def _encode_board_u8_to_history(
    bb_p1: cython.ulonglong,
    bb_p2: cython.ulonglong,
    heights_packed: cython.uint,
    current_player: cython.schar,
    entry: cython.pointer[HistoryEntry],
) -> cython.void:
    """Encode board as compact uint8 [126] into a slot HistoryEntry."""
    bb_cur: cython.ulonglong
    bb_opp: cython.ulonglong
    if current_player == 1:
        bb_cur = bb_p1
        bb_opp = bb_p2
    else:
        bb_cur = bb_p2
        bb_opp = bb_p1

    k: cython.int
    p2_val: cython.uchar = 1 if current_player == 1 else 0
    if cython.compiled:
        string.memset(entry[0].state, 0, 84)
        string.memset(entry[0].state + 84, p2_val, 42)
    else:
        for k in range(84):
            entry[0].state[k] = 0
        for k in range(84, 126):
            entry[0].state[k] = p2_val

    c: cython.int
    r: cython.int
    h: cython.int
    flat_idx: cython.int
    cur_col: cython.ulonglong
    opp_col: cython.ulonglong

    for c in range(7):
        h = cython.cast(cython.int, (heights_packed >> (c * 4)) & 0xF)
        if h > 0:
            cur_col = bb_cur >> (c * 7)
            opp_col = bb_opp >> (c * 7)
            for r in range(h):
                flat_idx = r * 7 + c
                entry[0].state[flat_idx] = cython.cast(
                    cython.uchar, (cur_col >> r) & 1
                )
                entry[0].state[42 + flat_idx] = cython.cast(
                    cython.uchar, (opp_col >> r) & 1
                )


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _slot_make_move(slot: cython.pointer[SelfPlaySlot], col: cython.int) -> cython.void:
    """Apply `col` in-place to `slot[0].game_*`."""
    row: cython.int = _col_height(slot[0].game_heights_packed, col)
    one64: cython.ulonglong = 1
    one32: cython.uint = 1
    bit: cython.ulonglong = one64 << (col * 7 + row)
    if slot[0].game_current_player == 1:
        slot[0].game_bb_p1 = slot[0].game_bb_p1 | bit
    else:
        slot[0].game_bb_p2 = slot[0].game_bb_p2 | bit
    slot[0].game_heights_packed = slot[0].game_heights_packed + (one32 << (col * 4))
    slot[0].game_last_move = cython.cast(cython.schar, col)
    slot[0].game_move_count = cython.cast(cython.uchar, slot[0].game_move_count + 1)
    slot[0].game_current_player = cython.cast(
        cython.schar, -slot[0].game_current_player
    )


@cython.cfunc
@cython.cdivision(True)
@cython.exceptval(check=False)
@cython.nogil
def _sample_opening_into_slot(
    slot: cython.pointer[SelfPlaySlot], random_opening_plies: cython.int
) -> cython.void:
    """Sample stochastic non-terminal opening board:
    50% 0 plies, 25% 1 ply, 25% uniform 2..=K (with terminal rollback).
    """
    slot[0].game_bb_p1 = 0
    slot[0].game_bb_p2 = 0
    slot[0].game_heights_packed = 0
    slot[0].game_current_player = 1
    slot[0].game_last_move = -1
    slot[0].game_move_count = 0

    max_plies: cython.int = random_opening_plies
    if max_plies > 40:
        max_plies = 40
    if max_plies <= 0:
        return

    u: cython.double = _rng_uniform_f64(slot)
    target_plies: cython.int = 0
    if max_plies == 1:
        target_plies = 0 if u < 0.50 else 1
    elif u < 0.50:
        target_plies = 0
    elif u < 0.75:
        target_plies = 1
    else:
        target_plies = 2 + _rng_range(slot, max_plies - 1)

    if target_plies == 0:
        return

    valid_cols: cython.int[7]
    if not cython.compiled:
        valid_cols = [0] * 7

    attempt: cython.int
    step: cython.int
    c: cython.int
    n_valid: cython.int
    chosen_col: cython.int
    ok: cython.bint

    prev_p1: cython.ulonglong
    prev_p2: cython.ulonglong
    prev_hp: cython.uint
    prev_cp: cython.schar
    prev_lm: cython.schar
    prev_mc: cython.uchar

    for attempt in range(32):
        slot[0].game_bb_p1 = 0
        slot[0].game_bb_p2 = 0
        slot[0].game_heights_packed = 0
        slot[0].game_current_player = 1
        slot[0].game_last_move = -1
        slot[0].game_move_count = 0
        ok = True

        for step in range(target_plies):
            n_valid = 0
            for c in range(7):
                if _col_height(slot[0].game_heights_packed, c) < 6:
                    valid_cols[n_valid] = c
                    n_valid += 1
            if n_valid == 0:
                ok = False
                break
            chosen_col = valid_cols[_rng_range(slot, n_valid)]
            _slot_make_move(slot, chosen_col)
            if (
                _terminal_solver_state(
                    slot[0].game_bb_p1,
                    slot[0].game_bb_p2,
                    slot[0].game_current_player,
                    slot[0].game_last_move,
                    slot[0].game_move_count,
                )
                != 0
            ):
                ok = False
                break
        if ok:
            return

    # Fallback: play up to target_plies, rolling back any terminal move
    slot[0].game_bb_p1 = 0
    slot[0].game_bb_p2 = 0
    slot[0].game_heights_packed = 0
    slot[0].game_current_player = 1
    slot[0].game_last_move = -1
    slot[0].game_move_count = 0

    for step in range(target_plies):
        n_valid = 0
        for c in range(7):
            if _col_height(slot[0].game_heights_packed, c) < 6:
                valid_cols[n_valid] = c
                n_valid += 1
        if n_valid == 0:
            break
        prev_p1 = slot[0].game_bb_p1
        prev_p2 = slot[0].game_bb_p2
        prev_hp = slot[0].game_heights_packed
        prev_cp = slot[0].game_current_player
        prev_lm = slot[0].game_last_move
        prev_mc = slot[0].game_move_count

        chosen_col = valid_cols[_rng_range(slot, n_valid)]
        _slot_make_move(slot, chosen_col)
        if (
            _terminal_solver_state(
                slot[0].game_bb_p1,
                slot[0].game_bb_p2,
                slot[0].game_current_player,
                slot[0].game_last_move,
                slot[0].game_move_count,
            )
            != 0
        ):
            slot[0].game_bb_p1 = prev_p1
            slot[0].game_bb_p2 = prev_p2
            slot[0].game_heights_packed = prev_hp
            slot[0].game_current_player = prev_cp
            slot[0].game_last_move = prev_lm
            slot[0].game_move_count = prev_mc
            break


# ── Flat Arena MCTS + Exact Solver + FPU + Virtual Losses ───────────────────


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _copy_node_fields(
    dst: cython.pointer[MCTSNode], src: cython.pointer[MCTSNode]
) -> cython.void:
    if cython.compiled:
        dst[0] = src[0]
    else:
        dst[0].bb_p1 = src[0].bb_p1
        dst[0].bb_p2 = src[0].bb_p2
        dst[0].total_value = src[0].total_value
        dst[0].prior = src[0].prior
        dst[0].heights_packed = src[0].heights_packed
        dst[0].parent = src[0].parent
        dst[0].first_child = src[0].first_child
        dst[0].visit_count = src[0].visit_count
        dst[0].virtual_losses = src[0].virtual_losses
        dst[0].current_player = src[0].current_player
        dst[0].last_move = src[0].last_move
        dst[0].move_count = src[0].move_count
        dst[0].num_children = src[0].num_children
        dst[0].is_expanded = src[0].is_expanded
        dst[0].solver_state = src[0].solver_state


@cython.cfunc
@cython.exceptval(check=False)
@cython.nogil
def _ensure_node_cap(
    slot: cython.pointer[SelfPlaySlot], min_cap: cython.int
) -> cython.void:
    if min_cap <= slot[0].nodes_cap:
        return
    new_cap: cython.int = slot[0].nodes_cap * 2
    if new_cap < min_cap:
        new_cap = min_cap + 64
    if cython.compiled:
        slot[0].nodes = cython.cast(
            cython.pointer[MCTSNode],
            stdlib.realloc(slot[0].nodes, new_cap * cython.sizeof(MCTSNode)),
        )
        slot[0].nodes_cap = new_cap
    else:
        slot[0].nodes._items.extend(
            [_new_py_node() for _ in range(new_cap - slot[0].nodes_cap)]
        )
        slot[0].nodes_cap = new_cap


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _reset_tree_to_slot_game(slot: cython.pointer[SelfPlaySlot]) -> cython.void:
    """Initialize `slot[0].nodes[0]` as a fresh unexpanded root from `slot[0].game_*`."""
    nodes: cython.pointer[MCTSNode] = slot[0].nodes
    nodes[0].bb_p1 = slot[0].game_bb_p1
    nodes[0].bb_p2 = slot[0].game_bb_p2
    nodes[0].total_value = 0.0
    nodes[0].prior = 0.0
    nodes[0].heights_packed = slot[0].game_heights_packed
    nodes[0].parent = -1
    nodes[0].first_child = -1
    nodes[0].visit_count = 0
    nodes[0].virtual_losses = 0
    nodes[0].current_player = slot[0].game_current_player
    nodes[0].last_move = slot[0].game_last_move
    nodes[0].move_count = slot[0].game_move_count
    nodes[0].num_children = 0
    nodes[0].is_expanded = 0
    nodes[0].solver_state = _terminal_solver_state(
        slot[0].game_bb_p1,
        slot[0].game_bb_p2,
        slot[0].game_current_player,
        slot[0].game_last_move,
        slot[0].game_move_count,
    )
    slot[0].nodes_len = 1


@cython.cfunc
@cython.inline
@cython.cdivision(True)
@cython.exceptval(check=False)
@cython.nogil
def _best_child(
    nodes: cython.pointer[MCTSNode],
    parent_idx: cython.int,
    c_puct: cython.double,
    c_fpu: cython.double,
    is_noisy_root: cython.bint,
) -> cython.int:
    """Select child maximizing PUCT + parent-relative FPU + exact MCTS-Solver priority."""
    fc: cython.int = nodes[parent_idx].first_child
    nc: cython.int = cython.cast(cython.int, nodes[parent_idx].num_children)

    all_proven_win: cython.bint = True
    visited_policy: cython.double = 0.0
    i: cython.int
    c_idx: cython.int
    cs: cython.schar

    # Single scan: immediate win check + all-losing check + visited policy sum
    for i in range(nc):
        c_idx = fc + i
        cs = nodes[c_idx].solver_state
        if cs == -1:  # SOLVER_LOSS for opponent => proven win for parent
            return c_idx
        if cs != 1:  # not SOLVER_WIN
            all_proven_win = False
        if nodes[c_idx].visit_count + nodes[c_idx].virtual_losses > 0:
            visited_policy += cython.cast(cython.double, nodes[c_idx].prior)

    fpu_neg_q: cython.double = 0.0
    parent_q: cython.double
    p_visits: cython.int = nodes[parent_idx].visit_count
    if (not is_noisy_root) and p_visits > 0:
        parent_q = nodes[parent_idx].total_value / cython.cast(cython.double, p_visits)
        if visited_policy < 0.0:
            visited_policy = 0.0
        fpu_neg_q = parent_q - c_fpu * cmath.sqrt(visited_policy)
        if fpu_neg_q < -1.0:
            fpu_neg_q = -1.0
        elif fpu_neg_q > 1.0:
            fpu_neg_q = 1.0

    eff_parent: cython.int = p_visits + nodes[parent_idx].virtual_losses
    if eff_parent < 1:
        eff_parent = 1
    c_puct_sqrt: cython.double = c_puct * cmath.sqrt(
        cython.cast(cython.double, eff_parent)
    )

    best_idx: cython.int = fc
    best_score: cython.double = -1.0e300
    v_cnt: cython.int
    v_loss: cython.int
    eff_visits: cython.int
    neg_q: cython.double
    score: cython.double

    for i in range(nc):
        c_idx = fc + i
        if (not all_proven_win) and nodes[c_idx].solver_state == 1:
            continue

        v_cnt = nodes[c_idx].visit_count
        v_loss = nodes[c_idx].virtual_losses
        eff_visits = v_cnt + v_loss
        if eff_visits == 0:
            neg_q = fpu_neg_q
        else:
            neg_q = -(
                nodes[c_idx].total_value + cython.cast(cython.double, v_loss)
            ) / cython.cast(cython.double, eff_visits)

        score = neg_q + (
            c_puct_sqrt
            * cython.cast(cython.double, nodes[c_idx].prior)
            / (1.0 + cython.cast(cython.double, eff_visits))
        )
        if score > best_score:
            best_score = score
            best_idx = c_idx

    return best_idx


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _select_leaf(
    nodes: cython.pointer[MCTSNode],
    root: cython.int,
    c_puct: cython.double,
    c_fpu: cython.double,
    is_noisy_root: cython.bint,
) -> cython.int:
    idx: cython.int = root
    at_noisy_root: cython.bint
    while True:
        if (
            nodes[idx].solver_state != 0
            or nodes[idx].is_expanded == 0
            or nodes[idx].num_children == 0
        ):
            return idx
        at_noisy_root = is_noisy_root and (idx == root)
        idx = _best_child(nodes, idx, c_puct, c_fpu, at_noisy_root)


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _propagate_solver_up(
    nodes: cython.pointer[MCTSNode], start_idx: cython.int
) -> cython.void:
    p_idx: cython.int = start_idx
    fc: cython.int
    nc: cython.int
    i: cython.int
    c_idx: cython.int
    cs: cython.schar
    any_child_loss: cython.bint
    all_children_win: cython.bint
    all_children_proven: cython.bint
    new_state: cython.schar

    while p_idx >= 0:
        if nodes[p_idx].solver_state != 0:
            p_idx = nodes[p_idx].parent
            continue
        nc = cython.cast(cython.int, nodes[p_idx].num_children)
        if nodes[p_idx].is_expanded == 0 or nc == 0:
            break

        fc = nodes[p_idx].first_child
        any_child_loss = False
        all_children_win = True
        all_children_proven = True

        for i in range(nc):
            c_idx = fc + i
            cs = nodes[c_idx].solver_state
            if cs == -1:  # SOLVER_LOSS
                any_child_loss = True
                all_children_win = False
                break
            elif cs == 1:  # SOLVER_WIN
                pass
            elif cs == 2:  # SOLVER_DRAW
                all_children_win = False
            else:  # SOLVER_ONGOING
                all_children_win = False
                all_children_proven = False

        if any_child_loss:
            new_state = 1  # SOLVER_WIN
        elif all_children_win:
            new_state = -1  # SOLVER_LOSS
        elif all_children_proven:
            new_state = 2  # SOLVER_DRAW
        else:
            break

        nodes[p_idx].solver_state = new_state
        p_idx = nodes[p_idx].parent


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _apply_virtual_loss(
    nodes: cython.pointer[MCTSNode], node_idx: cython.int
) -> cython.void:
    idx: cython.int = node_idx
    while idx >= 0:
        nodes[idx].virtual_losses += 1
        idx = nodes[idx].parent


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _revert_virtual_loss(
    nodes: cython.pointer[MCTSNode], node_idx: cython.int
) -> cython.void:
    idx: cython.int = node_idx
    while idx >= 0:
        if nodes[idx].virtual_losses > 0:
            nodes[idx].virtual_losses -= 1
        idx = nodes[idx].parent


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _backpropagate(
    nodes: cython.pointer[MCTSNode], node_idx: cython.int, value: cython.double
) -> cython.void:
    idx: cython.int = node_idx
    val: cython.double = value
    while idx >= 0:
        nodes[idx].visit_count += 1
        nodes[idx].total_value += val
        val = -val
        idx = nodes[idx].parent


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _proven_value_or_nan(
    solver_state: cython.schar, has_proven: cython.pointer[cython.bint]
) -> cython.double:
    if solver_state == 1:
        has_proven[0] = True
        return 1.0
    if solver_state == -1:
        has_proven[0] = True
        return -1.0
    if solver_state == 2:
        has_proven[0] = True
        return 0.0
    has_proven[0] = False
    return 0.0


@cython.cfunc
@cython.exceptval(check=False)
@cython.nogil
def _select_multiple_leaves(
    slot: cython.pointer[SelfPlaySlot],
    root: cython.int,
    c_puct: cython.double,
    c_fpu: cython.double,
    is_noisy_root: cython.bint,
    max_k: cython.int,
) -> cython.int:
    """Select up to `max_k` unique leaves into `slot[0].pending_nodes`.
    Returns `terminal_count` (leaves immediately resolved via solver/terminal).
    """
    nodes: cython.pointer[MCTSNode] = slot[0].nodes
    terminal_count: cython.int = 0
    n_pending: cython.int = 0
    step: cython.int
    leaf: cython.int
    ss: cython.schar
    term_ss: cython.schar
    val: cython.double

    for step in range(max_k):
        leaf = _select_leaf(nodes, root, c_puct, c_fpu, is_noisy_root)
        ss = nodes[leaf].solver_state
        if ss != 0:
            if ss == 1:
                val = 1.0
            elif ss == -1:
                val = -1.0
            else:
                val = 0.0
            _backpropagate(nodes, leaf, val)
            terminal_count += 1
            if nodes[root].solver_state != 0:
                break
        else:
            term_ss = _terminal_solver_state(
                nodes[leaf].bb_p1,
                nodes[leaf].bb_p2,
                nodes[leaf].current_player,
                nodes[leaf].last_move,
                nodes[leaf].move_count,
            )
            if term_ss != 0:
                nodes[leaf].solver_state = term_ss
                val = -1.0 if term_ss == -1 else 0.0
                _propagate_solver_up(nodes, nodes[leaf].parent)
                _backpropagate(nodes, leaf, val)
                terminal_count += 1
                if nodes[root].solver_state != 0:
                    break
            else:
                if nodes[leaf].virtual_losses > 0:
                    break
                _apply_virtual_loss(nodes, leaf)
                slot[0].pending_nodes[n_pending] = leaf
                n_pending += 1

    slot[0].n_pending = n_pending
    return terminal_count


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.exceptval(check=False)
@cython.nogil
def _expand_node(
    slot: cython.pointer[SelfPlaySlot],
    node_idx: cython.int,
    evals_flat: cython.float[::1],
    eval_base: cython.int,
) -> cython.void:
    """Expand `node_idx` in `slot` using `evals_flat[eval_base : eval_base + 7]`."""
    if slot[0].nodes[node_idx].is_expanded != 0:
        return

    # Reserve capacity up front before reading node fields so `nodes` cannot reallocate mid-loop
    _ensure_node_cap(slot, slot[0].nodes_len + 7)
    nodes: cython.pointer[MCTSNode] = slot[0].nodes

    p_bb1: cython.ulonglong = nodes[node_idx].bb_p1
    p_bb2: cython.ulonglong = nodes[node_idx].bb_p2
    p_hp: cython.uint = nodes[node_idx].heights_packed
    p_cp: cython.schar = nodes[node_idx].current_player
    p_lm: cython.schar = nodes[node_idx].last_move
    p_mc: cython.uchar = nodes[node_idx].move_count

    term_ss: cython.schar = _terminal_solver_state(p_bb1, p_bb2, p_cp, p_lm, p_mc)
    if term_ss != 0:
        nodes[node_idx].solver_state = term_ss
        nodes[node_idx].is_expanded = 1
        _propagate_solver_up(nodes, nodes[node_idx].parent)
        return

    fc: cython.int = slot[0].nodes_len
    nc: cython.int = 0
    col: cython.int
    row: cython.int
    child_idx: cython.int
    one64: cython.ulonglong = 1
    one32: cython.uint = 1
    bit: cython.ulonglong
    c_bb1: cython.ulonglong
    c_bb2: cython.ulonglong
    c_hp: cython.uint
    c_cp: cython.schar = cython.cast(cython.schar, -p_cp)
    c_mc: cython.uchar = cython.cast(cython.uchar, p_mc + 1)

    for col in range(7):
        row = cython.cast(cython.int, (p_hp >> (col * 4)) & 0xF)
        if row < 6:
            bit = one64 << (col * 7 + row)
            if p_cp == 1:
                c_bb1 = p_bb1 | bit
                c_bb2 = p_bb2
            else:
                c_bb1 = p_bb1
                c_bb2 = p_bb2 | bit
            c_hp = p_hp + (one32 << (col * 4))

            child_idx = fc + nc
            nodes[child_idx].bb_p1 = c_bb1
            nodes[child_idx].bb_p2 = c_bb2
            nodes[child_idx].total_value = 0.0
            nodes[child_idx].prior = evals_flat[eval_base + col]
            nodes[child_idx].heights_packed = c_hp
            nodes[child_idx].parent = node_idx
            nodes[child_idx].first_child = -1
            nodes[child_idx].visit_count = 0
            nodes[child_idx].virtual_losses = 0
            nodes[child_idx].current_player = c_cp
            nodes[child_idx].last_move = cython.cast(cython.schar, col)
            nodes[child_idx].move_count = c_mc
            nodes[child_idx].num_children = 0
            nodes[child_idx].is_expanded = 0
            nodes[child_idx].solver_state = _terminal_solver_state(
                c_bb1, c_bb2, c_cp, cython.cast(cython.schar, col), c_mc
            )
            nc += 1

    slot[0].nodes_len = fc + nc
    nodes[node_idx].first_child = fc if nc > 0 else -1
    nodes[node_idx].num_children = cython.cast(cython.uchar, nc)
    nodes[node_idx].is_expanded = 1

    _propagate_solver_up(nodes, node_idx)


@cython.cfunc
@cython.cdivision(True)
@cython.exceptval(check=False)
@cython.nogil
def _add_dirichlet_noise(
    slot: cython.pointer[SelfPlaySlot],
    node_idx: cython.int,
    alpha: cython.double,
    epsilon: cython.double,
) -> cython.void:
    nodes: cython.pointer[MCTSNode] = slot[0].nodes
    nc: cython.int = cython.cast(cython.int, nodes[node_idx].num_children)
    if nc < 2 or alpha <= 0.0 or epsilon <= 0.0:
        return

    raw: cython.double[7]
    if not cython.compiled:
        raw = [0.0] * 7

    total: cython.double = 0.0
    g: cython.double
    i: cython.int
    fc: cython.int = nodes[node_idx].first_child
    c_idx: cython.int
    prior: cython.double

    for i in range(nc):
        g = _rng_gamma(slot, alpha)
        raw[i] = g
        total += g

    if total > 0.0:
        for i in range(nc):
            c_idx = fc + i
            prior = cython.cast(cython.double, nodes[c_idx].prior)
            nodes[c_idx].prior = cython.cast(
                cython.float,
                (1.0 - epsilon) * prior + epsilon * (raw[i] / total),
            )


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _find_child_action(
    nodes: cython.pointer[MCTSNode], node_idx: cython.int, action: cython.int
) -> cython.int:
    fc: cython.int = nodes[node_idx].first_child
    nc: cython.int = cython.cast(cython.int, nodes[node_idx].num_children)
    i: cython.int
    c_idx: cython.int
    for i in range(nc):
        c_idx = fc + i
        if cython.cast(cython.int, nodes[c_idx].last_move) == action:
            return c_idx
    return -1


@cython.cfunc
@cython.exceptval(check=False)
@cython.nogil
def _rebase_tree(
    slot: cython.pointer[SelfPlaySlot], child_idx: cython.int
) -> cython.bint:
    """Compact live subtree under `child_idx` into `scratch_nodes` and swap pointers."""
    if child_idx < 0 or child_idx >= slot[0].nodes_len:
        return False

    needed_cap: cython.int = slot[0].nodes_cap
    if slot[0].scratch_cap < needed_cap:
        if cython.compiled:
            slot[0].scratch_nodes = cython.cast(
                cython.pointer[MCTSNode],
                stdlib.realloc(
                    slot[0].scratch_nodes, needed_cap * cython.sizeof(MCTSNode)
                ),
            )
            slot[0].scratch_cap = needed_cap
        else:
            slot[0].scratch_nodes._items.extend(
                [_new_py_node() for _ in range(needed_cap - slot[0].scratch_cap)]
            )
            slot[0].scratch_cap = needed_cap

    needed_q_cap: cython.int = 2 * needed_cap
    if slot[0].rebase_queue_cap < needed_q_cap:
        if cython.compiled:
            slot[0].rebase_queue = cython.cast(
                cython.pointer[cython.int],
                stdlib.realloc(
                    slot[0].rebase_queue, needed_q_cap * cython.sizeof(cython.int)
                ),
            )
            slot[0].rebase_queue_cap = needed_q_cap
        else:
            slot[0].rebase_queue._items.extend(
                [0] * (needed_q_cap - slot[0].rebase_queue_cap)
            )
            slot[0].rebase_queue_cap = needed_q_cap

    q: cython.pointer[cython.int] = slot[0].rebase_queue
    src_nodes: cython.pointer[MCTSNode] = slot[0].nodes
    dst_nodes: cython.pointer[MCTSNode] = slot[0].scratch_nodes

    q[0] = child_idx
    q[1] = -1
    q_len: cython.int = 1
    head: cython.int = 0
    old_idx: cython.int
    parent_new: cython.int
    nc: cython.int
    old_fc: cython.int
    new_fc: cython.int
    i: cython.int

    while head < q_len:
        old_idx = q[2 * head]
        parent_new = q[2 * head + 1]
        nc = cython.cast(cython.int, src_nodes[old_idx].num_children)
        old_fc = src_nodes[old_idx].first_child
        if nc > 0:
            new_fc = q_len
            for i in range(nc):
                q[2 * q_len] = old_fc + i
                q[2 * q_len + 1] = head
                q_len += 1
        else:
            new_fc = -1

        _copy_node_fields(
            cython.address(dst_nodes[head]), cython.address(src_nodes[old_idx])
        )
        dst_nodes[head].parent = parent_new
        dst_nodes[head].first_child = new_fc
        dst_nodes[head].virtual_losses = 0
        head += 1

    tmp_nodes: cython.pointer[MCTSNode] = slot[0].nodes
    slot[0].nodes = slot[0].scratch_nodes
    slot[0].scratch_nodes = tmp_nodes

    tmp_cap: cython.int = slot[0].nodes_cap
    slot[0].nodes_cap = slot[0].scratch_cap
    slot[0].scratch_cap = tmp_cap

    slot[0].nodes_len = q_len
    return True


# ── Policy target extraction, action sampling & continuous batching ───────────


@cython.cfunc
@cython.cdivision(True)
@cython.exceptval(check=False)
@cython.nogil
def _compute_root_policy_target(
    slot: cython.pointer[SelfPlaySlot], out_pol: cython.pointer[cython.float]
) -> cython.void:
    """Compute normalized decoupled policy target for `slot`'s current root (index 0)."""
    nodes: cython.pointer[MCTSNode] = slot[0].nodes
    root_solver: cython.schar = nodes[0].solver_state
    fc: cython.int = nodes[0].first_child
    nc: cython.int = cython.cast(cython.int, nodes[0].num_children)

    c: cython.int
    i: cython.int
    c_idx: cython.int
    col: cython.int
    for c in range(7):
        out_pol[c] = 0.0

    win_cols: cython.int[7]
    counts: cython.float[7]
    if not cython.compiled:
        win_cols = [0] * 7
        counts = [0.0] * 7

    n_win: cython.int = 0
    inv: cython.float
    has_safe_visits: cython.bint = False
    total: cython.float = 0.0
    n_valid: cython.int = 0

    if root_solver == 1:  # SOLVER_WIN
        for i in range(nc):
            c_idx = fc + i
            if nodes[c_idx].solver_state == -1:  # SOLVER_LOSS
                win_cols[n_win] = cython.cast(cython.int, nodes[c_idx].last_move)
                n_win += 1
        inv = 1.0 / cython.cast(cython.float, n_win if n_win > 0 else 1)
        for i in range(n_win):
            out_pol[win_cols[i]] = inv
        return

    for c in range(7):
        counts[c] = 0.0
    for i in range(nc):
        c_idx = fc + i
        col = cython.cast(cython.int, nodes[c_idx].last_move)
        counts[col] = cython.cast(cython.float, nodes[c_idx].visit_count)

    if root_solver == 2:  # SOLVER_DRAW
        for i in range(nc):
            c_idx = fc + i
            if nodes[c_idx].solver_state == 1:  # SOLVER_WIN for opponent
                counts[cython.cast(cython.int, nodes[c_idx].last_move)] = 0.0
    elif root_solver == 0:  # SOLVER_ONGOING
        for i in range(nc):
            c_idx = fc + i
            if nodes[c_idx].solver_state != 1 and nodes[c_idx].visit_count > 0:
                has_safe_visits = True
                break
        if has_safe_visits:
            for i in range(nc):
                c_idx = fc + i
                if nodes[c_idx].solver_state == 1:
                    counts[cython.cast(cython.int, nodes[c_idx].last_move)] = 0.0

    for c in range(7):
        total += counts[c]

    if total > 0.0:
        for c in range(7):
            out_pol[c] = counts[c] / total
    elif root_solver == 2:  # SOLVER_DRAW with 0 visits
        n_win = 0
        for i in range(nc):
            c_idx = fc + i
            if nodes[c_idx].solver_state == 2:
                win_cols[n_win] = cython.cast(cython.int, nodes[c_idx].last_move)
                n_win += 1
        inv = 1.0 / cython.cast(cython.float, n_win if n_win > 0 else 1)
        for i in range(n_win):
            out_pol[win_cols[i]] = inv
    else:
        for c in range(7):
            if _col_height(slot[0].game_heights_packed, c) < 6:
                n_valid += 1
        inv = 1.0 / cython.cast(cython.float, n_valid if n_valid > 0 else 1)
        for c in range(7):
            if _col_height(slot[0].game_heights_packed, c) < 6:
                out_pol[c] = inv


@cython.cfunc
@cython.inline
@cython.cdivision(True)
@cython.exceptval(check=False)
@cython.nogil
def _sample_action_from_policy(
    pol: cython.pointer[cython.float],
    move_count: cython.int,
    temp_threshold: cython.int,
    slot: cython.pointer[SelfPlaySlot],
) -> cython.int:
    c: cython.int
    best_col: cython.int = 0
    best_val: cython.float = -1.0e30
    p: cython.float

    if move_count >= temp_threshold:
        for c in range(7):
            p = pol[c]
            if p > best_val:
                best_val = p
                best_col = c
        return best_col

    r: cython.float = cython.cast(cython.float, _rng_uniform_f64(slot))
    cumsum: cython.float = 0.0
    last_nonzero: cython.int = 0
    for c in range(7):
        p = pol[c]
        if p > 0.0:
            last_nonzero = c
            cumsum += p
            if r < cumsum:
                return c
    return last_nonzero


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.exceptval(check=False)
@cython.nogil
def _emit_mirrored_game_u8(
    slot: cython.pointer[SelfPlaySlot],
    result: cython.float,
    current_player: cython.schar,
    comp_states_flat: cython.uchar[::1],
    comp_pol_flat: cython.float[::1],
    comp_val: cython.float[::1],
    start_idx: cython.int,
) -> cython.int:
    """Write `(orig, horiz_mirror)` training pairs for a finished game into output buffers."""
    hlen: cython.int = slot[0].history_len
    i: cython.int
    plane: cython.int
    r: cython.int
    c: cython.int
    row_off: cython.int
    idx_orig: cython.int
    idx_mirr: cython.int
    s_orig_base: cython.int
    s_mirr_base: cython.int
    p_orig_base: cython.int
    p_mirr_base: cython.int
    val: cython.float
    entry: cython.pointer[HistoryEntry]

    for i in range(hlen):
        entry = cython.address(slot[0].history[i])
        val = result if entry[0].player == current_player else -result

        idx_orig = start_idx + 2 * i
        idx_mirr = idx_orig + 1
        s_orig_base = idx_orig * 126
        s_mirr_base = idx_mirr * 126
        p_orig_base = idx_orig * 7
        p_mirr_base = idx_mirr * 7

        if cython.compiled:
            string.memcpy(
                cython.address(comp_states_flat[s_orig_base]),
                entry[0].state,
                126,
            )
        else:
            for c in range(126):
                comp_states_flat[s_orig_base + c] = entry[0].state[c]

        for plane in range(3):
            for r in range(6):
                row_off = plane * 42 + r * 7
                for c in range(7):
                    comp_states_flat[s_mirr_base + row_off + c] = entry[0].state[
                        row_off + (6 - c)
                    ]

        for c in range(7):
            comp_pol_flat[p_orig_base + c] = entry[0].policy[c]
            comp_pol_flat[p_mirr_base + c] = entry[0].policy[6 - c]

        comp_val[idx_orig] = val
        comp_val[idx_mirr] = val

    return hlen * 2


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
@cython.nogil
def _advance_ready_slots_c(
    slots: cython.pointer[SelfPlaySlot],
    num_games: cython.int,
    initial_games: cython.int,
    num_sims: cython.int,
    alpha: cython.double,
    eps: cython.double,
    temp_threshold: cython.int,
    active_count: cython.int,
    completed_examples: cython.int,
    comp_states_flat: cython.uchar[::1],
    comp_pol_flat: cython.float[::1],
    comp_val: cython.float[::1],
) -> AdvanceResult:
    """Advance any active slots whose root search has finished."""
    res: AdvanceResult
    if not cython.compiled:
        res = AdvanceResult(
            finished=0,
            active_count=active_count,
            completed_examples=completed_examples,
        )
    else:
        res.finished = 0
        res.active_count = active_count
        res.completed_examples = completed_examples

    if res.active_count == 0:
        return res

    is_noisy: cython.uchar = 1 if (alpha > 0.0 and eps > 0.0) else 0
    tail_sims_thresh: cython.int = (num_sims * 3) // 4

    s_idx: cython.int
    slot: cython.pointer[SelfPlaySlot]
    root_solver: cython.schar
    tail_straggler_done: cython.bint
    turn_done: cython.bint
    hlen: cython.int
    entry: cython.pointer[HistoryEntry]
    move_count: cython.int
    action: cython.int
    term_ss: cython.schar
    result: cython.float
    added: cython.int
    child_idx: cython.int

    for s_idx in range(num_games):
        slot = cython.address(slots[s_idx])
        if (
            slot[0].active == 0
            or slot[0].n_pending > 0
            or slot[0].pending_root_init != 0
        ):
            continue

        while True:
            root_solver = slot[0].nodes[0].solver_state
            if slot[0].nodes[0].is_expanded == 0 and root_solver == 0:
                break

            if slot[0].nodes[0].is_expanded != 0 and slot[0].noise_applied == 0:
                _add_dirichlet_noise(slot, 0, alpha, eps)
                slot[0].root_noisy = is_noisy
                slot[0].noise_applied = 1

            tail_straggler_done = (
                initial_games >= 64
                and res.active_count <= 4
                and slot[0].last_batch_leaves <= 2
                and slot[0].sims_done >= tail_sims_thresh
            )
            turn_done = (
                slot[0].sims_done >= num_sims
                or slot[0].nodes[0].solver_state != 0
                or tail_straggler_done
            )
            if not turn_done:
                break

            hlen = slot[0].history_len
            entry = cython.address(slot[0].history[hlen])
            _encode_board_u8_to_history(
                slot[0].game_bb_p1,
                slot[0].game_bb_p2,
                slot[0].game_heights_packed,
                slot[0].game_current_player,
                entry,
            )
            _compute_root_policy_target(slot, entry[0].policy)
            entry[0].player = slot[0].game_current_player
            slot[0].history_len = hlen + 1

            move_count = cython.cast(cython.int, slot[0].game_move_count)
            action = _sample_action_from_policy(
                entry[0].policy,
                move_count,
                temp_threshold,
                slot,
            )

            slot[0].sims_done = 0
            slot[0].root_noisy = 0
            slot[0].noise_applied = 0
            slot[0].pending_root_init = 0
            slot[0].last_batch_leaves = 16
            slot[0].n_pending = 0

            _slot_make_move(slot, action)

            term_ss = _terminal_solver_state(
                slot[0].game_bb_p1,
                slot[0].game_bb_p2,
                slot[0].game_current_player,
                slot[0].game_last_move,
                slot[0].game_move_count,
            )
            if term_ss != 0:
                slot[0].active = 0
                res.active_count -= 1
                res.finished += 1

                result = -1.0 if term_ss == -1 else 0.0
                added = _emit_mirrored_game_u8(
                    slot,
                    result,
                    slot[0].game_current_player,
                    comp_states_flat,
                    comp_pol_flat,
                    comp_val,
                    res.completed_examples,
                )
                res.completed_examples += added
                slot[0].history_len = 0
                break
            else:
                child_idx = _find_child_action(slot[0].nodes, 0, action)
                if child_idx >= 0:
                    if not _rebase_tree(slot, child_idx):
                        _reset_tree_to_slot_game(slot)
                else:
                    _reset_tree_to_slot_game(slot)

    return res


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def _select_slot_leaves_one(
    slot: cython.pointer[SelfPlaySlot],
    num_sims: cython.int,
    c_puct: cython.double,
    c_fpu: cython.double,
    alpha: cython.double,
    eps: cython.double,
    is_noisy: cython.uchar,
) -> cython.void:
    slot[0].n_pending = 0
    slot[0].pending_root_init = 0
    if slot[0].active == 0:
        return

    if slot[0].nodes[0].solver_state != 0:
        slot[0].sims_done = num_sims
        return

    if slot[0].nodes[0].is_expanded == 0:
        slot[0].pending_root_init = 1
        slot[0].pending_nodes[0] = 0
        slot[0].n_pending = 1
        return

    if slot[0].noise_applied == 0:
        _add_dirichlet_noise(slot, 0, alpha, eps)
        slot[0].root_noisy = is_noisy
        slot[0].noise_applied = 1

    if slot[0].sims_done >= num_sims:
        return

    remaining: cython.int = num_sims - slot[0].sims_done
    budget: cython.int = 16 if remaining > 16 else remaining
    term_cnt: cython.int = _select_multiple_leaves(
        slot,
        0,
        c_puct,
        c_fpu,
        slot[0].root_noisy != 0,
        budget,
    )
    slot[0].last_batch_leaves = slot[0].n_pending + term_cnt
    slot[0].sims_done += slot[0].last_batch_leaves
    if slot[0].nodes[0].solver_state != 0:
        slot[0].sims_done = num_sims


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.exceptval(check=False)
@cython.nogil
def _select_all_slots_c(
    slots: cython.pointer[SelfPlaySlot],
    num_games: cython.int,
    active_count: cython.int,
    num_sims: cython.int,
    c_puct: cython.double,
    c_fpu: cython.double,
    alpha: cython.double,
    eps: cython.double,
) -> cython.int:
    """Select leaves across all slots (parallel via OpenMP when active_count >= 8) and assign offsets."""
    is_noisy: cython.uchar = 1 if (alpha > 0.0 and eps > 0.0) else 0
    i: cython.int

    if cython.compiled and active_count >= 8:
        for i in prange(num_games, schedule="dynamic", chunksize=2):
            _select_slot_leaves_one(
                cython.address(slots[i]),
                num_sims,
                c_puct,
                c_fpu,
                alpha,
                eps,
                is_noisy,
            )
    else:
        for i in range(num_games):
            _select_slot_leaves_one(
                cython.address(slots[i]),
                num_sims,
                c_puct,
                c_fpu,
                alpha,
                eps,
                is_noisy,
            )

    total_count: cython.int = 0
    for i in range(num_games):
        slots[i].batch_offset = total_count
        total_count += slots[i].n_pending
    return total_count


@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.exceptval(check=False)
@cython.nogil
def _encode_slot_pending_one(
    slot: cython.pointer[SelfPlaySlot],
    states_flat: cython.uchar[::1],
    valid_flat: cython.uchar[::1],
) -> cython.void:
    n_pend: cython.int = slot[0].n_pending
    if n_pend <= 0:
        return
    offset: cython.int = slot[0].batch_offset
    nodes: cython.pointer[MCTSNode] = slot[0].nodes
    j: cython.int
    k: cython.int
    leaf: cython.int
    hp: cython.uint
    c: cython.int
    v_base: cython.int

    for j in range(n_pend):
        k = offset + j
        leaf = slot[0].pending_nodes[j]
        hp = nodes[leaf].heights_packed
        _encode_board_u8_to_slice(
            nodes[leaf].bb_p1,
            nodes[leaf].bb_p2,
            hp,
            nodes[leaf].current_player,
            states_flat,
            k * 126,
        )
        v_base = k * 7
        for c in range(7):
            valid_flat[v_base + c] = 1 if (((hp >> (c * 4)) & 0xF) < 6) else 0


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.exceptval(check=False)
@cython.nogil
def _encode_all_pending_c(
    slots: cython.pointer[SelfPlaySlot],
    num_games: cython.int,
    total_count: cython.int,
    states_flat: cython.uchar[::1],
    valid_flat: cython.uchar[::1],
) -> cython.void:
    i: cython.int
    if cython.compiled and total_count >= 64 and num_games >= 8:
        for i in prange(num_games, schedule="static"):
            _encode_slot_pending_one(
                cython.address(slots[i]), states_flat, valid_flat
            )
    else:
        for i in range(num_games):
            _encode_slot_pending_one(
                cython.address(slots[i]), states_flat, valid_flat
            )


@cython.cfunc
@cython.exceptval(check=False)
@cython.nogil
def _revert_all_pending_c(
    slots: cython.pointer[SelfPlaySlot], num_games: cython.int
) -> cython.void:
    i: cython.int
    j: cython.int
    slot: cython.pointer[SelfPlaySlot]
    for i in range(num_games):
        slot = cython.address(slots[i])
        if slot[0].pending_root_init == 0:
            for j in range(slot[0].n_pending):
                _revert_virtual_loss(slot[0].nodes, slot[0].pending_nodes[j])
        slot[0].pending_root_init = 0
        slot[0].n_pending = 0


@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.exceptval(check=False)
@cython.nogil
def _apply_slot_packed_one(
    slot: cython.pointer[SelfPlaySlot],
    evals_flat: cython.float[::1],
    num_sims: cython.int,
    alpha: cython.double,
    eps: cython.double,
    is_noisy: cython.uchar,
) -> cython.void:
    n_pend: cython.int = slot[0].n_pending
    if n_pend <= 0:
        return

    offset: cython.int = slot[0].batch_offset
    j: cython.int
    k: cython.int
    node_idx: cython.int
    row_base: cython.int
    ss: cython.schar
    bval: cython.double

    if slot[0].pending_root_init != 0:
        row_base = offset * 8
        _expand_node(slot, 0, evals_flat, row_base)
        _add_dirichlet_noise(slot, 0, alpha, eps)
        slot[0].root_noisy = is_noisy
        slot[0].noise_applied = 1
        slot[0].pending_root_init = 0
        slot[0].n_pending = 0
        if slot[0].nodes[0].solver_state != 0:
            slot[0].sims_done = num_sims
    else:
        for j in range(n_pend):
            k = offset + j
            row_base = k * 8
            node_idx = slot[0].pending_nodes[j]
            _revert_virtual_loss(slot[0].nodes, node_idx)
            _expand_node(slot, node_idx, evals_flat, row_base)
            ss = slot[0].nodes[node_idx].solver_state
            if ss == 1:
                bval = 1.0
            elif ss == -1:
                bval = -1.0
            elif ss == 2:
                bval = 0.0
            else:
                bval = cython.cast(cython.double, evals_flat[row_base + 7])
            _backpropagate(slot[0].nodes, node_idx, bval)

        slot[0].n_pending = 0
        if slot[0].nodes[0].solver_state != 0:
            slot[0].sims_done = num_sims


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.exceptval(check=False)
@cython.nogil
def _apply_all_packed_c(
    slots: cython.pointer[SelfPlaySlot],
    num_games: cython.int,
    active_count: cython.int,
    evals_flat: cython.float[::1],
    num_sims: cython.int,
    alpha: cython.double,
    eps: cython.double,
) -> cython.void:
    is_noisy: cython.uchar = 1 if (alpha > 0.0 and eps > 0.0) else 0
    i: cython.int
    if cython.compiled and active_count >= 8:
        for i in prange(num_games, schedule="dynamic", chunksize=2):
            _apply_slot_packed_one(
                cython.address(slots[i]),
                evals_flat,
                num_sims,
                alpha,
                eps,
                is_noisy,
            )
    else:
        for i in range(num_games):
            _apply_slot_packed_one(
                cython.address(slots[i]),
                evals_flat,
                num_sims,
                alpha,
                eps,
                is_noisy,
            )


# ── Public Extension Class: CythonBatchedSelfPlay ────────────────────────────


@cython.cclass
class CythonBatchedSelfPlay:
    """Barrier-free batched MCTS self-play engine in Cython Pure Python Mode."""

    _c_slots: cython.pointer[SelfPlaySlot]
    _py_slots: object
    _num_slots_alloc: cython.int
    _initial_games: cython.int
    _active_count: cython.int
    _pending_count: cython.int
    _completed_examples: cython.int
    _num_simulations: cython.int
    _c_puct: cython.double
    _c_fpu: cython.double
    _dirichlet_alpha: cython.double
    _dirichlet_epsilon: cython.double
    _temp_threshold: cython.int
    _comp_states: object
    _comp_policies: object
    _comp_values: object

    def __cinit__(self) -> None:
        if cython.compiled:
            self._c_slots = cython.NULL
            self._num_slots_alloc = 0

    def __init__(
        self,
        num_games: int,
        num_simulations: int,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.35,
        dirichlet_epsilon: float = 0.25,
        temp_threshold: int = 10,
        random_opening_plies: int = 0,
        c_fpu: float = 0.20,
    ) -> None:
        n_games: cython.int = max(0, int(num_games))
        n_sims: cython.int = max(0, int(num_simulations))
        r_plies: cython.int = max(0, int(random_opening_plies))

        self._initial_games = n_games
        self._active_count = n_games
        self._pending_count = 0
        self._completed_examples = 0
        self._num_simulations = n_sims
        self._c_puct = float(c_puct)
        self._c_fpu = float(c_fpu)
        self._dirichlet_alpha = float(dirichlet_alpha)
        self._dirichlet_epsilon = float(dirichlet_epsilon)
        self._temp_threshold = max(0, int(temp_threshold))

        # Exact upper bound on mirrored examples: <= 42 plies * 2 = 84 per game
        max_examples: int = max(1, n_games * 84)
        self._comp_states = np.empty((max_examples, 3, 6, 7), dtype=np.uint8)
        self._comp_policies = np.empty((max_examples, 7), dtype=np.float32)
        self._comp_values = np.empty((max_examples,), dtype=np.float32)

        if n_games == 0:
            self._py_slots = []
            return

        # Seed per-slot 64-bit SplitMix64 states from NumPy RNG (supports deterministic np.random.seed)
        seeds_arr = np.random.randint(
            1, (1 << 62) - 1, size=n_games, dtype=np.uint64
        )
        seeds_mv: cython.ulonglong[::1] = np.ascontiguousarray(
            seeds_arr, dtype=np.uint64
        )

        # Preallocate enough arena nodes per slot to avoid reallocs during standard games
        init_cap: cython.int = max(256, min(n_sims * 6 + 64, 16384))
        i: cython.int
        slot: cython.pointer[SelfPlaySlot]

        if cython.compiled:
            self._c_slots = cython.cast(
                cython.pointer[SelfPlaySlot],
                stdlib.calloc(n_games, cython.sizeof(SelfPlaySlot)),
            )
            if self._c_slots == cython.NULL:
                raise MemoryError("Failed to allocate SelfPlaySlot array")
            self._num_slots_alloc = n_games
            self._py_slots = None

            for i in range(n_games):
                slot = cython.address(self._c_slots[i])
                slot[0].nodes = cython.cast(
                    cython.pointer[MCTSNode],
                    stdlib.malloc(init_cap * cython.sizeof(MCTSNode)),
                )
                slot[0].scratch_nodes = cython.cast(
                    cython.pointer[MCTSNode],
                    stdlib.malloc(init_cap * cython.sizeof(MCTSNode)),
                )
                slot[0].rebase_queue = cython.cast(
                    cython.pointer[cython.int],
                    stdlib.malloc((2 * init_cap) * cython.sizeof(cython.int)),
                )
                if (
                    slot[0].nodes == cython.NULL
                    or slot[0].scratch_nodes == cython.NULL
                    or slot[0].rebase_queue == cython.NULL
                ):
                    raise MemoryError("Failed to allocate MCTSNode arena")

                slot[0].nodes_cap = init_cap
                slot[0].scratch_cap = init_cap
                slot[0].rebase_queue_cap = 2 * init_cap
                slot[0].rng_state = seeds_mv[i]
                slot[0].active = 1
                slot[0].last_batch_leaves = 16
                _sample_opening_into_slot(slot, r_plies)
                _reset_tree_to_slot_game(slot)
        else:
            py_init_cap: int = max(16, min(int(n_sims) * 4 + 16, 128))
            py_slots = [
                _new_py_slot(py_init_cap, int(seeds_arr[idx]))
                for idx in range(n_games)
            ]
            for i in range(n_games):
                slot = cython.address(py_slots[i])
                _sample_opening_into_slot(slot, r_plies)
                _reset_tree_to_slot_game(slot)
            self._py_slots = py_slots

    def __dealloc__(self) -> None:
        i: cython.int
        if cython.compiled:
            if self._c_slots != cython.NULL:
                for i in range(self._num_slots_alloc):
                    if self._c_slots[i].nodes != cython.NULL:
                        stdlib.free(self._c_slots[i].nodes)
                        self._c_slots[i].nodes = cython.NULL
                    if self._c_slots[i].scratch_nodes != cython.NULL:
                        stdlib.free(self._c_slots[i].scratch_nodes)
                        self._c_slots[i].scratch_nodes = cython.NULL
                    if self._c_slots[i].rebase_queue != cython.NULL:
                        stdlib.free(self._c_slots[i].rebase_queue)
                        self._c_slots[i].rebase_queue = cython.NULL
                stdlib.free(self._c_slots)
                self._c_slots = cython.NULL
                self._num_slots_alloc = 0

    def is_done(self) -> bool:
        """Return True if all self-play games have reached a terminal state."""
        return self._active_count == 0

    def num_active(self) -> int:
        """Return the number of currently active (unfinished) games."""
        return int(self._active_count)

    def num_pending(self) -> int:
        """Return the number of leaves currently awaiting neural network evaluation."""
        return int(self._pending_count)

    def _ensure_comp_capacity(self) -> None:
        """Ensure output buffers can hold all remaining active games' mirrored examples."""
        needed: int = int(self._completed_examples) + int(self._active_count) * 84
        if needed <= int(self._comp_values.shape[0]):
            return
        new_cap: int = max(needed, int(self._comp_values.shape[0]) * 2)
        new_s = np.empty((new_cap, 3, 6, 7), dtype=np.uint8)
        new_p = np.empty((new_cap, 7), dtype=np.float32)
        new_v = np.empty((new_cap,), dtype=np.float32)
        cur: int = int(self._completed_examples)
        if cur > 0:
            new_s[:cur] = self._comp_states[:cur]
            new_p[:cur] = self._comp_policies[:cur]
            new_v[:cur] = self._comp_values[:cur]
        self._comp_states = new_s
        self._comp_policies = new_p
        self._comp_values = new_v

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def collect_continuous_u8(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Collect next barrier-free batch of leaves & unexpanded roots as `uint8[K,3,6,7]`.

        Returns:
            `(states_u8[K,3,6,7], valid_bool[K,7], count, newly_finished_games)`.
        """
        if self._active_count == 0:
            self._pending_count = 0
            return (
                np.empty((0, 3, 6, 7), dtype=np.uint8),
                np.empty((0, 7), dtype=np.bool_),
                0,
                0,
            )

        self._ensure_comp_capacity()
        comp_s_flat: cython.uchar[::1] = self._comp_states.reshape(-1)
        comp_p_flat: cython.float[::1] = self._comp_policies.reshape(-1)
        comp_v_mv: cython.float[::1] = self._comp_values

        n_games: cython.int = self._initial_games
        num_sims: cython.int = self._num_simulations
        c_puct: cython.double = self._c_puct
        c_fpu: cython.double = self._c_fpu
        alpha: cython.double = self._dirichlet_alpha
        eps: cython.double = self._dirichlet_epsilon
        temp_thresh: cython.int = self._temp_threshold

        active_local: cython.int = self._active_count
        comp_local: cython.int = self._completed_examples
        total_finished: cython.int = 0
        total_count: cython.int = 0
        guard_limit: cython.int = num_sims + 64
        guard: cython.int
        adv_res: AdvanceResult

        if self._pending_count > 0:
            if cython.compiled:
                with cython.nogil:
                    _revert_all_pending_c(self._c_slots, n_games)
            else:
                _revert_all_pending_c(self._py_slots, n_games)
            self._pending_count = 0

        if cython.compiled:
            with cython.nogil:
                for guard in range(guard_limit):
                    adv_res = _advance_ready_slots_c(
                        self._c_slots,
                        n_games,
                        n_games,
                        num_sims,
                        alpha,
                        eps,
                        temp_thresh,
                        active_local,
                        comp_local,
                        comp_s_flat,
                        comp_p_flat,
                        comp_v_mv,
                    )
                    total_finished += adv_res.finished
                    active_local = adv_res.active_count
                    comp_local = adv_res.completed_examples
                    if active_local == 0:
                        total_count = 0
                        break

                    total_count = _select_all_slots_c(
                        self._c_slots,
                        n_games,
                        active_local,
                        num_sims,
                        c_puct,
                        c_fpu,
                        alpha,
                        eps,
                    )
                    if total_count > 0:
                        break
        else:
            for guard in range(guard_limit):
                adv_res = _advance_ready_slots_c(
                    self._py_slots,
                    n_games,
                    n_games,
                    num_sims,
                    alpha,
                    eps,
                    temp_thresh,
                    active_local,
                    comp_local,
                    comp_s_flat,
                    comp_p_flat,
                    comp_v_mv,
                )
                total_finished += adv_res.finished
                active_local = adv_res.active_count
                comp_local = adv_res.completed_examples
                if active_local == 0:
                    total_count = 0
                    break

                total_count = _select_all_slots_c(
                    self._py_slots,
                    n_games,
                    active_local,
                    num_sims,
                    c_puct,
                    c_fpu,
                    alpha,
                    eps,
                )
                if total_count > 0:
                    break

        self._active_count = active_local
        self._completed_examples = comp_local

        if total_count == 0:
            self._pending_count = 0
            return (
                np.empty((0, 3, 6, 7), dtype=np.uint8),
                np.empty((0, 7), dtype=np.bool_),
                0,
                int(total_finished),
            )

        states_arr = np.empty((total_count, 3, 6, 7), dtype=np.uint8)
        valid_u8 = np.empty((total_count, 7), dtype=np.uint8)
        states_flat: cython.uchar[::1] = states_arr.reshape(-1)
        valid_flat: cython.uchar[::1] = valid_u8.reshape(-1)

        if cython.compiled:
            with cython.nogil:
                _encode_all_pending_c(
                    self._c_slots, n_games, total_count, states_flat, valid_flat
                )
        else:
            _encode_all_pending_c(
                self._py_slots, n_games, total_count, states_flat, valid_flat
            )

        self._pending_count = total_count
        return (
            states_arr,
            valid_u8.view(np.bool_),
            int(total_count),
            int(total_finished),
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def apply_continuous_packed(self, evals_packed: np.ndarray) -> int:
        """Apply packed `float32[K, 8]` evaluations (`[policy_0..6, value]`) and advance ready slots."""
        if self._active_count == 0 and self._pending_count == 0:
            return 0

        self._ensure_comp_capacity()
        comp_s_flat: cython.uchar[::1] = self._comp_states.reshape(-1)
        comp_p_flat: cython.float[::1] = self._comp_policies.reshape(-1)
        comp_v_mv: cython.float[::1] = self._comp_values

        n_games: cython.int = self._initial_games
        num_sims: cython.int = self._num_simulations
        alpha: cython.double = self._dirichlet_alpha
        eps: cython.double = self._dirichlet_epsilon
        temp_thresh: cython.int = self._temp_threshold
        active_local: cython.int = self._active_count
        comp_local: cython.int = self._completed_examples
        adv_res: AdvanceResult

        if self._pending_count == 0:
            if cython.compiled:
                with cython.nogil:
                    adv_res = _advance_ready_slots_c(
                        self._c_slots,
                        n_games,
                        n_games,
                        num_sims,
                        alpha,
                        eps,
                        temp_thresh,
                        active_local,
                        comp_local,
                        comp_s_flat,
                        comp_p_flat,
                        comp_v_mv,
                    )
            else:
                adv_res = _advance_ready_slots_c(
                    self._py_slots,
                    n_games,
                    n_games,
                    num_sims,
                    alpha,
                    eps,
                    temp_thresh,
                    active_local,
                    comp_local,
                    comp_s_flat,
                    comp_p_flat,
                    comp_v_mv,
                )
            self._active_count = adv_res.active_count
            self._completed_examples = adv_res.completed_examples
            return int(adv_res.finished)

        if (
            not isinstance(evals_packed, np.ndarray)
            or evals_packed.dtype != np.float32
            or not evals_packed.flags["C_CONTIGUOUS"]
        ):
            evals_packed = np.ascontiguousarray(evals_packed, dtype=np.float32)

        flat_arr = evals_packed.reshape(-1)
        expected_len: int = int(self._pending_count) * 8
        if int(flat_arr.shape[0]) < expected_len:
            raise ValueError(
                f"Packed evaluation shape mismatch: pending={self._pending_count}, "
                f"expected_len={expected_len}, got={flat_arr.shape[0]}"
            )

        evals_flat: cython.float[::1] = flat_arr

        if cython.compiled:
            with cython.nogil:
                _apply_all_packed_c(
                    self._c_slots,
                    n_games,
                    active_local,
                    evals_flat,
                    num_sims,
                    alpha,
                    eps,
                )
                adv_res = _advance_ready_slots_c(
                    self._c_slots,
                    n_games,
                    n_games,
                    num_sims,
                    alpha,
                    eps,
                    temp_thresh,
                    active_local,
                    comp_local,
                    comp_s_flat,
                    comp_p_flat,
                    comp_v_mv,
                )
        else:
            _apply_all_packed_c(
                self._py_slots,
                n_games,
                active_local,
                evals_flat,
                num_sims,
                alpha,
                eps,
            )
            adv_res = _advance_ready_slots_c(
                self._py_slots,
                n_games,
                n_games,
                num_sims,
                alpha,
                eps,
                temp_thresh,
                active_local,
                comp_local,
                comp_s_flat,
                comp_p_flat,
                comp_v_mv,
            )

        self._pending_count = 0
        self._active_count = adv_res.active_count
        self._completed_examples = adv_res.completed_examples
        return int(adv_res.finished)

    def take_completed_u8(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Drain and return completed examples `(uint8[M,3,6,7], float32[M,7], float32[M])` including mirror pairs."""
        n: int = int(self._completed_examples)
        self._completed_examples = 0
        if n == 0:
            return (
                np.empty((0, 3, 6, 7), dtype=np.uint8),
                np.empty((0, 7), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        out_s = self._comp_states[:n]
        out_p = self._comp_policies[:n]
        out_v = self._comp_values[:n]

        rem_cap: int = max(1, int(self._active_count) * 84)
        self._comp_states = np.empty((rem_cap, 3, 6, 7), dtype=np.uint8)
        self._comp_policies = np.empty((rem_cap, 7), dtype=np.float32)
        self._comp_values = np.empty((rem_cap,), dtype=np.float32)

        return out_s, out_p, out_v

    def take_completed_f32(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Drain and return completed examples `(float32[M,3,6,7], float32[M,7], float32[M])` including mirror pairs."""
        s_u8, out_p, out_v = self.take_completed_u8()
        if s_u8.shape[0] == 0:
            return (
                np.empty((0, 3, 6, 7), dtype=np.float32),
                out_p,
                out_v,
            )
        return np.ascontiguousarray(s_u8, dtype=np.float32), out_p, out_v
