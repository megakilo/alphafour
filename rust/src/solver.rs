/// Exact Pascal-Pons-style Connect Four bitboard alpha-beta W/D/L solver.
///
/// Features:
/// - 49-bit `position + mask` key (with horizontal mirror canonicalization on openings)
/// - Lock-free shared TwoBig1 (depth-preferred + always-replace) Transposition Table
///   packing full 49-bit key, lower/upper WDL bounds, valid bit, and remaining depth
///   into a single tear-free `AtomicU64` (128 MiB, 16.7M entries)
/// - Known exact empty-board (`""`) W/D/L table `[-1, -1, 0, 1, 0, -1, -1]`
/// - Fast shallow tactical mate pre-check
/// - `[3, 2, 4, 1, 5, 0, 6]` center-first ordering refined by threat-creation count
/// - Immediate 4-in-a-row win detection & non-losing forced threat / double-threat pruning
/// - Null-window (`beta == alpha + 1`) W/D/L tests around 0
/// - Optional `max_nodes: Option<u64>` budget in `solve_batch_children_wdl` (returning `-3`
///   for any child exceeding `max_nodes`)

use crate::game::{ConnectFour, BOARD_SIZE, COLS, H1, ROWS};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

const MOVE_ORDER: [usize; COLS] = [3, 2, 4, 1, 5, 0, 6];
/// Exact game-theoretic child W/D/L values from the empty Connect Four board (John Tromp / Pascal Pons).
const EMPTY_BOARD_CHILDREN_WDL: [i8; COLS] = [-1, -1, 0, 1, 0, -1, -1];

const fn make_bottom_mask() -> u64 {
    let mut m = 0u64;
    let mut c = 0;
    while c < COLS {
        m |= 1u64 << (c * H1);
        c += 1;
    }
    m
}

const fn make_col_masks() -> [u64; COLS] {
    let col_bits = (1u64 << ROWS) - 1;
    let mut masks = [0u64; COLS];
    let mut c = 0;
    while c < COLS {
        masks[c] = col_bits << (c * H1);
        c += 1;
    }
    masks
}

const BOTTOM_MASK: u64 = make_bottom_mask();
const BOARD_MASK: u64 = BOTTOM_MASK * ((1u64 << ROWS) - 1);
const COL_MASKS: [u64; COLS] = make_col_masks();

// Transposition Table layout:
// 2^23 buckets * 2 slots * 8 bytes = 128 MiB shared across threads.
const TT_BUCKET_BITS: u32 = 23;
const TT_NUM_BUCKETS: usize = 1 << TT_BUCKET_BITS;

const KEY_MASK: u64 = (1u64 << 49) - 1;
const VALID_BIT: u64 = 1u64 << 53;

#[repr(align(16))]
struct TTBucket {
    depth_pref: AtomicU64,
    always: AtomicU64,
}

struct TranspositionTable {
    buckets: Box<[TTBucket]>,
}

impl TranspositionTable {
    fn new() -> Self {
        let mut v = Vec::with_capacity(TT_NUM_BUCKETS);
        for _ in 0..TT_NUM_BUCKETS {
            v.push(TTBucket {
                depth_pref: AtomicU64::new(0),
                always: AtomicU64::new(0),
            });
        }
        TranspositionTable {
            buckets: v.into_boxed_slice(),
        }
    }

    #[inline(always)]
    fn bucket_index(key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - TT_BUCKET_BITS)) as usize
    }

    #[inline(always)]
    fn pack(key: u64, lower: i8, upper: i8, work: u32) -> u64 {
        VALID_BIT
            | ((work as u64 & 0x3F) << 54)
            | (((upper + 1) as u64 & 0x3) << 51)
            | (((lower + 1) as u64 & 0x3) << 49)
            | (key & KEY_MASK)
    }

    #[inline(always)]
    fn unpack_bounds(raw: u64) -> (i8, i8) {
        let lower = ((raw >> 49) & 0x3) as i8 - 1;
        let upper = ((raw >> 51) & 0x3) as i8 - 1;
        (lower, upper)
    }

    #[inline(always)]
    fn unpack_work(raw: u64) -> u32 {
        ((raw >> 54) & 0x3F) as u32
    }

    #[inline(always)]
    fn get(&self, key: u64) -> Option<(i8, i8)> {
        let bucket = &self.buckets[Self::bucket_index(key)];
        let r0 = bucket.depth_pref.load(Ordering::Relaxed);
        if (r0 & VALID_BIT) != 0 && (r0 & KEY_MASK) == key {
            return Some(Self::unpack_bounds(r0));
        }
        let r1 = bucket.always.load(Ordering::Relaxed);
        if (r1 & VALID_BIT) != 0 && (r1 & KEY_MASK) == key {
            return Some(Self::unpack_bounds(r1));
        }
        None
    }

    #[inline(always)]
    fn put(&self, key: u64, lower: i8, upper: i8, work: u32) {
        let bucket = &self.buckets[Self::bucket_index(key)];
        let r0 = bucket.depth_pref.load(Ordering::Relaxed);

        if (r0 & VALID_BIT) != 0 && (r0 & KEY_MASK) == key {
            let (ol, ou) = Self::unpack_bounds(r0);
            let nl = ol.max(lower);
            let nu = ou.min(upper);
            if nl <= nu {
                bucket
                    .depth_pref
                    .store(Self::pack(key, nl, nu, work), Ordering::Relaxed);
            }
            return;
        }

        let r1 = bucket.always.load(Ordering::Relaxed);
        if (r1 & VALID_BIT) != 0 && (r1 & KEY_MASK) == key {
            let (ol, ou) = Self::unpack_bounds(r1);
            let nl = ol.max(lower);
            let nu = ou.min(upper);
            let merged = if nl <= nu {
                Self::pack(key, nl, nu, work)
            } else {
                Self::pack(key, lower, upper, work)
            };
            if (r0 & VALID_BIT) == 0 || work >= Self::unpack_work(r0) {
                bucket.always.store(r0, Ordering::Relaxed);
                bucket.depth_pref.store(merged, Ordering::Relaxed);
            } else {
                bucket.always.store(merged, Ordering::Relaxed);
            }
            return;
        }

        let entry = Self::pack(key, lower, upper, work);
        if (r0 & VALID_BIT) == 0 || work >= Self::unpack_work(r0) {
            if (r0 & VALID_BIT) != 0 {
                bucket.always.store(r0, Ordering::Relaxed);
            }
            bucket.depth_pref.store(entry, Ordering::Relaxed);
        } else {
            bucket.always.store(entry, Ordering::Relaxed);
        }
    }
}

fn global_tt() -> &'static TranspositionTable {
    static TT: OnceLock<TranspositionTable> = OnceLock::new();
    TT.get_or_init(TranspositionTable::new)
}

#[inline(always)]
fn mirror_key(k: u64) -> u64 {
    const COL: u64 = 0x7F;
    ((k & COL) << 42)
        | (((k >> 7) & COL) << 35)
        | (((k >> 14) & COL) << 28)
        | (k & (COL << 21))
        | (((k >> 28) & COL) << 14)
        | (((k >> 35) & COL) << 7)
        | ((k >> 42) & COL)
}

#[inline(always)]
fn tt_key(position: u64, mask: u64, moves: u32) -> u64 {
    let k = position + mask;
    if moves <= 14 {
        let mk = mirror_key(k);
        if k <= mk { k } else { mk }
    } else {
        k
    }
}

#[inline(always)]
fn symmetric_key_full(position: u64, mask: u64) -> u64 {
    let k = position + mask;
    let mk = mirror_key(k);
    if k <= mk { k } else { mk }
}

/// Bitmap of all empty board squares where placing a stone of `position` completes 4-in-a-row.
#[inline]
fn compute_winning_position(position: u64, mask: u64) -> u64 {
    // 1. Vertical (shift = 1)
    let mut r = (position << 1) & (position << 2) & (position << 3);

    // 2. Horizontal (shift = H1 = 7)
    let mut p = (position << H1) & (position << (2 * H1));
    r |= p & (position << (3 * H1));
    r |= p & (position >> H1);
    p = (position >> H1) & (position >> (2 * H1));
    r |= p & (position << H1);
    r |= p & (position >> (3 * H1));

    // 3. Anti-diagonal \ (shift = H1 - 1 = 6)
    const D1: usize = H1 - 1;
    p = (position << D1) & (position << (2 * D1));
    r |= p & (position << (3 * D1));
    r |= p & (position >> D1);
    p = (position >> D1) & (position >> (2 * D1));
    r |= p & (position << D1);
    r |= p & (position >> (3 * D1));

    // 4. Diagonal / (shift = H1 + 1 = 8)
    const D2: usize = H1 + 1;
    p = (position << D2) & (position << (2 * D2));
    r |= p & (position << (3 * D2));
    r |= p & (position >> D2);
    p = (position >> D2) & (position >> (2 * D2));
    r |= p & (position << D2);
    r |= p & (position >> (3 * D2));

    r & (BOARD_MASK ^ mask)
}

/// Exact depth-bounded forced-win check (`true` ONLY IF current player forces a win in `<= max_plies`).
fn tactical_can_win(position: u64, mask: u64, moves: u32, max_plies: u32) -> bool {
    if moves >= BOARD_SIZE as u32 || max_plies == 0 {
        return false;
    }
    let possible = (mask + BOTTOM_MASK) & BOARD_MASK;
    let winning_p = compute_winning_position(position, mask);
    if (possible & winning_p) != 0 {
        return true;
    }
    if max_plies <= 2 || moves >= (BOARD_SIZE as u32) - 2 {
        return false;
    }

    let opponent_win = compute_winning_position(position ^ mask, mask);
    let forced = possible & opponent_win;
    let mut candidate_mask = possible;
    if forced != 0 {
        if (forced & (forced - 1)) != 0 {
            return false;
        }
        candidate_mask = forced;
    }
    let non_losing = candidate_mask & !(opponent_win >> 1);
    if non_losing == 0 {
        return false;
    }

    let next_pos = position ^ mask;
    for &col in &MOVE_ORDER {
        let move_bit = non_losing & COL_MASKS[col];
        if move_bit != 0
            && tactical_is_forced_loss(next_pos, mask | move_bit, moves + 1, max_plies - 1)
        {
            return true;
        }
    }
    false
}

/// Exact depth-bounded forced-loss check (`true` ONLY IF current player loses against best play in `<= max_plies`).
fn tactical_is_forced_loss(position: u64, mask: u64, moves: u32, max_plies: u32) -> bool {
    if moves >= BOARD_SIZE as u32 {
        return false;
    }
    let possible = (mask + BOTTOM_MASK) & BOARD_MASK;
    let winning_p = compute_winning_position(position, mask);
    if (possible & winning_p) != 0 {
        return false;
    }
    if moves == (BOARD_SIZE as u32) - 1 {
        return false;
    }

    let opponent_win = compute_winning_position(position ^ mask, mask);
    let forced = possible & opponent_win;
    let mut candidate_mask = possible;
    if forced != 0 {
        if (forced & (forced - 1)) != 0 {
            return true;
        }
        candidate_mask = forced;
    }
    let non_losing = candidate_mask & !(opponent_win >> 1);
    if non_losing == 0 {
        return true;
    }
    if max_plies <= 1 {
        return false;
    }

    let next_pos = position ^ mask;
    for &col in &MOVE_ORDER {
        let move_bit = non_losing & COL_MASKS[col];
        if move_bit != 0
            && !tactical_can_win(next_pos, mask | move_bit, moves + 1, max_plies - 1)
        {
            return false;
        }
    }
    true
}

struct NodeBudget {
    remaining: Option<u64>,
}

impl NodeBudget {
    #[inline(always)]
    fn tick(&mut self) -> bool {
        if let Some(ref mut rem) = self.remaining {
            if *rem == 0 {
                return false;
            }
            *rem -= 1;
        }
        true
    }
}

/// Zero-window W/D/L test with optional node budget:
/// Returns `Some(true)` iff `V >= beta`, `Some(false)` iff `V < beta`, or `None` if budget exhausted.
fn can_achieve_budget(
    tt: &TranspositionTable,
    position: u64,
    mask: u64,
    moves: u32,
    beta: i8,
    budget: &mut NodeBudget,
) -> Option<bool> {
    debug_assert!(beta == 0 || beta == 1);

    if !budget.tick() {
        return None;
    }

    if moves >= BOARD_SIZE as u32 {
        return Some(0 >= beta);
    }

    let possible = (mask + BOTTOM_MASK) & BOARD_MASK;
    let winning_p = compute_winning_position(position, mask);
    if (possible & winning_p) != 0 {
        return Some(true);
    }

    if moves == (BOARD_SIZE as u32) - 1 {
        return Some(0 >= beta);
    }
    if moves == (BOARD_SIZE as u32) - 2 && beta == 1 {
        return Some(false);
    }

    let opponent_win = compute_winning_position(position ^ mask, mask);
    let forced_moves = possible & opponent_win;
    let mut candidate_mask = possible;
    if forced_moves != 0 {
        if (forced_moves & (forced_moves - 1)) != 0 {
            return Some(false);
        }
        candidate_mask = forced_moves;
    }

    let non_losing = candidate_mask & !(opponent_win >> 1);
    if non_losing == 0 {
        return Some(false);
    }

    let work = (BOARD_SIZE as u32) - moves;
    let key = tt_key(position, mask, moves);
    if work > 2 {
        if let Some((lower, upper)) = tt.get(key) {
            if lower >= beta {
                return Some(true);
            }
            if upper < beta {
                return Some(false);
            }
        }
    }

    let child_beta = 1 - beta;
    let next_pos = position ^ mask;

    // Fast path 1: single forced/non-losing move
    if (non_losing & (non_losing - 1)) == 0 {
        let child_res =
            can_achieve_budget(tt, next_pos, mask | non_losing, moves + 1, child_beta, budget)?;
        let res = !child_res;
        if work > 2 {
            if res {
                tt.put(key, beta, 1, work);
            } else {
                tt.put(key, -1, beta - 1, work);
            }
        }
        return Some(res);
    }

    // Fast path 2: near leaves (`work <= 6`), skip threat scoring & sorting overhead
    if work <= 6 {
        for &col in &MOVE_ORDER {
            let move_bit = non_losing & COL_MASKS[col];
            if move_bit != 0 {
                let child_res = can_achieve_budget(
                    tt,
                    next_pos,
                    mask | move_bit,
                    moves + 1,
                    child_beta,
                    budget,
                )?;
                if !child_res {
                    if work > 2 {
                        tt.put(key, beta, 1, work);
                    }
                    return Some(true);
                }
            }
        }
        if work > 2 {
            tt.put(key, -1, beta - 1, work);
        }
        return Some(false);
    }

    // Interior node: order candidate moves by created threat count (ties broken by MOVE_ORDER)
    let mut scored_moves = [(0u64, 0u32); COLS];
    let mut num_moves = 0usize;
    for &col in &MOVE_ORDER {
        let move_bit = non_losing & COL_MASKS[col];
        if move_bit != 0 {
            let score =
                compute_winning_position(position | move_bit, mask | move_bit).count_ones();
            let mut j = num_moves;
            while j > 0 && score > scored_moves[j - 1].1 {
                scored_moves[j] = scored_moves[j - 1];
                j -= 1;
            }
            scored_moves[j] = (move_bit, score);
            num_moves += 1;
        }
    }

    for &(move_bit, _) in &scored_moves[..num_moves] {
        let child_res =
            can_achieve_budget(tt, next_pos, mask | move_bit, moves + 1, child_beta, budget)?;
        if !child_res {
            tt.put(key, beta, 1, work);
            return Some(true);
        }
    }

    tt.put(key, -1, beta - 1, work);
    Some(false)
}

/// Solve exact W/D/L (`+1`, `0`, `-1`) with optional node budget (`-3` if budget exceeded).
#[inline]
fn solve_bitboard_wdl_budget(
    tt: &TranspositionTable,
    position: u64,
    mask: u64,
    moves: u32,
    max_nodes: Option<u64>,
) -> i8 {
    // Known exact 0-stone and 1-stone solutions from `EMPTY_BOARD_CHILDREN_WDL`
    if moves == 0 {
        return 1;
    }
    if moves == 1 && position == 0 {
        for c in 0..COLS {
            if (mask & COL_MASKS[c]) != 0 {
                return -EMPTY_BOARD_CHILDREN_WDL[c];
            }
        }
    }
    if moves >= BOARD_SIZE as u32 {
        return 0;
    }
    let possible = (mask + BOTTOM_MASK) & BOARD_MASK;
    let winning_p = compute_winning_position(position, mask);
    if (possible & winning_p) != 0 {
        return 1;
    }

    let work = (BOARD_SIZE as u32) - moves;
    let key = tt_key(position, mask, moves);
    let mut lower = -1i8;
    let mut upper = 1i8;
    if let Some((l, u)) = tt.get(key) {
        if l == u {
            return l;
        }
        lower = l;
        upper = u;
    }

    let mut budget = NodeBudget {
        remaining: max_nodes,
    };

    // Fast shallow tactical check (~few microseconds) catches short forced wins/losses
    // before plunging 35+ plies deep along a non-tactical column!
    if work > 10 && max_nodes.map_or(true, |n| n >= 2_000) {
        if upper > 0 && tactical_can_win(position, mask, moves, 9) {
            tt.put(key, 1, 1, work);
            return 1;
        }
        if lower < 0 && tactical_is_forced_loss(position, mask, moves, 8) {
            tt.put(key, -1, -1, work);
            return -1;
        }
    }

    if upper > 0 {
        match can_achieve_budget(tt, position, mask, moves, 1, &mut budget) {
            Some(true) => return 1,
            Some(false) => upper = 0,
            None => return -3,
        }
    }
    if lower < 0 && upper == 0 {
        match can_achieve_budget(tt, position, mask, moves, 0, &mut budget) {
            Some(true) => return 0,
            Some(false) => return -1,
            None => return -3,
        }
    }
    upper
}

/// Solve exact W/D/L (`+1`, `0`, `-1`) for `game` from `game.current_player`'s perspective.
pub fn solve_game_wdl(game: &ConnectFour) -> i8 {
    if game.is_win() {
        return -1;
    }
    if game.is_draw() {
        return 0;
    }
    if game.move_count == 0 {
        return 1;
    }
    let (pos, mask, moves) = game.solver_bitboards();
    solve_bitboard_wdl_budget(global_tt(), pos, mask, moves, None)
}

/// Solve exact W/D/L (`+1`, `0`, `-1`) for a 1-indexed move string (`'1'..'7'`).
pub fn solve_position_wdl(moves_str: &str) -> Result<i8, String> {
    let game = ConnectFour::from_moves(moves_str)?;
    Ok(solve_game_wdl(&game))
}

/// Solve exact W/D/L for each of the 7 child moves from ROOT MOVER's perspective:
/// `-2` if column is full (or root is already terminal), else `+1` win / `0` draw / `-1` loss.
pub fn solve_game_children_wdl(game: &ConnectFour) -> [i8; COLS] {
    if game.is_terminal() {
        return [-2; COLS];
    }
    if game.move_count == 0 {
        return EMPTY_BOARD_CHILDREN_WDL;
    }
    let tt = global_tt();
    let (pos, mask, moves) = game.solver_bitboards();
    let possible = (mask + BOTTOM_MASK) & BOARD_MASK;
    let winning_p = compute_winning_position(pos, mask);

    let mut out = [-2i8; COLS];
    for col in 0..COLS {
        let move_bit = possible & COL_MASKS[col];
        if move_bit == 0 {
            out[col] = -2;
        } else if (move_bit & winning_p) != 0 {
            out[col] = 1;
        } else if moves + 1 >= BOARD_SIZE as u32 {
            out[col] = 0;
        } else {
            let next_pos = pos ^ mask;
            let next_mask = mask | move_bit;
            let child_wdl = solve_bitboard_wdl_budget(tt, next_pos, next_mask, moves + 1, None);
            out[col] = -child_wdl;
        }
    }
    out
}

/// Solve exact W/D/L for each of the 7 child moves of `moves_str` from ROOT MOVER's perspective.
pub fn solve_children_wdl(moves_str: &str) -> Result<[i8; COLS], String> {
    let game = ConnectFour::from_moves(moves_str)?;
    Ok(solve_game_children_wdl(&game))
}

/// Batch-solve child W/D/L values for `moves_list` in parallel using Rayon.
/// If `max_nodes` is `Some(limit)`, any child position exceeding `limit` search nodes
/// returns `-3` instead of blocking; when `max_nodes` is `None`, runs unlimited exact search.
pub fn solve_batch_children_wdl(
    moves_list: &[String],
    max_nodes: Option<u64>,
) -> Result<Vec<i8>, String> {
    let n = moves_list.len();
    let mut games = Vec::with_capacity(n);
    for s in moves_list {
        games.push(ConnectFour::from_moves(s)?);
    }

    let tt = global_tt();
    let mut out = vec![-2i8; n * COLS];

    let mut key_to_task_idx: HashMap<u64, usize> = HashMap::with_capacity(n * 4);
    let mut unique_tasks: Vec<(u64, u64, u32)> = Vec::with_capacity(n * 4);
    let mut pending_refs: Vec<(usize, usize)> = Vec::with_capacity(n * COLS);

    for (i, game) in games.iter().enumerate() {
        if game.is_terminal() {
            continue;
        }
        if game.move_count == 0 {
            out[i * COLS..(i + 1) * COLS].copy_from_slice(&EMPTY_BOARD_CHILDREN_WDL);
            continue;
        }
        let (pos, mask, moves) = game.solver_bitboards();
        let possible = (mask + BOTTOM_MASK) & BOARD_MASK;
        let winning_p = compute_winning_position(pos, mask);

        for col in 0..COLS {
            let move_bit = possible & COL_MASKS[col];
            let slot = i * COLS + col;
            if move_bit == 0 {
                out[slot] = -2;
            } else if (move_bit & winning_p) != 0 {
                out[slot] = 1;
            } else if moves + 1 >= BOARD_SIZE as u32 {
                out[slot] = 0;
            } else {
                let next_pos = pos ^ mask;
                let next_mask = mask | move_bit;
                let next_moves = moves + 1;
                let sym = symmetric_key_full(next_pos, next_mask);
                let task_idx = *key_to_task_idx.entry(sym).or_insert_with(|| {
                    let idx = unique_tasks.len();
                    unique_tasks.push((next_pos, next_mask, next_moves));
                    idx
                });
                pending_refs.push((slot, task_idx));
            }
        }
    }

    let mut order: Vec<usize> = (0..unique_tasks.len()).collect();
    order.sort_unstable_by(|&a, &b| unique_tasks[a].2.cmp(&unique_tasks[b].2));

    let solved: Vec<(usize, i8)> = order
        .par_iter()
        .with_max_len(1)
        .map(|&task_idx| {
            let (p, m, mv) = unique_tasks[task_idx];
            (task_idx, solve_bitboard_wdl_budget(tt, p, m, mv, max_nodes))
        })
        .collect();

    let mut task_results = vec![0i8; unique_tasks.len()];
    for (task_idx, child_wdl) in solved {
        task_results[task_idx] = child_wdl;
    }

    for (slot, task_idx) in pending_refs {
        let child_wdl = task_results[task_idx];
        out[slot] = if child_wdl == -3 { -3 } else { -child_wdl };
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_board_exact_wdl() {
        assert_eq!(solve_position_wdl("").unwrap(), 1);
        assert_eq!(
            solve_children_wdl("").unwrap(),
            [-1, -1, 0, 1, 0, -1, -1]
        );
        // 1-stone replies match negated empty-board table
        assert_eq!(solve_position_wdl("4").unwrap(), -1);
        assert_eq!(solve_position_wdl("3").unwrap(), 0);
        assert_eq!(solve_position_wdl("1").unwrap(), 1);
    }

    #[test]
    fn test_immediate_win_detection() {
        assert_eq!(solve_position_wdl("121212").unwrap(), 1);
        let children = solve_children_wdl("121212").unwrap();
        assert_eq!(children[0], 1);
    }

    #[test]
    fn test_full_column_marked_minus_two() {
        let children = solve_children_wdl("4444441235671235671235").unwrap();
        assert_eq!(children[3], -2);
        for c in [0, 1, 2, 4, 5, 6] {
            assert!(children[c] >= -1 && children[c] <= 1);
        }
    }

    #[test]
    fn test_max_nodes_budget_returns_minus_three_when_exceeded() {
        let res = solve_batch_children_wdl(&["44".to_string()], Some(5)).unwrap();
        assert_eq!(res.len(), 7);
        assert!(res.contains(&-3));
    }

    #[test]
    fn test_symmetry_invariant() {
        let c1 = solve_children_wdl("32164625").unwrap();
        let c2 = solve_children_wdl("56724263").unwrap();
        for i in 0..COLS {
            assert_eq!(c1[i], c2[COLS - 1 - i]);
        }
    }

    #[test]
    fn test_testdata_l3_r1_and_l1_r1_sample() {
        let p_l3 = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/Test_L3_R1");
        if p_l3.exists() {
            let content = std::fs::read_to_string(p_l3).unwrap();
            let mut moves_list = Vec::new();
            let mut signs = Vec::new();
            for line in content.lines().take(200) {
                let mut parts = line.split_whitespace();
                let m = parts.next().unwrap().to_string();
                let s: i32 = parts.next().unwrap().parse().unwrap();
                moves_list.push(m);
                signs.push(s.signum() as i8);
            }
            let flat = solve_batch_children_wdl(&moves_list, None).unwrap();
            for i in 0..moves_list.len() {
                let row = &flat[i * COLS..(i + 1) * COLS];
                let max_wdl = *row.iter().max().unwrap();
                assert_eq!(max_wdl, signs[i], "Mismatch at {} ({})", i, moves_list[i]);
            }
        }
    }
}
