/// PyO3 Python bindings for the `alphafour-engine` Rust crate.
///
/// Exposes:
/// - `RustConnectFour` (including `from_moves`)
/// - `RustBatchedSelfPlay`
/// - `RustBatchedMCTS`
/// - `batch_encode`, `batch_valid_moves`
/// - `solve_position_wdl`, `solve_children_wdl`, `solve_batch_children_wdl`
///
/// Uses safe `readonly().as_slice()` slice views (`PyResult`) and zero-copy
/// `Array::from_shape_vec` buffer transfers throughout (zero `unsafe` blocks).

use numpy::ndarray::{Array2, Array3, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod batch_mcts;
mod batch_self_play;
mod game;
mod mcts;
mod solver;

use batch_mcts::BatchedMCTS;
use batch_self_play::BatchedSelfPlay;
use game::{ConnectFour, BOARD_SIZE, COLS, ROWS};

// ── ConnectFour bindings ────────────────────────────────────────────────

#[pyclass(name = "RustConnectFour")]
#[derive(Clone)]
struct PyConnectFour {
    inner: ConnectFour,
}

#[pymethods]
impl PyConnectFour {
    #[new]
    fn new() -> Self {
        PyConnectFour {
            inner: ConnectFour::new(),
        }
    }

    #[staticmethod]
    fn from_moves(moves_str: &str) -> PyResult<Self> {
        ConnectFour::from_moves(moves_str)
            .map(|inner| PyConnectFour { inner })
            .map_err(PyValueError::new_err)
    }

    fn make_move(&mut self, col: u8) -> PyResult<()> {
        if col as usize >= COLS {
            return Err(PyValueError::new_err(format!(
                "Column index {} out of range 0..{}",
                col, COLS
            )));
        }
        if (self.inner.heights[col as usize] as usize) >= ROWS {
            return Err(PyValueError::new_err(format!("Column {} is full", col)));
        }
        self.inner.make_move(col);
        Ok(())
    }

    #[getter]
    fn current_player(&self) -> i8 {
        self.inner.current_player
    }

    #[getter]
    fn move_count(&self) -> u8 {
        self.inner.move_count
    }

    #[getter]
    fn last_move(&self) -> Option<u8> {
        self.inner.last_move
    }

    fn is_win(&self) -> bool {
        self.inner.is_win()
    }

    fn is_draw(&self) -> bool {
        self.inner.is_draw()
    }

    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    fn get_result(&self) -> Option<f64> {
        self.inner.get_result()
    }

    fn get_valid_moves<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let valid = self.inner.get_valid_moves();
        PyArray1::from_vec(py, valid.to_vec())
    }

    fn get_valid_move_indices(&self) -> Vec<u8> {
        self.inner.get_valid_move_indices()
    }

    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        let flat = self.inner.encode();
        Array3::from_shape_vec((3, ROWS, COLS), flat.to_vec())
            .expect("valid (3, 6, 7) shape")
            .into_pyarray(py)
    }

    fn copy(&self) -> PyConnectFour {
        PyConnectFour {
            inner: self.inner.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RustConnectFour(move_count={}, player={})",
            self.inner.move_count, self.inner.current_player
        )
    }
}

// ── Batch & Solver functions ────────────────────────────────────────────

#[pyfunction]
fn batch_encode<'py>(
    py: Python<'py>,
    games: Vec<PyRef<'_, PyConnectFour>>,
) -> Bound<'py, PyArray4<f32>> {
    let n = games.len();
    let mut flat = Vec::with_capacity(n * 3 * BOARD_SIZE);
    for game in &games {
        flat.extend_from_slice(&game.inner.encode());
    }
    vec_to_states(py, flat, n)
}

#[pyfunction]
fn batch_valid_moves<'py>(
    py: Python<'py>,
    games: Vec<PyRef<'_, PyConnectFour>>,
) -> Bound<'py, PyArray2<bool>> {
    let n = games.len();
    let mut flat = Vec::with_capacity(n * COLS);
    for game in &games {
        flat.extend_from_slice(&game.inner.get_valid_moves());
    }
    vec_to_valid(py, flat, n)
}

#[pyfunction]
fn solve_position_wdl(py: Python<'_>, moves_str: &str) -> PyResult<i8> {
    let owned = moves_str.to_string();
    py.allow_threads(move || solver::solve_position_wdl(&owned))
        .map_err(PyValueError::new_err)
}

#[pyfunction]
fn solve_children_wdl(py: Python<'_>, moves_str: &str) -> PyResult<Vec<i8>> {
    let owned = moves_str.to_string();
    py.allow_threads(move || solver::solve_children_wdl(&owned).map(|arr| arr.to_vec()))
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (moves_list, max_nodes=None))]
fn solve_batch_children_wdl<'py>(
    py: Python<'py>,
    moves_list: Vec<String>,
    max_nodes: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let n = moves_list.len();
    let flat = py
        .allow_threads(move || solver::solve_batch_children_wdl(&moves_list, max_nodes))
        .map_err(PyValueError::new_err)?;
    let arr = Array2::from_shape_vec((n, COLS), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

// ── BatchedSelfPlay bindings ────────────────────────────────────────────

#[pyclass(name = "RustBatchedSelfPlay")]
struct PyBatchedSelfPlay {
    inner: BatchedSelfPlay,
}

#[pymethods]
impl PyBatchedSelfPlay {
    #[new]
    #[pyo3(signature = (num_games, num_simulations, c_puct=1.5, dirichlet_alpha=0.35, dirichlet_epsilon=0.25, temp_threshold=10, random_opening_plies=0, c_fpu=0.20))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_games: usize,
        num_simulations: usize,
        c_puct: f32,
        dirichlet_alpha: f64,
        dirichlet_epsilon: f64,
        temp_threshold: usize,
        random_opening_plies: usize,
        c_fpu: f32,
    ) -> Self {
        PyBatchedSelfPlay {
            inner: BatchedSelfPlay::new(
                num_games,
                num_simulations,
                c_puct,
                dirichlet_alpha,
                dirichlet_epsilon,
                temp_threshold,
                random_opening_plies,
                c_fpu,
            ),
        }
    }

    fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    fn is_step_done(&self) -> bool {
        self.inner.is_step_done()
    }

    fn num_active(&self) -> usize {
        self.inner.num_active()
    }

    /// Get root states as `(states: ndarray[N,3,6,7], valid: ndarray[N,7])`.
    fn get_root_states<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray4<f32>>, Bound<'py, PyArray2<bool>>) {
        let n = self.inner.num_active();
        let (states_flat, valid_flat) = py.allow_threads(|| self.inner.get_root_states());
        (
            vec_to_states(py, states_flat, n),
            vec_to_valid(py, valid_flat, n),
        )
    }

    /// Get ONLY `!is_expanded` active root states as `(states[U,3,6,7], valid[U,7], U)`.
    fn get_unexpanded_root_states<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray4<f32>>, Bound<'py, PyArray2<bool>>, usize) {
        let (states_flat, valid_flat, count) =
            py.allow_threads(|| self.inner.get_unexpanded_root_states());
        (
            vec_to_states(py, states_flat, count),
            vec_to_valid(py, valid_flat, count),
            count,
        )
    }

    /// Expand root nodes with policies from neural network.
    fn init_roots(
        &mut self,
        py: Python<'_>,
        policies: &Bound<'_, PyArray2<f32>>,
    ) -> PyResult<()> {
        let expected = self.inner.num_active() * COLS;
        let pol_ro = policies.readonly();
        let pol_slice = pol_ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("policies must be contiguous: {e}")))?;
        if pol_slice.len() < expected {
            return Err(PyValueError::new_err(format!(
                "Expected policies slice of len >= {}, got {}",
                expected,
                pol_slice.len()
            )));
        }
        let inner = &mut self.inner;
        py.allow_threads(|| inner.init_roots(pol_slice));
        Ok(())
    }

    /// Expand ONLY `!is_expanded` active roots (`policies` shape `(U, 7)`).
    fn init_unexpanded_roots(
        &mut self,
        py: Python<'_>,
        policies: &Bound<'_, PyArray2<f32>>,
    ) -> PyResult<()> {
        let expected = self.inner.num_unexpanded_roots() * COLS;
        if expected == 0 {
            return Ok(());
        }
        let pol_ro = policies.readonly();
        let pol_slice = pol_ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("policies must be contiguous: {e}")))?;
        if pol_slice.len() < expected {
            return Err(PyValueError::new_err(format!(
                "Expected policies slice of len >= {}, got {}",
                expected,
                pol_slice.len()
            )));
        }
        let inner = &mut self.inner;
        py.allow_threads(|| inner.init_unexpanded_roots(pol_slice));
        Ok(())
    }

    /// Add Dirichlet noise to active roots.
    fn add_noise(&mut self, py: Python<'_>) {
        let inner = &mut self.inner;
        py.allow_threads(|| inner.add_noise());
    }

    /// One multi-leaf batch selection step across active games.
    /// Returns `(states: ndarray[K,3,6,7], valid: ndarray[K,7], count: int)`.
    fn collect_leaves<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray4<f32>>, Bound<'py, PyArray2<bool>>, usize) {
        let inner = &mut self.inner;
        let (states_flat, valid_flat, count) = py.allow_threads(|| inner.collect_leaves());
        (
            vec_to_states(py, states_flat, count),
            vec_to_valid(py, valid_flat, count),
            count,
        )
    }

    /// Apply neural network evaluations to pending leaves.
    fn apply_evaluations(
        &mut self,
        py: Python<'_>,
        policies: &Bound<'_, PyArray2<f32>>,
        values: &Bound<'_, PyArray1<f32>>,
    ) -> PyResult<()> {
        let count = self.inner.num_pending();
        if count == 0 {
            return Ok(());
        }
        let pol_ro = policies.readonly();
        let val_ro = values.readonly();
        let pol_slice = pol_ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("policies must be contiguous: {e}")))?;
        let val_slice = val_ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("values must be contiguous: {e}")))?;
        if pol_slice.len() < count * COLS || val_slice.len() < count {
            return Err(PyValueError::new_err(format!(
                "Evaluation shape mismatch: pending={}, policies_len={}, values_len={}",
                count,
                pol_slice.len(),
                val_slice.len()
            )));
        }
        let inner = &mut self.inner;
        py.allow_threads(|| inner.apply_evaluations(pol_slice, val_slice));
        Ok(())
    }

    /// Advance games: sample actions, return finished training examples (zero-copy).
    /// Returns `(states: ndarray[M,3,6,7], policies: ndarray[M,7], values: ndarray[M])`.
    fn advance_games<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
    ) {
        let inner = &mut self.inner;
        let (states_flat, policies_flat, values_vec, n) =
            py.allow_threads(|| inner.advance_games_flat());
        let states = vec_to_states(py, states_flat, n);
        let policies = Array2::from_shape_vec((n, COLS), policies_flat)
            .expect("valid (n, 7) policies shape")
            .into_pyarray(py);
        let values = PyArray1::from_vec(py, values_vec);
        (states, policies, values)
    }

    /// Initialize new roots after advancing (`policies` shape `(num_active, 7)`).
    fn init_new_roots(
        &mut self,
        py: Python<'_>,
        policies: &Bound<'_, PyArray2<f32>>,
    ) -> PyResult<()> {
        let expected = self.inner.num_active() * COLS;
        if expected == 0 {
            return Ok(());
        }
        let pol_ro = policies.readonly();
        let pol_slice = pol_ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("policies must be contiguous: {e}")))?;
        if pol_slice.len() < expected {
            return Err(PyValueError::new_err(format!(
                "Expected policies slice of len >= {}, got {}",
                expected,
                pol_slice.len()
            )));
        }
        let inner = &mut self.inner;
        py.allow_threads(|| inner.init_new_roots(pol_slice));
        Ok(())
    }
}

// ── BatchedMCTS bindings ────────────────────────────────────────────────

#[pyclass(name = "RustBatchedMCTS")]
struct PyBatchedMCTS {
    inner: BatchedMCTS,
}

#[pymethods]
impl PyBatchedMCTS {
    #[new]
    #[pyo3(signature = (games, num_simulations, c_puct=1.5, c_fpu=0.20))]
    fn new(
        games: Vec<PyRef<'_, PyConnectFour>>,
        num_simulations: usize,
        c_puct: f32,
        c_fpu: f32,
    ) -> Self {
        let rust_games: Vec<ConnectFour> = games.iter().map(|g| g.inner.clone()).collect();
        PyBatchedMCTS {
            inner: BatchedMCTS::new(rust_games, num_simulations, c_puct, c_fpu),
        }
    }

    fn get_root_states<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray4<f32>>, Bound<'py, PyArray2<bool>>) {
        let n = self.inner.num_games();
        let (states_flat, valid_flat) = py.allow_threads(|| self.inner.get_root_states());
        (
            vec_to_states(py, states_flat, n),
            vec_to_valid(py, valid_flat, n),
        )
    }

    fn init_roots(
        &mut self,
        py: Python<'_>,
        policies: &Bound<'_, PyArray2<f32>>,
    ) -> PyResult<()> {
        let expected = self.inner.num_games() * COLS;
        if expected == 0 {
            return Ok(());
        }
        let pol_ro = policies.readonly();
        let pol_slice = pol_ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("policies must be contiguous: {e}")))?;
        if pol_slice.len() < expected {
            return Err(PyValueError::new_err(format!(
                "Expected policies slice of len >= {}, got {}",
                expected,
                pol_slice.len()
            )));
        }
        let inner = &mut self.inner;
        py.allow_threads(|| inner.init_roots(pol_slice));
        Ok(())
    }

    fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    fn collect_leaves<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray4<f32>>, Bound<'py, PyArray2<bool>>, usize) {
        let inner = &mut self.inner;
        let (states_flat, valid_flat, count) = py.allow_threads(|| inner.collect_leaves());
        (
            vec_to_states(py, states_flat, count),
            vec_to_valid(py, valid_flat, count),
            count,
        )
    }

    fn apply_evaluations(
        &mut self,
        py: Python<'_>,
        policies: &Bound<'_, PyArray2<f32>>,
        values: &Bound<'_, PyArray1<f32>>,
    ) -> PyResult<()> {
        let count = self.inner.num_pending();
        if count == 0 {
            return Ok(());
        }
        let pol_ro = policies.readonly();
        let val_ro = values.readonly();
        let pol_slice = pol_ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("policies must be contiguous: {e}")))?;
        let val_slice = val_ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("values must be contiguous: {e}")))?;
        if pol_slice.len() < count * COLS || val_slice.len() < count {
            return Err(PyValueError::new_err(format!(
                "Evaluation shape mismatch: pending={}, policies_len={}, values_len={}",
                count,
                pol_slice.len(),
                val_slice.len()
            )));
        }
        let inner = &mut self.inner;
        py.allow_threads(|| inner.apply_evaluations(pol_slice, val_slice));
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn get_results<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray2<f32>>,
    ) {
        let n = self.inner.num_games();
        let (vc_flat, best_actions_vec, best_values_vec, child_q_flat) =
            py.allow_threads(|| self.inner.get_results());
        let vc = Array2::from_shape_vec((n, COLS), vc_flat)
            .expect("valid (n, 7) visit_counts shape")
            .into_pyarray(py);
        let best_actions = PyArray1::from_vec(py, best_actions_vec);
        let best_values = PyArray1::from_vec(py, best_values_vec);
        let child_q = Array2::from_shape_vec((n, COLS), child_q_flat)
            .expect("valid (n, 7) child_q_values shape")
            .into_pyarray(py);
        (vc, best_actions, best_values, child_q)
    }
}

// ── Zero-copy vector-to-ndarray helpers ─────────────────────────────────

#[inline]
fn vec_to_states<'py>(py: Python<'py>, flat: Vec<f32>, n: usize) -> Bound<'py, PyArray4<f32>> {
    Array4::from_shape_vec((n, 3, ROWS, COLS), flat)
        .expect("valid (n, 3, 6, 7) states shape")
        .into_pyarray(py)
}

#[inline]
fn vec_to_valid<'py>(py: Python<'py>, flat: Vec<bool>, n: usize) -> Bound<'py, PyArray2<bool>> {
    Array2::from_shape_vec((n, COLS), flat)
        .expect("valid (n, 7) valid_moves shape")
        .into_pyarray(py)
}

// ── Module ──────────────────────────────────────────────────────────────

#[pymodule]
fn alphafour_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConnectFour>()?;
    m.add_class::<PyBatchedSelfPlay>()?;
    m.add_class::<PyBatchedMCTS>()?;
    m.add_function(wrap_pyfunction!(batch_encode, m)?)?;
    m.add_function(wrap_pyfunction!(batch_valid_moves, m)?)?;
    m.add_function(wrap_pyfunction!(solve_position_wdl, m)?)?;
    m.add_function(wrap_pyfunction!(solve_children_wdl, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch_children_wdl, m)?)?;
    Ok(())
}
