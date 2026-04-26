/// PyO3 Python bindings for the alphafour-engine Rust crate.
///
/// Exposes the ConnectFour game engine and BatchedSelfPlay to Python
/// with numpy-compatible methods.

use numpy::ndarray::{Array1, Array2, Array3, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods};
use pyo3::prelude::*;

mod batch_self_play;
mod game;
mod mcts;

use batch_self_play::{BatchedSelfPlay, Example};
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

    fn make_move(&mut self, col: u8) {
        self.inner.make_move(col);
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
        let mut arr = Array3::<f32>::zeros((3, ROWS, COLS));
        for plane in 0..3 {
            for r in 0..ROWS {
                for c in 0..COLS {
                    arr[[plane, r, c]] = flat[plane * BOARD_SIZE + r * COLS + c];
                }
            }
        }
        arr.into_pyarray(py)
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

// ── Batch helpers ───────────────────────────────────────────────────────

#[pyfunction]
fn batch_encode<'py>(
    py: Python<'py>,
    games: Vec<PyRef<'_, PyConnectFour>>,
) -> Bound<'py, PyArray4<f32>> {
    let n = games.len();
    let mut arr = Array4::<f32>::zeros((n, 3, ROWS, COLS));
    for (i, game) in games.iter().enumerate() {
        let flat = game.inner.encode();
        for plane in 0..3 {
            for r in 0..ROWS {
                for c in 0..COLS {
                    arr[[i, plane, r, c]] = flat[plane * BOARD_SIZE + r * COLS + c];
                }
            }
        }
    }
    arr.into_pyarray(py)
}

#[pyfunction]
fn batch_valid_moves<'py>(
    py: Python<'py>,
    games: Vec<PyRef<'_, PyConnectFour>>,
) -> Bound<'py, PyArray2<bool>> {
    let n = games.len();
    let mut arr = Array2::<bool>::default((n, COLS));
    for (i, game) in games.iter().enumerate() {
        let valid = game.inner.get_valid_moves();
        for c in 0..COLS {
            arr[[i, c]] = valid[c];
        }
    }
    arr.into_pyarray(py)
}

// ── BatchedSelfPlay bindings ────────────────────────────────────────────

#[pyclass(name = "RustBatchedSelfPlay")]
struct PyBatchedSelfPlay {
    inner: BatchedSelfPlay,
}

#[pymethods]
impl PyBatchedSelfPlay {
    #[new]
    #[pyo3(signature = (num_games, num_simulations, c_puct=1.5, dirichlet_alpha=1.0, dirichlet_epsilon=0.25, temp_threshold=30))]
    fn new(
        num_games: usize,
        num_simulations: usize,
        c_puct: f32,
        dirichlet_alpha: f64,
        dirichlet_epsilon: f64,
        temp_threshold: usize,
    ) -> Self {
        PyBatchedSelfPlay {
            inner: BatchedSelfPlay::new(
                num_games,
                num_simulations,
                c_puct,
                dirichlet_alpha,
                dirichlet_epsilon,
                temp_threshold,
            ),
        }
    }

    fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    fn num_active(&self) -> usize {
        self.inner.num_active()
    }

    /// Get root states as (states: ndarray[N,3,6,7], valid: ndarray[N,7]).
    fn get_root_states<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray4<f32>>, Bound<'py, PyArray2<bool>>) {
        let (states_flat, valid_flat) = self.inner.get_root_states();
        let n = self.inner.num_active();
        let states = flat_to_states(py, &states_flat, n);
        let valid = flat_to_valid(py, &valid_flat, n);
        (states, valid)
    }

    /// Expand root nodes with policies from neural network.
    fn init_roots(&mut self, policies: &Bound<'_, PyArray2<f32>>) {
        let policies = unsafe { policies.as_slice().unwrap() };
        self.inner.init_roots(policies);
    }

    /// Add Dirichlet noise to active roots.
    fn add_noise(&mut self) {
        self.inner.add_noise();
    }

    /// One simulation step: select leaves, return states needing evaluation.
    /// Returns (states: ndarray[K,3,6,7], valid: ndarray[K,7], count: int).
    fn collect_leaves<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray4<f32>>, Bound<'py, PyArray2<bool>>, usize) {
        let (states_flat, valid_flat, count) = self.inner.collect_leaves();
        let states = flat_to_states(py, &states_flat, count);
        let valid = flat_to_valid(py, &valid_flat, count);
        (states, valid, count)
    }

    /// Apply neural network evaluations to pending leaves.
    fn apply_evaluations(
        &mut self,
        policies: &Bound<'_, PyArray2<f32>>,
        values: &Bound<'_, PyArray1<f32>>,
    ) {
        let policies = unsafe { policies.as_slice().unwrap() };
        let values = unsafe { values.as_slice().unwrap() };
        self.inner.apply_evaluations(policies, values);
    }

    /// Advance games: sample actions, return finished training examples.
    /// Returns (states: ndarray[M,3,6,7], policies: ndarray[M,7], values: ndarray[M]).
    fn advance_games<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
    ) {
        let examples = self.inner.advance_games();
        examples_to_numpy(py, &examples)
    }

    /// Initialize new roots after advancing (same as init_roots).
    fn init_new_roots(&mut self, policies: &Bound<'_, PyArray2<f32>>) {
        let policies = unsafe { policies.as_slice().unwrap() };
        self.inner.init_new_roots(policies);
    }
}

// ── Helper functions ────────────────────────────────────────────────────

fn flat_to_states<'py>(py: Python<'py>, flat: &[f32], n: usize) -> Bound<'py, PyArray4<f32>> {
    let mut arr = Array4::<f32>::zeros((n, 3, ROWS, COLS));
    for i in 0..n {
        for plane in 0..3 {
            for r in 0..ROWS {
                for c in 0..COLS {
                    arr[[i, plane, r, c]] =
                        flat[i * 3 * BOARD_SIZE + plane * BOARD_SIZE + r * COLS + c];
                }
            }
        }
    }
    arr.into_pyarray(py)
}

fn flat_to_valid<'py>(py: Python<'py>, flat: &[bool], n: usize) -> Bound<'py, PyArray2<bool>> {
    let mut arr = Array2::<bool>::default((n, COLS));
    for i in 0..n {
        for c in 0..COLS {
            arr[[i, c]] = flat[i * COLS + c];
        }
    }
    arr.into_pyarray(py)
}

fn examples_to_numpy<'py>(
    py: Python<'py>,
    examples: &[Example],
) -> (
    Bound<'py, PyArray4<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
) {
    let n = examples.len();
    let mut states = Array4::<f32>::zeros((n, 3, ROWS, COLS));
    let mut policies = Array2::<f32>::zeros((n, COLS));
    let mut values = Array1::<f32>::zeros(n);

    for (i, ex) in examples.iter().enumerate() {
        for plane in 0..3 {
            for r in 0..ROWS {
                for c in 0..COLS {
                    states[[i, plane, r, c]] =
                        ex.state[plane * BOARD_SIZE + r * COLS + c];
                }
            }
        }
        for c in 0..COLS {
            policies[[i, c]] = ex.policy[c];
        }
        values[i] = ex.value;
    }

    (
        states.into_pyarray(py),
        policies.into_pyarray(py),
        values.into_pyarray(py),
    )
}

// ── Module ──────────────────────────────────────────────────────────────

#[pymodule]
fn alphafour_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConnectFour>()?;
    m.add_class::<PyBatchedSelfPlay>()?;
    m.add_function(wrap_pyfunction!(batch_encode, m)?)?;
    m.add_function(wrap_pyfunction!(batch_valid_moves, m)?)?;
    Ok(())
}
