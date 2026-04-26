"""Integration tests for Rust BatchedSelfPlay engine.

Verifies that the Rust self-play engine produces valid training data:
- Correct output shapes
- Valid policy distributions (sum to ~1)
- Value labels in [-1, 1]
- Augmentation doubles the data
- Games actually terminate

Run with: uv run pytest tests/test_rust_mcts.py -v
"""

import numpy as np
import numpy.testing as npt
import pytest

try:
    from alphafour_engine import RustBatchedSelfPlay

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE,
    reason="Rust engine not built. Run: cd rust && maturin develop --release",
)

COLS = 7


class TestBatchedSelfPlay:
    """Test the Rust BatchedSelfPlay engine with dummy NN evaluations."""

    def _run_engine(self, num_games=5, num_sims=10):
        """Run a complete self-play session with uniform random policies."""
        engine = RustBatchedSelfPlay(num_games, num_sims, 1.5, 1.0, 0.25, 30)

        # Init roots
        states, valid = engine.get_root_states()
        assert states.shape == (num_games, 3, 6, 7)
        assert valid.shape == (num_games, 7)

        n = engine.num_active()
        policies = np.full((n, COLS), 1.0 / COLS, dtype=np.float32)
        engine.init_roots(policies)

        all_states = []
        all_policies = []
        all_values = []

        max_iters = 500
        iters = 0
        while not engine.is_done() and iters < max_iters:
            engine.add_noise()

            for _ in range(num_sims):
                leaf_states, leaf_valid, count = engine.collect_leaves()
                if count > 0:
                    p = np.full((count, COLS), 1.0 / COLS, dtype=np.float32)
                    v = np.zeros(count, dtype=np.float32)
                    engine.apply_evaluations(p, v)

            ex_states, ex_policies, ex_values = engine.advance_games()
            if len(ex_states) > 0:
                all_states.append(ex_states)
                all_policies.append(ex_policies)
                all_values.append(ex_values)

            n = engine.num_active()
            if n > 0:
                states, valid = engine.get_root_states()
                policies = np.full((n, COLS), 1.0 / COLS, dtype=np.float32)
                engine.init_new_roots(policies)

            iters += 1

        assert engine.is_done(), f"Games didn't finish in {max_iters} iterations"

        states = np.concatenate(all_states) if all_states else np.empty((0, 3, 6, 7))
        policies = np.concatenate(all_policies) if all_policies else np.empty((0, 7))
        values = np.concatenate(all_values) if all_values else np.empty((0,))
        return states, policies, values

    def test_basic_completion(self):
        """Engine should complete all games."""
        states, policies, values = self._run_engine(num_games=5, num_sims=10)
        assert len(states) > 0
        assert len(states) == len(policies) == len(values)

    def test_output_shapes(self):
        """Output arrays should have correct shapes."""
        states, policies, values = self._run_engine(num_games=3, num_sims=5)
        assert states.ndim == 4
        assert states.shape[1:] == (3, 6, 7)
        assert policies.ndim == 2
        assert policies.shape[1] == 7
        assert values.ndim == 1

    def test_augmentation_doubles_data(self):
        """Rust engine includes augmentation, so data count should be even."""
        states, policies, values = self._run_engine(num_games=5, num_sims=10)
        assert len(states) % 2 == 0, "Augmentation should produce pairs"

    def test_policies_are_valid(self):
        """Policy vectors should sum to ~1."""
        states, policies, values = self._run_engine(num_games=3, num_sims=10)
        sums = policies.sum(axis=1)
        npt.assert_allclose(sums, 1.0, atol=0.01)

    def test_values_in_range(self):
        """Values should be in [-1, 1]."""
        states, policies, values = self._run_engine(num_games=5, num_sims=10)
        assert np.all(values >= -1.0)
        assert np.all(values <= 1.0)

    def test_states_are_valid_encodings(self):
        """State encodings should have valid structure."""
        states, policies, values = self._run_engine(num_games=3, num_sims=5)
        for s in states:
            # Planes 0 and 1 should be binary (0 or 1)
            assert np.all((s[0] == 0) | (s[0] == 1))
            assert np.all((s[1] == 0) | (s[1] == 1))
            # No overlap between planes 0 and 1
            assert np.all(s[0] * s[1] == 0)
            # Plane 2 is all 0s or all 1s
            assert s[2].sum() == 0 or s[2].sum() == 42

    def test_augmented_pairs_are_mirrored(self):
        """Consecutive example pairs should be horizontal mirrors."""
        states, policies, values = self._run_engine(num_games=3, num_sims=10)
        for i in range(0, len(states) - 1, 2):
            original = states[i]
            mirrored = states[i + 1]
            # Mirrored state = original flipped along column axis
            npt.assert_array_equal(mirrored, original[:, :, ::-1])
            # Mirrored policy = original reversed
            npt.assert_array_almost_equal(policies[i + 1], policies[i][::-1])
            # Same value
            assert values[i] == values[i + 1]

    def test_many_games(self):
        """Test with more games to stress-test."""
        states, policies, values = self._run_engine(num_games=20, num_sims=10)
        assert len(states) > 0
        # Expect at least some examples per game (min ~8 moves per game × 2 for augment)
        assert len(states) >= 20
