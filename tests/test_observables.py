# tests/test_observables.py
#
# ============================================================
# UNIT TESTS: Physical Observables
# ============================================================
#
# WHAT WE'RE TESTING:
#   The observables module computes quantum expectation values from MCMC
#   samples. Because these are averages over ±1 spin values, many properties
#   can be checked analytically without running VMC.
#
# TEST STRATEGY:
#   We construct controlled sample arrays (not from MCMC — just numpy arrays)
#   that represent simple known spin configurations. For these we can compute
#   exact expected values:
#
#   ALL-UP:       samples = [[+1,+1,+1,+1], ...]  → m = 1.0, C[i,j] = 1.0
#   ALL-DOWN:     samples = [[-1,-1,-1,-1], ...]  → m = -1.0, C[i,j] = 1.0
#   ALTERNATING:  samples = [[+1,-1,+1,-1], ...]  → m = 0.0, structure factor peaks at k=π
#
# WHY THIS APPROACH WORKS:
#   The observable functions treat their input as a numpy array of shape
#   (n_samples, n_spins) with values ±1 — they don't know or care whether
#   the samples came from MCMC or were hand-crafted. So we can test the
#   math without running the full sampling pipeline.
#
# KEY PHYSICAL IDENTITIES TESTED:
#   - C[i,i] = ⟨σᵢ²⟩ = 1  (since σᵢ = ±1 implies σᵢ² = 1 always)
#   - C[i,j] = C[j,i]  (symmetry of correlator)
#   - Σ_k S(k) = N  (Parseval's theorem for ±1 spins)
#   - S(k) >= 0  (structure factor is a power spectrum)
#
# ============================================================

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.observables import (
    magnetization,
    magnetization_per_site,
    spin_spin_correlation,
    correlation_matrix,
    connected_correlation_matrix,
    structure_factor,
    compute_all_observables,
)


# ============================================================
# SECTION: Helper — Controlled Sample Arrays
# ============================================================

def _all_up(n_spins, n_samples):
    """All-up spin samples: every entry is +1."""
    return np.ones((n_samples, n_spins), dtype=np.int8)

def _all_down(n_spins, n_samples):
    """All-down spin samples: every entry is -1."""
    return -np.ones((n_samples, n_spins), dtype=np.int8)

def _alternating(n_spins, n_samples):
    """Alternating (Néel) pattern: [+1,-1,+1,-1,...] repeated."""
    row = np.array([(+1 if i % 2 == 0 else -1) for i in range(n_spins)])
    return np.tile(row, (n_samples, 1)).astype(np.int8)


# ============================================================
# SECTION: Magnetization Tests
# ============================================================

class TestMagnetization(unittest.TestCase):
    """Test the mean z-magnetization per site."""

    def test_all_up_magnetization(self):
        """
        All-up samples → m = 1.0.

        PHYSICS: If all spins point up (σᵢ = +1 for all i), the system is
        fully polarized ferromagnetically. The order parameter m = 1.0.
        """
        samples = _all_up(n_spins=4, n_samples=100)
        self.assertAlmostEqual(magnetization(samples), 1.0, places=10)

    def test_all_down_magnetization(self):
        """All-down samples → m = -1.0 (fully polarized in the opposite direction)."""
        samples = _all_down(n_spins=4, n_samples=100)
        self.assertAlmostEqual(magnetization(samples), -1.0, places=10)

    def test_alternating_magnetization(self):
        """
        Alternating [+1,-1,+1,-1] → m = 0.0 for even N.

        PHYSICS: The Néel state has equal numbers of up and down spins,
        so the net magnetization is zero. This is the antiferromagnetic
        order — structurally ordered but magnetically unbiased.
        """
        samples = _alternating(n_spins=4, n_samples=50)
        self.assertAlmostEqual(magnetization(samples), 0.0, places=10)

    def test_magnetization_in_valid_range(self):
        """
        |m| <= 1.0 for any ±1 sample array.

        MATH: m = mean of ±1 values. Mean of values bounded to [-1,1]
        is itself bounded to [-1,1].
        """
        rng = np.random.default_rng(0)
        samples = rng.choice([-1, 1], size=(200, 8)).astype(np.int8)
        m = magnetization(samples)
        self.assertGreaterEqual(m, -1.0)
        self.assertLessEqual(m, 1.0)

    def test_magnetization_per_site_shape(self):
        """magnetization_per_site returns shape (n_spins,)."""
        samples = _all_up(n_spins=6, n_samples=50)
        m_per_site = magnetization_per_site(samples)
        self.assertEqual(m_per_site.shape, (6,))

    def test_magnetization_per_site_all_up(self):
        """All-up samples → m_per_site = [1.0, 1.0, ..., 1.0]."""
        samples = _all_up(n_spins=5, n_samples=50)
        m_per_site = magnetization_per_site(samples)
        np.testing.assert_array_almost_equal(m_per_site, np.ones(5), decimal=10)


# ============================================================
# SECTION: Correlation Matrix Tests
# ============================================================

class TestCorrelationMatrix(unittest.TestCase):
    """Test spin-spin correlations and the full correlation matrix."""

    def test_spin_spin_correlation_same_site(self):
        """
        C(i, i) = ⟨σᵢ² ⟩ = 1.0 for any ±1 spin samples.

        PHYSICS: Any spin variable squared equals 1 (σ² = 1 for σ = ±1).
        This is a fundamental identity of spin-1/2 systems.
        """
        rng = np.random.default_rng(1)
        samples = rng.choice([-1, 1], size=(100, 6)).astype(np.int8)
        for i in range(6):
            with self.subTest(i=i):
                c_ii = spin_spin_correlation(samples, i, i)
                self.assertAlmostEqual(c_ii, 1.0, places=10,
                                       msg=f"C({i},{i}) should be 1.0")

    def test_correlation_matrix_shape(self):
        """correlation_matrix returns shape (n_spins, n_spins)."""
        samples = _all_up(n_spins=6, n_samples=50)
        C = correlation_matrix(samples)
        self.assertEqual(C.shape, (6, 6))

    def test_correlation_matrix_diagonal_ones(self):
        """
        Diagonal entries C[i,i] = 1.0 for all i.

        PHYSICS: Each spin squared = 1, so ⟨σᵢ²⟩ = 1.0 exactly.
        This should hold regardless of the sample values.
        """
        rng = np.random.default_rng(2)
        samples = rng.choice([-1, 1], size=(200, 8)).astype(np.int8)
        C = correlation_matrix(samples)
        np.testing.assert_array_almost_equal(
            np.diag(C), np.ones(8), decimal=10,
            err_msg="Diagonal of correlation matrix must be all 1.0"
        )

    def test_correlation_matrix_symmetric(self):
        """
        C[i,j] = C[j,i] — the correlator is symmetric.

        MATH: ⟨σᵢσⱼ⟩ = ⟨σⱼσᵢ⟩ since multiplication is commutative.
        """
        rng = np.random.default_rng(3)
        samples = rng.choice([-1, 1], size=(100, 6)).astype(np.int8)
        C = correlation_matrix(samples)
        np.testing.assert_array_almost_equal(
            C, C.T, decimal=10,
            err_msg="Correlation matrix must be symmetric"
        )

    def test_correlation_matrix_bounds(self):
        """
        All entries: -1 <= C[i,j] <= 1.

        MATH: Cauchy-Schwarz: |⟨σᵢσⱼ⟩| <= √(⟨σᵢ²⟩⟨σⱼ²⟩) = 1.
        """
        rng = np.random.default_rng(4)
        samples = rng.choice([-1, 1], size=(200, 8)).astype(np.int8)
        C = correlation_matrix(samples)
        self.assertTrue(np.all(C >= -1.0), "All correlations must be >= -1")
        self.assertTrue(np.all(C <= +1.0), "All correlations must be <= +1")

    def test_all_up_correlations_equal_one(self):
        """
        All-up samples: every pair correlates perfectly → C[i,j] = 1.0.

        PHYSICS: If all spins are +1, then σᵢσⱼ = 1 for every (i,j).
        This is maximum ferromagnetic order.
        """
        samples = _all_up(n_spins=4, n_samples=100)
        C = correlation_matrix(samples)
        np.testing.assert_array_almost_equal(
            C, np.ones((4, 4)), decimal=10,
            err_msg="All-up samples should give all-ones correlation matrix"
        )

    def test_connected_correlation_shape(self):
        """connected_correlation_matrix returns shape (n_spins, n_spins)."""
        samples = _all_up(n_spins=5, n_samples=50)
        Cc = connected_correlation_matrix(samples)
        self.assertEqual(Cc.shape, (5, 5))

    def test_connected_correlation_near_zero_for_disordered(self):
        """
        For strongly disordered (random) samples with large N, connected correlation
        matrix approaches zero (no long-range order).

        PHYSICS: For a completely random spin chain (infinite temperature),
        ⟨σᵢσⱼ⟩ → ⟨σᵢ⟩⟨σⱼ⟩ = 0 (no connected correlations).
        With 500 samples, we expect off-diagonal entries to be close to 0.
        """
        rng = np.random.default_rng(5)
        # Random samples simulate infinite temperature with no order
        samples = rng.choice([-1, 1], size=(5000, 8)).astype(np.int8)
        Cc = connected_correlation_matrix(samples)
        # Off-diagonal entries should be small (within Monte Carlo noise ~0.1)
        off_diag = Cc - np.diag(np.diag(Cc))
        self.assertTrue(np.all(np.abs(off_diag) < 0.1),
                        "Random samples should have ~zero connected correlations")


# ============================================================
# SECTION: Structure Factor Tests
# ============================================================

class TestStructureFactor(unittest.TestCase):
    """Test S(k) — the Fourier transform of spin-spin correlations."""

    def test_structure_factor_returns_two_arrays(self):
        """structure_factor must return a (k_values, S_k) tuple."""
        samples = _all_up(n_spins=4, n_samples=50)
        result = structure_factor(samples)
        self.assertEqual(len(result), 2, "structure_factor should return (k_values, S_k)")

    def test_structure_factor_shape(self):
        """Both k_values and S_k must have shape (n_spins,)."""
        n_spins = 6
        samples = _all_up(n_spins=n_spins, n_samples=50)
        k_values, S_k = structure_factor(samples)
        self.assertEqual(k_values.shape, (n_spins,))
        self.assertEqual(S_k.shape, (n_spins,))

    def test_structure_factor_nonneg(self):
        """
        S(k) >= 0 for all k.

        MATH: S(k) = (1/N) ⟨|FFT(σ)|²⟩. Absolute values squared are always
        non-negative, and the average of non-negative numbers is non-negative.
        """
        rng = np.random.default_rng(6)
        samples = rng.choice([-1, 1], size=(200, 8)).astype(np.int8)
        _, S_k = structure_factor(samples)
        self.assertTrue(np.all(S_k >= -1e-12),    # allow tiny float errors
                        "S(k) must be non-negative for all k")

    def test_structure_factor_sum_rule(self):
        """
        Σ_k S(k) = N (Parseval's theorem for ±1 spins).

        DERIVATION:
          S(k) = (1/N) ⟨|σ̂(k)|²⟩
          Σ_k S(k) = (1/N) ⟨Σ_k |σ̂(k)|²⟩
                   = (1/N) ⟨N Σᵢ σᵢ²⟩  [Parseval]
                   = ⟨Σᵢ σᵢ²⟩  = N  [since σᵢ² = 1]

        This is a model-independent identity that must hold for any ±1 spin samples.
        """
        n_spins = 8
        rng = np.random.default_rng(7)
        samples = rng.choice([-1, 1], size=(300, n_spins)).astype(np.int8)
        _, S_k = structure_factor(samples)
        self.assertAlmostEqual(np.sum(S_k), float(n_spins), places=8,
                               msg="Sum of S(k) must equal N by Parseval's theorem")

    def test_ferromagnetic_peak_at_k0(self):
        """
        All-up samples → S(k) peaks at k=0 (ferromagnetic order).

        PHYSICS: For σᵢ = +1 for all i, FFT gives σ̂(k=0) = N, all others = 0.
          S(k=0) = N²/N = N,  S(k≠0) = 0.
        The peak at k=0 is the signature of ferromagnetic long-range order.
        """
        n_spins = 4
        samples = _all_up(n_spins=n_spins, n_samples=100)
        _, S_k = structure_factor(samples)

        # S(k=0) should be N, all others should be ~0
        self.assertAlmostEqual(S_k[0], float(n_spins), places=8,
                               msg="All-up state should have S(k=0) = N")
        for k_idx in range(1, n_spins):
            self.assertAlmostEqual(S_k[k_idx], 0.0, places=8,
                                   msg=f"All-up state should have S(k≠0) = 0, but S(k_{k_idx}) = {S_k[k_idx]}")

    def test_antiferromagnetic_peak_at_k_pi(self):
        """
        Alternating (Néel) samples → peak at k=π (antiferromagnetic order).

        PHYSICS: For [+1,-1,+1,-1,...], the FFT has nonzero component only at
          k = π (the N/2-th frequency mode). This signals antiferromagnetic order.
        The Heisenberg model ground state has significant weight at k=π.
        """
        n_spins = 4   # even N required for clean Néel state
        samples = _alternating(n_spins=n_spins, n_samples=100)
        _, S_k = structure_factor(samples)

        # For n_spins=4, the π mode is at index N/2 = 2
        pi_idx = n_spins // 2
        self.assertAlmostEqual(S_k[pi_idx], float(n_spins), places=8,
                               msg="Néel state should have S(k=π) = N")
        for k_idx in range(n_spins):
            if k_idx != pi_idx:
                self.assertAlmostEqual(S_k[k_idx], 0.0, places=8,
                                       msg=f"Néel state should have S(k≠π) = 0")

    def test_k_values_range(self):
        """
        k_values should be in [0, 2π) with spacing 2π/N.

        PHYSICS: The discrete momentum values for a chain of length N are
          k_n = 2π n / N  for n = 0, 1, ..., N-1.
        """
        n_spins = 8
        samples = _all_up(n_spins=n_spins, n_samples=10)
        k_values, _ = structure_factor(samples)
        expected = 2 * np.pi * np.arange(n_spins) / n_spins
        np.testing.assert_array_almost_equal(k_values, expected, decimal=10)


# ============================================================
# SECTION: compute_all_observables Tests
# ============================================================

class TestComputeAllObservables(unittest.TestCase):
    """Test the combined observable computation function."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.samples = rng.choice([-1, 1], size=(100, 6)).astype(np.int8)

    def test_returns_dict(self):
        """compute_all_observables must return a dict."""
        obs = compute_all_observables(self.samples)
        self.assertIsInstance(obs, dict)

    def test_expected_keys_present(self):
        """
        The returned dict must have all expected observable keys.

        These keys are what run_vmc.py and plot_phase_diagram.py read.
        Missing keys would cause KeyError in downstream scripts.
        """
        obs = compute_all_observables(self.samples)
        expected_keys = {
            'magnetization',
            'magnetization_per_site',
            'correlation_matrix',
            'connected_correlation',
            'k_values',
            'structure_factor',
        }
        for key in expected_keys:
            with self.subTest(key=key):
                self.assertIn(key, obs, f"Key '{key}' missing from observables dict")

    def test_shapes_are_consistent(self):
        """All array-valued observables must have shapes consistent with n_spins=6."""
        n_spins = 6
        obs = compute_all_observables(self.samples)
        self.assertEqual(obs['magnetization_per_site'].shape, (n_spins,))
        self.assertEqual(obs['correlation_matrix'].shape, (n_spins, n_spins))
        self.assertEqual(obs['connected_correlation'].shape, (n_spins, n_spins))
        self.assertEqual(obs['k_values'].shape, (n_spins,))
        self.assertEqual(obs['structure_factor'].shape, (n_spins,))

    def test_magnetization_is_scalar(self):
        """obs['magnetization'] must be a Python float."""
        obs = compute_all_observables(self.samples)
        self.assertIsInstance(obs['magnetization'], float)


# ============================================================
# SECTION: Entry Point
# ============================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
