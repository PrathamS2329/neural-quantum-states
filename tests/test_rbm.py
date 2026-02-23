# tests/test_rbm.py
#
# ============================================================
# UNIT TESTS: Restricted Boltzmann Machine (RBM) Ansatz
# ============================================================
#
# WHAT WE'RE TESTING:
#   The RBM is the central ML component of the project. Its correctness is
#   critical — if log_psi or grad_log_psi are wrong, every VMC result is wrong.
#
# TEST STRATEGY:
#   1. STRUCTURAL TESTS: parameter count, shapes, update round-trips.
#
#   2. ANALYTICAL GRADIENT TESTS: the gradient formulas
#        d(log_psi)/d(a_i) = sigma_i
#      are simple enough to verify by hand for specific values.
#
#   3. FINITE-DIFFERENCE GRADIENT CHECK: the gold standard for gradient testing.
#      We perturb each parameter by ±epsilon, measure the change in log_psi,
#      and compare to the analytical gradient. Agreement to 1e-4 rtol means
#      the gradient code is correct.
#
# WHY THE FINITE-DIFFERENCE CHECK IS IMPORTANT:
#   The VMC energy gradient depends on grad_log_psi through:
#     grad_E = 2 Re(<O_k* E_loc> - <O_k*><E_loc>)
#   A silent bug in grad_log_psi would make the optimizer climb instead of
#   descend, and the training would slowly diverge. This test catches that.
#
# ============================================================

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ansatz import RBM


# ============================================================
# SECTION: Structural Tests (Parameter Count, Shapes)
# ============================================================

class TestRBMStructure(unittest.TestCase):
    """Verify parameter count and tensor shapes for the RBM."""

    def setUp(self):
        """Small N=4, alpha=2 RBM used throughout this test class."""
        self.N = 4
        self.alpha = 2
        self.M = self.alpha * self.N    # 8 hidden units
        self.rbm = RBM(n_spins=self.N, alpha=self.alpha, seed=0)

    def test_n_params_formula(self):
        """
        n_params = N + M + N*M  where M = alpha * N.

        MATH: The RBM has:
          - a: visible biases, shape (N,)    → N parameters
          - b: hidden biases,  shape (M,)    → M parameters
          - W: weight matrix,  shape (N, M)  → N*M parameters
          Total: N + M + N*M = N(1 + alpha + alpha*N) ≈ alpha * N² for large N.
        """
        expected = self.N + self.M + self.N * self.M    # 4 + 8 + 32 = 44
        self.assertEqual(self.rbm.n_params, expected)

    def test_parameters_array_length(self):
        """len(parameters) must equal n_params."""
        self.assertEqual(len(self.rbm.parameters), self.rbm.n_params)

    def test_grad_log_psi_shape(self):
        """grad_log_psi must return a flat array of shape (n_params,)."""
        spins = np.array([+1, -1, +1, -1])
        grad = self.rbm.grad_log_psi(spins)
        self.assertEqual(grad.shape, (self.rbm.n_params,))

    def test_log_psi_scalar(self):
        """log_psi must return a scalar, not an array."""
        spins = np.array([+1, +1, -1, +1])
        lp = self.rbm.log_psi(spins)
        # A scalar has no length
        self.assertFalse(hasattr(lp, '__len__'), "log_psi should return a scalar")

    def test_log_psi_finite(self):
        """log_psi must be finite (no NaN or inf) for valid spin inputs."""
        spins = np.array([+1, -1, -1, +1])
        lp = self.rbm.log_psi(spins)
        self.assertTrue(np.isfinite(lp), "log_psi must be finite")


# ============================================================
# SECTION: Gradient Tests
# ============================================================

class TestRBMGradients(unittest.TestCase):
    """Verify the analytical gradients of the RBM log-wavefunction."""

    def setUp(self):
        self.N = 4
        self.alpha = 2
        self.rbm = RBM(n_spins=self.N, alpha=self.alpha, seed=7)
        self.spins = np.array([+1, -1, +1, -1])

    def test_grad_a_equals_spins(self):
        """
        d(log_psi)/d(a_i) = sigma_i — the gradient of the visible bias term.

        DERIVATION:
          log_psi = Σ_i a_i sigma_i + Σ_j log(2 cosh(theta_j))
          d(log_psi)/d(a_i) = sigma_i  (the log(2 cosh) term doesn't depend on a_i)

        We check the first N entries of grad_log_psi against the spin values.
        """
        grad = self.rbm.grad_log_psi(self.spins)
        grad_a = grad[:self.N]    # first N entries are d(log_psi)/d(a_i)
        np.testing.assert_array_almost_equal(
            np.real(grad_a), self.spins.astype(float), decimal=10,
            err_msg="Gradient w.r.t. visible biases should equal spin values"
        )

    def test_grad_b_is_tanh_of_theta(self):
        """
        d(log_psi)/d(b_j) = tanh(theta_j) where theta_j = b_j + W[:,j] · sigma.

        We compute theta manually and check against the gradient values.
        """
        theta = self.rbm.b + self.rbm.W.T @ self.spins    # shape (M,)
        expected_grad_b = np.tanh(theta)

        grad = self.rbm.grad_log_psi(self.spins)
        grad_b = grad[self.N : self.N + self.rbm.n_hidden]    # next M entries

        np.testing.assert_array_almost_equal(
            np.real(grad_b), expected_grad_b, decimal=10,
            err_msg="Gradient w.r.t. hidden biases should equal tanh(theta)"
        )

    def test_finite_difference_gradient(self):
        """
        Numerically validate grad_log_psi via central finite differences.

        ALGORITHM:
          For each parameter k:
            grad_numerical[k] = (log_psi(theta + eps*e_k) - log_psi(theta - eps*e_k)) / (2*eps)

          Compare to analytical grad_log_psi. Agreement to rtol=1e-4 confirms
          that the gradient code is correctly computing the true derivative.

        WHY THIS MATTERS:
          A bug in any of a, b, or W gradients would cause wrong parameter
          updates during VMC training, leading to slow or failed convergence.
          This test catches any such bug immediately.
        """
        eps = 1e-6
        spins = self.spins
        analytical = np.real(self.rbm.grad_log_psi(spins))

        n_params = self.rbm.n_params
        numerical = np.zeros(n_params)

        for k in range(n_params):
            # Step forward: theta_k → theta_k + eps
            delta_fwd = np.zeros(n_params)
            delta_fwd[k] = eps
            self.rbm.update_parameters(delta_fwd)
            lp_fwd = np.real(self.rbm.log_psi(spins))

            # Step backward: theta_k → theta_k - 2*eps (currently at +eps)
            delta_bwd = np.zeros(n_params)
            delta_bwd[k] = -2.0 * eps
            self.rbm.update_parameters(delta_bwd)
            lp_bwd = np.real(self.rbm.log_psi(spins))

            # Restore to original: theta_k → theta_k + eps (currently at -eps)
            delta_restore = np.zeros(n_params)
            delta_restore[k] = eps
            self.rbm.update_parameters(delta_restore)

            numerical[k] = (lp_fwd - lp_bwd) / (2.0 * eps)

        np.testing.assert_allclose(
            analytical, numerical, rtol=1e-4, atol=1e-8,
            err_msg="Analytical gradient disagrees with finite-difference gradient"
        )


# ============================================================
# SECTION: Parameter Management Tests
# ============================================================

class TestRBMParameterManagement(unittest.TestCase):
    """Verify that parameter updates are applied correctly."""

    def setUp(self):
        self.rbm = RBM(n_spins=4, alpha=2, seed=99)

    def test_update_zero_delta_is_identity(self):
        """Applying a zero update must leave all parameters unchanged."""
        params_before = self.rbm.parameters.copy()
        self.rbm.update_parameters(np.zeros(self.rbm.n_params))
        np.testing.assert_array_equal(
            self.rbm.parameters, params_before,
            err_msg="Zero delta should not change parameters"
        )

    def test_update_then_undo_is_identity(self):
        """Applying delta then -delta returns to original parameters."""
        params_before = self.rbm.parameters.copy()
        delta = np.random.default_rng(5).normal(0, 1, size=self.rbm.n_params)

        self.rbm.update_parameters(delta)
        self.rbm.update_parameters(-delta)

        np.testing.assert_array_almost_equal(
            self.rbm.parameters, params_before, decimal=12,
            err_msg="Delta then -delta should restore original parameters"
        )

    def test_parameters_flat_vector(self):
        """parameters property must return a 1D flat array."""
        params = self.rbm.parameters
        self.assertEqual(params.ndim, 1, "parameters must be a 1D array")

    def test_update_changes_log_psi(self):
        """A non-zero update must change log_psi (confirms parameters affect output)."""
        spins = np.array([+1, -1, +1, +1])
        lp_before = self.rbm.log_psi(spins)

        # Apply a large update to force a visible change
        delta = np.ones(self.rbm.n_params) * 0.5
        self.rbm.update_parameters(delta)
        lp_after = self.rbm.log_psi(spins)

        self.assertNotAlmostEqual(
            np.real(lp_before), np.real(lp_after), places=3,
            msg="Non-zero update should change log_psi"
        )


# ============================================================
# SECTION: Entry Point
# ============================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
