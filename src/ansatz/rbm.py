# src/ansatz/rbm.py
#
# Restricted Boltzmann Machine (RBM) as a neural quantum state ansatz.
#
# The RBM is the original NQS architecture from Carleo & Troyer (2017). It has
# N visible units (physical spins) and M = alpha*N hidden units that capture
# multi-spin correlations — effectively encoding quantum entanglement. The hidden
# units can be analytically summed out, giving a closed-form log-wavefunction:
#
#   log psi(sigma) = sum_i a_i*sigma_i + sum_j log(2*cosh(b_j + W[:,j] . sigma))
#
# Gradients are also analytical, making this very efficient for VMC.

import numpy as np
from .base import Ansatz


class RBM(Ansatz):
    """
    Restricted Boltzmann Machine as a neural quantum state ansatz.

    Parameters:
        a (N,):    visible biases — local field on each spin
        b (M,):    hidden biases
        W (N, M):  weight matrix connecting visible to hidden units

    Total parameters: N + M + N*M
    """

    def __init__(self, n_spins: int, alpha: int = 1, seed: int = 42):
        """
        Args:
            n_spins: Number of visible units (= physical spins).
            alpha:   Hidden unit density. M = alpha * n_spins.
                     Higher alpha = more expressive but slower. Typical: 1-4.
            seed:    Random seed for reproducibility.
        """
        self.n_spins = n_spins
        self.alpha = alpha
        self.n_hidden = alpha * n_spins

        rng = np.random.default_rng(seed)

        # Small Gaussian init keeps initial wavefunction close to uniform
        sigma_init = 0.01
        self.a = rng.normal(0, sigma_init, size=n_spins)
        self.b = rng.normal(0, sigma_init, size=self.n_hidden)
        self.W = rng.normal(0, sigma_init, size=(n_spins, self.n_hidden))

    def log_psi(self, spins: np.ndarray) -> complex:
        """
        Compute log(psi(sigma)) for a given spin configuration.

        log psi = a . sigma + sum_j log(2*cosh(theta_j))
        where theta_j = b_j + W[:,j] . sigma are hidden pre-activations.

        Args:
            spins: array of shape (n_spins,) with values +1 or -1

        Returns:
            Log amplitude as a float (real-valued RBM).
        """
        theta = self.b + self.W.T @ spins
        visible_term = self.a @ spins
        # logaddexp(x, -x) = log(2*cosh(x)), numerically stable for large |x|
        hidden_term = np.sum(np.logaddexp(theta, -theta))
        return visible_term + hidden_term

    def grad_log_psi(self, spins: np.ndarray) -> np.ndarray:
        """
        Compute d(log psi)/d(params) for all parameters.

        Analytical gradients:
            d/d(a_i)   = sigma_i
            d/d(b_j)   = tanh(theta_j)
            d/d(W_ij)  = sigma_i * tanh(theta_j)

        Returns:
            Flat array [grad_a | grad_b | grad_W.flatten()], shape (n_params,).
        """
        theta = self.b + self.W.T @ spins

        grad_a = spins.copy()
        grad_b = np.tanh(theta)
        grad_W = np.outer(spins, grad_b)

        return np.concatenate([grad_a, grad_b, grad_W.flatten()])

    @property
    def parameters(self) -> np.ndarray:
        """All parameters as a flat vector: [a | b | W.flatten()]."""
        return np.concatenate([self.a, self.b, self.W.flatten()])

    def update_parameters(self, delta: np.ndarray) -> None:
        """Apply parameter update: theta <- theta + delta."""
        n = self.n_spins
        m = self.n_hidden
        self.a += delta[:n]
        self.b += delta[n:n + m]
        self.W += delta[n + m:].reshape(n, m)

    def __repr__(self) -> str:
        return (
            f"RBM(n_visible={self.n_spins}, n_hidden={self.n_hidden}, "
            f"alpha={self.alpha}, n_params={self.n_params})"
        )
