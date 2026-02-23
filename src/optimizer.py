# src/optimizer.py
#
# Parameter optimizers for VMC: Stochastic Reconfiguration (SR), Adam, and SGD.
#
# Standard gradient descent treats all parameter directions equally, but for
# wavefunctions this is wrong — different parameter changes can have wildly
# different effects on the quantum state. SR fixes this by using the quantum
# geometric tensor (Fisher information matrix) to compute a "natural gradient"
# that accounts for the geometry of the wavefunction manifold. It's equivalent
# to imaginary-time evolution projected onto the variational space, and
# typically converges 5-10x faster than plain SGD.

import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Abstract base class for VMC parameter optimizers.

    Takes grad_E (energy gradient) and grad_logs (log-derivatives per sample),
    returns delta (parameter update). SR uses both; Adam and SGD ignore grad_logs.
    """

    @abstractmethod
    def compute_update(self, grad_E: np.ndarray, grad_logs: np.ndarray) -> np.ndarray:
        """
        Compute the parameter update delta.

        Args:
            grad_E:    energy gradient, shape (n_params,)
            grad_logs: log-derivatives for each sample, shape (n_samples, n_params)

        Returns:
            delta: parameter update, shape (n_params,)
        """
        pass


class StochasticReconfiguration(Optimizer):
    """
    Stochastic Reconfiguration — the natural gradient for quantum states.

    Solves the linear system:
        (S + epsilon * I) * delta = -lr * grad_E

    where S is the quantum geometric tensor:
        S_kk' = <O_k* O_k'> - <O_k*><O_k'>

    O_k = d(log psi)/d(theta_k) are the log-derivatives from the ansatz.
    The S matrix encodes the Riemannian geometry of the quantum state manifold —
    using it as a preconditioner is the "natural gradient" method (Amari 1998),
    independently discovered for quantum systems by Sorella (1998).
    """

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 0.01):
        """
        Args:
            learning_rate: Step size. SR is less sensitive to lr than SGD because
                           the natural gradient accounts for parameter curvature.
            epsilon:       Tikhonov regularization added to diagonal of S.
                           Prevents instability when S is nearly singular.
                           Typical range: 1e-4 to 1e-1.
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def compute_update(self, grad_E: np.ndarray, grad_logs: np.ndarray) -> np.ndarray:
        """
        Compute the SR parameter update.

        1. Build S matrix from centered log-derivatives (covariance form)
        2. Regularize: S_reg = S + epsilon * I
        3. Solve: S_reg * delta = -lr * grad_E via least-squares

        Args:
            grad_E:    energy gradient, shape (n_params,)
            grad_logs: O_k(sigma_i) for each sample, shape (n_samples, n_params)

        Returns:
            delta: parameter update, shape (n_params,)
        """
        n_samples, n_params = grad_logs.shape

        # S matrix: covariance of log-derivatives = <O*O> - <O*><O>
        # Computing from centered O is numerically better
        mean_grad = np.mean(grad_logs, axis=0)
        O_centered = grad_logs - mean_grad[np.newaxis, :]
        S = (np.conj(O_centered).T @ O_centered) / n_samples

        # Regularize to keep S invertible
        S_reg = S + self.epsilon * np.eye(n_params)

        # Solve linear system (lstsq is robust to near-singularity)
        rhs = -self.learning_rate * grad_E
        delta, _, _, _ = np.linalg.lstsq(S_reg, rhs, rcond=None)

        return delta


class Adam(Optimizer):
    """
    Adam optimizer (Kingma & Ba, 2014) adapted for VMC.

    Maintains exponential moving averages of the gradient (first moment)
    and squared gradient (second moment) for per-parameter adaptive learning
    rates. Included as a comparison to SR — both minimize the energy, but
    SR uses quantum geometric information while Adam uses only gradient statistics.

    Ignores grad_logs (the SR-specific input) — only uses grad_E.
    """

    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        """
        Args:
            learning_rate: Step size. Default: 0.001 (standard Adam default).
            beta1:         Decay rate for first moment (gradient mean).
            beta2:         Decay rate for second moment (gradient variance).
            epsilon:       Numerical stability constant.
        """
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Moment estimates — initialized lazily on first call
        self.m = None     # first moment (mean of gradient)
        self.v = None     # second moment (mean of squared gradient)
        self.t = 0        # timestep counter (for bias correction)

    def compute_update(self, grad_E: np.ndarray, grad_logs: np.ndarray) -> np.ndarray:
        """Compute Adam update. Uses only grad_E (grad_logs is ignored)."""
        if self.m is None:
            self.m = np.zeros_like(grad_E)
            self.v = np.zeros_like(grad_E)

        self.t += 1

        # Update biased moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_E
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_E ** 2

        # Bias correction (important in early steps when m, v are near 0)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return -self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class SGD(Optimizer):
    """
    Plain Stochastic Gradient Descent — the simplest baseline optimizer.

    delta = -learning_rate * grad_E

    Included for comparison: plotting SR vs Adam vs SGD convergence on the
    same system shows concretely why SR is superior for quantum states.
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def compute_update(self, grad_E: np.ndarray, grad_logs: np.ndarray) -> np.ndarray:
        """SGD update: delta = -lr * grad_E. Ignores grad_logs."""
        return -self.learning_rate * grad_E
