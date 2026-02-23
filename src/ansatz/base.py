# src/ansatz/base.py
#
# Abstract base class for neural network quantum state ansatze.
# An "ansatz" is a parameterized wavefunction psi(sigma; theta) used to
# approximate the true quantum ground state. Both RBM and CNN implement
# this interface, so the trainer/sampler can swap between them freely.

from abc import ABC, abstractmethod
import numpy as np


class Ansatz(ABC):
    """
    Abstract base class for neural quantum state ansatze.

    Every ansatz must provide:
      - log_psi(spins): log amplitude of the wavefunction
      - grad_log_psi(spins): gradients w.r.t. all parameters (for VMC updates)
      - parameters property: flat array of all trainable parameters
      - update_parameters(delta): apply a parameter update
    """

    @abstractmethod
    def log_psi(self, spins: np.ndarray) -> complex:
        """
        Compute log(psi(spins)).

        We use log amplitudes for numerical stability and efficient ratio
        computation: psi(s')/psi(s) = exp(log_psi(s') - log_psi(s)).

        Args:
            spins: array of shape (n_spins,) with values +1 or -1

        Returns:
            log(psi(spins)) as a complex scalar.
        """
        pass

    @abstractmethod
    def grad_log_psi(self, spins: np.ndarray) -> np.ndarray:
        """
        Compute d(log psi)/d(theta_k) for all parameters.

        These "log-derivatives" O_k are the key ingredient in the VMC gradient:
            grad_E = 2 * Re( <O_k* E_loc> - <O_k*> <E_loc> )

        Args:
            spins: array of shape (n_spins,) with values +1 or -1

        Returns:
            Flat array of shape (n_params,).
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> np.ndarray:
        """Return all trainable parameters as a single flat array."""
        pass

    @abstractmethod
    def update_parameters(self, delta: np.ndarray) -> None:
        """
        Apply a parameter update: theta <- theta + delta.

        Args:
            delta: flat array of shape (n_params,), same layout as `parameters`
        """
        pass

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return len(self.parameters)
