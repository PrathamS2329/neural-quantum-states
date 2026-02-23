# src/hamiltonians/base.py
#
# Abstract base class for quantum Hamiltonians.
# Every Hamiltonian (Ising, Heisenberg, etc.) inherits from this class,
# so the trainer and sampler can work with any system polymorphically.

from abc import ABC, abstractmethod
import numpy as np


class Hamiltonian(ABC):
    """
    Abstract base class for quantum Hamiltonians.

    Subclasses must implement `local_energy`, which computes
    E_loc(sigma) = <sigma|H|psi> / <sigma|psi> — the central quantity in VMC.
    The variational energy is the Monte Carlo average of E_loc over samples.
    """

    @abstractmethod
    def local_energy(self, spins: np.ndarray, log_psi_func) -> complex:
        """
        Compute the local energy E_loc(sigma) = <sigma|H|psi> / <sigma|psi>.

        Args:
            spins:        numpy array of shape (n_spins,), values in {+1, -1}
            log_psi_func: callable — takes a spin array, returns log(psi(spins))

        Returns:
            Local energy as a complex number.
        """
        pass

    @abstractmethod
    def n_spins(self) -> int:
        """Return the number of spins in the system."""
        pass
