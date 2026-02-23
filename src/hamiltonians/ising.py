# src/hamiltonians/ising.py
#
# 1D Transverse Field Ising Model (TFIM).
#
# H = -J * sum_i(sigma_i^z * sigma_{i+1}^z) - Gamma * sum_i(sigma_i^x)
#
# Two competing terms:
#   - J term (diagonal): neighboring spins want to align (ferromagnetic)
#   - Gamma term (off-diagonal): transverse field flips spins into superpositions
#
# Quantum phase transition at Gamma/J = 1.0:
#   Gamma/J < 1 → ordered (spins aligned), Gamma/J > 1 → disordered
#
# Has an exact analytical solution (Jordan-Wigner), so we can verify NQS results.

import numpy as np
from .base import Hamiltonian


class IsingHamiltonian(Hamiltonian):
    """
    1D Transverse Field Ising Model with periodic boundary conditions.

    H = -J * sum_i (sigma_i^z * sigma_{i+1}^z) - Gamma * sum_i (sigma_i^x)
    """

    def __init__(self, n_spins: int, J: float = 1.0, gamma: float = 1.0):
        """
        Args:
            n_spins: Number of spins in the chain.
            J:       Ferromagnetic coupling strength (J > 0).
            gamma:   Transverse field strength. Phase transition at gamma/J = 1.0.
        """
        self._n_spins = n_spins
        self.J = J
        self.gamma = gamma

    @property
    def n_spins(self) -> int:
        return self._n_spins

    def local_energy(self, spins: np.ndarray, log_psi_func) -> complex:
        """
        Compute E_loc(sigma) for the TFIM.

        Two contributions:
          1. Diagonal (ZZ): -J * sum_i sigma_i * sigma_{i+1}
             Just a product of neighboring spin values — no wavefunction needed.

          2. Off-diagonal (X): -Gamma * sum_i psi(sigma^(i)) / psi(sigma)
             sigma^(i) = config with spin i flipped. Uses log-ratio for stability.

        Args:
            spins:        array of shape (n_spins,) with values +1 or -1
            log_psi_func: callable returning log(psi(spins))

        Returns:
            Local energy as a complex scalar.
        """
        # Diagonal: -J * sum of nearest-neighbor products (periodic BC via roll)
        neighbors = np.roll(spins, -1)
        diagonal = -self.J * np.sum(spins * neighbors)

        # Off-diagonal: flip each spin and sum the amplitude ratios
        log_psi_current = log_psi_func(spins)

        off_diagonal = 0.0 + 0j
        for i in range(self._n_spins):
            flipped = spins.copy()
            flipped[i] *= -1
            log_ratio = log_psi_func(flipped) - log_psi_current
            off_diagonal += np.exp(log_ratio)

        off_diagonal *= -self.gamma

        return diagonal + off_diagonal
