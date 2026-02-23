# src/hamiltonians/heisenberg.py
#
# 1D Heisenberg XXX Model — full SU(2) symmetric spin interactions.
#
# Unlike the Ising model (only Z interactions), the Heisenberg model couples all
# three spin components equally: XX + YY + ZZ. This makes the physics richer —
# the ground state is a quantum superposition of Neel-like configurations, and
# the model has an exact solution via the Bethe Ansatz (E/N -> -0.4431 J).
#
# KEY PHYSICS: The AFM ground state has a non-trivial SIGN STRUCTURE described
# by the Marshall sign rule: psi(sigma) = (-1)^{N_up_even} * |psi(sigma)|.
# Since our RBM uses real parameters (producing positive amplitudes only), we
# apply the Marshall sign explicitly in local_energy. This lets the RBM learn
# just the magnitude |psi|, while the sign is handled analytically.
#
# Including this model proves our NQS framework is general: the same RBM + VMC
# loop works on a completely different physical system with zero code changes.

import numpy as np
from .base import Hamiltonian


class HeisenbergHamiltonian(Hamiltonian):
    """
    1D Heisenberg XXX Model with periodic boundary conditions.

    H = J * sum_i [ sigma_i^z * sigma_{i+1}^z
                   + 2(S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+) ]

    Convention: J > 0 is antiferromagnetic, J < 0 is ferromagnetic.
    """

    def __init__(self, n_spins: int, J: float = 1.0):
        """
        Args:
            n_spins: Number of spins in the chain.
            J:       Exchange coupling (J > 0 = antiferromagnetic).
        """
        self._n_spins = n_spins
        self.J = J

    @property
    def n_spins(self) -> int:
        return self._n_spins

    def local_energy(self, spins: np.ndarray, log_psi_func) -> complex:
        """
        Compute E_loc(sigma) for the Heisenberg model with Marshall sign rule.

        The full ansatz is psi(sigma) = marshall_sign(sigma) * psi_RBM(sigma),
        where marshall_sign = (-1)^{N_up_even}. For all nearest-neighbor swaps
        on the 1D chain, the sign ratio is always -1 (since exactly one of the
        two swapped sites is on an even sublattice).

        Per bond (i, i+1):
          - ZZ term (diagonal): J * sigma_i * sigma_j  (no sign change)
          - Swap term (off-diagonal): -2J * exp(log_ratio)  (note the minus
            sign from the Marshall sign rule — this is what allows the real-
            valued RBM to represent the AFM ground state)

        Args:
            spins:        array of shape (n_spins,) with values +1 or -1
            log_psi_func: callable returning log(psi_RBM(spins))

        Returns:
            Local energy as a complex scalar.
        """
        log_psi_current = log_psi_func(spins)
        local_e = 0.0 + 0j

        for i in range(self._n_spins):
            j = (i + 1) % self._n_spins
            sigma_i = spins[i]
            sigma_j = spins[j]

            # ZZ term (diagonal) — unaffected by Marshall sign
            local_e += self.J * sigma_i * sigma_j

            # XX + YY term (off-diagonal): swaps antiparallel pairs.
            # The bare matrix element is +2J, but the Marshall sign rule
            # contributes a factor of -1 for all nearest-neighbor swaps,
            # giving a net coefficient of -2J. This negative sign is what
            # allows the positive-definite RBM to capture the AFM ground state.
            if sigma_i != sigma_j:
                swapped = spins.copy()
                swapped[i] = sigma_j
                swapped[j] = sigma_i
                log_ratio = log_psi_func(swapped) - log_psi_current
                local_e += -2.0 * self.J * np.exp(log_ratio)

        return local_e
