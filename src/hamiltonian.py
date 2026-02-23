import numpy as np


class IsingHamiltonian:
    """
    1D Transverse Field Ising Model (TFIM)

    H = -J * sum_i (sigma_i^z * sigma_{i+1}^z)  -  Gamma * sum_i (sigma_i^x)

    - First term:  neighboring spins want to align (ferromagnetic coupling J)
    - Second term: transverse field Gamma fights that alignment by flipping spins

    Spins are represented as +1 (up) or -1 (down).
    Periodic boundary conditions: the last spin connects back to the first.

    The quantum phase transition happens at Gamma/J = 1.0:
      - Gamma/J < 1: ordered phase (spins tend to align)
      - Gamma/J > 1: disordered phase (field dominates, spins fluctuate)
    """

    def __init__(self, n_spins, J=1.0, gamma=1.0):
        """
        Args:
            n_spins: number of spins in the chain
            J:       coupling strength between neighboring spins
            gamma:   transverse magnetic field strength
        """
        self.n_spins = n_spins
        self.J = J
        self.gamma = gamma

    def local_energy(self, spins, log_psi_func):
        """
        Compute the local energy E_loc(sigma) = <sigma|H|psi> / <sigma|psi>

        This is the quantity we average over many spin samples to estimate
        the total energy expectation value <E> = <psi|H|psi> / <psi|psi>.

        The local energy has two parts:

        1. Diagonal term: -J * sum_i sigma_i * sigma_{i+1}
           Easy to compute — just multiply neighboring spin values.

        2. Off-diagonal term: -Gamma * sum_i psi(sigma^(i)) / psi(sigma)
           sigma^(i) means the configuration with spin i flipped.
           We use the RBM to compute this ratio via log amplitudes:
           psi(flipped) / psi(current) = exp(log_psi(flipped) - log_psi(current))

        Args:
            spins:        numpy array of shape (n_spins,) with values +1 or -1
            log_psi_func: function that takes a spin array and returns log(psi(spins))
                          this will be the RBM's forward pass

        Returns:
            local energy as a float
        """
        # --- Diagonal term: -J * sum_i sigma_i * sigma_{i+1} ---
        # np.roll(spins, -1) shifts the array left by 1, giving us the right neighbor.
        # With periodic boundaries, the last spin's neighbor is the first spin.
        diagonal = -self.J * np.sum(spins * np.roll(spins, -1))

        # --- Off-diagonal term: -Gamma * sum_i psi(sigma^i) / psi(sigma) ---
        log_psi_current = log_psi_func(spins)

        off_diagonal = 0.0
        for i in range(self.n_spins):
            # Flip spin i to get a new configuration
            flipped = spins.copy()
            flipped[i] *= -1

            # Compute ratio using log amplitudes (numerically stable)
            log_ratio = log_psi_func(flipped) - log_psi_current
            off_diagonal += np.exp(log_ratio)

        off_diagonal *= -self.gamma

        return diagonal + off_diagonal
