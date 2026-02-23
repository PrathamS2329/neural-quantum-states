# src/sampler.py
#
# Metropolis-Hastings MCMC sampler for spin configurations.
#
# In VMC, we need to estimate <E> = sum_sigma |psi(sigma)|^2 * E_loc(sigma),
# but summing over all 2^N configurations is intractable. Instead, we use MCMC
# to draw samples from |psi(sigma)|^2 and estimate <E> as a sample average.
# The Metropolis algorithm proposes single spin flips and accepts/rejects based
# on the amplitude ratio |psi(new)/psi(old)|^2, satisfying detailed balance.

import numpy as np
from .ansatz.base import Ansatz


class MetropolisSampler:
    """
    Metropolis-Hastings sampler for spin configurations from |psi(sigma)|^2.

    Proposes single spin flips, and optionally exchange (pair-swap) moves for
    models like Heisenberg where the ground state lives in a fixed total-Sz sector.
    Acceptance ratio computed in log-space for stability.
    Healthy acceptance rate: 0.3 - 0.7.
    """

    def __init__(self, ansatz: Ansatz, n_spins: int, seed: int = 42,
                 use_exchange: bool = False):
        """
        Args:
            ansatz:       Neural network wavefunction (must implement log_psi).
            n_spins:      Number of spins in the chain.
            seed:         Random seed for reproducibility.
            use_exchange: If True, ALL proposals are non-local pair swaps of
                          antiparallel spins (conserves total Sz). Essential
                          for Heisenberg-type models where single flips change
                          total Sz and get rejected once the wavefunction peaks.
        """
        self.ansatz = ansatz
        self.n_spins = n_spins
        self.use_exchange = use_exchange
        self.rng = np.random.default_rng(seed)

        self.current_spins = self.rng.choice([-1, 1], size=n_spins)
        self.current_log_psi = self.ansatz.log_psi(self.current_spins)

        self._n_proposed = 0
        self._n_accepted = 0

    def _metropolis_step(self) -> None:
        """
        One Metropolis step: propose a move and accept/reject.

        When use_exchange=True (Heisenberg), ALL proposals are non-local pair
        swaps of any two antiparallel spins. Single flips change total Sz and
        get rejected once the wavefunction peaks on the Sz=0 sector, so they're
        excluded entirely. Non-local swaps (not just adjacent) dramatically
        improve MCMC mixing across the Hilbert space.

        When use_exchange=False (Ising), proposals are single spin flips.

        Acceptance ratio: A = |psi(proposed)|^2 / |psi(current)|^2
        Computed as log(A) = 2 * Re(log_psi(proposed) - log_psi(current)).
        """
        proposed_spins = self.current_spins.copy()

        if self.use_exchange:
            # --- Adjacent pair swap: exchange two neighboring antiparallel spins ---
            # Adjacent swaps have higher acceptance than non-local swaps because
            # the wavefunction ratio for nearby spins is closer to 1. They also
            # mirror the Hamiltonian's nearest-neighbor structure.
            antiparallel = [
                i for i in range(self.n_spins)
                if self.current_spins[i] != self.current_spins[(i + 1) % self.n_spins]
            ]
            if len(antiparallel) > 0:
                i = int(self.rng.choice(antiparallel))
                j = (i + 1) % self.n_spins
                proposed_spins[i], proposed_spins[j] = proposed_spins[j], proposed_spins[i]
            else:
                # Fully polarized — fall back to single flip
                proposed_spins[self.rng.integers(0, self.n_spins)] *= -1
        else:
            # --- Single spin flip (original behavior, for Ising) ---
            proposed_spins[self.rng.integers(0, self.n_spins)] *= -1

        proposed_log_psi = self.ansatz.log_psi(proposed_spins)
        log_acceptance = 2.0 * np.real(proposed_log_psi - self.current_log_psi)

        self._n_proposed += 1

        # Accept if A >= 1, or with probability A otherwise.
        # Using log(random) < log(A) avoids computing exp().
        if log_acceptance >= 0 or np.log(self.rng.random()) < log_acceptance:
            self.current_spins = proposed_spins
            self.current_log_psi = proposed_log_psi
            self._n_accepted += 1

    def burn_in(self, n_steps: int) -> None:
        """Run n_steps without recording to let the chain reach equilibrium."""
        for _ in range(n_steps):
            self._metropolis_step()

    def sample(self, n_samples: int, sweep_size: int = None) -> np.ndarray:
        """
        Collect n_samples spin configurations from |psi(sigma)|^2.

        Between consecutive samples, performs `sweep_size` Metropolis steps
        to reduce autocorrelation. One "sweep" = N steps (each spin gets
        one flip attempt on average).

        Args:
            n_samples:  Number of configurations to collect.
            sweep_size: Steps between samples. Default: n_spins (one sweep).

        Returns:
            Array of shape (n_samples, n_spins) with values +1 or -1.
        """
        if sweep_size is None:
            sweep_size = self.n_spins

        samples = np.empty((n_samples, self.n_spins), dtype=np.int8)

        for k in range(n_samples):
            for _ in range(sweep_size):
                self._metropolis_step()
            samples[k] = self.current_spins

        return samples

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed moves that were accepted."""
        if self._n_proposed == 0:
            return 0.0
        return self._n_accepted / self._n_proposed

    def reset_acceptance_stats(self) -> None:
        """Reset acceptance counters (call at the start of each epoch)."""
        self._n_proposed = 0
        self._n_accepted = 0

    def refresh(self) -> None:
        """
        Recompute cached log_psi for the current spin configuration.

        Must be called after the optimizer updates ansatz parameters, otherwise
        the cached log_psi is stale (computed under old params). This makes the
        Metropolis acceptance ratio use the wrong denominator, which can freeze
        the chain on sharply peaked wavefunctions (e.g., Heisenberg model).
        """
        self.current_log_psi = self.ansatz.log_psi(self.current_spins)

    def reset_state(self, spins: np.ndarray = None) -> None:
        """
        Reset the sampler's spin configuration (and recompute cached log_psi).

        Args:
            spins: New configuration. If None, generates a random one.
        """
        if spins is None:
            self.current_spins = self.rng.choice([-1, 1], size=self.n_spins)
        else:
            self.current_spins = spins.copy()

        self.current_log_psi = self.ansatz.log_psi(self.current_spins)
        self.reset_acceptance_stats()
