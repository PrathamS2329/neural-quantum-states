# src/trainer.py
#
# VMC training loop — the heart of the project. Ties together the Hamiltonian,
# RBM ansatz, MCMC sampler, and optimizer into the full variational Monte Carlo
# algorithm. Each epoch: sample spin configs from |psi|^2, compute local energies
# and log-derivatives, estimate the energy gradient, and update parameters.
#
# The VMC energy gradient uses the log-derivative trick (same as REINFORCE in RL):
#   grad_E_k = 2 * Re[ <O_k* E_loc> - <O_k*><E_loc> ]
# where O_k = d(log psi)/d(theta_k) and <.> denotes MC averages over |psi|^2.

import numpy as np
import os
import time

from .ansatz.base import Ansatz
from .hamiltonians.base import Hamiltonian
from .sampler import MetropolisSampler
from .optimizer import Optimizer
from .utils import Logger, Checkpointer, make_progress_bar


class VMCTrainer:
    """
    Variational Monte Carlo trainer.

    Orchestrates sampling, energy estimation, gradient computation, and
    parameter updates. Each epoch approximates one step of imaginary-time
    evolution (when using SR) toward the quantum ground state.
    """

    def __init__(self,
                 ansatz: Ansatz,
                 hamiltonian: Hamiltonian,
                 sampler: MetropolisSampler,
                 optimizer: Optimizer,
                 n_samples: int = 500,
                 n_burn: int = 200,
                 n_therm: int = None,
                 sweep_size: int = None,
                 checkpoint_every: int = 50,
                 log_every: int = 10,
                 results_dir: str = None):
        """
        Args:
            ansatz:           Neural network wavefunction (RBM or CNN).
            hamiltonian:      Quantum Hamiltonian (Ising or Heisenberg).
            sampler:          MetropolisSampler initialized with the same ansatz.
            optimizer:        SR, Adam, or SGD optimizer.
            n_samples:        MCMC samples per epoch. More samples = lower variance
                              gradient but slower per epoch. Typical: 200-2000.
            n_burn:           Initial burn-in steps before first epoch. Ensures the
                              chain reaches the stationary distribution |psi|^2.
            n_therm:          Thermalization steps between epochs (lets chain adjust
                              to updated ansatz). Default: n_spins (one sweep).
            sweep_size:       Metropolis steps between consecutive samples within
                              an epoch (reduces autocorrelation). Default: n_spins.
            checkpoint_every: Save every N epochs. 0 or None to disable.
            log_every:        Print summary every N epochs.
            results_dir:      Directory for checkpoints and logs. None = no saving.
        """
        self.ansatz        = ansatz
        self.hamiltonian   = hamiltonian
        self.sampler       = sampler
        self.optimizer     = optimizer
        self.n_samples     = n_samples
        self.n_burn        = n_burn
        self.n_therm       = n_therm if n_therm is not None else ansatz.n_spins
        self.sweep_size    = sweep_size
        self.log_every     = log_every
        self.results_dir   = results_dir

        self.checkpoint_every = checkpoint_every or 0

        self.checkpointer = None
        if results_dir is not None:
            self.checkpointer = Checkpointer(save_dir=results_dir)

        self.logger = Logger()

        # Track best (lowest) energy for best-checkpoint saving
        self._best_energy = np.inf
        self._best_epoch  = -1

    def _compute_local_quantities(self, samples: np.ndarray):
        """
        Compute local energies E_loc(sigma_k) and log-derivatives O_k(sigma_k)
        for all MCMC samples. This is the most expensive step: O(K * N * M).

        Returns:
            energies:  complex array of shape (n_samples,)
            grad_logs: array of shape (n_samples, n_params)
        """
        n_samples = len(samples)
        n_params  = self.ansatz.n_params

        energies  = np.empty(n_samples, dtype=complex)
        grad_logs = np.empty((n_samples, n_params), dtype=float)

        for k, sigma in enumerate(samples):
            energies[k] = self.hamiltonian.local_energy(sigma, self.ansatz.log_psi)
            grad_logs[k] = self.ansatz.grad_log_psi(sigma)

        return energies, grad_logs

    def _compute_energy_gradient(self,
                                 energies: np.ndarray,
                                 grad_logs: np.ndarray) -> np.ndarray:
        """
        Compute the VMC energy gradient from Monte Carlo samples.

        grad_E_k = 2 * Re[ <O_k* E_loc> - <O_k*><E_loc> ]

        The subtraction of <O_k*><E_loc> is a control variate — it doesn't
        change the gradient mean but significantly reduces variance.

        Returns:
            grad_E: shape (n_params,)
        """
        mean_E = np.mean(energies)
        mean_O_conj = np.mean(np.conj(grad_logs), axis=0)

        # Cross term: <O_k* E_loc> for each parameter k
        # energies[:, newaxis] broadcasts (n_samples,) -> (n_samples, 1)
        OstarE = np.conj(grad_logs) * energies[:, np.newaxis]
        mean_OstarE = np.mean(OstarE, axis=0)

        return 2.0 * np.real(mean_OstarE - mean_O_conj * mean_E)

    def train(self, n_epochs: int, exact_energy: float = None) -> Logger:
        """
        Run the full VMC training loop for n_epochs.

        Each epoch: thermalize -> sample -> compute energies/gradients ->
        optimizer step -> log metrics -> checkpoint.

        Args:
            n_epochs:     Number of training epochs.
            exact_energy: Known exact energy (for error tracking in output).

        Returns:
            Logger with full training history.
        """
        # Initial burn-in: long thermalization run once before training
        print(f"Running initial burn-in ({self.n_burn} steps)...")
        self.sampler.burn_in(self.n_burn)
        print("Burn-in complete. Starting VMC training.\n")

        epoch_iter = make_progress_bar(range(1, n_epochs + 1),
                                       desc="VMC Training",
                                       leave=True)

        for epoch in epoch_iter:
            t_start = time.perf_counter()

            # Step 1 — Thermalize: brief re-equilibration for current ansatz
            self.sampler.reset_acceptance_stats()
            self.sampler.burn_in(self.n_therm)

            # Step 2 — Sample K spin configurations from |psi|^2
            samples = self.sampler.sample(
                n_samples=self.n_samples,
                sweep_size=self.sweep_size
            )
            accept_rate = self.sampler.acceptance_rate

            # Step 3 — Compute E_loc and O_k for each sample
            energies, grad_logs = self._compute_local_quantities(samples)

            # Step 4 — Energy estimate and gradient
            mean_E = float(np.mean(np.real(energies)))
            std_E  = float(np.std(np.real(energies)))
            grad_E = self._compute_energy_gradient(energies, grad_logs)
            grad_norm = float(np.linalg.norm(grad_E))

            # Step 5 — Gradient clipping: on hard models (Heisenberg), early
            # gradients can be huge (|grad| > 20), causing wavefunction collapse
            # that freezes the MCMC chain permanently. Standard in NQS and deep learning.
            max_grad_norm = 5.0
            if grad_norm > max_grad_norm:
                grad_E = grad_E * (max_grad_norm / grad_norm)

            # Step 6 — Optimizer: SR passes grad_logs too (for the S matrix);
            # Adam and SGD ignore grad_logs.
            delta = self.optimizer.compute_update(grad_E, grad_logs)

            # Step 6b — Clip the optimizer output too. SR solves S^{-1} * grad,
            # and an ill-conditioned S matrix can amplify a clipped gradient back
            # into a huge parameter update. This caps the actual step size.
            delta_norm = float(np.linalg.norm(delta))
            max_delta_norm = 1.0
            if delta_norm > max_delta_norm:
                delta = delta * (max_delta_norm / delta_norm)

            # Step 7 — Apply update and refresh sampler's cached log_psi
            # (stale cache causes wrong acceptance ratios -> chain freeze)
            self.ansatz.update_parameters(delta)
            self.sampler.refresh()

            # Step 7b — Chain rescue: if acceptance collapsed, the chain is stuck
            # on one configuration and all future gradients will be zero. Reset
            # to a random state and re-equilibrate to break the deadlock.
            if accept_rate < 0.01:
                self.sampler.reset_state()
                self.sampler.burn_in(self.n_burn)

            # Step 8 — Log metrics
            self.logger.record(
                epoch       = epoch,
                energy      = mean_E,
                energy_std  = std_E,
                acceptance_rate = accept_rate,
                grad_norm   = grad_norm
            )

            # Step 9 — Checkpoint
            if self.checkpointer and self.checkpoint_every > 0:
                if epoch % self.checkpoint_every == 0:
                    path = self.checkpointer.save(self.ansatz, epoch)

                # Also track and save the best (lowest energy) checkpoint
                if mean_E < self._best_energy:
                    self._best_energy = mean_E
                    self._best_epoch  = epoch
                    if self.checkpointer:
                        self.checkpointer.save(self.ansatz, epoch, tag='best')

            # Step 10 — Print epoch summary
            t_elapsed = time.perf_counter() - t_start
            if epoch % self.log_every == 0 or epoch == 1 or epoch == n_epochs:
                self._print_epoch(epoch, n_epochs, mean_E, std_E,
                                  accept_rate, grad_norm, t_elapsed, exact_energy)

            # Update tqdm postfix with live energy
            try:
                postfix = {'E': f'{mean_E:.5f}', 'accept': f'{accept_rate:.3f}'}
                if exact_energy is not None:
                    err = abs(mean_E - exact_energy)
                    postfix['err'] = f'{err:.2e}'
                epoch_iter.set_postfix(postfix)
            except AttributeError:
                pass    # not a tqdm object, skip

        # Training complete — print summary
        print(f"\nTraining complete.")
        print(f"  Final energy:   {self.logger.energies[-1]:.6f}")
        if exact_energy is not None:
            err = abs(self.logger.energies[-1] - exact_energy)
            print(f"  Exact energy:   {exact_energy:.6f}")
            print(f"  Error vs exact: {err:.6f}  ({100*err/abs(exact_energy):.3f}%)")
        if self._best_epoch >= 0:
            print(f"  Best energy:    {self._best_energy:.6f}  (epoch {self._best_epoch})")

        if self.results_dir is not None:
            log_path = os.path.join(self.results_dir, "training_history.npz")
            self.logger.save(log_path)
            print(f"  History saved:  {log_path}")
            self.checkpointer.save(self.ansatz, n_epochs, tag='final')

        return self.logger

    def _print_epoch(self, epoch: int, n_epochs: int,
                     mean_E: float, std_E: float,
                     accept_rate: float, grad_norm: float,
                     t_elapsed: float,
                     exact_energy: float = None) -> None:
        """Print a one-line epoch summary."""
        line = (
            f"Epoch {epoch:4d}/{n_epochs} | "
            f"E = {mean_E:+.6f} +/- {std_E:.6f} | "
            f"accept = {accept_rate:.3f} | "
            f"|grad| = {grad_norm:.2e} | "
            f"{t_elapsed:.2f}s"
        )
        if exact_energy is not None:
            err = abs(mean_E - exact_energy)
            line += f" | err = {err:.2e}"
        print(line)
