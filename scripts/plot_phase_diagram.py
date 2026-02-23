#!/usr/bin/env python3
# scripts/plot_phase_diagram.py
#
# ============================================================
# TFIM QUANTUM PHASE DIAGRAM — The Showcase Result
# ============================================================
#
# USAGE:
#   python scripts/plot_phase_diagram.py                    # default settings
#   python scripts/plot_phase_diagram.py --n_spins 8 --n_epochs 200
#   python scripts/plot_phase_diagram.py --n_gammas 20 --save_dir results/pd_N12 --n_spins 12
#
# WHAT THIS SCRIPT DOES:
#   Scans Gamma/J (the transverse field strength) from gamma_min to gamma_max,
#   runs a VMC training run at each point, and plots:
#     1. Ground state energy per site vs Gamma/J  (VMC + exact)
#     2. Magnetization |⟨σᶻ⟩| per site vs Gamma/J  (the order parameter)
#
# THE PHYSICS WE'RE OBSERVING:
#   The TFIM undergoes a quantum phase transition at Gamma/J = 1.0.
#   By scanning Gamma/J and measuring the magnetization, we can SEE this
#   transition: |m| is large for Gamma/J < 1 and drops to zero for Gamma/J > 1.
#   This is the visual proof that the NQS correctly captures the quantum phase.
#
# ADIABATIC CONTINUATION (WARM STARTING):
#   Instead of training each RBM from random initialization, we warm-start
#   each Gamma/J point using the trained parameters from the previous point.
#   This works because the ground state wavefunction changes CONTINUOUSLY as
#   Gamma/J varies — the warm-started RBM is already close to the answer,
#   so fewer epochs are needed. This is "adiabatic continuation," a standard
#   technique in computational condensed matter physics.
#
# ============================================================

import argparse
import sys
import os
import io
import numpy as np
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.hamiltonians import IsingHamiltonian
from src.ansatz import RBM
from src.sampler import MetropolisSampler
from src.optimizer import StochasticReconfiguration
from src.trainer import VMCTrainer
from src.exact import exact_ground_state_energy
from src.observables import magnetization
from src.utils import plot_phase_diagram


# ============================================================
# SECTION: Single-Point VMC
# ============================================================

def run_vmc_at_gamma(n_spins: int, alpha: int, J: float, gamma: float,
                     n_epochs: int, n_samples: int, seed: int,
                     warm_start_params: np.ndarray = None) -> tuple:
    """
    Run a VMC training run for the TFIM at a single Gamma/J value.

    PHYSICS: At each Gamma/J point, we train an RBM from scratch (or from
    the previous point's parameters via warm-starting) and measure the
    ground state energy and magnetization.

    ADIABATIC CONTINUATION: When warm_start_params is given, we load those
    parameters into the RBM before training begins. The RBM then starts from
    a wavefunction that's already a good approximation of the nearby ground
    state, and converges faster.

    Args:
        n_spins:           Number of spins.
        alpha:             RBM hidden unit density.
        J:                 Ising coupling.
        gamma:             Transverse field. The Gamma/J ratio determines the phase.
        n_epochs:          Training epochs at this Gamma/J value.
        n_samples:         MCMC samples per epoch.
        seed:              Random seed.
        warm_start_params: Optional parameter array from the previous Gamma/J point.

    Returns:
        energy:      Final VMC energy (float).
        mag:         |⟨σᶻ⟩| per site at the end of training (float).
        final_params: Trained RBM parameters (to pass as warm_start for the next point).
    """
    hamiltonian = IsingHamiltonian(n_spins=n_spins, J=J, gamma=gamma)
    rbm         = RBM(n_spins=n_spins, alpha=alpha, seed=seed)

    # Warm-start: load previous gamma's trained parameters
    if warm_start_params is not None:
        delta = warm_start_params - rbm.parameters
        rbm.update_parameters(delta)

    sampler   = MetropolisSampler(ansatz=rbm, n_spins=n_spins, seed=seed)
    # Fresh SR optimizer at each gamma (stateless, so warm-starting is just the RBM)
    optimizer = StochasticReconfiguration(learning_rate=0.01, epsilon=0.01)

    trainer = VMCTrainer(
        ansatz           = rbm,
        hamiltonian      = hamiltonian,
        sampler          = sampler,
        optimizer        = optimizer,
        n_samples        = n_samples,
        n_burn           = 100,          # shorter burn-in per point (warm-starting helps)
        log_every        = n_epochs + 1, # only print first and last epoch
        checkpoint_every = 0,            # no checkpointing during the scan
        results_dir      = None,
    )

    # Suppress training's stdout output (burn-in messages, epoch summaries)
    # The tqdm progress bar is on stderr, so it's still visible.
    with redirect_stdout(io.StringIO()):
        logger = trainer.train(n_epochs=n_epochs)

    # Measure the magnetization from the final trained wavefunction
    sampler.burn_in(50)
    obs_samples = sampler.sample(n_samples=300)
    mag = abs(magnetization(obs_samples))

    final_energy = logger.energies[-1]
    return final_energy, mag, rbm.parameters.copy()


# ============================================================
# SECTION: Main Phase Diagram Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Scan Gamma/J and plot the TFIM quantum phase diagram.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plot_phase_diagram.py
  python scripts/plot_phase_diagram.py --n_spins 12 --n_epochs 200 --n_gammas 20
  python scripts/plot_phase_diagram.py --no_exact --n_spins 20
        """
    )
    parser.add_argument('--n_spins',   type=int,   default=8,
                        help='Number of spins (default: 8)')
    parser.add_argument('--alpha',     type=int,   default=2,
                        help='RBM hidden unit density (default: 2)')
    parser.add_argument('--n_epochs',  type=int,   default=150,
                        help='VMC training epochs per Gamma/J point (default: 150)')
    parser.add_argument('--n_gammas',  type=int,   default=15,
                        help='Number of Gamma/J values to scan (default: 15)')
    parser.add_argument('--gamma_min', type=float, default=0.2,
                        help='Minimum Gamma/J (default: 0.2)')
    parser.add_argument('--gamma_max', type=float, default=3.0,
                        help='Maximum Gamma/J (default: 3.0)')
    parser.add_argument('--n_samples', type=int,   default=500,
                        help='MCMC samples per epoch (default: 500)')
    parser.add_argument('--J',         type=float, default=1.0,
                        help='Ising coupling strength (default: 1.0)')
    parser.add_argument('--seed',      type=int,   default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--save_dir',  default='results/phase_diagram/',
                        help='Directory to save results (default: results/phase_diagram/)')
    parser.add_argument('--no_exact',  action='store_true',
                        help='Skip exact diagonalization (faster but no reference line)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    gammas = np.linspace(args.gamma_min, args.gamma_max, args.n_gammas)

    # ---- Print experiment header -------------------------------------------
    print("=" * 62)
    print("  TFIM Quantum Phase Diagram Scan")
    print("=" * 62)
    print(f"  N = {args.n_spins} spins  |  alpha = {args.alpha}  |  J = {args.J}")
    print(f"  Gamma/J range: {args.gamma_min:.2f} to {args.gamma_max:.2f}  "
          f"({args.n_gammas} points)")
    print(f"  Epochs per point: {args.n_epochs}  |  Samples: {args.n_samples}")
    print(f"  Adiabatic continuation: ON (warm-starting between gamma points)")
    print(f"  Exact diagonalization:  {'OFF' if args.no_exact else 'ON'}")
    print(f"  Save directory: {args.save_dir}")
    print("=" * 62)
    print()

    # ---- Scan loop ---------------------------------------------------------
    vmc_energies       = []
    vmc_magnetizations = []
    exact_energies     = []

    warm_params = None    # warm-start parameter array (updated after each gamma point)

    for idx, gamma in enumerate(gammas):
        gamma_ratio = gamma / args.J
        print(f"[{idx+1:2d}/{args.n_gammas}]  Gamma/J = {gamma_ratio:.3f}", end='  ')

        # Exact diagonalization reference
        exact_e = None
        if not args.no_exact and args.n_spins <= 20:
            ham_exact = IsingHamiltonian(n_spins=args.n_spins, J=args.J, gamma=gamma)
            exact_e = exact_ground_state_energy(ham_exact)
            exact_energies.append(exact_e)

        # VMC training (training output suppressed; tqdm bar still visible)
        vmc_e, mag, warm_params = run_vmc_at_gamma(
            n_spins          = args.n_spins,
            alpha            = args.alpha,
            J                = args.J,
            gamma            = gamma,
            n_epochs         = args.n_epochs,
            n_samples        = args.n_samples,
            seed             = args.seed,
            warm_start_params = warm_params,
        )

        vmc_energies.append(vmc_e)
        vmc_magnetizations.append(mag)

        # Print one-line result summary
        per_site = vmc_e / args.n_spins
        summary = f"E/site = {per_site:.4f}  |  |m| = {mag:.4f}"
        if exact_e is not None:
            err = abs(vmc_e - exact_e)
            summary += f"  |  err = {err:.4f}"
        print(summary)

    print()
    print("Scan complete.")

    # ---- Save raw data ----------------------------------------------------
    data_path = os.path.join(args.save_dir, 'phase_diagram.npz')
    np.savez(
        data_path,
        gammas             = gammas,
        vmc_energies       = np.array(vmc_energies),
        vmc_magnetizations = np.array(vmc_magnetizations),
        exact_energies     = np.array(exact_energies) if exact_energies else np.array([]),
        n_spins            = args.n_spins,
        J                  = args.J,
    )
    print(f"Raw data saved: {data_path}")

    # ---- Plot the phase diagram -------------------------------------------
    plot_path = os.path.join(args.save_dir, 'phase_diagram.png')
    plot_phase_diagram(
        gammas          = gammas / args.J,          # normalize to Gamma/J ratios
        energies        = vmc_energies,
        magnetizations  = vmc_magnetizations,
        n_spins         = args.n_spins,
        save_path       = plot_path,
    )

    # ---- Print summary table ---------------------------------------------
    print()
    print(f"{'Gamma/J':>8} | {'E/site (VMC)':>14} | {'|m| (VMC)':>10}", end='')
    if exact_energies:
        print(f" | {'E/site (exact)':>14} | {'error':>8}")
    else:
        print()
    print("-" * (50 + (26 if exact_energies else 0)))
    for i, gamma in enumerate(gammas):
        line = (f"{gamma/args.J:>8.3f} | "
                f"{vmc_energies[i]/args.n_spins:>14.6f} | "
                f"{vmc_magnetizations[i]:>10.4f}")
        if exact_energies:
            exact_per_site = exact_energies[i] / args.n_spins
            err = abs(vmc_energies[i] - exact_energies[i])
            line += f" | {exact_per_site:>14.6f} | {err:>8.4f}"
        print(line)

    print(f"\nPhase diagram plot saved: {plot_path}")


if __name__ == '__main__':
    main()
