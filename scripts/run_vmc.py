#!/usr/bin/env python3
# scripts/run_vmc.py
#
# ============================================================
# CLI ENTRY POINT — Run a single VMC experiment from a YAML config
# ============================================================
#
# USAGE:
#   python scripts/run_vmc.py configs/ising_small.yaml
#   python scripts/run_vmc.py configs/heisenberg.yaml
#   python scripts/run_vmc.py configs/ising_large.yaml
#
# WHAT THIS SCRIPT DOES:
#   1. Loads the YAML config to get all hyperparameters
#   2. Constructs the Hamiltonian, RBM ansatz, sampler, and optimizer
#   3. Optionally runs exact diagonalization to get the reference energy
#   4. Runs the VMC training loop (with tqdm progress bar)
#   5. Plots and saves the energy convergence curve
#   6. Computes and prints final physical observables
#   7. Saves everything to the results directory specified in the config
#
# WHY A YAML CONFIG (not just command-line args)?
#   Reproducibility. A config file records EVERY hyperparameter in one place.
#   Someone can clone the repo, run this script with the same config, and get
#   exactly our results. This is essential for scientific credibility.
#
# ============================================================

import argparse
import sys
import os
import numpy as np

# Add project root to path so we can import src/ regardless of where
# the script is called from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.hamiltonians import IsingHamiltonian, HeisenbergHamiltonian
from src.ansatz import RBM
from src.sampler import MetropolisSampler
from src.optimizer import StochasticReconfiguration, Adam, SGD
from src.trainer import VMCTrainer
from src.exact import exact_ground_state_energy
from src.observables import compute_all_observables
from src.utils import load_config, plot_energy_convergence


# ============================================================
# SECTION: Object Builders (Config → Python Objects)
# ============================================================

def build_hamiltonian(cfg: dict):
    """Construct the Hamiltonian from the 'system' section of the config."""
    s = cfg['system']
    if s['hamiltonian'] == 'ising':
        return IsingHamiltonian(n_spins=s['n_spins'], J=s['J'], gamma=s['gamma'])
    elif s['hamiltonian'] == 'heisenberg':
        return HeisenbergHamiltonian(n_spins=s['n_spins'], J=s['J'])
    else:
        raise ValueError(
            f"Unknown hamiltonian '{s['hamiltonian']}'. "
            f"Choose 'ising' or 'heisenberg'."
        )


def build_ansatz(cfg: dict):
    """Construct the neural network ansatz from the 'ansatz' section."""
    a = cfg['ansatz']
    n = cfg['system']['n_spins']
    if a['type'] == 'rbm':
        return RBM(n_spins=n, alpha=a['alpha'], seed=a.get('seed', 42))
    else:
        raise ValueError(
            f"Unknown ansatz type '{a['type']}'. "
            f"Only 'rbm' is implemented (cnn is the stretch goal)."
        )


def build_optimizer(cfg: dict):
    """Construct the optimizer from the 'optimizer' section."""
    o = cfg['optimizer']
    lr = o['learning_rate']
    if o['type'] == 'sr':
        return StochasticReconfiguration(
            learning_rate=lr,
            epsilon=o.get('epsilon', 0.01)
        )
    elif o['type'] == 'adam':
        return Adam(learning_rate=lr)
    elif o['type'] == 'sgd':
        return SGD(learning_rate=lr)
    else:
        raise ValueError(
            f"Unknown optimizer '{o['type']}'. "
            f"Choose 'sr', 'adam', or 'sgd'."
        )


# ============================================================
# SECTION: Experiment Summary Printer
# ============================================================

def print_experiment_summary(cfg: dict, ansatz) -> None:
    """Print a human-readable summary of the experiment configuration."""
    s = cfg['system']
    a = cfg['ansatz']
    o = cfg['optimizer']
    t = cfg['training']

    print("=" * 62)
    print("  Neural Quantum States — VMC Experiment")
    print("=" * 62)
    print(f"  System:      {s['hamiltonian'].upper()}  |  N = {s['n_spins']} spins")
    if s['hamiltonian'] == 'ising':
        print(f"               J = {s['J']},  Gamma = {s['gamma']}  "
              f"(Gamma/J = {s['gamma']/s['J']:.2f})")
    else:
        print(f"               J = {s['J']}")
    print(f"  Ansatz:      {a['type'].upper()}  |  alpha = {a['alpha']}  "
          f"|  {ansatz.n_params} parameters")
    print(f"  Optimizer:   {o['type'].upper()}  |  lr = {o['learning_rate']}")
    print(f"  Training:    {t['n_epochs']} epochs  |  "
          f"{cfg['sampler']['n_samples']} samples/epoch")
    print(f"  Output:      {cfg.get('output', {}).get('results_dir', 'None')}")
    print("=" * 62)
    print()


# ============================================================
# SECTION: Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run a VMC experiment from a YAML config file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_vmc.py configs/ising_small.yaml
  python scripts/run_vmc.py configs/heisenberg.yaml
        """
    )
    parser.add_argument(
        'config',
        help='Path to a YAML config file (e.g. configs/ising_small.yaml)'
    )
    args = parser.parse_args()

    # ---- Load config ----------------------------------------------------------
    cfg = load_config(args.config)
    out = cfg.get('output', {})
    results_dir = out.get('results_dir', 'results/default/')

    # ---- Build all objects ----------------------------------------------------
    hamiltonian = build_hamiltonian(cfg)
    ansatz      = build_ansatz(cfg)
    sampler     = MetropolisSampler(
        ansatz       = ansatz,
        n_spins      = hamiltonian.n_spins,
        seed         = cfg['ansatz'].get('seed', 42),
        use_exchange = cfg['sampler'].get('use_exchange', False),
    )
    optimizer   = build_optimizer(cfg)

    # ---- Print experiment summary --------------------------------------------
    print_experiment_summary(cfg, ansatz)

    # ---- Optional: Exact Diagonalization ------------------------------------
    # Run before training so we can display the target energy during training.
    exact_energy = None
    if out.get('run_exact', True) and hamiltonian.n_spins <= 20:
        print(f"Running exact diagonalization (N={hamiltonian.n_spins})...")
        exact_energy = exact_ground_state_energy(hamiltonian)
        print(f"  Exact ground state energy:  {exact_energy:.6f}")
        print(f"  Exact energy per site:      {exact_energy / hamiltonian.n_spins:.6f}")
        print()

    # ---- Build Trainer -------------------------------------------------------
    samp  = cfg['sampler']
    train = cfg['training']

    trainer = VMCTrainer(
        ansatz           = ansatz,
        hamiltonian      = hamiltonian,
        sampler          = sampler,
        optimizer        = optimizer,
        n_samples        = samp['n_samples'],
        n_burn           = samp['n_burn'],
        sweep_size       = samp.get('sweep_size', None),
        checkpoint_every = train.get('checkpoint_every', 50),
        log_every        = train.get('log_every', 10),
        results_dir      = results_dir,
    )

    # ---- Train ---------------------------------------------------------------
    logger = trainer.train(
        n_epochs     = train['n_epochs'],
        exact_energy = exact_energy,
    )

    # ---- Plot energy convergence ---------------------------------------------
    plot_path = os.path.join(results_dir, 'energy_convergence.png')
    plot_energy_convergence(
        history      = logger.history,
        exact_energy = exact_energy,
        n_spins      = hamiltonian.n_spins,
        save_path    = plot_path,
    )

    # ---- Final comparison with exact ----------------------------------------
    if exact_energy is not None:
        final_E = logger.energies[-1]
        error   = abs(final_E - exact_energy)
        rel_err = 100 * error / abs(exact_energy)
        print(f"\nAccuracy vs exact diagonalization:")
        print(f"  VMC final energy:   {final_E:.6f}")
        print(f"  Exact energy:       {exact_energy:.6f}")
        print(f"  Absolute error:     {error:.6f}")
        print(f"  Relative error:     {rel_err:.4f}%")

    # ---- Observables ---------------------------------------------------------
    if out.get('compute_observables', True):
        print("\nComputing physical observables from trained wavefunction...")
        # Thermalize the chain to the final trained wavefunction
        sampler.burn_in(200)
        obs_samples = sampler.sample(n_samples=1000)
        obs = compute_all_observables(obs_samples)

        N = hamiltonian.n_spins
        print(f"  |magnetization|       = {abs(obs['magnetization']):.4f}")
        print(f"  C(0, N/2) = C(0,{N//2}) = {obs['correlation_matrix'][0, N//2]:.4f}")
        print(f"  S(k=0)                = {obs['structure_factor'][0]:.4f}")
        print(f"  S(k=pi) ~ S(k={obs['k_values'][N//2]:.2f})    = "
              f"{obs['structure_factor'][N//2]:.4f}")

        # Save observables to disk
        obs_path = os.path.join(results_dir, 'observables.npz')
        np.savez(obs_path,
                 magnetization        = obs['magnetization'],
                 correlation_matrix   = obs['correlation_matrix'],
                 connected_correlation= obs['connected_correlation'],
                 k_values             = obs['k_values'],
                 structure_factor     = obs['structure_factor'])
        print(f"\n  Observables saved: {obs_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
