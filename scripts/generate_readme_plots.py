#!/usr/bin/env python
"""
Generate all plots referenced in README.md from saved results + quick live training.

Produces:
  results/energy_convergence_both.png   (~1 sec, from saved data)
  results/phase_diagram_notebook.png    (~1 sec, from saved data)
  results/correlations_structure.png    (~1 sec, from saved data)
  results/optimizer_comparison.png      (~2-3 min, trains 3 small RBMs)
"""

import sys, os, io
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.hamiltonians import IsingHamiltonian
from src.ansatz import RBM
from src.sampler import MetropolisSampler
from src.optimizer import StochasticReconfiguration, Adam, SGD
from src.trainer import VMCTrainer
from src.exact import exact_ground_state_energy

plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')


# ============================================================
# 1. Energy Convergence (Ising + Heisenberg side-by-side)
# ============================================================
def plot_energy_convergence():
    print('[1/4] Energy convergence plot...', end=' ', flush=True)
    ising_hist = np.load(os.path.join(RESULTS, 'ising_small', 'training_history.npz'))
    heisen_hist = np.load(os.path.join(RESULTS, 'heisenberg', 'training_history.npz'))

    N = 8
    ising_exact = exact_ground_state_energy(IsingHamiltonian(n_spins=N, J=1.0, gamma=1.0))
    from src.hamiltonians import HeisenbergHamiltonian
    heisen_exact = exact_ground_state_energy(HeisenbergHamiltonian(n_spins=N, J=1.0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, hist, exact, title in [
        (axes[0], ising_hist, ising_exact, 'Ising (N=8, $\\Gamma/J=1$)'),
        (axes[1], heisen_hist, heisen_exact, 'Heisenberg (N=8, $J=1$)'),
    ]:
        epochs = hist['epochs']
        E = hist['energies'] / N
        std = hist['energy_stds'] / N
        exact_per_site = exact / N

        ax.plot(epochs, E, color='royalblue', lw=1.5, label='VMC $\\langle E \\rangle$')
        ax.fill_between(epochs, E - std, E + std, alpha=0.2, color='royalblue', label='$\\pm 1\\sigma$')
        ax.axhline(exact_per_site, color='crimson', ls='--', lw=1.5,
                   label=f'Exact: {exact_per_site:.4f}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Energy per site')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'energy_convergence_both.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('done.')


# ============================================================
# 2. Phase Diagram
# ============================================================
def plot_phase_diagram():
    print('[2/4] Phase diagram plot...', end=' ', flush=True)
    pd = np.load(os.path.join(RESULTS, 'phase_diagram', 'phase_diagram.npz'))
    gammas = pd['gammas']
    vmc_E = pd['vmc_energies']
    vmc_mag = pd['vmc_magnetizations']
    exact_E_pd = pd['exact_energies'] if len(pd['exact_energies']) > 0 else None
    N_pd = int(pd['n_spins'])
    J_pd = float(pd['J'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    gamma_ratio = gammas / J_pd

    # Energy panel
    ax = axes[0]
    ax.plot(gamma_ratio, vmc_E / N_pd, 'o-', color='royalblue', ms=5, lw=1.5, label='VMC')
    if exact_E_pd is not None and len(exact_E_pd) == len(gammas):
        ax.plot(gamma_ratio, exact_E_pd / N_pd, 's--', color='crimson', ms=4, lw=1.2, label='Exact')
    ax.axvline(1.0, color='gray', ls=':', alpha=0.7, label='$\\Gamma/J = 1$ (QPT)')
    ax.set_xlabel('$\\Gamma/J$')
    ax.set_ylabel('Energy per site')
    ax.set_title('Ground State Energy')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Magnetization panel
    ax = axes[1]
    ax.plot(gamma_ratio, np.abs(vmc_mag), 'o-', color='seagreen', ms=5, lw=1.5)
    ax.axvline(1.0, color='gray', ls=':', alpha=0.7, label='$\\Gamma/J = 1$ (QPT)')
    ax.set_xlabel('$\\Gamma/J$')
    ax.set_ylabel('$|\\langle \\sigma^z \\rangle|$')
    ax.set_title('Order Parameter (Magnetization)')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'phase_diagram_notebook.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('done.')


# ============================================================
# 3. Correlations + Structure Factor
# ============================================================
def plot_correlations():
    print('[3/4] Correlations + structure factor plot...', end=' ', flush=True)
    ising_obs = np.load(os.path.join(RESULTS, 'ising_small', 'observables.npz'))
    heisen_obs = np.load(os.path.join(RESULTS, 'heisenberg', 'observables.npz'))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, obs, title in [
        (axes[0, 0], ising_obs, 'Ising ($\\Gamma/J = 1$)'),
        (axes[0, 1], heisen_obs, 'Heisenberg ($J = 1$)'),
    ]:
        C = obs['correlation_matrix']
        vmax = max(abs(C.min()), abs(C.max()), 1e-10)
        im = ax.imshow(C, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_title(f'Correlation $C_{{ij}}$ \u2014 {title}')
        ax.set_xlabel('Site $j$')
        ax.set_ylabel('Site $i$')
        plt.colorbar(im, ax=ax, shrink=0.8)

    for ax, obs, title in [
        (axes[1, 0], ising_obs, 'Ising ($\\Gamma/J = 1$)'),
        (axes[1, 1], heisen_obs, 'Heisenberg ($J = 1$)'),
    ]:
        k = obs['k_values']
        Sk = obs['structure_factor']
        ax.bar(k, Sk, width=0.4, color='teal', alpha=0.8)
        ax.set_xlabel('$k$')
        ax.set_ylabel('$S(k)$')
        ax.set_title(f'Structure Factor \u2014 {title}')
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                      ['0', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'correlations_structure.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('done.')


# ============================================================
# 4. Optimizer Comparison (SR vs Adam vs SGD)
# ============================================================
def plot_optimizer_comparison():
    print('[4/4] Optimizer comparison (training 3 RBMs)...', flush=True)
    N = 8
    n_epochs = 150

    ham = IsingHamiltonian(n_spins=N, J=1.0, gamma=1.0)
    exact_E = exact_ground_state_energy(ham)

    optimizers = {
        'SR':   StochasticReconfiguration(learning_rate=0.01, epsilon=0.01),
        'Adam': Adam(learning_rate=0.005),
        'SGD':  SGD(learning_rate=0.01),
    }

    results = {}
    for name, opt in optimizers.items():
        rbm = RBM(n_spins=N, alpha=2, seed=42)
        sampler = MetropolisSampler(ansatz=rbm, n_spins=N, seed=42)
        trainer = VMCTrainer(
            ansatz=rbm, hamiltonian=ham, sampler=sampler, optimizer=opt,
            n_samples=500, n_burn=200, log_every=9999, checkpoint_every=0,
        )

        with redirect_stdout(io.StringIO()):
            logger = trainer.train(n_epochs=n_epochs)

        results[name] = logger.history
        final_E = logger.energies[-1]
        err = abs(final_E - exact_E) / abs(exact_E) * 100
        print(f'  {name:5s}: final E = {final_E:.4f}, error = {err:.3f}%')

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'SR': 'royalblue', 'Adam': 'darkorange', 'SGD': 'seagreen'}

    for name, hist in results.items():
        E = hist['energies'] / N
        ax.plot(hist['epochs'], E, lw=1.5, label=name, color=colors[name])

    ax.axhline(exact_E / N, color='crimson', ls='--', lw=1.5, label=f'Exact: {exact_E/N:.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Energy per site')
    ax.set_title('Optimizer Comparison: SR vs Adam vs SGD (Ising N=8, $\\Gamma/J=1$)')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'optimizer_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  done.')


if __name__ == '__main__':
    print('=' * 50)
    print('  Generating README plots')
    print('=' * 50)
    plot_energy_convergence()
    plot_phase_diagram()
    plot_correlations()
    plot_optimizer_comparison()
    print('=' * 50)
    print('  All plots saved to results/')
    print('=' * 50)
