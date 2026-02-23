# src/utils.py
#
# Infrastructure utilities: logging, checkpointing, plotting, and config loading.
#
# The two key plots tell the whole story of the project:
#   1. Energy convergence: energy vs epoch with exact solution as a dashed line
#   2. Phase diagram: energy and magnetization vs Gamma/J showing the QPT
#
# matplotlib and tqdm are wrapped in try/except so the core physics code
# still works even if they're not installed.

import numpy as np
import os

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ============================================================
# Training Logger
# ============================================================

class Logger:
    """
    Records training metrics epoch by epoch.

    Tracked quantities:
      - energies: mean <E> per epoch (should decrease toward ground state)
      - energy_stds: MC statistical error (shrinks with more samples)
      - acceptance_rates: MCMC acceptance fraction (healthy: 0.3-0.7)
      - grad_norms: ||grad_E||_2 (signals training plateau when near zero)

    Uses Python lists internally (O(1) append) and converts to numpy on demand.
    """

    def __init__(self):
        self.epochs         = []
        self.energies       = []
        self.energy_stds    = []
        self.acceptance_rates = []
        self.grad_norms     = []

    def record(self, epoch: int, energy: float, energy_std: float,
               acceptance_rate: float, grad_norm: float) -> None:
        """Record metrics for one epoch."""
        self.epochs.append(epoch)
        self.energies.append(energy)
        self.energy_stds.append(energy_std)
        self.acceptance_rates.append(acceptance_rate)
        self.grad_norms.append(grad_norm)

    @property
    def history(self) -> dict:
        """Return all metrics as a dict of numpy arrays (ready for plotting)."""
        return {
            'epochs':           np.array(self.epochs),
            'energies':         np.array(self.energies),
            'energy_stds':      np.array(self.energy_stds),
            'acceptance_rates': np.array(self.acceptance_rates),
            'grad_norms':       np.array(self.grad_norms),
        }

    def save(self, path: str) -> None:
        """Save training history to a .npz file (numpy compressed archive)."""
        np.savez(path, **self.history)

    @classmethod
    def load(cls, path: str) -> 'Logger':
        """Load training history from a .npz file."""
        logger = cls()
        data = np.load(path)
        logger.epochs           = list(data['epochs'])
        logger.energies         = list(data['energies'])
        logger.energy_stds      = list(data['energy_stds'])
        logger.acceptance_rates = list(data['acceptance_rates'])
        logger.grad_norms       = list(data['grad_norms'])
        return logger

    def summary(self, last_n: int = 10) -> None:
        """Print a formatted summary of the last n recorded epochs."""
        if not self.energies:
            print("Logger: no data recorded yet.")
            return
        n = min(last_n, len(self.energies))
        print(f"Training summary (last {n} epochs):")
        for i in range(-n, 0):
            print(
                f"  Epoch {self.epochs[i]:4d} | "
                f"E = {self.energies[i]:+.6f} +/- {self.energy_stds[i]:.6f} | "
                f"accept = {self.acceptance_rates[i]:.3f} | "
                f"|grad| = {self.grad_norms[i]:.2e}"
            )


# ============================================================
# Parameter Checkpointing
# ============================================================

class Checkpointer:
    """
    Saves and loads ansatz parameters as .npy files.

    Serves three purposes:
      1. Fault tolerance: resume training if it crashes
      2. Best-model tracking: save the lowest-energy state
      3. Reproducibility: share trained wavefunctions
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, ansatz, epoch: int, tag: str = None) -> str:
        """
        Save ansatz parameters to a .npy file.

        Args:
            ansatz: Any Ansatz subclass with a .parameters property.
            epoch:  Current epoch (used in filename).
            tag:    Optional label like 'best' or 'final'.

        Returns:
            Path to the saved file.
        """
        suffix = tag if tag else f"epoch{epoch:04d}"
        path = os.path.join(self.save_dir, f"checkpoint_{suffix}.npy")
        np.save(path, ansatz.parameters)
        return path

    def load(self, ansatz, path: str) -> None:
        """Load parameters from a checkpoint into an ansatz."""
        saved_params = np.load(path)
        delta = saved_params - ansatz.parameters
        ansatz.update_parameters(delta)

    def latest(self) -> str | None:
        """Return path of the most recent checkpoint, or None."""
        files = [
            os.path.join(self.save_dir, f)
            for f in os.listdir(self.save_dir)
            if f.startswith("checkpoint_") and f.endswith(".npy")
        ]
        if not files:
            return None
        return max(files, key=os.path.getmtime)


# ============================================================
# Plotting
# ============================================================

def plot_energy_convergence(history: dict,
                            exact_energy: float = None,
                            n_spins: int = None,
                            save_path: str = None) -> None:
    """
    Plot energy vs training epoch with +/-1 std shaded band.

    Left panel: energy convergence toward the ground state, with optional
    exact solution as a dashed red line.
    Right panel: MCMC acceptance rate (healthy range 0.3-0.7 marked).

    Args:
        history:      Dict from Logger.history.
        exact_energy: Ground state energy from exact diag (optional).
        n_spins:      If given, normalizes energies to per-site values.
        save_path:    File path to save figure. None = plt.show().
    """
    if not HAS_MPL:
        print("matplotlib not installed — skipping plot_energy_convergence")
        return

    epochs   = history['epochs']
    energies = history['energies'].copy()
    stds     = history['energy_stds'].copy()

    y_label = "Energy"
    if n_spins is not None:
        energies /= n_spins
        stds     /= n_spins
        if exact_energy is not None:
            exact_energy = exact_energy / n_spins
        y_label = "Energy per site"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: energy convergence
    ax = axes[0]
    ax.plot(epochs, energies, color='royalblue', linewidth=1.5, label='VMC ⟨E⟩')
    ax.fill_between(epochs,
                    energies - stds,
                    energies + stds,
                    alpha=0.25, color='royalblue', label='±1 std (MC error)')

    if exact_energy is not None:
        ax.axhline(exact_energy, color='crimson', linestyle='--', linewidth=1.5,
                   label=f'Exact: {exact_energy:.4f}')

    ax.set_xlabel('Training Epoch')
    ax.set_ylabel(y_label)
    ax.set_title('VMC Energy Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: acceptance rate
    ax = axes[1]
    if 'acceptance_rates' in history:
        rates = history['acceptance_rates']
        ax.plot(epochs, rates, color='seagreen', linewidth=1.5, label='Acceptance rate')
        ax.axhline(0.3, color='orange', linestyle=':', linewidth=1.2, alpha=0.8)
        ax.axhline(0.7, color='orange', linestyle=':', linewidth=1.2, alpha=0.8,
                   label='Healthy range [0.3, 0.7]')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('MCMC Acceptance Rate')
        ax.set_title('Metropolis Acceptance Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_phase_diagram(gammas: np.ndarray,
                       energies: np.ndarray,
                       magnetizations: np.ndarray = None,
                       n_spins: int = None,
                       save_path: str = None) -> None:
    """
    Plot the TFIM quantum phase diagram: energy and magnetization vs Gamma/J.

    The 1D TFIM has a quantum phase transition at Gamma/J = 1.0:
      - Gamma/J < 1: ordered (ferromagnetic), high |<sigma^z>|
      - Gamma/J > 1: disordered (paramagnetic), |<sigma^z>| -> 0
    Seeing a clear transition near Gamma/J = 1 is the visual proof that the
    NQS correctly captures the quantum phase transition.

    Args:
        gammas:         Array of Gamma/J values scanned.
        energies:       Ground state energy at each Gamma/J.
        magnetizations: |<sigma^z>| at each Gamma/J (optional but recommended).
        n_spins:        If given, normalizes energies to per-site.
        save_path:      File path to save. None = plt.show().
    """
    if not HAS_MPL:
        print("matplotlib not installed — skipping plot_phase_diagram")
        return

    energies = np.array(energies)
    if n_spins is not None:
        energies = energies / n_spins

    n_panels = 2 if magnetizations is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    # Energy panel
    ax = axes[0]
    ax.plot(gammas, energies, 'o-', color='royalblue', markersize=5, linewidth=1.5)
    ax.axvline(1.0, color='crimson', linestyle='--', alpha=0.8,
               label='Critical point (Γ/J = 1)')
    y_label = 'Energy per site' if n_spins else 'Energy'
    ax.set_xlabel('Γ/J  (transverse field strength)')
    ax.set_ylabel(y_label)
    ax.set_title('Ground State Energy vs Transverse Field')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Magnetization panel
    if magnetizations is not None:
        ax = axes[1]
        ax.plot(gammas, np.abs(magnetizations), 'o-', color='seagreen',
                markersize=5, linewidth=1.5)
        ax.axvline(1.0, color='crimson', linestyle='--', alpha=0.8,
                   label='Critical point (Γ/J = 1)')
        ax.set_xlabel('Γ/J  (transverse field strength)')
        ax.set_ylabel('|⟨σᶻ⟩|  (magnetization)')
        ax.set_title('Order Parameter vs Transverse Field')
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ============================================================
# Configuration Loading
# ============================================================

def load_config(path: str) -> dict:
    """
    Load a YAML experiment config file and return as a dict.

    YAML supports inline comments, making experiment configs self-documenting
    and reproducible — someone can clone the repo, run the script with a
    config file, and get exactly our results.
    """
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required for config loading. Install with: pip install pyyaml"
        )
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ============================================================
# Progress Bar
# ============================================================

def make_progress_bar(iterable, desc: str = "", total: int = None, **kwargs):
    """Wrap an iterable with tqdm if available, else return it unchanged."""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total, **kwargs)
    return iterable
