# src/observables.py
#
# Physical observables computed from MCMC samples — beyond just the energy.
#
# Energy is one number. These observables characterize the full quantum phase:
# magnetization is the order parameter for the TFIM phase transition, correlations
# reveal the spatial structure of entanglement, and the structure factor shows
# ordering in momentum space. All are diagonal in the sigma^z basis, so they're
# cheap to compute — just averages over sample spin values, no extra wavefunction
# evaluations needed.

import numpy as np


# ============================================================
# Magnetization
# ============================================================

def magnetization(samples: np.ndarray) -> float:
    """
    Compute the mean z-magnetization per site: m = (1/N) sum_i <sigma_i^z>.

    This is the order parameter of the TFIM:
      - Ordered phase (Gamma/J < 1): |m| -> 1 (spins aligned)
      - Disordered phase (Gamma/J > 1): m -> 0 (quantum fluctuations win)

    Note: the true ground state has <sigma_i^z> = 0 by Z2 symmetry, but VMC
    with MCMC spontaneously breaks this symmetry (chain gets trapped near one
    sector), giving nonzero |m| in the ordered phase — which is the physically
    meaningful observable for finite systems.

    Args:
        samples: MCMC spin samples, shape (n_samples, n_spins), values +/-1.

    Returns:
        Mean magnetization per site (float in [-1, +1]).
    """
    return float(np.mean(samples))


def magnetization_per_site(samples: np.ndarray) -> np.ndarray:
    """
    Compute <sigma_i^z> for each site i separately.

    For translationally invariant systems, all sites should give the same
    value — large site-to-site variation suggests poor MCMC convergence.

    Returns:
        Array of shape (n_spins,).
    """
    return np.mean(samples, axis=0)


# ============================================================
# Spin-Spin Correlations
# ============================================================

def spin_spin_correlation(samples: np.ndarray, i: int, j: int) -> float:
    """
    Compute C(i,j) = <sigma_i^z sigma_j^z> between sites i and j.

    Measures how much spin i "knows" about spin j:
      - Ordered phase: C(i,j) -> m^2 != 0 at long range
      - Disordered phase: C(i,j) decays exponentially with |i-j|
      - Critical point (Gamma/J=1): power-law decay C ~ |i-j|^{-1/4}
    """
    return float(np.mean(samples[:, i] * samples[:, j]))


def correlation_matrix(samples: np.ndarray) -> np.ndarray:
    """
    Full correlation matrix C[i,j] = <sigma_i^z sigma_j^z> for all site pairs.

    Computed via matrix multiplication (O(K * N^2)) rather than per-pair loops.
    For a translationally invariant chain with PBC, C[i,j] depends only on |i-j|.

    Returns:
        Array of shape (n_spins, n_spins).
    """
    samples_f = samples.astype(float)
    return (samples_f.T @ samples_f) / len(samples)


def connected_correlation_matrix(samples: np.ndarray) -> np.ndarray:
    """
    Connected correlation: C_conn[i,j] = <sigma_i sigma_j> - <sigma_i><sigma_j>.

    Measures purely quantum correlations beyond classical spin alignment.
    In the disordered phase (where <sigma_i> = 0), C_conn equals the full C.
    In the ordered phase, C_conn decays even though C -> m^2 != 0.

    Returns:
        Array of shape (n_spins, n_spins).
    """
    C = correlation_matrix(samples)
    m = magnetization_per_site(samples)
    return C - np.outer(m, m)


# ============================================================
# Structure Factor
# ============================================================

def structure_factor(samples: np.ndarray) -> tuple:
    """
    Compute the static structure factor S(k) — Fourier transform of correlations.

        S(k) = (1/N) sum_{i,j} <sigma_i sigma_j> exp(ik(i-j))

    Efficiently computed via FFT using the convolution theorem:
        S(k) = (1/N) <|FFT[spins]|^2>
    This is O(K * N log N) instead of O(K * N^2) for the direct double sum.

    Peak interpretation:
      - k=0 peak: ferromagnetic order (TFIM ordered phase)
      - k=pi peak: antiferromagnetic order (Heisenberg ground state)
      - Flat S(k): disordered phase (no long-range order)

    Sum rule: sum_k S(k) = N (Parseval's theorem for +/-1 spins).

    Args:
        samples: MCMC spin samples, shape (n_samples, n_spins), values +/-1.

    Returns:
        (k_values, S_k): momentum grid k_n = 2*pi*n/N and structure factor,
                         both arrays of shape (n_spins,).
    """
    n_samples, n_spins = samples.shape
    samples_f = samples.astype(float)

    # FFT each spin config along the spatial axis
    spin_k = np.fft.fft(samples_f, axis=1)

    # S(k) = (1/N) * <|FFT|^2> averaged over samples
    S_k = np.mean(np.abs(spin_k) ** 2, axis=0) / n_spins

    k_values = 2 * np.pi * np.arange(n_spins) / n_spins

    return k_values, np.real(S_k)


# ============================================================
# Summary
# ============================================================

def compute_all_observables(samples: np.ndarray) -> dict:
    """
    Compute all observables at once. Convenience wrapper for phase diagram scans.

    Returns dict with keys: 'magnetization', 'magnetization_per_site',
    'correlation_matrix', 'connected_correlation', 'k_values', 'structure_factor'.
    """
    k_values, S_k = structure_factor(samples)

    return {
        'magnetization':          magnetization(samples),
        'magnetization_per_site': magnetization_per_site(samples),
        'correlation_matrix':     correlation_matrix(samples),
        'connected_correlation':  connected_correlation_matrix(samples),
        'k_values':               k_values,
        'structure_factor':       S_k,
    }
