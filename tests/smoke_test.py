# tests/smoke_test.py
#
# Quick sanity check: verifies that all modules built so far
# import correctly, produce outputs of the right shape, and
# don't crash. This is not a correctness test — just checking
# that the plumbing works end to end.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# ── 1. Hamiltonians ────────────────────────────────────────────────────────────
print("=" * 60)
print("Testing hamiltonians...")

from src.hamiltonians import IsingHamiltonian, HeisenbergHamiltonian

N = 6  # small chain for fast testing

ising = IsingHamiltonian(n_spins=N, J=1.0, gamma=1.0)
heis  = HeisenbergHamiltonian(n_spins=N, J=1.0)

assert ising.n_spins == N, "IsingHamiltonian.n_spins wrong"
assert heis.n_spins  == N, "HeisenbergHamiltonian.n_spins wrong"
print(f"  IsingHamiltonian(N={N}, J=1.0, gamma=1.0) ... OK")
print(f"  HeisenbergHamiltonian(N={N}, J=1.0)       ... OK")

# ── 2. RBM ────────────────────────────────────────────────────────────────────
print("\nTesting RBM ansatz...")

from src.ansatz import RBM

rbm = RBM(n_spins=N, alpha=2, seed=0)
print(f"  {rbm}")

# Check parameter count: N + alpha*N + N*(alpha*N) = N(1 + alpha + alpha*N)
expected_params = N + rbm.n_hidden + N * rbm.n_hidden
assert rbm.n_params == expected_params, (
    f"n_params mismatch: got {rbm.n_params}, expected {expected_params}"
)
print(f"  n_params = {rbm.n_params} (expected {expected_params}) ... OK")

# Check log_psi returns a scalar
spins = np.array([1, -1, 1, 1, -1, 1])
log_p = rbm.log_psi(spins)
assert np.isscalar(log_p) or log_p.shape == (), f"log_psi should be scalar, got shape {np.array(log_p).shape}"
print(f"  log_psi({spins}) = {log_p:.6f} ... OK")

# Check grad_log_psi returns correct shape
grad = rbm.grad_log_psi(spins)
assert grad.shape == (rbm.n_params,), (
    f"grad shape mismatch: got {grad.shape}, expected ({rbm.n_params},)"
)
print(f"  grad_log_psi shape = {grad.shape} ... OK")

# Check parameter update works
params_before = rbm.parameters.copy()
delta = np.ones(rbm.n_params) * 0.001
rbm.update_parameters(delta)
params_after = rbm.parameters
assert np.allclose(params_after, params_before + delta), "update_parameters failed"
rbm.update_parameters(-delta)  # revert
print(f"  update_parameters ... OK")

# ── 3. Local energy ───────────────────────────────────────────────────────────
print("\nTesting local energy computation...")

e_ising = ising.local_energy(spins, rbm.log_psi)
e_heis  = heis.local_energy(spins, rbm.log_psi)

assert np.isfinite(e_ising), f"Ising local energy is not finite: {e_ising}"
assert np.isfinite(e_heis),  f"Heisenberg local energy is not finite: {e_heis}"
print(f"  Ising local energy     = {e_ising:.6f} ... OK")
print(f"  Heisenberg local energy = {e_heis:.6f} ... OK")

# ── 4. Sampler ────────────────────────────────────────────────────────────────
print("\nTesting MetropolisSampler...")

from src.sampler import MetropolisSampler

sampler = MetropolisSampler(ansatz=rbm, n_spins=N, seed=42)

# Burn in
sampler.burn_in(n_steps=100)
print(f"  burn_in(100) ... OK")

# Collect samples
n_samples = 50
samples = sampler.sample(n_samples=n_samples)
assert samples.shape == (n_samples, N), (
    f"samples shape mismatch: got {samples.shape}, expected ({n_samples}, {N})"
)
assert set(np.unique(samples)).issubset({-1, 1}), "samples contain values outside {-1, +1}"
print(f"  sample(n_samples={n_samples}) -> shape {samples.shape} ... OK")
print(f"  acceptance_rate = {sampler.acceptance_rate:.3f} ... OK")

# ── 5. End-to-end energy estimate ─────────────────────────────────────────────
print("\nTesting end-to-end energy estimate...")

energies = np.array([
    ising.local_energy(s, rbm.log_psi) for s in samples
])
mean_energy = np.mean(np.real(energies))
energy_std  = np.std(np.real(energies))

assert np.isfinite(mean_energy), f"Mean energy is not finite: {mean_energy}"
print(f"  <E> over {n_samples} samples = {mean_energy:.6f} ± {energy_std:.6f} ... OK")
print(f"  (per site: {mean_energy / N:.6f})")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("All smoke tests passed.")
print("=" * 60)
