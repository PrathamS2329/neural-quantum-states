# CLAUDE.md — Project Context for Claude Code Sessions

This file exists so that Claude Code is never contextless at the start of a new session.
Read this entire file before doing anything. Every decision recorded here was made
deliberately — do not second-guess the structure without asking the user first.

---

## CRITICAL RULE — KEEPING THIS FILE UP TO DATE

**Anytime there is a major change to:**
- The project structure (files added, removed, renamed, reorganized)
- The project vision or goals
- A design decision (why we chose one approach over another)
- The build order or current status of any module

**Claude MUST update this CLAUDE.md file immediately after the change.**

This is non-negotiable. Pratham will not remember every decision made across sessions.
This file is the single source of truth. If it's not in here, it's lost.

---

## Who is building this and why

**Builder:** Pratham (student, ~1 week timeline, first serious GitHub project from scratch)
**Audience:** Recruiters and practitioners in ML, Physics, and Software Engineering — all three equally
**Goal:** A polished, impressive GitHub project that demonstrates real depth at the intersection
of machine learning, quantum physics, and software engineering. Not a tutorial. Not filler.
Every file earns its place.

**This is also a learning project.** Every time Claude writes code or runs a command, it must
explain **what** it is doing and **why** — covering the physics, the ML, and the software
design reasoning. No silent changes. Ever.

---

## What this project is

**Neural Quantum States (NQS)** — implementing the method from:
> Carleo & Troyer, *"Solving the Quantum Many-Body Problem with Artificial Neural Networks"*,
> Science 355, 602 (2017).

The core idea: use neural networks as variational ansätze (parameterized guesses) for quantum
ground state wavefunctions. Then use **Variational Monte Carlo (VMC)** to optimize the network
parameters until the energy is minimized — which by the variational principle means we've found
an approximation to the true ground state.

**Physical systems implemented:**
1. **1D Transverse Field Ising Model (TFIM)** — ferromagnetic coupling vs transverse field,
   quantum phase transition at Gamma/J = 1.0, exact solution known (for verification)
2. **1D Heisenberg XXX Model** — SU(2) symmetric, richer correlations, no simple exact solution
   for all observables (shows the abstractions work across different physics)

**Neural network ansätze implemented:**
1. **RBM** (Restricted Boltzmann Machine) — the original NQS architecture from the 2017 paper
2. **CNN-NQS** (1D Convolutional Network) — bakes in translational symmetry of the spin chain,
   a genuine research-level architectural choice (stretch goal, build after core is working)

**Why this is impressive to each audience:**
- *ML people:* Two architectures, three optimizers, training dynamics comparison, hyperparameter study
- *Physics people:* Two Hamiltonians, phase diagram, observables (magnetization, correlations,
  structure factor), exact diagonalization verification
- *SWE people:* Abstract base classes, YAML config system, CLI entry point, unit tests,
  checkpointing, clean modular architecture

---

## Coding style rules (IMPORTANT — always follow these)

1. **Comment well, not excessively.** Every non-trivial function gets a concise docstring.
   Add physics/ML context where it genuinely helps understanding — not on every function.
   Complex functions (SR optimizer, gradient formula, structure factor) deserve more context.
   Simple functions (magnetization, SGD, reset methods) need just a brief docstring.

2. **Short file-level context.** Each file starts with a 2-4 sentence comment block
   explaining what the file does and why it matters. Not a full lecture.

3. **Section headers where they help navigation.** Use sparingly:
   ```python
   # ============================================================
   # Section Name
   # ============================================================
   ```

4. **Inline comments on non-obvious lines only.** If a line of math or numpy isn't
   immediately obvious, add a comment. Don't comment obvious operations.

5. **No magic numbers.** All constants are named and explained.

6. **No silent changes.** Claude tells the user what it is doing and why before doing it.

7. **Complexity without bloat.** Every file earns its place. No filler. No random extras.

---

## Full project structure (target)

```
neural-quantum-states/
│
├── CLAUDE.md                          # This file — AI session context
├── README.md                          # GitHub showcase (write last)
├── requirements.txt                   # numpy, scipy, matplotlib, tqdm
│
├── configs/                           # YAML experiment configs (reproducibility)
│   ├── ising_small.yaml               # e.g. N=8, alpha=2, quick test
│   ├── ising_large.yaml               # e.g. N=20, alpha=4, serious run
│   └── heisenberg.yaml                # Heisenberg model config
│
├── src/
│   ├── hamiltonians/                  # Package: physical systems
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract Hamiltonian base class
│   │   ├── ising.py                   # TFIM (refactored from hamiltonian.py)
│   │   └── heisenberg.py              # Heisenberg XXX model
│   │
│   ├── ansatz/                        # Package: neural network wavefunctions
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract ansatz base class
│   │   ├── rbm.py                     # Restricted Boltzmann Machine
│   │   └── cnn.py                     # 1D CNN with translational symmetry (stretch goal)
│   │
│   ├── sampler.py                     # Metropolis-Hastings MCMC
│   ├── optimizer.py                   # Stochastic Reconfiguration + SGD + Adam
│   ├── trainer.py                     # VMC training loop (ties everything together)
│   ├── observables.py                 # Magnetization, correlations, structure factor
│   ├── exact.py                       # Exact diagonalization via scipy sparse
│   └── utils.py                       # Logging, checkpointing, plotting
│
├── scripts/
│   ├── run_vmc.py                     # CLI entry point
│   └── plot_phase_diagram.py          # Scans Gamma/J, plots the phase transition
│
├── notebooks/
│   └── demo.ipynb                     # Interactive walkthrough with plots
│
├── results/
│   └── .gitkeep                       # Gitignored except .gitkeep
│
└── tests/
    ├── test_hamiltonians.py
    ├── test_rbm.py
    ├── test_sampler.py
    └── test_observables.py
```

---

## Module descriptions (detailed)

### `src/hamiltonians/base.py` — STATUS: DONE
- Abstract base class `Hamiltonian` that all Hamiltonians must inherit from
- Forces every Hamiltonian to implement `local_energy(spins, log_psi_func)`
- Why: the trainer and sampler don't need to know which Hamiltonian they're using —
  they just call `local_energy`. This is the open/closed principle in action.

### `src/hamiltonians/ising.py` — STATUS: DONE
- Refactored from the existing `src/hamiltonian.py`
- Class: `IsingHamiltonian(Hamiltonian)`
- H = -J * sum_i(sigma_i^z * sigma_{i+1}^z) - Gamma * sum_i(sigma_i^x)
- Diagonal term: ZZ coupling between neighbors
- Off-diagonal term: transverse field flips each spin
- Phase transition at Gamma/J = 1.0

### `src/hamiltonians/heisenberg.py` — STATUS: DONE
- Class: `HeisenbergHamiltonian(Hamiltonian)`
- H = J * sum_i [ σᵢᶻσⱼᶻ + 2(S⁺S⁻ + S⁻S⁺) ] with periodic BC
- **Marshall sign rule** baked into `local_energy()`: the AFM ground state has sign
  structure ψ(σ) = (-1)^{N_↑_even} × |ψ(σ)|. Since our RBM is real-valued (ψ > 0 always),
  we absorb the sign into the Hamiltonian. For all nearest-neighbor swaps on a 1D chain,
  the Marshall sign ratio is -1, so the off-diagonal coefficient becomes -2J (not +2J).
  This was the KEY FIX that unlocked Heisenberg convergence (see design decisions below).
- Demonstrates that the same VMC + RBM framework works on a different system

### `src/ansatz/base.py` — STATUS: DONE
- Abstract base class `Ansatz` that all neural network ansätze must inherit from
- Forces every ansatz to implement:
  - `log_psi(spins)` — log amplitude of the wavefunction
  - `grad_log_psi(spins)` — gradients w.r.t. all parameters (for VMC updates)
  - `parameters` property — flat array of all trainable parameters
  - `update_parameters(delta)` — apply a parameter update

### `src/ansatz/rbm.py` — STATUS: DONE
- Class: `RBM(Ansatz)`
- Visible layer: N spin variables (the physical spins, values +1/-1)
- Hidden layer: M = alpha * N hidden units (alpha is the hidden unit density hyperparameter)
- Parameters: visible biases `a` (N,), hidden biases `b` (M,), weight matrix `W` (N, M)
- log psi(sigma) = sum_i(a_i * sigma_i) + sum_j log(2 cosh(b_j + sum_i W_ij * sigma_i))
- Gradients: d(log psi)/da_i = sigma_i, d(log psi)/db_j = tanh(theta_j),
  d(log psi)/dW_ij = sigma_i * tanh(theta_j) where theta_j = b_j + W^T sigma

### `src/ansatz/cnn.py` — STATUS: STRETCH GOAL (build after core works)
- Class: `CNN(Ansatz)`
- 1D convolutional architecture with periodic padding (respects translational symmetry)
- Why better than RBM for some systems: CNNs share weights across sites, encoding the
  physical fact that all sites in the chain are equivalent
- Must implement the same `log_psi` and `grad_log_psi` interface as RBM
- Enables RBM vs CNN comparison: same system, same training, different architectures

### `src/sampler.py` — STATUS: DONE
- Class: `MetropolisSampler`
- Uses Metropolis-Hastings MCMC to sample spin configs from |psi(sigma)|^2
- Two proposal modes:
  - **Single spin flip** (`use_exchange=False`): for Ising-type models
  - **Adjacent pair swap** (`use_exchange=True`): 100% exchange moves for Heisenberg.
    Swaps two neighboring antiparallel spins, conserving total Sz. Essential because
    single flips change Sz and get rejected once ψ peaks on the Sz=0 sector.
- Acceptance: min(1, exp(2 * Re(log_psi(new) - log_psi(old))))
- `refresh()`: recomputes cached log_psi after optimizer updates (prevents stale cache → chain freeze)
- `reset_state()`: reinitializes to random config when acceptance collapses
- Key method: `sample(n_samples)` → array of shape (n_samples, n_spins)

### `src/optimizer.py` — STATUS: DONE
- Class: `StochasticReconfiguration`
- The "secret sauce" — natural gradient for quantum systems
- Computes the quantum geometric tensor S_kk' = <O_k* O_k'> - <O_k*><O_k'>
  where O_k = d(log psi)/d(theta_k) are the parameter gradients
- Solves (S + epsilon*I) * delta_theta = -learning_rate * grad_E
  (epsilon is Tikhonov regularization to handle ill-conditioning)
- Also implements `SGD` and `Adam` as fallback optimizers for comparison
- Why SR beats SGD: it uses the geometry of the wavefunction manifold, equivalent to
  imaginary-time evolution projected onto the variational space

### `src/trainer.py` — STATUS: DONE
- Class: `VMCTrainer`
- Orchestrates the full VMC loop:
  1. Sample spin configurations: MetropolisSampler → {sigma_1, ..., sigma_N}
  2. Compute local energies: Hamiltonian.local_energy for each sample
  3. Compute energy gradient: grad_E = 2 * Re(<O_k* E_loc> - <O_k*><E_loc>)
  4. Update parameters: StochasticReconfiguration → new theta
  5. Log energy, acceptance rate, gradient norm
- Loads config from YAML (so experiments are reproducible)
- Saves checkpoints via utils.py

### `src/observables.py` — STATUS: DONE
- Functions to compute physical observables from MCMC samples + ansatz:
  - `magnetization(samples, ansatz)` — <sigma^z> average spin polarization
  - `correlation(samples, ansatz, i, j)` — <sigma_i^z * sigma_j^z> spin-spin correlation
  - `structure_factor(samples, ansatz)` — Fourier transform of correlations S(k)
- These let us visualize the phase transition (magnetization drops to 0 at Gamma/J = 1)
- Why impressive: energy alone is one number; these are full physical characterizations

### `src/exact.py` — STATUS: DONE
- Function: `exact_diagonalization(hamiltonian, n_spins)` → ground state energy
- Builds the full 2^N x 2^N Hamiltonian matrix using scipy sparse matrices
- Uses scipy.sparse.linalg.eigsh to find the lowest eigenvalue
- Only feasible for small N (N <= 20) but that's enough for verification
- The killer result: "our RBM matches the exact answer to 4 decimal places"

### `src/utils.py` — STATUS: DONE
- `Logger` class: tracks energy history, acceptance rates, gradient norms per epoch
- `Checkpointer` class: saves/loads ansatz parameters to/from .npy files
- `plot_energy_convergence(history)`: energy vs epoch curve with ±1 std band
- `plot_phase_diagram(gammas, energies, magnetizations)`: the showcase plot
- `load_config(path)`: YAML → dict
- `make_progress_bar()`: tqdm wrapper (graceful fallback if not installed)

---

## Key physics concepts (reference)

**Variational Principle:** For any state |psi>, <psi|H|psi> >= E_ground.
Minimizing <E> over our parameterized ansatz approaches the true ground state energy.

**Variational Monte Carlo (VMC):**
- Can't sum over all 2^N spin configurations (exponentially many for large N)
- Instead: sample sigma ~ |psi(sigma)|^2 using MCMC
- <E> ≈ (1/N_samples) * sum_samples E_loc(sigma) — unbiased estimator
- Gradients computed similarly: averages over the same samples

**Why RBM?**
- RBMs can represent highly entangled quantum states with polynomial parameters
- Hidden units capture multi-spin correlations that simple mean-field theory misses
- log_psi is O(N*M) to compute — very cheap compared to storing the full 2^N state vector

**Stochastic Reconfiguration (SR):**
- Plain gradient descent treats all parameter directions equally — wrong for wavefunctions
- SR accounts for the fact that different parameter changes have different effects on the state
- The S matrix (quantum Fisher information) encodes the geometry of the wavefunction manifold
- Result: much faster convergence, more stable training

**Marshall Sign Rule (Heisenberg):**
- The AFM Heisenberg ground state on a bipartite lattice has sign structure:
  ψ(σ) = (-1)^{number of up spins on even sublattice} × |ψ(σ)|
- Real-valued RBMs produce ψ > 0, so they can't learn the sign directly
- Solution: absorb the sign into the Hamiltonian's off-diagonal term (factor of -1)
- For nearest-neighbor swaps on 1D chain, the Marshall sign ratio is always -1
- This is a standard technique in NQS literature (Carleo & Troyer used it too)

**Quantum Phase Transition (TFIM):**
- At Gamma/J << 1: spins align (ordered/ferromagnetic phase), magnetization ≠ 0
- At Gamma/J >> 1: transverse field dominates, spins fluctuate (disordered phase), magnetization → 0
- At Gamma/J = 1: quantum critical point — correlations become long-range
- We see this in our results: magnetization plot vs Gamma/J shows a clear transition

---

## Build order (follow this sequence)

1. `src/hamiltonians/` — base.py → ising.py (refactor) → heisenberg.py
2. `src/ansatz/` — base.py → rbm.py
3. `src/sampler.py`
4. `src/optimizer.py`
5. `src/trainer.py` + `src/utils.py`
6. `src/exact.py` + `src/observables.py`
7. `configs/` + `scripts/`
8. `tests/`
9. `src/ansatz/cnn.py` ← stretch goal, only after everything above works
10. `notebooks/demo.ipynb` + `README.md` ← write last

---

## Current status

| File | Status | Notes |
|------|--------|-------|
| `src/hamiltonian.py` | Done (old) | Will be deleted once hamiltonians/ is committed |
| `src/hamiltonians/__init__.py` | Done | |
| `src/hamiltonians/base.py` | Done | Abstract base class |
| `src/hamiltonians/ising.py` | Done | Refactored from `src/hamiltonian.py` |
| `src/hamiltonians/heisenberg.py` | Done | |
| `src/ansatz/__init__.py` | Done | |
| `src/ansatz/base.py` | Done | Abstract base class |
| `src/ansatz/rbm.py` | Done | Core ML component |
| `src/ansatz/cnn.py` | Not started | Stretch goal |
| `src/sampler.py` | Done | Metropolis-Hastings MCMC |
| `src/optimizer.py` | Done | SR + Adam + SGD |
| `src/trainer.py` | Done | VMCTrainer: full training loop, gradient formula, checkpointing |
| `src/observables.py` | Done | magnetization, correlation_matrix, structure_factor (FFT), compute_all_observables |
| `src/exact.py` | Done | exact_diagonalization (scipy sparse Lanczos), ising_exact_energy_thermodynamic |
| `src/utils.py` | Done | Logger, Checkpointer, plot_energy_convergence, plot_phase_diagram, load_config |
| `configs/` | Done | ising_small.yaml, ising_large.yaml, heisenberg.yaml |
| `scripts/` | Done | run_vmc.py (CLI entry point), plot_phase_diagram.py (phase scan) |
| `tests/` | Done | 64 tests — test_hamiltonians.py, test_rbm.py, test_sampler.py, test_observables.py |
| `notebooks/demo.ipynb` | Not started | See visualization plan below |
| `README.md` | Not started | Write last, after notebook generates real plots |

---

## Remaining work & visualization plan

### Step 1 — Run experiments (get real results) — DONE
Experiments completed with verified results:
  - `python scripts/run_vmc.py configs/ising_small.yaml`   → DONE (0.011% error)
  - `python scripts/run_vmc.py configs/heisenberg.yaml`    → DONE (0.007% error)
  - `python scripts/plot_phase_diagram.py`                 → TODO (~5min, produces phase_diagram.png)

### Step 2 — notebooks/demo.ipynb
The notebook tells the story end-to-end with inline plots. It should include:

**Existing visualizations (already implemented in src/utils.py):**
- Energy convergence plot (energy vs epoch, ±1σ band, acceptance rate panel)
- Phase diagram (E/site + |magnetization| vs Gamma/J showing QPT at Gamma/J=1)

**New visualizations to add (for non-physics audiences):**
These are high priority — they are what makes the project accessible to ML/SWE recruiters:
- Spin configuration heatmap: red/blue grid of ±1 spins, contrasting ordered vs disordered phase
- Correlation matrix heatmap: color grid of C[i,j] = <σᵢσⱼ> (like a covariance matrix, but quantum)
- SR vs SGD vs Adam convergence comparison: same experiment, three optimizers — why SR wins
- RBM weight matrix heatmap: W[i,j] visualized — what did the network actually learn?
- (Optional) Wavefunction probability bar chart: |ψ(σ)|² for all 2^N configs (N=6 max)

**Implementation:** New visualization functions go in src/utils.py or a new src/visualization.py.
Adding a new plot = one new function + one new notebook cell. Nothing else changes.

### Step 3 — README.md
Write last. Built around the actual plots from Step 1 + Step 2. Structure:
  - One-line hook, badges
  - What this project does (physics + ML angle)
  - Key results with embedded images
  - How to run it
  - Project structure

### Step 4 — src/ansatz/cnn.py (stretch goal, optional)
Only if time allows. Adds translational symmetry via 1D convolutions.
Enables RBM vs CNN comparison plot as an additional visualization.

---

## Design decisions log

### Marshall sign rule in heisenberg.py (critical fix)
**Problem:** RBM uses real parameters → log_psi is always real → ψ(σ) > 0 for all σ.
But the AFM Heisenberg ground state has negative amplitudes: ψ(σ) = (-1)^{N_↑_even} × |ψ(σ)|.
A positive-definite wavefunction cannot represent this state. Energy plateaued at E ≈ -4.0
(the best any positive wavefunction can achieve), far from exact E = -14.6.

**Solution:** Apply Marshall sign analytically in `local_energy()`. The full ansatz is
ψ(σ) = marshall_sign(σ) × ψ_RBM(σ). For nearest-neighbor swaps on a 1D chain, the sign
ratio is always -1 (exactly one swapped site is on an even sublattice). This changes the
off-diagonal coefficient from +2J to -2J. The RBM only needs to learn |ψ|, not the sign.

**Result:** Heisenberg converges to 0.007% error (E = -14.603 vs exact -14.604).

### Exchange moves in sampler.py
**Problem:** Single spin flips change total Sz. Once the wavefunction peaks on the Sz=0
sector, all single-flip proposals get rejected → acceptance drops to 0 → chain freezes.

**Solution:** When `use_exchange=True`, 100% of proposals are adjacent pair swaps of
antiparallel spins (conserves Sz). Non-local swaps were tried but had worse acceptance
(4% vs 40%) because wavefunction ratios for distant pairs are further from 1.

### Training stability (trainer.py)
Three mechanisms prevent wavefunction collapse:
1. **Gradient clipping** (max norm 5.0): prevents huge early gradients
2. **Delta clipping** (max norm 1.0): prevents SR from amplifying clipped gradients via ill-conditioned S matrix
3. **Chain rescue** (accept < 1%): resets to random config + re-burn-in if chain freezes

---

## Verified experiment results

| Experiment | VMC Energy | Exact Energy | Error | Acceptance |
|-----------|-----------|-------------|-------|------------|
| Ising (N=8, Γ/J=1) | -8.2413 | -8.2425 | 0.011% | ~50% |
| Heisenberg (N=8, J=1) | -14.6033 | -14.6044 | 0.007% | ~40% |

Both models converge with the same RBM + VMC framework — different Hamiltonians, zero
training code changes. Results saved in `results/ising_small/` and `results/heisenberg/`.

---

## How to resume a session

1. Read this entire CLAUDE.md file.
2. Run `git log --oneline` to see what has been committed.
3. Run `git status` to see untracked/modified files.
4. Check the status table above to find where we left off.
5. Tell the user the current state and suggest the next step.
6. Always explain what you are about to do before doing it.
7. If any major decision was made this session, update this file before the session ends.
