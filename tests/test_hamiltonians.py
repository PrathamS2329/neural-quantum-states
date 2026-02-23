# tests/test_hamiltonians.py
#
# ============================================================
# UNIT TESTS: Hamiltonians (Ising and Heisenberg)
# ============================================================
#
# WHAT WE'RE TESTING:
#   The core physics computation of both Hamiltonians is local_energy().
#   We verify it against analytically known results for simple spin configs.
#
# TEST STRATEGY:
#   We use a "constant wavefunction" (log_psi = 0 for all configs) in most
#   tests. This makes exp(log_ratio) = exp(0) = 1 for every off-diagonal
#   element, so we can compute the expected local energy by hand.
#
#   We also run one exact diagonalization test per model to confirm that
#   the matrix built in exact.py gives the physically correct ground state.
#
# ============================================================

import sys
import os
import unittest
import numpy as np

# Add project root to path so we can import src/ from anywhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.hamiltonians import IsingHamiltonian, HeisenbergHamiltonian
from src.exact import exact_ground_state_energy


# ============================================================
# SECTION: Ising Hamiltonian Tests
# ============================================================

class TestIsingHamiltonian(unittest.TestCase):
    """Tests for IsingHamiltonian.local_energy() against analytical results."""

    # A constant log_psi: psi(sigma) = 1 for all sigma.
    # This makes all wavefunction ratios = 1, so we can evaluate local_energy
    # purely from the spin values — no neural network needed.
    CONST_LOG_PSI = staticmethod(lambda spins: 0.0)

    def setUp(self):
        """Set up a small N=4 Ising chain used in multiple tests."""
        self.N = 4
        self.J = 1.0

    # ------------------------------------------------------------------
    # Test 1: Diagonal term (J coupling only)
    # ------------------------------------------------------------------

    def test_all_up_no_field_diagonal(self):
        """
        All-up spins with no transverse field: diagonal term only.

        PHYSICS: H = -J Σ σᵢ σᵢ₊₁ with gamma=0.
          All bonds aligned → each contributes -J.
          For N=4 bonds with periodic BC: E_loc = -J * 4 = -4.0
        """
        ham = IsingHamiltonian(n_spins=self.N, J=self.J, gamma=0.0)
        spins = np.array([+1, +1, +1, +1])
        E_loc = ham.local_energy(spins, self.CONST_LOG_PSI)
        self.assertAlmostEqual(np.real(E_loc), -4.0, places=10)

    def test_alternating_no_field_diagonal(self):
        """
        Alternating spins (Néel-like) with no transverse field: all bonds anti-aligned.

        PHYSICS: -J Σ σᵢ σᵢ₊₁ with all products = -1.
          For N=4: E_loc = -J * 4 * (-1) = +4.0
          This is the MAXIMUM energy configuration for ferromagnetic Ising.
        """
        ham = IsingHamiltonian(n_spins=self.N, J=self.J, gamma=0.0)
        spins = np.array([+1, -1, +1, -1])
        E_loc = ham.local_energy(spins, self.CONST_LOG_PSI)
        self.assertAlmostEqual(np.real(E_loc), +4.0, places=10)

    # ------------------------------------------------------------------
    # Test 2: Off-diagonal term (transverse field only)
    # ------------------------------------------------------------------

    def test_all_up_field_only_off_diagonal(self):
        """
        All-up spins with J=0, gamma=1: off-diagonal term only.

        PHYSICS: H = -Gamma Σ σᵢˣ with J=0.
          σᵢˣ flips spin i: |↑↑↑↑⟩ → |config with one spin flipped⟩
          For constant psi: each spin contributes -gamma * exp(0) = -1.
          Total for N=4 spins: E_loc = -gamma * 4 = -4.0
        """
        ham = IsingHamiltonian(n_spins=self.N, J=0.0, gamma=1.0)
        spins = np.array([+1, +1, +1, +1])
        E_loc = ham.local_energy(spins, self.CONST_LOG_PSI)
        self.assertAlmostEqual(np.real(E_loc), -4.0, places=10)

    # ------------------------------------------------------------------
    # Test 3: Both terms (J=1, gamma=1, critical point)
    # ------------------------------------------------------------------

    def test_all_up_critical_point(self):
        """
        All-up spins at the critical point J=gamma=1, constant psi.

        PHYSICS: Diagonal = -J*N = -4; off-diagonal = -gamma*N = -4.
          Total E_loc = -4 + (-4) = -8.0  (for N=4)
        """
        ham = IsingHamiltonian(n_spins=self.N, J=1.0, gamma=1.0)
        spins = np.array([+1, +1, +1, +1])
        E_loc = ham.local_energy(spins, self.CONST_LOG_PSI)
        self.assertAlmostEqual(np.real(E_loc), -8.0, places=10)

    # ------------------------------------------------------------------
    # Test 4: Symmetry — flipping all spins leaves energy unchanged
    # ------------------------------------------------------------------

    def test_global_flip_symmetry(self):
        """
        Flipping all spins leaves the local energy invariant (Z2 symmetry).

        PHYSICS: H commutes with the global spin-flip operator Π σᵢˣ.
          So E_loc(+all) == E_loc(-all) for a constant wavefunction
          (which also has Z2 symmetry).
        """
        ham = IsingHamiltonian(n_spins=self.N, J=1.0, gamma=1.0)
        spins_up = np.array([+1, +1, +1, +1])
        spins_down = np.array([-1, -1, -1, -1])
        E_up = ham.local_energy(spins_up, self.CONST_LOG_PSI)
        E_down = ham.local_energy(spins_down, self.CONST_LOG_PSI)
        self.assertAlmostEqual(np.real(E_up), np.real(E_down), places=10)

    # ------------------------------------------------------------------
    # Test 5: Return type
    # ------------------------------------------------------------------

    def test_returns_scalar(self):
        """local_energy must return a scalar (complex or float), not an array."""
        ham = IsingHamiltonian(n_spins=self.N, J=1.0, gamma=1.0)
        spins = np.array([+1, -1, +1, +1])
        E_loc = ham.local_energy(spins, self.CONST_LOG_PSI)
        # Must be a Python/numpy scalar, not a numpy array
        self.assertFalse(hasattr(E_loc, '__len__'), "local_energy should be scalar")

    # ------------------------------------------------------------------
    # Test 6: n_spins property
    # ------------------------------------------------------------------

    def test_n_spins_property(self):
        """n_spins property should return the value passed to __init__."""
        ham = IsingHamiltonian(n_spins=8, J=1.0, gamma=0.5)
        self.assertEqual(ham.n_spins, 8)

    # ------------------------------------------------------------------
    # Test 7: Exact diagonalization for trivial case
    # ------------------------------------------------------------------

    def test_exact_ising_pure_zz(self):
        """
        Pure Ising (gamma=0): exact ground state energy = -J * N.

        PHYSICS: Without a transverse field, the Hamiltonian is diagonal.
          Ground state is |↑↑↑↑⟩ or |↓↓↓↓⟩ with energy = -J * N.
          This is exact and provides a sanity check for exact.py.
        """
        ham = IsingHamiltonian(n_spins=4, J=1.0, gamma=0.0)
        exact_E = exact_ground_state_energy(ham)
        # Ground state energy should be -J * N = -4.0
        self.assertAlmostEqual(exact_E, -4.0, places=8)

    def test_exact_ising_energy_negative(self):
        """
        At the critical point (gamma=J=1), exact energy must be negative.
        The ground state energy per site → -2/π ≈ -0.637 in the thermodynamic limit.
        """
        ham = IsingHamiltonian(n_spins=6, J=1.0, gamma=1.0)
        exact_E = exact_ground_state_energy(ham)
        self.assertLess(exact_E, 0.0, "Ising ground state energy must be negative")


# ============================================================
# SECTION: Heisenberg Hamiltonian Tests
# ============================================================

class TestHeisenbergHamiltonian(unittest.TestCase):
    """Tests for HeisenbergHamiltonian.local_energy() against analytical results."""

    CONST_LOG_PSI = staticmethod(lambda spins: 0.0)

    def setUp(self):
        self.N = 4
        self.J = 1.0

    # ------------------------------------------------------------------
    # Test 1: All-parallel spins — ZZ only, no spin flips
    # ------------------------------------------------------------------

    def test_all_up_zz_term_only(self):
        """
        All-up spins: only the ZZ diagonal term contributes.

        PHYSICS: H = J Σ σᵢ σᵢ₊₁ + spin-flip terms.
          For all-parallel spins, spin-flip terms are zero (S⁺S⁻ requires antiparallel bond).
          ZZ term: J * 4 bonds * (+1) = +4.0  (ferromagnetic state has high energy for J>0 AFM)
        """
        ham = HeisenbergHamiltonian(n_spins=self.N, J=self.J)
        spins = np.array([+1, +1, +1, +1])
        E_loc = ham.local_energy(spins, self.CONST_LOG_PSI)
        self.assertAlmostEqual(np.real(E_loc), +4.0, places=10)

    # ------------------------------------------------------------------
    # Test 2: Néel state with constant psi — ZZ and spin-flip terms cancel
    # ------------------------------------------------------------------

    def test_neel_state_constant_psi(self):
        """
        Néel state with constant wavefunction and Marshall sign rule.

        PHYSICS: For spins [+1,-1,+1,-1] with all bonds antiparallel:
          - ZZ contribution per bond: J * (-1) = -J → total ZZ = -4.0
          - XX+YY term with Marshall sign: -2J * exp(0) = -2J per bond → total = -8.0
            (Marshall sign gives factor -1 for all NN swaps on the 1D chain)
          - Net E_loc = -12.0

        The Marshall sign rule converts the frustrated off-diagonal contribution
        from positive to negative, allowing the real-valued RBM to access the
        full quantum ground state energy.
        """
        ham = HeisenbergHamiltonian(n_spins=self.N, J=self.J)
        spins = np.array([+1, -1, +1, -1])
        E_loc = ham.local_energy(spins, self.CONST_LOG_PSI)
        self.assertAlmostEqual(np.real(E_loc), -12.0, places=10)

    # ------------------------------------------------------------------
    # Test 3: n_spins property
    # ------------------------------------------------------------------

    def test_n_spins_property(self):
        """n_spins property should return the value passed to __init__."""
        ham = HeisenbergHamiltonian(n_spins=6, J=2.0)
        self.assertEqual(ham.n_spins, 6)

    # ------------------------------------------------------------------
    # Test 4: Exact diagonalization — ground state is negative for AFM
    # ------------------------------------------------------------------

    def test_exact_heisenberg_energy_negative(self):
        """
        Antiferromagnetic Heisenberg (J>0) must have negative ground state energy.

        PHYSICS: The quantum ground state is a singlet with lower energy than any
          classical (product) state. For our N=4 chain, E_ground < 0 always holds.
        """
        ham = HeisenbergHamiltonian(n_spins=4, J=1.0)
        exact_E = exact_ground_state_energy(ham)
        self.assertLess(exact_E, 0.0, "Heisenberg AFM ground state must be negative")

    def test_exact_heisenberg_energy_finite(self):
        """Exact diagonalization must return a finite (non-NaN, non-inf) energy."""
        ham = HeisenbergHamiltonian(n_spins=4, J=1.0)
        exact_E = exact_ground_state_energy(ham)
        self.assertTrue(np.isfinite(exact_E), "Exact energy must be finite")


# ============================================================
# SECTION: Entry Point
# ============================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
