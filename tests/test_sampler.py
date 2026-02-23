# tests/test_sampler.py
#
# ============================================================
# UNIT TESTS: Metropolis-Hastings MCMC Sampler
# ============================================================
#
# WHAT WE'RE TESTING:
#   The MetropolisSampler is the data generator for VMC training.
#   Its correctness is about statistical properties, not exact values.
#
# TEST STRATEGY:
#   We test three categories:
#
#   1. OUTPUT SHAPE AND TYPE: samples array has the right shape, all entries
#      are exactly ±1 (the discrete spin values we expect).
#
#   2. ACCEPTANCE RATE: must be in [0, 1]. The Metropolis algorithm by
#      construction can only accept or reject — never accept more than 100%.
#
#   3. RESET AND STATE MANAGEMENT: reset_state correctly restores the chain,
#      and reset_acceptance_stats correctly zeroes the counters.
#
# NOTE ON TESTING MCMC:
#   We can't easily test that the samples are drawn from |psi|^2 without
#   running a full convergence test. Instead we test:
#   - The chain produces valid spin configurations (±1, right shape)
#   - The acceptance rate is in a valid range
#   - The state management methods work correctly
#   These give confidence the core logic is right without requiring slow
#   statistical convergence tests in a unit test suite.
#
# ============================================================

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ansatz import RBM
from src.sampler import MetropolisSampler


# ============================================================
# SECTION: Helper — Trivial Ansatz for Sampler Tests
# ============================================================

def _make_sampler(n_spins=4, alpha=1, seed=42):
    """Create a sampler with a small RBM for testing."""
    rbm = RBM(n_spins=n_spins, alpha=alpha, seed=seed)
    return MetropolisSampler(ansatz=rbm, n_spins=n_spins, seed=seed)


# ============================================================
# SECTION: Output Shape and Type Tests
# ============================================================

class TestSamplerOutput(unittest.TestCase):
    """Verify that sample() returns the correct shape and spin values."""

    def test_sample_shape_default(self):
        """
        samples.shape must be (n_samples, n_spins).

        PHYSICS: Each row is one spin configuration — a snapshot of the
        N-spin chain. We collect n_samples of these.
        """
        n_spins, n_samples = 6, 50
        sampler = _make_sampler(n_spins=n_spins)
        samples = sampler.sample(n_samples=n_samples)
        self.assertEqual(samples.shape, (n_samples, n_spins))

    def test_sample_values_are_binary(self):
        """
        Every spin value in the sample array must be exactly +1 or -1.

        PHYSICS: We work in the σᶻ eigenbasis. There are only two eigenvalues:
          σᶻ|↑⟩ = +|↑⟩  and  σᶻ|↓⟩ = -|↓⟩
        The sampler should only ever produce these values, never 0 or fractions.
        """
        sampler = _make_sampler(n_spins=4)
        samples = sampler.sample(n_samples=100)
        unique_values = set(np.unique(samples))
        self.assertEqual(unique_values, {-1, 1},
                         f"Samples should only contain ±1, got: {unique_values}")

    def test_sample_different_n_samples(self):
        """Sampling 1, 10, and 500 samples all produce the right shape."""
        sampler = _make_sampler(n_spins=4)
        for n in [1, 10, 500]:
            with self.subTest(n_samples=n):
                samples = sampler.sample(n_samples=n)
                self.assertEqual(samples.shape[0], n)

    def test_sample_with_custom_sweep_size(self):
        """Passing a custom sweep_size should work and produce the right shape."""
        n_spins = 4
        sampler = _make_sampler(n_spins=n_spins)
        samples = sampler.sample(n_samples=20, sweep_size=2)
        self.assertEqual(samples.shape, (20, n_spins))


# ============================================================
# SECTION: Acceptance Rate Tests
# ============================================================

class TestSamplerAcceptanceRate(unittest.TestCase):
    """Verify Metropolis acceptance rate is valid."""

    def test_acceptance_rate_in_unit_interval(self):
        """
        Acceptance rate must be in [0, 1].

        ALGORITHM: The Metropolis acceptance probability is min(1, A) where
        A >= 0. So the acceptance rate can never exceed 1 or go below 0.
        """
        sampler = _make_sampler(n_spins=6)
        sampler.sample(n_samples=200)
        rate = sampler.acceptance_rate
        self.assertGreaterEqual(rate, 0.0, "Acceptance rate must be >= 0")
        self.assertLessEqual(rate, 1.0, "Acceptance rate must be <= 1")

    def test_acceptance_rate_nonzero_after_sampling(self):
        """
        After sampling, at least some proposals must have been made.

        A zero acceptance rate after sampling could indicate the chain
        is completely stuck — a sign of a critical bug.
        """
        sampler = _make_sampler(n_spins=4)
        sampler.sample(n_samples=100)
        # After 100 samples with n_spins sweeps each, there should be many proposals
        self.assertGreater(sampler._n_proposed, 0,
                           "Some proposals must have been made during sampling")

    def test_acceptance_rate_before_sampling_is_zero(self):
        """
        Before any sampling, acceptance rate should be 0.0 (no proposals made).
        """
        sampler = _make_sampler(n_spins=4)
        self.assertEqual(sampler.acceptance_rate, 0.0)

    def test_reset_acceptance_stats(self):
        """
        reset_acceptance_stats() must zero the acceptance counters.

        USE CASE: In VMC training, we reset at the start of each epoch so
        the logged acceptance rate reflects only the current epoch.
        """
        sampler = _make_sampler(n_spins=4)
        sampler.sample(n_samples=50)
        self.assertGreater(sampler._n_proposed, 0)    # confirm proposals were made
        sampler.reset_acceptance_stats()
        self.assertEqual(sampler._n_proposed, 0)
        self.assertEqual(sampler._n_accepted, 0)
        self.assertEqual(sampler.acceptance_rate, 0.0)


# ============================================================
# SECTION: State Management Tests
# ============================================================

class TestSamplerStateManagement(unittest.TestCase):
    """Verify burn_in and reset_state work correctly."""

    def test_burn_in_does_not_crash(self):
        """burn_in(n) must complete without raising an exception."""
        sampler = _make_sampler(n_spins=4)
        sampler.burn_in(100)    # should not raise

    def test_burn_in_makes_proposals(self):
        """burn_in should make exactly n_steps proposals."""
        sampler = _make_sampler(n_spins=4)
        sampler.burn_in(50)
        self.assertEqual(sampler._n_proposed, 50)

    def test_reset_state_with_explicit_spins(self):
        """
        reset_state(spins) must set the chain's current position to `spins`.

        PHYSICS: Occasionally the ansatz parameters change so much that the
        current chain position is in a low-probability region. Reset gets us
        to a known starting point for re-thermalization.
        """
        n_spins = 4
        sampler = _make_sampler(n_spins=n_spins)
        sampler.sample(n_samples=50)    # advance the chain

        target_spins = np.array([+1, -1, +1, -1])
        sampler.reset_state(target_spins)

        np.testing.assert_array_equal(
            sampler.current_spins, target_spins,
            err_msg="reset_state should set current_spins to the given array"
        )

    def test_reset_state_clears_acceptance_stats(self):
        """reset_state also resets acceptance counters."""
        sampler = _make_sampler(n_spins=4)
        sampler.sample(n_samples=50)
        sampler.reset_state()    # random reset
        self.assertEqual(sampler._n_proposed, 0)
        self.assertEqual(sampler._n_accepted, 0)

    def test_reset_state_random(self):
        """reset_state() with no argument produces valid ±1 spins."""
        n_spins = 6
        sampler = _make_sampler(n_spins=n_spins)
        sampler.reset_state()
        spins = sampler.current_spins
        self.assertEqual(spins.shape, (n_spins,))
        self.assertTrue(
            set(np.unique(spins)).issubset({-1, 1}),
            "reset_state() should produce ±1 spins"
        )


# ============================================================
# SECTION: Entry Point
# ============================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
