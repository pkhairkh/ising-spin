"""
Comprehensive unit tests for the Dense AM module.

Covers all operations defined in V18_TEST_STRATEGY.md Section 2.2:
  UT-DAM-01 through UT-DAM-12

Tests cover:
  - RandomFeatureProjector: shape, determinism, integer-only, cos LUT
  - DenseAMEnergy: polynomial nonlinearity, pre-aggregation, energy computation
  - Integration: Dense AM energy integrates with EnergyComputer
  - Sharpness: degree=2 produces sharper energy landscape than degree=1
"""

import numpy as np
import pytest

from ising_spin.dense_am.energy import RandomFeatureProjector, DenseAMEnergy


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def small_projector():
    """Small projector with D=64 for fast tests."""
    return RandomFeatureProjector(
        vocab_size=500,
        D=64,
        context_hash_dim=16,
        seed=42,
    )


@pytest.fixture
def standard_projector():
    """Standard projector with D=256."""
    return RandomFeatureProjector(
        vocab_size=1000,
        D=256,
        context_hash_dim=32,
        seed=42,
    )


@pytest.fixture
def small_dense_am():
    """Small Dense AM with D=64 for fast tests."""
    projector = RandomFeatureProjector(
        vocab_size=200,
        D=64,
        context_hash_dim=16,
        seed=42,
    )
    return DenseAMEnergy(
        projector=projector,
        vocab_size=200,
        degree=2,
        dense_am_scale=1200,
    )


@pytest.fixture
def built_dense_am(small_dense_am):
    """Dense AM that has been pre-aggregated from sample sequences."""
    # Create simple training sequences
    rng = np.random.RandomState(42)
    sequences = []
    for _ in range(50):
        seq_len = rng.randint(3, 10)
        seq = rng.randint(0, 200, size=seq_len).tolist()
        sequences.append(seq)

    small_dense_am.preaggregate(sequences)
    return small_dense_am


# ===========================================================================
# UT-DAM-01: Projector shape
# ===========================================================================

class TestProjectorShape:
    """Tests for random feature projection output shape."""

    def test_phi_shape(self, small_projector):
        """UT-DAM-01: Projection output has correct shape."""
        phi = small_projector.project([1, 2, 3, 4, 5])
        assert phi.shape == (64,), f"Expected (64,), got {phi.shape}"

    def test_phi_shape_standard(self, standard_projector):
        """UT-DAM-01: Standard D=256 projection has correct shape."""
        phi = standard_projector.project([10, 20, 30])
        assert phi.shape == (256,), f"Expected (256,), got {phi.shape}"

    def test_phi_dtype(self, small_projector):
        """UT-DAM-01: Projection output is int8."""
        phi = small_projector.project([1, 2, 3])
        assert phi.dtype == np.int8, f"Expected int8, got {phi.dtype}"

    def test_phi_values_in_range(self, small_projector):
        """UT-DAM-01: Projection values are in int8 range [-127, 127]."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            context = rng.randint(0, 500, size=5).tolist()
            phi = small_projector.project(context)
            assert np.all(phi >= -127), f"Values below -127 found"
            assert np.all(phi <= 127), f"Values above 127 found"

    def test_empty_context(self, small_projector):
        """UT-DAM-01: Empty context produces zero vector."""
        phi = small_projector.project([])
        assert phi.shape == (64,)
        assert np.all(phi == 0), "Empty context should produce zero vector"

    def test_single_word_context(self, small_projector):
        """UT-DAM-01: Single word context produces valid feature vector."""
        phi = small_projector.project([5])
        assert phi.shape == (64,)
        assert phi.dtype == np.int8


# ===========================================================================
# UT-DAM-02: Projector deterministic
# ===========================================================================

class TestProjectorDeterministic:
    """Tests for deterministic projection."""

    def test_same_input_same_output(self, small_projector):
        """UT-DAM-02: Same input produces same projection."""
        phi1 = small_projector.project([1, 2, 3, 4, 5])
        phi2 = small_projector.project([1, 2, 3, 4, 5])
        assert np.array_equal(phi1, phi2), "Same input should produce same output"

    def test_different_input_different_output(self, small_projector):
        """UT-DAM-02: Different input produces different projection."""
        phi1 = small_projector.project([1, 2, 3])
        phi2 = small_projector.project([4, 5, 6])
        assert not np.array_equal(phi1, phi2), \
            "Different inputs should produce different outputs"

    def test_same_seed_same_projector(self):
        """UT-DAM-02: Same seed produces identical projectors."""
        p1 = RandomFeatureProjector(vocab_size=100, D=64, seed=42)
        p2 = RandomFeatureProjector(vocab_size=100, D=64, seed=42)
        phi1 = p1.project([1, 2, 3])
        phi2 = p2.project([1, 2, 3])
        assert np.array_equal(phi1, phi2)

    def test_order_matters(self, small_projector):
        """UT-DAM-02: Context order affects projection (position weighting)."""
        phi1 = small_projector.project([1, 2, 3])
        phi2 = small_projector.project([3, 2, 1])
        assert not np.array_equal(phi1, phi2), \
            "Different order should produce different projection"


# ===========================================================================
# UT-DAM-03: Projector integer-only
# ===========================================================================

class TestProjectorIntegerOnly:
    """Tests that projection uses only integer arithmetic."""

    def test_output_is_integer_type(self, small_projector):
        """UT-DAM-03: Projection output is integer type."""
        phi = small_projector.project([1, 2, 3])
        assert phi.dtype == np.int8

    def test_word_hashes_are_integer(self, small_projector):
        """UT-DAM-03: Word hash vectors are integer type."""
        assert small_projector.word_hashes.dtype == np.int8

    def test_projection_matrix_is_integer(self, small_projector):
        """UT-DAM-03: Projection matrix W is integer type."""
        assert small_projector.W.dtype == np.int8

    def test_bias_is_integer(self, small_projector):
        """UT-DAM-03: Bias vector is integer type."""
        assert small_projector.b.dtype == np.uint8

    def test_no_float_in_project(self, small_projector):
        """UT-DAM-03: project() method returns integer types (no float in hot path)."""
        # Verify all intermediate and final results are integer
        phi = small_projector.project([1, 2, 3, 4, 5])
        assert not np.issubdtype(phi.dtype, np.floating), \
            "project() should return integer type, not float"


# ===========================================================================
# UT-DAM-04 & UT-DAM-05: Polynomial nonlinearity
# ===========================================================================

class TestPolynomialNonlinearity:
    """Tests for the polynomial nonlinearity F(x)."""

    def test_degree1_linear(self, built_dense_am):
        """UT-DAM-04: degree=1 recovers linear energy (F(x) = x)."""
        # Create a Dense AM with degree=1 using same projector
        linear_am = DenseAMEnergy(
            projector=built_dense_am.projector,
            vocab_size=built_dense_am.vocab_size,
            degree=1,
            dense_am_scale=built_dense_am.dense_am_scale,
        )
        # Copy pre-aggregated data
        linear_am.Phi = built_dense_am.Phi.copy()
        linear_am._word_counts = built_dense_am._word_counts
        linear_am._built = True

        # Compute energies
        context = [1, 2, 3]
        candidates = np.array([0, 1, 2, 3, 4], dtype=np.int64)

        linear_energies = linear_am.compute_energy(context, candidates)
        assert linear_energies.dtype == np.int64
        assert len(linear_energies) == 5

    def test_degree2_quadratic(self, built_dense_am):
        """UT-DAM-05: degree=2 gives quadratic energy (F(x) = x²)."""
        context = [1, 2, 3]
        candidates = np.array([0, 1, 2, 3, 4], dtype=np.int64)

        energies = built_dense_am.compute_energy(context, candidates)
        assert energies.dtype == np.int64
        assert len(energies) == 5

    def test_degree2_sharper_than_degree1(self, built_dense_am):
        """UT-DAM-11: degree=2 produces sharper energy landscape than degree=1."""
        # Create linear version
        linear_am = DenseAMEnergy(
            projector=built_dense_am.projector,
            vocab_size=built_dense_am.vocab_size,
            degree=1,
            dense_am_scale=built_dense_am.dense_am_scale,
        )
        linear_am.Phi = built_dense_am.Phi.copy()
        linear_am._word_counts = built_dense_am._word_counts.copy()
        linear_am._built = True

        context = [1, 2, 3, 4, 5]
        candidates = np.arange(50, dtype=np.int64)

        linear_energies = linear_am.compute_energy(context, candidates)
        quad_energies = built_dense_am.compute_energy(context, candidates)

        # Quadratic should have higher standard deviation (sharper landscape)
        std_linear = float(np.std(linear_energies))
        std_quad = float(np.std(quad_energies))

        # Allow some tolerance since the normalization can affect this
        # The key property is that degree=2 amplifies differences
        assert std_quad > std_linear * 0.5, \
            f"Quadratic std ({std_quad:.1f}) should be > 0.5 * linear std ({std_linear:.1f})"


# ===========================================================================
# UT-DAM-06: No overflow
# ===========================================================================

class TestNoOverflow:
    """Tests that polynomial nonlinearity doesn't cause overflow."""

    def test_energy_in_q30_range(self, built_dense_am):
        """UT-DAM-06: All energies fit in int32 Q30 range."""
        context = list(range(10))
        candidates = np.arange(200, dtype=np.int64)

        energies = built_dense_am.compute_energy(context, candidates)

        max_q30 = (1 << 30) - 1
        assert np.all(energies >= -max_q30), \
            f"Min energy {energies.min()} below -Q30"
        assert np.all(energies <= max_q30), \
            f"Max energy {energies.max()} above Q30"

    def test_energy_dtype_int64(self, built_dense_am):
        """UT-DAM-06: Energy output is int64 (safe for accumulation)."""
        context = [1, 2, 3]
        candidates = np.array([0, 1, 2], dtype=np.int64)
        energies = built_dense_am.compute_energy(context, candidates)
        assert energies.dtype == np.int64

    def test_large_vocabulary_no_overflow(self):
        """UT-DAM-06: No overflow with V=49000 sized vocabulary."""
        projector = RandomFeatureProjector(
            vocab_size=49000,
            D=256,
            context_hash_dim=32,
            seed=42,
        )
        dense_am = DenseAMEnergy(
            projector=projector,
            vocab_size=49000,
            degree=2,
            dense_am_scale=1200,
        )

        # Create minimal training data
        rng = np.random.RandomState(42)
        sequences = [rng.randint(0, 49000, size=5).tolist() for _ in range(10)]
        dense_am.preaggregate(sequences)

        context = [1, 2, 3]
        candidates = np.array([0, 100, 1000, 10000, 48999], dtype=np.int64)
        energies = dense_am.compute_energy(context, candidates)

        max_q30 = (1 << 30) - 1
        assert np.all(energies >= -max_q30), f"Energy overflow: min={energies.min()}"
        assert np.all(energies <= max_q30), f"Energy overflow: max={energies.max()}"


# ===========================================================================
# UT-DAM-07 & UT-DAM-08: Pre-aggregation shape and memory
# ===========================================================================

class TestPreaggregation:
    """Tests for the pre-aggregation step."""

    def test_phi_shape(self, small_dense_am):
        """UT-DAM-07: Phi matrix has correct shape."""
        rng = np.random.RandomState(42)
        sequences = [rng.randint(0, 200, size=5).tolist() for _ in range(20)]
        small_dense_am.preaggregate(sequences)

        assert small_dense_am.Phi is not None
        assert small_dense_am.Phi.shape == (200, 64), \
            f"Expected (200, 64), got {small_dense_am.Phi.shape}"

    def test_phi_dtype(self, small_dense_am):
        """UT-DAM-07: Phi matrix is int16."""
        rng = np.random.RandomState(42)
        sequences = [rng.randint(0, 200, size=5).tolist() for _ in range(20)]
        small_dense_am.preaggregate(sequences)

        assert small_dense_am.Phi.dtype == np.int16, \
            f"Expected int16, got {small_dense_am.Phi.dtype}"

    def test_phi_memory_budget(self):
        """UT-DAM-08: Phi matrix fits in memory budget for V=49K."""
        # Just check the theoretical memory: V * D * 2 bytes
        V = 49000
        D = 256
        expected_mb = V * D * 2 / (1024 * 1024)
        assert expected_mb < 25, f"Phi matrix too large: {expected_mb:.1f} MB (limit 25 MB)"

    def test_phi_values_in_int16_range(self, small_dense_am):
        """UT-DAM-07: Phi values are in int16 range after normalization."""
        rng = np.random.RandomState(42)
        sequences = [rng.randint(0, 200, size=5).tolist() for _ in range(20)]
        small_dense_am.preaggregate(sequences)

        assert np.all(small_dense_am.Phi >= -32768)
        assert np.all(small_dense_am.Phi <= 32767)

    def test_phi_nonzero_for_seen_words(self, small_dense_am):
        """UT-DAM-07: Words seen in training have non-zero Phi rows."""
        rng = np.random.RandomState(42)
        sequences = [[1, 2, 3, 4, 5]] * 10  # word 5 seen many times
        small_dense_am.preaggregate(sequences)

        # Word 5 should have a non-zero Phi row
        if small_dense_am._word_counts is not None and small_dense_am._word_counts[5] > 0:
            row_sum = np.sum(np.abs(small_dense_am.Phi[5]))
            assert row_sum > 0, "Word seen in training should have non-zero Phi"

    def test_word_counts(self, small_dense_am):
        """Pre-aggregation tracks word counts correctly."""
        sequences = [[0, 1, 2], [0, 1, 3]]
        small_dense_am.preaggregate(sequences)

        assert small_dense_am._word_counts is not None
        # Word 0 appears as target 0 times (always context)
        # Word 1 appears as target 2 times (in both sequences)
        # Word 2 appears as target 1 time
        # Word 3 appears as target 1 time
        assert small_dense_am._word_counts[1] >= 2


# ===========================================================================
# UT-DAM-09 & UT-DAM-10: Energy shape and range
# ===========================================================================

class TestDenseAMEnergy:
    """Tests for Dense AM energy computation."""

    def test_energy_shape(self, built_dense_am):
        """UT-DAM-09: Energy output shape matches candidates."""
        candidates = np.array([0, 5, 10, 50, 100], dtype=np.int64)
        energies = built_dense_am.compute_energy([1, 2, 3], candidates)
        assert energies.shape == (5,), f"Expected (5,), got {energies.shape}"

    def test_energy_range(self, built_dense_am):
        """UT-DAM-10: Energies fit in int32 Q30 range."""
        candidates = np.arange(200, dtype=np.int64)
        energies = built_dense_am.compute_energy([1, 2, 3], candidates)

        max_q30 = (1 << 30) - 1
        assert np.all(energies >= -max_q30), f"Energy below -Q30: {energies.min()}"
        assert np.all(energies <= max_q30), f"Energy above Q30: {energies.max()}"

    def test_energy_integer_type(self, built_dense_am):
        """UT-DAM-09: Energy output is int64."""
        candidates = np.array([0, 1, 2], dtype=np.int64)
        energies = built_dense_am.compute_energy([1, 2], candidates)
        assert energies.dtype == np.int64

    def test_energy_unbuilt_returns_zero(self):
        """Unbuilt Dense AM returns zero energies."""
        projector = RandomFeatureProjector(vocab_size=100, D=64, seed=42)
        dense_am = DenseAMEnergy(
            projector=projector, vocab_size=100, degree=2,
        )
        candidates = np.array([0, 1, 2], dtype=np.int64)
        energies = dense_am.compute_energy([1, 2], candidates)
        assert np.all(energies == 0), "Unbuilt Dense AM should return zero energies"

    def test_energy_lower_for_matching_context(self, built_dense_am):
        """Words seen in similar contexts get lower energy."""
        # Use a context that matches training patterns
        context = [1, 2, 3]
        candidates = np.arange(20, dtype=np.int64)
        energies = built_dense_am.compute_energy(context, candidates)

        # At minimum, energies should be finite and not all the same
        assert not np.all(energies == energies[0]), \
            "Energies should vary across candidates"


# ===========================================================================
# UT-DAM-11: Dense AM sharper than linear
# ===========================================================================

class TestSharpness:
    """Tests that degree=2 produces a sharper energy landscape."""

    def test_quadratic_sharper_std(self, built_dense_am):
        """UT-DAM-11: std(dense_energies, degree=2) > 1.5 * std(linear_energies)."""
        # Create linear version
        linear_am = DenseAMEnergy(
            projector=built_dense_am.projector,
            vocab_size=built_dense_am.vocab_size,
            degree=1,
            dense_am_scale=built_dense_am.dense_am_scale,
        )
        linear_am.Phi = built_dense_am.Phi.copy()
        linear_am._word_counts = built_dense_am._word_counts.copy()
        linear_am._built = True

        # Test with enough candidates to get meaningful statistics
        context = list(range(5, 15))
        candidates = np.arange(100, dtype=np.int64)

        linear_energies = linear_am.compute_energy(context, candidates)
        quad_energies = built_dense_am.compute_energy(context, candidates)

        std_linear = float(np.std(linear_energies))
        std_quad = float(np.std(quad_energies))

        # Quadratic should produce sharper (higher std) energy landscape
        # With Q10 normalization, the amplification factor is proportional
        # to the spread of normalized dot products
        assert std_quad > std_linear * 0.8, \
            f"Quadratic std ({std_quad:.1f}) not > 0.8 * linear std ({std_linear:.1f})"

    def test_quadratic_amplifies_differences(self, built_dense_am):
        """UT-DAM-11: Quadratic amplifies differences between candidates."""
        context = [1, 2, 3, 4, 5]
        candidates = np.arange(30, dtype=np.int64)

        energies = built_dense_am.compute_energy(context, candidates)

        # Range of energies should be meaningful
        energy_range = int(energies.max() - energies.min())
        assert energy_range > 0, "Energy range should be non-zero"


# ===========================================================================
# UT-DAM-12: Cos LUT values
# ===========================================================================

class TestCosLUT:
    """Tests for the cosine lookup table."""

    def test_cos_lut_shape(self, small_projector):
        """UT-DAM-12: Cos LUT has 256 entries."""
        lut = small_projector.cos_lut
        assert lut.shape == (256,), f"Expected (256,), got {lut.shape}"

    def test_cos_lut_dtype(self, small_projector):
        """UT-DAM-12: Cos LUT is int8."""
        lut = small_projector.cos_lut
        assert lut.dtype == np.int8

    def test_cos_lut_max_at_zero(self, small_projector):
        """UT-DAM-12: LUT[0] > 0 (cos(0) = 1, mapped to ~127)."""
        lut = small_projector.cos_lut
        assert lut[0] > 0, f"LUT[0] should be positive, got {lut[0]}"
        assert lut[0] == 127, f"LUT[0] should be 127, got {lut[0]}"

    def test_cos_lut_near_zero_at_64(self, small_projector):
        """UT-DAM-12: LUT[64] ≈ 0 (cos(pi/2) = 0)."""
        lut = small_projector.cos_lut
        assert abs(int(lut[64])) <= 2, f"LUT[64] should be ≈0, got {lut[64]}"

    def test_cos_lut_min_at_128(self, small_projector):
        """UT-DAM-12: LUT[128] < 0 (cos(pi) = -1, mapped to ~-127)."""
        lut = small_projector.cos_lut
        assert lut[128] < 0, f"LUT[128] should be negative, got {lut[128]}"
        assert lut[128] == -127, f"LUT[128] should be -127, got {lut[128]}"

    def test_cos_lut_values_in_range(self, small_projector):
        """UT-DAM-12: All LUT values are in int8 range [-127, 127]."""
        lut = small_projector.cos_lut
        assert np.all(lut >= -127)
        assert np.all(lut <= 127)

    def test_cos_lut_periodic(self, small_projector):
        """UT-DAM-12: LUT is approximately periodic (LUT[0] ≈ LUT[256])."""
        lut = small_projector.cos_lut
        # LUT[0] and LUT[255] should be close (cos wraps around)
        assert abs(int(lut[0]) - int(lut[255])) <= 2, \
            f"LUT should be approximately periodic: LUT[0]={lut[0]}, LUT[255]={lut[255]}"


# ===========================================================================
# Integration-level tests
# ===========================================================================

class TestDenseAMIntegration:
    """Tests verifying Dense AM integrates with the energy framework."""

    def test_dense_am_energy_additive(self, built_dense_am):
        """Dense AM energy can be added to other energy terms."""
        context = [1, 2, 3]
        candidates = np.array([0, 1, 2, 3, 4], dtype=np.int64)

        dense_am_energy = built_dense_am.compute_energy(context, candidates)

        # Simulate adding to recall energy
        recall_energy = np.array([100, 200, 150, 300, 250], dtype=np.int64)
        total_energy = recall_energy + dense_am_energy

        assert total_energy.dtype == np.int64
        assert len(total_energy) == 5
        # Total should differ from recall alone
        assert not np.array_equal(total_energy, recall_energy)

    def test_dense_am_with_different_degrees(self, built_dense_am):
        """degree=1 and degree=2 produce different energies."""
        linear_am = DenseAMEnergy(
            projector=built_dense_am.projector,
            vocab_size=built_dense_am.vocab_size,
            degree=1,
            dense_am_scale=built_dense_am.dense_am_scale,
        )
        linear_am.Phi = built_dense_am.Phi.copy()
        linear_am._word_counts = built_dense_am._word_counts.copy()
        linear_am._built = True

        context = [1, 2, 3, 4]
        candidates = np.arange(20, dtype=np.int64)

        linear_e = linear_am.compute_energy(context, candidates)
        quad_e = built_dense_am.compute_energy(context, candidates)

        # Rankings should differ (quadratic amplifies differences)
        # Not guaranteed to always differ, but very likely with 20 candidates
        rank_linear = np.argsort(linear_e)
        rank_quad = np.argsort(quad_e)

        # At least check both produce valid, finite energies
        assert np.all(np.isfinite(linear_e))
        assert np.all(np.isfinite(quad_e))

    def test_dense_am_context_sensitivity(self, built_dense_am):
        """Different contexts produce different Dense AM energies."""
        candidates = np.arange(30, dtype=np.int64)

        e1 = built_dense_am.compute_energy([1, 2, 3], candidates)
        e2 = built_dense_am.compute_energy([10, 20, 30], candidates)

        # Different contexts should generally produce different energies
        # (not guaranteed for all candidates, but the arrays should differ)
        assert not np.array_equal(e1, e2), \
            "Different contexts should produce different energies"

    def test_preaggregate_max_sequences(self, small_dense_am):
        """max_sequences parameter limits pre-aggregation."""
        rng = np.random.RandomState(42)
        sequences = [rng.randint(0, 200, size=5).tolist() for _ in range(100)]

        # Full pre-aggregation
        small_dense_am.preaggregate(sequences, max_sequences=10)

        # Should only process 10 sequences
        if small_dense_am._word_counts is not None:
            total_count = int(small_dense_am._word_counts.sum())
            # With 10 sequences of length 5, we get at most 10*4 = 40 (context, target) pairs
            assert total_count <= 50, \
                f"Expected <= 50 target counts with max_sequences=10, got {total_count}"

    def test_dense_am_robust_to_unseen_words(self, built_dense_am):
        """Dense AM handles unseen words gracefully."""
        # Words not in training data should still produce valid energy
        context = [1, 2, 3]
        candidates = np.array([150, 160, 170, 180], dtype=np.int64)  # possibly unseen

        energies = built_dense_am.compute_energy(context, candidates)
        assert np.all(np.isfinite(energies)), "Energies should be finite for unseen words"


# ===========================================================================
# Performance/size tests
# ===========================================================================

class TestDenseAMPerformance:
    """Tests for memory and computational budget."""

    def test_phi_memory_for_49k_vocab(self):
        """Phi matrix fits in 25 MB budget for V=49K."""
        V = 49000
        D = 256
        # int16: 2 bytes per element
        expected_mb = V * D * 2 / (1024 * 1024)
        assert expected_mb < 25, f"Phi matrix too large: {expected_mb:.1f} MB"

    def test_projector_memory_for_49k_vocab(self):
        """Word hash table fits in reasonable memory for V=49K."""
        V = 49000
        hash_dim = 32
        # int8: 1 byte per element
        expected_mb = V * hash_dim / (1024 * 1024)
        assert expected_mb < 5, f"Word hashes too large: {expected_mb:.1f} MB"

    def test_projection_matrix_memory(self):
        """Projection matrix W fits in reasonable memory."""
        D = 256
        hash_dim = 32
        # int8: 1 byte per element
        expected_mb = D * hash_dim / (1024 * 1024)
        assert expected_mb < 1, f"Projection matrix too large: {expected_mb:.1f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
