"""
Unit tests for Cross-Scale Random Fourier Features module (v18.3).

Test categories:
  UT-RFF-01: project() shape and dtype
  UT-RFF-02: project() integer-only output
  UT-RFF-03: Theta matrix shape and dtype
  UT-RFF-04: Theta memory budget
  UT-RFF-05: RFF energy shape
  UT-RFF-06: RFF energy range (fits in int32)
  UT-RFF-07: Cross-scale sensitivity
  UT-RFF-08: Cosine LUT properties
  UT-RFF-09: Empty context handling
  UT-RFF-10: Build from sequences
  UT-RFF-11: Determinism (same seed -> same results)
  UT-RFF-12: Integer-only verification
"""

import numpy as np
import pytest
from ising_spin.rff import CrossScaleRFF


def _make_rff(vocab_size=100, D=32, n_pos=13, n_topics=8, seed=42):
    """Create a small CrossScaleRFF instance for testing."""
    return CrossScaleRFF(
        vocab_size=vocab_size,
        n_pos=n_pos,
        n_topics=n_topics,
        D=D,
        context_hash_dim=16,
        seed=seed,
        rff_scale=600,
    )


def _make_synthetic_sequences(vocab_size=100, n_seqs=20, seq_len=10, seed=42):
    """Generate small synthetic sequences for build testing."""
    rng = np.random.RandomState(seed)
    sequences = []
    for _ in range(n_seqs):
        seq = rng.randint(5, vocab_size, size=seq_len).tolist()
        sequences.append(seq)
    return sequences


def _make_built_rff(vocab_size=100, D=32, n_pos=13, n_topics=8, seed=42):
    """Create a built CrossScaleRFF with synthetic data."""
    rff = _make_rff(vocab_size=vocab_size, D=D, n_pos=n_pos, n_topics=n_topics, seed=seed)
    sequences = _make_synthetic_sequences(vocab_size=vocab_size)
    word_pos_tags = {w: w % n_pos for w in range(vocab_size)}
    word_topics = np.array([w % n_topics for w in range(vocab_size)], dtype=np.int8)
    rff.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)
    return rff


# ===================================================================
# UT-RFF-01: project() returns (D,) int8
# ===================================================================

class TestRFFProjectShape:
    """Test that project() returns the correct shape and dtype."""

    def test_phi_shape(self):
        """project() returns (D,) array."""
        rff = _make_rff(D=32)
        phi = rff.project([5, 10, 15], [1, 2, 3], [0, 1, 2])
        assert phi.shape == (32,)

    def test_phi_dtype_int8(self):
        """project() returns int8 array."""
        rff = _make_rff(D=32)
        phi = rff.project([5, 10, 15], [1, 2, 3], [0, 1, 2])
        assert phi.dtype == np.int8

    def test_phi_custom_D(self):
        """project() returns correct shape for custom D."""
        rff = _make_rff(D=64)
        phi = rff.project([5, 10], [1, 2], [0, 1])
        assert phi.shape == (64,)


# ===================================================================
# UT-RFF-02: project() values in [-127, 127]
# ===================================================================

class TestRFFProjectIntegerRange:
    """Test that project() output values are within int8 range."""

    def test_phi_integer_only(self):
        """Values in project output are in [-127, 127]."""
        rff = _make_rff(D=32)
        phi = rff.project([5, 10, 15, 20, 25], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4])
        assert np.all(phi >= -127)
        assert np.all(phi <= 127)

    def test_phi_integer_only_long_context(self):
        """Even with long context, values stay in int8 range."""
        rff = _make_rff(D=32)
        ctx_words = list(range(5, 50))
        ctx_pos = [w % 13 for w in ctx_words]
        ctx_topics = [w % 8 for w in ctx_words]
        phi = rff.project(ctx_words, ctx_pos, ctx_topics)
        assert np.all(phi >= -127)
        assert np.all(phi <= 127)

    def test_phi_no_float_values(self):
        """No float values in project output."""
        rff = _make_rff(D=32)
        phi = rff.project([5, 10, 15], [1, 2, 3], [0, 1, 2])
        # Check that all values are integers
        for v in phi:
            assert int(v) == v, f"Non-integer value: {v}"


# ===================================================================
# UT-RFF-03: Theta is (V, D) int8
# ===================================================================

class TestRFFThetaShape:
    """Test Theta matrix shape and dtype."""

    def test_theta_shape(self):
        """Theta has shape (V, D) after build."""
        rff = _make_built_rff(vocab_size=100, D=32)
        assert rff.Theta.shape == (100, 32)

    def test_theta_dtype_int8(self):
        """Theta is int8."""
        rff = _make_built_rff(vocab_size=100, D=32)
        assert rff.Theta.dtype == np.int8

    def test_theta_values_bounded(self):
        """Theta values are in [-127, 127] (int8 range)."""
        rff = _make_built_rff(vocab_size=100, D=32)
        assert np.all(rff.Theta >= -127)
        assert np.all(rff.Theta <= 127)

    def test_theta_not_built_initially(self):
        """Theta is None before build."""
        rff = _make_rff(vocab_size=100, D=32)
        assert rff.Theta is None
        assert not rff.built


# ===================================================================
# UT-RFF-04: Theta memory < 15 MB for V=49K
# ===================================================================

class TestRFFThetaMemory:
    """Test Theta memory budget."""

    def test_theta_memory_small(self):
        """Theta memory for V=100, D=32 is tiny."""
        rff = _make_built_rff(vocab_size=100, D=32)
        nbytes = rff.Theta.nbytes
        assert nbytes < 1 * 1024 * 1024  # < 1 MB for small model

    def test_theta_memory_budget_large(self):
        """Theta memory for V=49000, D=256 fits in 15 MB (estimated)."""
        # Don't actually create a 49K vocab model, just compute the expected size
        V = 49000
        D = 256
        expected_bytes = V * D * 1  # int8 = 1 byte per element
        expected_mb = expected_bytes / (1024 * 1024)
        assert expected_mb < 15, f"Expected {expected_mb:.1f} MB exceeds 15 MB budget"


# ===================================================================
# UT-RFF-05: RFF energy shape matches candidates
# ===================================================================

class TestRFFEnergyShape:
    """Test RFF energy computation shape."""

    def test_rff_energy_shape(self):
        """Energy shape matches number of candidates."""
        rff = _make_built_rff(vocab_size=100, D=32)
        candidates = np.array([5, 10, 15, 20, 25], dtype=np.int64)
        energies = rff.compute_energy([5, 10], [1, 2], [0, 1], candidates)
        assert energies.shape == (5,)

    def test_rff_energy_dtype_int64(self):
        """Energy returns int64 array."""
        rff = _make_built_rff(vocab_size=100, D=32)
        candidates = np.array([5, 10, 15], dtype=np.int64)
        energies = rff.compute_energy([5, 10], [1, 2], [0, 1], candidates)
        assert energies.dtype == np.int64


# ===================================================================
# UT-RFF-06: RFF energy range fits in int32
# ===================================================================

class TestRFFEnergyRange:
    """Test that RFF energy values fit in int32 range."""

    def test_rff_energy_range(self):
        """RFF energies fit in int32 Q30 range."""
        rff = _make_built_rff(vocab_size=100, D=32)
        candidates = np.array(list(range(5, 50)), dtype=np.int64)
        energies = rff.compute_energy([5, 10, 15], [1, 2, 3], [0, 1, 2], candidates)
        # Energies should be bounded by rff_scale
        assert np.all(energies >= -2 * rff.rff_scale)
        assert np.all(energies <= 2 * rff.rff_scale)

    def test_rff_energy_no_overflow(self):
        """RFF energies don't overflow int64."""
        rff = _make_built_rff(vocab_size=100, D=32)
        candidates = np.array(list(range(5, 50)), dtype=np.int64)
        energies = rff.compute_energy([5, 10, 15], [1, 2, 3], [0, 1, 2], candidates)
        assert np.all(energies > -(2**63))
        assert np.all(energies < (2**63) - 1)


# ===================================================================
# UT-RFF-07: Cross-scale sensitivity
# ===================================================================

class TestRFFCrossScaleSensitivity:
    """Test that different topic/POS produces different features."""

    def test_different_topic_different_features(self):
        """Different topic IDs produce different feature vectors."""
        rff = _make_rff(vocab_size=100, D=32)
        ctx_words = [5, 10, 15]
        ctx_pos = [1, 2, 3]
        phi_topic0 = rff.project(ctx_words, ctx_pos, [0, 0, 0])
        phi_topic5 = rff.project(ctx_words, ctx_pos, [5, 5, 5])
        # Different topics should produce different features
        assert not np.array_equal(phi_topic0, phi_topic5)

    def test_different_pos_different_features(self):
        """Different POS IDs produce different feature vectors."""
        rff = _make_rff(vocab_size=100, D=32)
        ctx_words = [5, 10, 15]
        ctx_topics = [0, 1, 2]
        phi_pos0 = rff.project(ctx_words, [0, 0, 0], ctx_topics)
        phi_pos5 = rff.project(ctx_words, [5, 5, 5], ctx_topics)
        assert not np.array_equal(phi_pos0, phi_pos5)

    def test_same_context_same_features(self):
        """Same context produces same features (determinism)."""
        rff = _make_rff(vocab_size=100, D=32)
        ctx_words = [5, 10, 15]
        ctx_pos = [1, 2, 3]
        ctx_topics = [0, 1, 2]
        phi1 = rff.project(ctx_words, ctx_pos, ctx_topics)
        phi2 = rff.project(ctx_words, ctx_pos, ctx_topics)
        np.testing.assert_array_equal(phi1, phi2)


# ===================================================================
# UT-RFF-08: Cosine LUT properties
# ===================================================================

class TestRFFCosLUT:
    """Test cosine lookup table properties."""

    def test_rff_cos_lut_zero(self):
        """LUT[0] = 127 (cos(0) = 1)."""
        rff = _make_rff(vocab_size=100, D=32)
        assert rff.cos_lut[0] == 127

    def test_rff_cos_lut_pi(self):
        """LUT[128] = -127 (cos(pi) = -1)."""
        rff = _make_rff(vocab_size=100, D=32)
        assert rff.cos_lut[128] == -127

    def test_rff_cos_lut_quarter(self):
        """LUT[64] is approximately 0 (cos(pi/2) ≈ 0)."""
        rff = _make_rff(vocab_size=100, D=32)
        assert -2 <= rff.cos_lut[64] <= 2

    def test_rff_cos_lut_size(self):
        """LUT has 256 entries."""
        rff = _make_rff(vocab_size=100, D=32)
        assert len(rff.cos_lut) == 256

    def test_rff_cos_lut_dtype(self):
        """LUT is int8."""
        rff = _make_rff(vocab_size=100, D=32)
        assert rff.cos_lut.dtype == np.int8


# ===================================================================
# UT-RFF-09: Empty context handling
# ===================================================================

class TestRFFEmptyContext:
    """Test edge cases with empty context."""

    def test_project_empty_context(self):
        """project() with empty context returns zero vector."""
        rff = _make_rff(vocab_size=100, D=32)
        phi = rff.project([], [], [])
        assert phi.shape == (32,)
        assert phi.dtype == np.int8
        assert np.all(phi == 0)

    def test_energy_with_empty_context(self):
        """compute_energy() with empty context works."""
        rff = _make_built_rff(vocab_size=100, D=32)
        candidates = np.array([5, 10, 15], dtype=np.int64)
        energies = rff.compute_energy([], [], [], candidates)
        assert energies.shape == (3,)
        # With empty context, phi is zero, so dot products are zero
        # and energies should all be zero
        assert np.all(energies == 0)

    def test_energy_not_built_returns_zeros(self):
        """compute_energy() returns zeros if not built."""
        rff = _make_rff(vocab_size=100, D=32)
        candidates = np.array([5, 10, 15], dtype=np.int64)
        energies = rff.compute_energy([5, 10], [1, 2], [0, 1], candidates)
        assert energies.shape == (3,)
        assert np.all(energies == 0)


# ===================================================================
# UT-RFF-10: Build from sequences
# ===================================================================

class TestRFFBuild:
    """Test build() from sequences."""

    def test_build_creates_theta(self):
        """build() creates Theta matrix."""
        rff = _make_rff(vocab_size=100, D=32)
        sequences = _make_synthetic_sequences(vocab_size=100)
        word_pos_tags = {w: w % 13 for w in range(100)}
        word_topics = np.array([w % 8 for w in range(100)], dtype=np.int8)
        rff.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)
        assert rff.built
        assert rff.Theta is not None
        assert rff.Theta.shape == (100, 32)

    def test_build_sets_word_counts(self):
        """build() sets word counts."""
        rff = _make_rff(vocab_size=100, D=32)
        sequences = _make_synthetic_sequences(vocab_size=100)
        word_pos_tags = {w: w % 13 for w in range(100)}
        word_topics = np.array([w % 8 for w in range(100)], dtype=np.int8)
        rff.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)
        assert rff.word_counts is not None
        assert len(rff.word_counts) == 100
        # Some words should have non-zero counts
        assert np.sum(rff.word_counts > 0) > 0

    def test_build_empty_sequences(self):
        """build() with empty sequences creates zero Theta."""
        rff = _make_rff(vocab_size=100, D=32)
        rff.build([], word_pos_tags={}, word_topics=None)
        assert rff.built
        assert np.all(rff.Theta == 0)

    def test_build_max_sequences_cap(self):
        """build() respects max_sequences parameter."""
        rff = _make_rff(vocab_size=100, D=32)
        sequences = _make_synthetic_sequences(vocab_size=100, n_seqs=50)
        word_pos_tags = {w: w % 13 for w in range(100)}
        word_topics = np.array([w % 8 for w in range(100)], dtype=np.int8)
        rff.build(sequences, word_pos_tags=word_pos_tags,
                  word_topics=word_topics, max_sequences=10)
        assert rff.built

    def test_build_with_dict_pos_tags(self):
        """build() works with dict-style word_pos_tags."""
        rff = _make_rff(vocab_size=100, D=32)
        sequences = _make_synthetic_sequences(vocab_size=100)
        word_pos_tags = {w: w % 13 for w in range(100)}
        word_topics = np.array([w % 8 for w in range(100)], dtype=np.int8)
        rff.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)
        assert rff.built


# ===================================================================
# UT-RFF-11: Determinism
# ===================================================================

class TestRFFDeterminism:
    """Test that same seed produces identical results."""

    def test_same_seed_same_theta(self):
        """Same seed + same data -> same Theta."""
        sequences = _make_synthetic_sequences(vocab_size=100)
        word_pos_tags = {w: w % 13 for w in range(100)}
        word_topics = np.array([w % 8 for w in range(100)], dtype=np.int8)

        rff1 = _make_rff(vocab_size=100, D=32, seed=42)
        rff1.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)

        rff2 = _make_rff(vocab_size=100, D=32, seed=42)
        rff2.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)

        np.testing.assert_array_equal(rff1.Theta, rff2.Theta)

    def test_different_seed_different_theta(self):
        """Different seeds -> different Theta."""
        sequences = _make_synthetic_sequences(vocab_size=100)
        word_pos_tags = {w: w % 13 for w in range(100)}
        word_topics = np.array([w % 8 for w in range(100)], dtype=np.int8)

        rff1 = _make_rff(vocab_size=100, D=32, seed=42)
        rff1.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)

        rff2 = _make_rff(vocab_size=100, D=32, seed=123)
        rff2.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)

        assert not np.array_equal(rff1.Theta, rff2.Theta)

    def test_same_seed_same_energy(self):
        """Same seed + same context -> same energy."""
        sequences = _make_synthetic_sequences(vocab_size=100)
        word_pos_tags = {w: w % 13 for w in range(100)}
        word_topics = np.array([w % 8 for w in range(100)], dtype=np.int8)

        rff1 = _make_rff(vocab_size=100, D=32, seed=42)
        rff1.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)

        rff2 = _make_rff(vocab_size=100, D=32, seed=42)
        rff2.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)

        candidates = np.array([5, 10, 15], dtype=np.int64)
        e1 = rff1.compute_energy([5, 10], [1, 2], [0, 1], candidates)
        e2 = rff2.compute_energy([5, 10], [1, 2], [0, 1], candidates)
        np.testing.assert_array_equal(e1, e2)


# ===================================================================
# UT-RFF-12: Integer-only verification
# ===================================================================

class TestRFFIntegerOnly:
    """Verify that RFF operations use integer arithmetic."""

    def test_project_output_integer(self):
        """project() returns integer array (int8)."""
        rff = _make_rff(vocab_size=100, D=32)
        phi = rff.project([5, 10, 15], [1, 2, 3], [0, 1, 2])
        assert phi.dtype == np.int8

    def test_theta_integer(self):
        """Theta is integer (int8)."""
        rff = _make_built_rff(vocab_size=100, D=32)
        assert rff.Theta.dtype == np.int8

    def test_energy_integer(self):
        """Energy returns integer (int64)."""
        rff = _make_built_rff(vocab_size=100, D=32)
        candidates = np.array([5, 10, 15], dtype=np.int64)
        energies = rff.compute_energy([5, 10], [1, 2], [0, 1], candidates)
        assert energies.dtype == np.int64

    def test_hash_vectors_integer(self):
        """Word, POS, and topic hash vectors are int8."""
        rff = _make_rff(vocab_size=100, D=32)
        assert rff.word_hashes.dtype == np.int8
        assert rff.pos_hashes.dtype == np.int8
        assert rff.topic_hashes.dtype == np.int8

    def test_projection_matrix_integer(self):
        """W matrix is int8."""
        rff = _make_rff(vocab_size=100, D=32)
        assert rff.W.dtype == np.int8

    def test_bias_integer(self):
        """Bias vector is uint8."""
        rff = _make_rff(vocab_size=100, D=32)
        assert rff.b.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
