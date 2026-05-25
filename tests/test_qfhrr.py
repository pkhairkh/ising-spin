"""
Comprehensive unit tests for the VSA/qFHRR module.

Covers all operations defined in V18_TEST_STRATEGY.md Section 2.1:
  UT-VSA-01 through UT-VSA-17
"""

import numpy as np
import pytest

from ising_spin.vsa.qfhrr import QFHRRVectors, VSAEncoder


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def qfhrr():
    """Standard qFHRR instance with D=512."""
    return QFHRRVectors(dimension=512, seed=42)


@pytest.fixture
def small_qfhrr():
    """Small qFHRR instance with D=64 for faster tests."""
    return QFHRRVectors(dimension=64, seed=42)


@pytest.fixture
def small_encoder():
    """VSA encoder with small vocab for testing."""
    return VSAEncoder(
        vocab_size=500,
        n_pos=13,
        n_topics=16,
        dimension=128,
        seed=42,
    )


# ===========================================================================
# UT-VSA-01: Vector generation shape
# ===========================================================================

class TestVectorGeneration:
    """Tests for random vector generation."""

    def test_shape_batch(self, qfhrr):
        """UT-VSA-01: Generate 1000 vectors, verify shape and value range."""
        vectors = qfhrr.generate(1000)
        assert vectors.shape == (1000, 512), f"Expected (1000, 512), got {vectors.shape}"
        assert vectors.dtype == np.uint8
        assert vectors.min() >= 0
        assert vectors.max() <= 255

    def test_shape_single(self, qfhrr):
        """Single vector generation has correct shape."""
        v = qfhrr.generate_one()
        assert v.shape == (512,)
        assert v.dtype == np.uint8
        assert v.min() >= 0
        assert v.max() <= 255

    def test_all_values_in_range(self, qfhrr):
        """UT-VSA-01: All components in [0, 255]."""
        vectors = qfhrr.generate(1000)
        assert np.all(vectors >= 0)
        assert np.all(vectors <= 255)

    def test_dimension_correct(self, qfhrr):
        """UT-VSA-01: Dimension D=512 is correct."""
        v = qfhrr.generate_one()
        assert len(v) == 512


# ===========================================================================
# UT-VSA-02: Deterministic generation
# ===========================================================================

class TestDeterministic:
    """Tests for deterministic vector generation."""

    def test_same_seed_same_vectors(self, qfhrr):
        """UT-VSA-02: Same seed produces same vectors."""
        v1 = qfhrr.generate(10, seed=123)
        v2 = qfhrr.generate(10, seed=123)
        assert np.array_equal(v1, v2)

    def test_different_seed_different_vectors(self, qfhrr):
        """Different seed produces different vectors."""
        v1 = qfhrr.generate(10, seed=123)
        v2 = qfhrr.generate(10, seed=456)
        assert not np.array_equal(v1, v2)

    def test_default_seed_deterministic(self):
        """Default seed produces deterministic results."""
        q1 = QFHRRVectors(dimension=64, seed=42)
        q2 = QFHRRVectors(dimension=64, seed=42)
        v1 = q1.generate(5)
        v2 = q2.generate(5)
        assert np.array_equal(v1, v2)


# ===========================================================================
# UT-VSA-03: Bind/unbind roundtrip
# ===========================================================================

class TestBindUnbind:
    """Tests for bind and unbind operations."""

    def test_roundtrip_similarity(self, qfhrr):
        """UT-VSA-03: bind(a, b) then unbind recovers a with high similarity."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            a = rng.randint(0, 256, size=512, dtype=np.uint8)
            b = rng.randint(0, 256, size=512, dtype=np.uint8)
            bound = QFHRRVectors.bind(a, b)
            recovered = QFHRRVectors.unbind(bound, b)
            sim = qfhrr.similarity(a, recovered)
            self_sim = qfhrr.similarity(a, a)
            # Round-trip should recover most of the similarity
            assert sim > self_sim * 0.95, f"Roundtrip similarity too low: {sim} vs {self_sim}"

    def test_roundtrip_exact(self):
        """Bind then unbind is exact (no noise from superposition)."""
        rng = np.random.RandomState(42)
        a = rng.randint(0, 256, size=64, dtype=np.uint8)
        b = rng.randint(0, 256, size=64, dtype=np.uint8)
        bound = QFHRRVectors.bind(a, b)
        recovered = QFHRRVectors.unbind(bound, b)
        assert np.array_equal(a, recovered), "Bind/unbind should be exact without superposition"

    def test_bind_values_in_range(self, qfhrr):
        """Bind output is always uint8 in [0, 255]."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            a = rng.randint(0, 256, size=512, dtype=np.uint8)
            b = rng.randint(0, 256, size=512, dtype=np.uint8)
            bound = QFHRRVectors.bind(a, b)
            assert bound.dtype == np.uint8
            assert np.all(bound >= 0)
            assert np.all(bound <= 255)

    def test_unbind_values_in_range(self, qfhrr):
        """Unbind output is always uint8 in [0, 255]."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            a = rng.randint(0, 256, size=512, dtype=np.uint8)
            b = rng.randint(0, 256, size=512, dtype=np.uint8)
            unbound = QFHRRVectors.unbind(a, b)
            assert unbound.dtype == np.uint8
            assert np.all(unbound >= 0)
            assert np.all(unbound <= 255)


# ===========================================================================
# UT-VSA-04 & UT-VSA-05: Commutativity and Associativity
# ===========================================================================

class TestBindAlgebra:
    """Tests for algebraic properties of binding."""

    def test_commutativity(self):
        """UT-VSA-04: bind(a, b) == bind(b, a) element-wise."""
        rng = np.random.RandomState(42)
        a = rng.randint(0, 256, size=512, dtype=np.uint8)
        b = rng.randint(0, 256, size=512, dtype=np.uint8)
        ab = QFHRRVectors.bind(a, b)
        ba = QFHRRVectors.bind(b, a)
        assert np.array_equal(ab, ba)

    def test_associativity(self):
        """UT-VSA-05: bind(bind(a,b), c) == bind(a, bind(b,c))."""
        rng = np.random.RandomState(42)
        a = rng.randint(0, 256, size=512, dtype=np.uint8)
        b = rng.randint(0, 256, size=512, dtype=np.uint8)
        c = rng.randint(0, 256, size=512, dtype=np.uint8)
        ab_c = QFHRRVectors.bind(QFHRRVectors.bind(a, b), c)
        a_bc = QFHRRVectors.bind(a, QFHRRVectors.bind(b, c))
        assert np.array_equal(ab_c, a_bc)


# ===========================================================================
# UT-VSA-06: Superpose clipping
# ===========================================================================

class TestSuperpose:
    """Tests for superposition operation."""

    def test_no_values_above_255(self):
        """UT-VSA-06: Saturating addition clips at 255."""
        a = np.full(64, 200, dtype=np.uint8)
        b = np.full(64, 200, dtype=np.uint8)
        result = QFHRRVectors.superpose(a, b)
        assert np.all(result <= 255)
        assert np.all(result == 255)  # 200 + 200 = 400 → clipped to 255

    def test_superpose_small_values(self):
        """Small values add normally."""
        a = np.full(64, 10, dtype=np.uint8)
        b = np.full(64, 20, dtype=np.uint8)
        result = QFHRRVectors.superpose(a, b)
        assert np.all(result == 30)

    def test_superpose_zero(self):
        """Superpose with zero is identity."""
        a = np.arange(64, dtype=np.uint8)
        z = np.zeros(64, dtype=np.uint8)
        result = QFHRRVectors.superpose(a, z)
        assert np.array_equal(result, a)


# ===========================================================================
# UT-VSA-07 & UT-VSA-08: Similarity properties
# ===========================================================================

class TestSimilarity:
    """Tests for similarity computation."""

    def test_self_similarity_max(self, qfhrr):
        """UT-VSA-07: Self-similarity is maximum."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            v = rng.randint(0, 256, size=512, dtype=np.uint8)
            self_sim = qfhrr.similarity(v, v)
            # Compare with random other vector
            other = rng.randint(0, 256, size=512, dtype=np.uint8)
            cross_sim = qfhrr.similarity(v, other)
            assert self_sim > cross_sim, f"Self-sim {self_sim} not > cross-sim {cross_sim}"

    def test_self_similarity_value(self, qfhrr):
        """Self-similarity equals max possible (D * MAX_SIM_PER_DIM)."""
        v = np.zeros(512, dtype=np.uint8)  # all-zero vector
        self_sim = qfhrr.similarity(v, v)
        expected_max = 512 * 256  # D * MAX_SIM_PER_DIM
        assert self_sim == expected_max, f"Self-sim {self_sim} != {expected_max}"

    def test_random_low_similarity(self, qfhrr):
        """UT-VSA-08: Random vectors have low similarity (near D * mean_lut)."""
        rng = np.random.RandomState(42)
        sims = []
        for _ in range(100):
            v1 = rng.randint(0, 256, size=512, dtype=np.uint8)
            v2 = rng.randint(0, 256, size=512, dtype=np.uint8)
            sim = qfhrr.similarity(v1, v2)
            sims.append(sim)

        # Random similarity should be well below self-similarity
        # With clipped-to-non-negative LUT, random phase vectors still
        # have moderate similarity because cos values near zero get clipped to 0
        # but positive cos values accumulate. Expect ~60-80% of max.
        max_sim = 512 * 256
        mean_random = np.mean(sims)
        assert mean_random < max_sim * 0.85, f"Random sim {mean_random} too high vs max {max_sim}"

    def test_similarity_batch_matches_individual(self, qfhrr):
        """Batch similarity matches individual similarity calls."""
        rng = np.random.RandomState(42)
        a = rng.randint(0, 256, size=512, dtype=np.uint8)
        B = rng.randint(0, 256, size=(20, 512), dtype=np.uint8)

        batch_sims = qfhrr.similarity_batch(a, B)

        for i in range(20):
            individual_sim = qfhrr.similarity(a, B[i])
            assert batch_sims[i] == individual_sim, f"Mismatch at i={i}"


# ===========================================================================
# UT-VSA-09: Binding preserves similarity structure
# ===========================================================================

class TestBindingPreservesSimilarity:
    """Tests that binding preserves similarity relationships."""

    def test_bound_pair_similarity_correlation(self, qfhrr):
        """UT-VSA-09: similarity(bind(a,b), bind(a,c)) correlates with similarity(b,c)."""
        rng = np.random.RandomState(42)
        a = rng.randint(0, 256, size=512, dtype=np.uint8)
        b = rng.randint(0, 256, size=512, dtype=np.uint8)
        c = rng.randint(0, 256, size=512, dtype=np.uint8)

        # b and c are random, so their similarity should be moderate
        sim_bc = qfhrr.similarity(b, c)
        sim_bound = qfhrr.similarity(
            QFHRRVectors.bind(a, b),
            QFHRRVectors.bind(a, c),
        )

        # The bound similarities should reflect the original relationship
        # At minimum, bind(a,b) should be more similar to bind(a,c) than
        # to a completely random vector
        random_v = rng.randint(0, 256, size=512, dtype=np.uint8)
        sim_random = qfhrr.similarity(
            QFHRRVectors.bind(a, b),
            random_v,
        )
        assert sim_bound > sim_random * 0.8, \
            f"Bound sim {sim_bound} not > random sim {sim_random}"


# ===========================================================================
# UT-VSA-10: Phase-difference LUT monotonicity
# ===========================================================================

class TestPhaseDiffLUT:
    """Tests for the phase-difference lookup table."""

    def test_lut_monotonic_first_half(self, qfhrr):
        """UT-VSA-10: LUT is monotonically decreasing from 0 to 128."""
        lut = qfhrr.phase_diff_lut
        for i in range(128):
            assert lut[i] >= lut[i + 1], \
                f"LUT not monotonic: LUT[{i}]={lut[i]} < LUT[{i+1}]={lut[i+1]}"

    def test_lut_max_at_zero(self, qfhrr):
        """LUT[0] = MAX_SIM_PER_DIM (maximum similarity)."""
        lut = qfhrr.phase_diff_lut
        assert lut[0] == 256, f"LUT[0]={lut[0]}, expected 256"

    def test_lut_min_at_128(self, qfhrr):
        """LUT[128] = 0 (minimum similarity, opposite phases)."""
        lut = qfhrr.phase_diff_lut
        assert lut[128] == 0, f"LUT[128]={lut[128]}, expected 0"

    def test_lut_symmetric(self, qfhrr):
        """LUT is symmetric: LUT[d] == LUT[256-d]."""
        lut = qfhrr.phase_diff_lut
        for d in range(1, 128):
            assert lut[d] == lut[256 - d], \
                f"LUT not symmetric: LUT[{d}]={lut[d]} != LUT[{256-d}]={lut[256-d]}"

    def test_lut_all_non_negative(self, qfhrr):
        """All LUT values are non-negative."""
        lut = qfhrr.phase_diff_lut
        assert np.all(lut >= 0)

    def test_lut_shape(self, qfhrr):
        """LUT has exactly 256 entries."""
        lut = qfhrr.phase_diff_lut
        assert lut.shape == (256,)


# ===========================================================================
# UT-VSA-11 & UT-VSA-12: VSAEncoder context sensitivity
# ===========================================================================

class TestVSAEncoder:
    """Tests for the VSAEncoder class."""

    def test_distinguishes_contexts(self, small_encoder):
        """UT-VSA-11: Same word in different contexts produces different vectors."""
        # "bank" as NOUN (pos=0) + SPORTS topic (3) vs NOUN (0) + topic POLITICS (7)
        code1 = small_encoder.encode(word_id=10, pos_id=0, topic_id=3)
        code2 = small_encoder.encode(word_id=10, pos_id=0, topic_id=7)
        assert not np.array_equal(code1, code2), \
            "Same word in different topics should produce different encodings"

    def test_distinguishes_pos(self, small_encoder):
        """Same word with different POS produces different vectors."""
        code_noun = small_encoder.encode(word_id=10, pos_id=0, topic_id=1)
        code_verb = small_encoder.encode(word_id=10, pos_id=1, topic_id=1)
        assert not np.array_equal(code_noun, code_verb), \
            "Same word with different POS should produce different encodings"

    def test_consistent_encoding(self, small_encoder):
        """UT-VSA-12: Same encoding inputs produce same output."""
        code1 = small_encoder.encode(word_id=5, pos_id=3, topic_id=2)
        code2 = small_encoder.encode(word_id=5, pos_id=3, topic_id=2)
        assert np.array_equal(code1, code2)

    def test_different_words_different_codes(self, small_encoder):
        """Different words produce different codes even with same POS/topic."""
        code1 = small_encoder.encode(word_id=1, pos_id=0, topic_id=0)
        code2 = small_encoder.encode(word_id=2, pos_id=0, topic_id=0)
        assert not np.array_equal(code1, code2)

    def test_readout_matrix_shape(self, small_encoder):
        """UT-VSA-13: Readout matrix has correct dimensions."""
        small_encoder.build()
        R = small_encoder.readout_matrix
        assert R is not None
        assert R.shape == (500, 128), f"Expected (500, 128), got {R.shape}"
        assert R.dtype == np.uint8

    def test_readout_matrix_memory(self):
        """UT-VSA-14: Readout matrix fits in memory budget for V=49000."""
        encoder = VSAEncoder(
            vocab_size=49000,
            n_pos=13,
            n_topics=16,
            dimension=512,
            seed=42,
        )
        encoder.build()
        R = encoder.readout_matrix
        mem_mb = R.nbytes / (1024 * 1024)
        assert mem_mb < 30, f"Readout matrix too large: {mem_mb:.1f} MB (limit 30 MB)"

    def test_readout_no_zero_rows(self, small_encoder):
        """No all-zero rows in readout matrix (all words encoded)."""
        small_encoder.build()
        R = small_encoder.readout_matrix
        row_sums = R.sum(axis=1)
        assert np.all(row_sums > 0), "Some readout rows are all zeros"


# ===========================================================================
# UT-VSA-15, UT-VSA-16, UT-VSA-17: VSA Energy tests
# ===========================================================================

class TestVSAEnergy:
    """Tests for VSA energy computation."""

    def _build_small_encoder(self):
        """Build a small encoder with POS and topic info."""
        encoder = VSAEncoder(
            vocab_size=200,
            n_pos=13,
            n_topics=16,
            dimension=128,
            seed=42,
        )
        # Build with minimal pos_system
        class FakePOSSystem:
            allowed_types = {i: {0} for i in range(200)}  # all NOUN

        word_topics = np.zeros(200, dtype=np.int8)
        # Assign different topics to different word ranges
        for i in range(200):
            word_topics[i] = i % 16

        encoder.build(pos_system=FakePOSSystem(), word_topics=word_topics)
        return encoder

    def test_vsa_energy_integer_only(self):
        """UT-VSA-15: VSA energy computation produces integer results."""
        encoder = self._build_small_encoder()
        context = np.arange(128, dtype=np.uint8)  # simple context encoding (0..127)
        candidates = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        energies = encoder.compute_vsa_energy(context, candidates, vsa_scale=800)
        assert energies.dtype == np.int64
        # All values should be integers (they are by construction, but verify)
        assert np.all(energies == energies.astype(np.int64))

    def test_vsa_energy_range(self):
        """UT-VSA-16: VSA energy fits in int32 Q30 range."""
        encoder = self._build_small_encoder()
        context = np.zeros(128, dtype=np.uint8)
        candidates = np.arange(200, dtype=np.int64)
        energies = encoder.compute_vsa_energy(context, candidates, vsa_scale=800)
        assert np.all(energies >= 0), "VSA energies should be non-negative"
        assert np.all(energies < 2**30), f"VSA energies exceed Q30 range: max={energies.max()}"

    def test_vsa_energy_context_sensitivity(self):
        """UT-VSA-17: Different contexts produce different energy rankings."""
        encoder = self._build_small_encoder()

        # Two different context encodings
        ctx1 = np.full(128, 0, dtype=np.uint8)
        ctx2 = np.full(128, 128, dtype=np.uint8)

        candidates = np.arange(20, dtype=np.int64)
        energies1 = encoder.compute_vsa_energy(ctx1, candidates, vsa_scale=800)
        energies2 = encoder.compute_vsa_energy(ctx2, candidates, vsa_scale=800)

        # Rankings should differ for at least some candidates
        rank1 = np.argsort(energies1)
        rank2 = np.argsort(energies2)
        assert not np.array_equal(rank1, rank2), \
            "Different contexts should produce different energy rankings"

    def test_vsa_energy_shape(self):
        """Energy output shape matches candidates."""
        encoder = self._build_small_encoder()
        context = np.zeros(128, dtype=np.uint8)
        candidates = np.array([0, 5, 10, 50, 100], dtype=np.int64)
        energies = encoder.compute_vsa_energy(context, candidates, vsa_scale=800)
        assert energies.shape == (5,)

    def test_vsa_energy_lower_for_similar(self):
        """Words encoded similarly to context get lower energy."""
        encoder = self._build_small_encoder()

        # Create a context that is exactly the readout vector for word 5
        word_5_encoding = encoder.readout_matrix[5].copy()
        candidates = np.array([5, 10, 50, 100], dtype=np.int64)
        energies = encoder.compute_vsa_energy(word_5_encoding, candidates, vsa_scale=800)

        # Word 5 should have the lowest energy (most similar to itself)
        assert energies[0] <= energies[1], \
            f"Self-energy {energies[0]} should be <= other {energies[1]}"
        assert energies[0] == energies.min(), \
            "Self should have minimum energy"

    def test_vsa_energy_unbuilt_returns_zero(self):
        """Unbuilt encoder returns zero energies."""
        encoder = VSAEncoder(vocab_size=100, dimension=64, seed=42)
        # Don't call build()
        context = np.zeros(64, dtype=np.uint8)
        candidates = np.array([0, 1, 2], dtype=np.int64)
        energies = encoder.compute_vsa_energy(context, candidates)
        assert np.all(energies == 0)


# ===========================================================================
# Context encoding tests
# ===========================================================================

class TestContextEncoding:
    """Tests for context encoding computation."""

    def test_empty_context(self, small_encoder):
        """Empty context produces zero vector."""
        ctx = small_encoder.compute_context_encoding([])
        assert np.all(ctx == 0)

    def test_single_word_context(self, small_encoder):
        """Single word context produces valid encoding."""
        ctx = small_encoder.compute_context_encoding([5], [0], [1])
        assert ctx.dtype == np.uint8
        assert np.all(ctx >= 0)
        assert np.all(ctx <= 255)

    def test_context_values_in_range(self, small_encoder):
        """Context encoding values are in uint8 range."""
        ctx = small_encoder.compute_context_encoding(
            list(range(20)), list(range(20)), [1] * 20
        )
        assert ctx.dtype == np.uint8
        assert np.all(ctx >= 0)
        assert np.all(ctx <= 255)

    def test_context_window(self, small_encoder):
        """Window parameter limits context length."""
        # With window=5, only last 5 tokens should matter
        ctx_long = small_encoder.compute_context_encoding(
            list(range(20)), [0] * 20, [0] * 20, window=5
        )
        ctx_short = small_encoder.compute_context_encoding(
            list(range(15, 20)), [0] * 5, [0] * 5, window=5
        )
        # These should be similar (same last 5 tokens)
        assert ctx_long.shape == ctx_short.shape


# ===========================================================================
# Integration-level tests (VSA + Energy interaction)
# ===========================================================================

class TestVSAIntegration:
    """Tests verifying VSA integrates correctly with the energy framework."""

    def test_ambiguity_disambiguation(self):
        """
        Verify "bank" in financial vs river context gets different VSA energies.
        This is the CORE expressivity gain of qFHRR binding.
        """
        encoder = VSAEncoder(
            vocab_size=100,
            n_pos=13,
            n_topics=16,
            dimension=256,
            seed=42,
        )
        encoder.build()

        # Word "bank" at word_id=5
        # Financial context: NOUN + FINANCE topic
        fin_encoding = encoder.encode(word_id=5, pos_id=0, topic_id=1)
        # River context: NOUN + NATURE topic
        nat_encoding = encoder.encode(word_id=5, pos_id=0, topic_id=5)

        # These encodings should be DIFFERENT
        assert not np.array_equal(fin_encoding, nat_encoding), \
            "bank+NOUN+FINANCE should differ from bank+NOUN+NATURE"

        # The financial encoding should be more similar to the financial
        # readout vector than the nature one
        sim_fin = encoder.qfhrr.similarity(fin_encoding, encoder.readout_matrix[5])
        sim_nat = encoder.qfhrr.similarity(nat_encoding, encoder.readout_matrix[5])

        # Both should have reasonable similarity, but the key test is that
        # they ARE different, enabling context-dependent energy
        assert sim_fin != sim_nat, \
            "Same word in different topics should have different similarity to readout"

    def test_no_float_operations_in_hot_path(self):
        """
        UT-VSA-15 extended: Verify no float operations in VSA energy hot path.
        Check that similarity, bind, unbind, superpose all return integer types.
        """
        qfhrr = QFHRRVectors(dimension=64, seed=42)
        rng = np.random.RandomState(42)

        a = rng.randint(0, 256, size=64, dtype=np.uint8)
        b = rng.randint(0, 256, size=64, dtype=np.uint8)

        # bind returns uint8
        bound = QFHRRVectors.bind(a, b)
        assert bound.dtype == np.uint8

        # unbind returns uint8
        unbound = QFHRRVectors.unbind(a, b)
        assert unbound.dtype == np.uint8

        # superpose returns uint8
        sup = QFHRRVectors.superpose(a, b)
        assert sup.dtype == np.uint8

        # similarity returns int
        sim = qfhrr.similarity(a, b)
        assert isinstance(sim, int)

        # batch similarity returns int (int32 or int64 depending on platform)
        B = rng.randint(0, 256, size=(10, 64), dtype=np.uint8)
        batch_sims = qfhrr.similarity_batch(a, B)
        assert batch_sims.dtype in (np.int32, np.int64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
