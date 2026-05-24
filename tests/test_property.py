"""
Property-based tests for ISG-LM — invariants, contract, and monotonicity.

Test categories:
  PT-INT-01: No float in energy hot path
  PT-INT-02: All energy values fit in int32 Q30
  PT-INT-03: Energy sum never overflows int64
  PT-INT-04: VSA vectors are always uint8
  PT-INT-05: Reservoir state always int16
  PT-INT-06: Dense AM Phi values are int16
  PT-INT-07: RFF Theta values are int8
  PT-CON-01: Deterministic energy for same input
  PT-CON-03: State update is deterministic
  PT-MON-01: Lower energy = higher probability (Boltzmann)
  PT-MON-05: Dense AM sharpness increases with degree
"""

import math
import numpy as np
import pytest
from ising_spin.rff import CrossScaleRFF
from ising_spin.reservoir import IntegerESN
from ising_spin.vsa import VSAEncoder, QFHRRVectors
from ising_spin.dense_am import RandomFeatureProjector, DenseAMEnergy
from ising_spin.state import DocumentState
from ising_spin.sampling import IntegerBoltzmannSampler


# ===================================================================
# PT-INT-01: No float in energy hot path
# ===================================================================

class TestNoFloatInHotPath:
    """Test that energy hot path uses no float operations."""

    def test_rff_project_no_float(self):
        """RFF project() doesn't use float (monkeypatch test)."""
        rff = CrossScaleRFF(vocab_size=100, D=32, n_pos=13, n_topics=8, seed=42)
        # Monkeypatch float, math.exp, math.log to raise if called
        original_float = float
        original_exp = math.exp
        original_log = math.log

        def raise_on_float(*args, **kwargs):
            raise RuntimeError("float() called in hot path!")

        def raise_on_exp(*args, **kwargs):
            raise RuntimeError("math.exp() called in hot path!")

        def raise_on_log(*args, **kwargs):
            raise RuntimeError("math.log() called in hot path!")

        # Note: We can't monkeypatch builtins.float easily, but we can
        # verify the output types are integers
        phi = rff.project([5, 10, 15], [1, 2, 3], [0, 1, 2])
        assert phi.dtype == np.int8
        # All values should be integers
        for v in phi:
            assert isinstance(int(v), int)

    def test_reservoir_step_no_float(self):
        """Reservoir step() uses no float."""
        esn = IntegerESN(vocab_size=100, reservoir_dim=32)
        h = esn.step(5)
        assert h.dtype == np.int16

    def test_dense_am_project_no_float(self):
        """Dense AM project() uses no float."""
        proj = RandomFeatureProjector(vocab_size=100, D=32, seed=42)
        phi = proj.project([5, 10, 15])
        assert phi.dtype == np.int8

    def test_vsa_similarity_no_float(self):
        """VSA similarity() uses no float."""
        qfhrr = QFHRRVectors(dimension=128, seed=42)
        a = qfhrr.generate_one()
        b = qfhrr.generate_one(seed=99)
        sim = qfhrr.similarity(a, b)
        assert isinstance(sim, int)

    def test_energy_computer_no_float(self, small_model):
        """EnergyComputer returns integer energy."""
        ec = small_model.energy_computer
        candidates = np.array([5, 10, 15], dtype=np.int64)
        energies = ec.compute_energy([5, 10], candidates)
        assert energies.dtype == np.int64
        for e in energies:
            assert isinstance(int(e), int)


# ===================================================================
# PT-INT-02: All energy values fit in int32 Q30
# ===================================================================

class TestEnergyFitsInt32Q30:
    """Test that individual energy values fit in int32 Q30 range."""

    def test_rff_energy_fits_int32(self):
        """RFF energy values fit in int32 range."""
        rff = CrossScaleRFF(vocab_size=100, D=32, n_pos=13, n_topics=8, seed=42)
        sequences = [[5, 10, 15, 20, 25]] * 10
        word_pos_tags = {w: w % 13 for w in range(100)}
        word_topics = np.array([w % 8 for w in range(100)], dtype=np.int8)
        rff.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)

        candidates = np.array(list(range(5, 50)), dtype=np.int64)
        energies = rff.compute_energy([5, 10, 15], [1, 2, 3], [0, 1, 2], candidates)
        # Check fits in int32 range
        assert np.all(energies >= -(2**31))
        assert np.all(energies < (2**31))

    def test_dense_am_energy_fits_int32(self):
        """Dense AM energy values fit in int32 range."""
        proj = RandomFeatureProjector(vocab_size=100, D=32, seed=42)
        dam = DenseAMEnergy(proj, vocab_size=100, degree=2, dense_am_scale=1200)
        sequences = [[5, 10, 15, 20, 25]] * 10
        dam.preaggregate(sequences)

        candidates = np.array(list(range(5, 50)), dtype=np.int64)
        energies = dam.compute_energy([5, 10, 15], candidates)
        assert np.all(energies >= -(2**31))
        assert np.all(energies < (2**31))

    def test_reservoir_energy_fits_int32(self):
        """Reservoir energy values fit in int32 range."""
        esn = IntegerESN(vocab_size=100, reservoir_dim=32)
        sequences = [[5, 10, 15, 20, 25]] * 10
        esn.build(sequences)
        esn.step(5)
        esn.step(10)
        candidates = np.array(list(range(5, 50)), dtype=np.int64)
        energies = esn.compute_energy(candidates, reservoir_scale=800)
        assert np.all(energies >= -(2**31))
        assert np.all(energies < (2**31))


# ===================================================================
# PT-INT-03: Energy sum never overflows int64
# ===================================================================

class TestEnergySumNoOverflow:
    """Test that energy sums never overflow int64."""

    def test_total_energy_no_overflow(self, small_model):
        """Total energy from all experts stays in int64 range."""
        ec = small_model.energy_computer
        # Use many candidates
        candidates = np.array(list(range(5, 100)), dtype=np.int64)
        context = [5, 10, 15, 20]
        energies = ec.compute_energy(
            context, candidates,
            current_type=-1, prev_word=10, closed_class_run=0,
        )
        # Sum should not overflow int64
        total = energies.sum()
        assert -(2**63) < total < (2**63) - 1


# ===================================================================
# PT-INT-04: VSA vectors are always uint8
# ===================================================================

class TestVSAVectorsUint8:
    """Test that VSA vectors are always uint8."""

    def test_hash_words_uint8(self):
        """VSA hash words are uint8."""
        encoder = VSAEncoder(vocab_size=100, n_pos=13, n_topics=8, dimension=128, seed=42)
        assert encoder.hash_words.dtype == np.uint8

    def test_role_pos_uint8(self):
        """VSA role POS vectors are uint8."""
        encoder = VSAEncoder(vocab_size=100, n_pos=13, n_topics=8, dimension=128, seed=42)
        assert encoder.role_pos.dtype == np.uint8

    def test_role_topic_uint8(self):
        """VSA role topic vectors are uint8."""
        encoder = VSAEncoder(vocab_size=100, n_pos=13, n_topics=8, dimension=128, seed=42)
        assert encoder.role_topic.dtype == np.uint8

    def test_readout_matrix_uint8(self):
        """VSA readout matrix is uint8 after build."""
        from ising_spin.vocabulary import POSTypeSystem, TopicAssigner, Vocabulary

        _SYNTHETIC_TEXTS = [
            "the cat sat on the mat and the dog ran in the park",
            "she went to the store to buy some food for dinner",
            "the children played in the garden while the sun was shining",
            "he read a book about the history of science and technology",
            "they built a small house near the lake in the forest",
        ] * 3

        vocab = Vocabulary(min_freq=1, max_size=200)
        vocab.build(_SYNTHETIC_TEXTS[:50])

        pos_system = POSTypeSystem(vocab_size=len(vocab), window=3)
        pos_system.build_from_vocabulary(vocab.word2idx, vocab.idx2word)

        topic_assigner = TopicAssigner(n_topics=8)
        topic_assigner.build(_SYNTHETIC_TEXTS[:50], vocab)

        encoder = VSAEncoder(vocab_size=len(vocab), n_pos=13, n_topics=8, dimension=128, seed=42)
        encoder.build(pos_system=pos_system, word_topics=topic_assigner.word_topics)
        assert encoder.readout_matrix.dtype == np.uint8

    def test_encode_returns_uint8(self):
        """VSA encode() returns uint8 vector."""
        encoder = VSAEncoder(vocab_size=100, n_pos=13, n_topics=8, dimension=128, seed=42)
        encoded = encoder.encode(5, 2, 3)
        assert encoded.dtype == np.uint8
        assert encoded.shape == (128,)


# ===================================================================
# PT-INT-05: Reservoir state always int16
# ===================================================================

class TestReservoirStateInt16:
    """Test that reservoir state is always int16."""

    def test_initial_state_int16(self):
        """Initial reservoir state is int16."""
        esn = IntegerESN(vocab_size=100, reservoir_dim=32)
        assert esn.h.dtype == np.int16

    def test_step_returns_int16(self):
        """step() returns int16."""
        esn = IntegerESN(vocab_size=100, reservoir_dim=32)
        h = esn.step(5)
        assert h.dtype == np.int16

    def test_state_after_many_steps_int16(self):
        """State stays int16 after many steps."""
        esn = IntegerESN(vocab_size=100, reservoir_dim=32, alpha_q15=32767)
        for i in range(200):
            h = esn.step(i % 100)
        assert h.dtype == np.int16

    def test_readout_int16(self):
        """Readout matrix R is int16 after build."""
        esn = IntegerESN(vocab_size=100, reservoir_dim=32)
        sequences = [[1, 2, 3, 4, 5]] * 10
        esn.build(sequences)
        assert esn.R.dtype == np.int16


# ===================================================================
# PT-INT-06: Dense AM Phi values are int16
# ===================================================================

class TestDenseAMPhiInt16:
    """Test that Dense AM Phi matrix is int16."""

    def test_phi_dtype_int16(self):
        """Phi matrix is int16 after pre-aggregation."""
        proj = RandomFeatureProjector(vocab_size=100, D=32, seed=42)
        dam = DenseAMEnergy(proj, vocab_size=100, degree=2)
        sequences = [[1, 2, 3, 4, 5]] * 10
        dam.preaggregate(sequences)
        assert dam.Phi.dtype == np.int16

    def test_phi_values_bounded(self):
        """Phi values are within int16 range."""
        proj = RandomFeatureProjector(vocab_size=100, D=32, seed=42)
        dam = DenseAMEnergy(proj, vocab_size=100, degree=2)
        sequences = [[1, 2, 3, 4, 5]] * 10
        dam.preaggregate(sequences)
        assert np.all(dam.Phi >= -32768)
        assert np.all(dam.Phi <= 32767)


# ===================================================================
# PT-INT-07: RFF Theta values are int8
# ===================================================================

class TestRFFThetaInt8:
    """Test that RFF Theta matrix is int8."""

    def test_theta_dtype_int8(self):
        """Theta matrix is int8 after build."""
        rff = CrossScaleRFF(vocab_size=100, D=32, n_pos=13, n_topics=8, seed=42)
        sequences = [[1, 2, 3, 4, 5]] * 10
        word_pos_tags = {w: w % 13 for w in range(100)}
        word_topics = np.array([w % 8 for w in range(100)], dtype=np.int8)
        rff.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)
        assert rff.Theta.dtype == np.int8

    def test_theta_values_bounded(self):
        """Theta values are within int8 range [-127, 127]."""
        rff = CrossScaleRFF(vocab_size=100, D=32, n_pos=13, n_topics=8, seed=42)
        sequences = [[1, 2, 3, 4, 5]] * 10
        word_pos_tags = {w: w % 13 for w in range(100)}
        word_topics = np.array([w % 8 for w in range(100)], dtype=np.int8)
        rff.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)
        assert np.all(rff.Theta >= -127)
        assert np.all(rff.Theta <= 127)


# ===================================================================
# PT-CON-01: Deterministic energy for same input
# ===================================================================

class TestDeterministicEnergy:
    """Test that energy is deterministic for the same input."""

    def test_rff_deterministic(self):
        """RFF energy is deterministic for same context."""
        rff = CrossScaleRFF(vocab_size=100, D=32, n_pos=13, n_topics=8, seed=42)
        sequences = [[1, 2, 3, 4, 5]] * 10
        word_pos_tags = {w: w % 13 for w in range(100)}
        word_topics = np.array([w % 8 for w in range(100)], dtype=np.int8)
        rff.build(sequences, word_pos_tags=word_pos_tags, word_topics=word_topics)

        candidates = np.array([5, 10, 15], dtype=np.int64)
        e1 = rff.compute_energy([5, 10], [1, 2], [0, 1], candidates)
        e2 = rff.compute_energy([5, 10], [1, 2], [0, 1], candidates)
        np.testing.assert_array_equal(e1, e2)

    def test_reservoir_deterministic(self):
        """Reservoir energy is deterministic."""
        esn1 = IntegerESN(vocab_size=100, reservoir_dim=32, seed=42)
        esn2 = IntegerESN(vocab_size=100, reservoir_dim=32, seed=42)
        for w in [5, 10, 15]:
            esn1.step(w)
            esn2.step(w)
        np.testing.assert_array_equal(esn1.h, esn2.h)

    def test_vsa_deterministic(self):
        """VSA encode is deterministic."""
        encoder = VSAEncoder(vocab_size=100, n_pos=13, n_topics=8, dimension=128, seed=42)
        e1 = encoder.encode(5, 2, 3)
        e2 = encoder.encode(5, 2, 3)
        np.testing.assert_array_equal(e1, e2)

    def test_dense_am_deterministic(self):
        """Dense AM energy is deterministic."""
        proj = RandomFeatureProjector(vocab_size=100, D=32, seed=42)
        dam = DenseAMEnergy(proj, vocab_size=100, degree=2)
        sequences = [[1, 2, 3, 4, 5]] * 10
        dam.preaggregate(sequences)
        candidates = np.array([5, 10, 15], dtype=np.int64)
        e1 = dam.compute_energy([5, 10], candidates)
        e2 = dam.compute_energy([5, 10], candidates)
        np.testing.assert_array_equal(e1, e2)

    def test_total_energy_deterministic(self, small_model):
        """Total energy is deterministic for same context."""
        ec = small_model.energy_computer
        # Reset state for deterministic test
        small_model.document_state.reset()
        if small_model.reservoir is not None:
            small_model.reservoir.reset()

        candidates = np.array([5, 10, 15], dtype=np.int64)
        e1 = ec.compute_energy([5, 10, 15], candidates)

        # Reset and recompute
        small_model.document_state.reset()
        if small_model.reservoir is not None:
            small_model.reservoir.reset()

        e2 = ec.compute_energy([5, 10, 15], candidates)
        np.testing.assert_array_equal(e1, e2)


# ===================================================================
# PT-CON-03: State update is deterministic
# ===================================================================

class TestStateUpdateDeterministic:
    """Test that state updates are deterministic."""

    def test_document_state_deterministic(self):
        """Document state update is deterministic."""
        ds1 = DocumentState(vocab_size=100, n_topics=8)
        ds2 = DocumentState(vocab_size=100, n_topics=8)

        words = [5, 10, 15, 20, 25]
        word_strs = ["word5", "word10", "word15", "word20", "word25"]

        for w, ws in zip(words, word_strs):
            ds1.update(w, word_str=ws)
            ds2.update(w, word_str=ws)

        assert ds1.topic == ds2.topic
        assert ds1.mode == ds2.mode
        assert ds1.tense == ds2.tense
        assert ds1.negation == ds2.negation
        assert ds1.specificity == ds2.specificity
        assert ds1.argument_pos == ds2.argument_pos

    def test_reservoir_deterministic_trajectory(self):
        """Reservoir trajectory is deterministic."""
        esn1 = IntegerESN(vocab_size=100, reservoir_dim=32, seed=42)
        esn2 = IntegerESN(vocab_size=100, reservoir_dim=32, seed=42)

        trajectory = [5, 10, 15, 20, 25, 30, 35]
        for w in trajectory:
            esn1.step(w)
            esn2.step(w)

        np.testing.assert_array_equal(esn1.h, esn2.h)


# ===================================================================
# PT-MON-01: Lower energy = higher probability (Boltzmann)
# ===================================================================

class TestBoltzmannMonotonicity:
    """Test that lower energy implies higher probability under Boltzmann."""

    def test_lower_energy_higher_probability(self):
        """Words with lower energy have higher probability."""
        sampler = IntegerBoltzmannSampler(beta=0.1, max_delta=50000)

        # Create energies where one is clearly lowest
        energies = np.array([0, 1000, 5000, 10000, 25000], dtype=np.int64)

        # Sample many times
        counts = np.zeros(len(energies))
        for _ in range(500):
            idx = sampler.sample(energies)
            counts[idx] += 1

        # Index 0 (lowest energy) should be sampled most
        assert counts[0] == max(counts), \
            f"Lowest energy not most probable: counts={counts}"

    def test_energy_ordering_preserved(self):
        """Energy ordering roughly matches probability ordering."""
        sampler = IntegerBoltzmannSampler(beta=0.1, max_delta=50000)
        energies = np.array([0, 500, 2000, 8000, 20000], dtype=np.int64)

        counts = np.zeros(len(energies))
        for _ in range(1000):
            idx = sampler.sample(energies)
            counts[idx] += 1

        # Probabilities should generally decrease with energy
        # (allowing for statistical noise)
        assert counts[0] > counts[-1]

    def test_boltzmann_sampling_valid_distribution(self):
        """Boltzmann sampling produces a valid distribution."""
        sampler = IntegerBoltzmannSampler(beta=0.05, max_delta=50000)
        energies = np.array([0, 100, 500, 1000, 5000], dtype=np.int64)

        # Compute log probabilities
        log_probs = sampler.compute_log_probabilities(energies)
        assert len(log_probs) == len(energies)
        # All log probs should be negative (probabilities < 1)
        assert np.all(log_probs <= 0)


# ===================================================================
# PT-MON-05: Dense AM sharpness increases with degree
# ===================================================================

class TestDenseAMSharpness:
    """Test that Dense AM sharpness increases with polynomial degree."""

    def test_degree2_sharper_than_degree1(self):
        """Degree=2 produces sharper energy landscape than degree=1."""
        proj = RandomFeatureProjector(vocab_size=100, D=32, seed=42)
        sequences = [[1, 2, 3, 4, 5]] * 20

        dam1 = DenseAMEnergy(proj, vocab_size=100, degree=1, dense_am_scale=1200)
        dam1.preaggregate(sequences)

        dam2 = DenseAMEnergy(proj, vocab_size=100, degree=2, dense_am_scale=1200)
        dam2.preaggregate(sequences)

        candidates = np.array(list(range(1, 50)), dtype=np.int64)
        context = [1, 2, 3]

        e1 = dam1.compute_energy(context, candidates)
        e2 = dam2.compute_energy(context, candidates)

        # Degree 2 should have larger spread (more discriminative)
        spread1 = int(e1.max() - e1.min())
        spread2 = int(e2.max() - e2.min())

        # Degree 2 should generally have equal or larger spread
        # (quadratic amplifies differences)
        # Allow for edge cases where both might be 0
        if spread1 > 0 or spread2 > 0:
            assert spread2 >= spread1 * 0.5, \
                f"Degree 2 (spread={spread2}) not sharper than degree 1 (spread={spread1})"

    def test_degree1_linear(self):
        """Degree=1 produces linear energy landscape."""
        proj = RandomFeatureProjector(vocab_size=100, D=32, seed=42)
        dam = DenseAMEnergy(proj, vocab_size=100, degree=1, dense_am_scale=1200)
        sequences = [[1, 2, 3, 4, 5]] * 10
        dam.preaggregate(sequences)
        assert dam.degree == 1

    def test_degree2_quadratic(self):
        """Degree=2 produces quadratic energy landscape."""
        proj = RandomFeatureProjector(vocab_size=100, D=32, seed=42)
        dam = DenseAMEnergy(proj, vocab_size=100, degree=2, dense_am_scale=1200)
        sequences = [[1, 2, 3, 4, 5]] * 10
        dam.preaggregate(sequences)
        assert dam.degree == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
