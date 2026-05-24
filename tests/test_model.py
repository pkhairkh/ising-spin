"""
Comprehensive test suite for the Ising Spin Glass Language Model.

Test levels:
  1. Unit tests — each DDD module in isolation
  2. Integration tests — modules working together (energy pipeline, generator loop)
  3. Edge case / error handling tests — invalid inputs, boundary conditions
  4. Determinism tests — integer-only arithmetic produces identical results

Run with:
  pytest tests/ -v
"""

import numpy as np
import pytest

# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_texts():
    """Small corpus for testing."""
    return [
        "the cat sat on the mat and the dog ran in the park",
        "science and technology are important for research and development",
        "the weather was warm and sunny during the summer months",
        "research shows that exercise improves health and well being",
        "the history of art reflects the culture of different societies",
    ]


@pytest.fixture
def vocab(sample_texts):
    """Pre-built vocabulary."""
    from ising_spin import Vocabulary
    v = Vocabulary(min_freq=1, max_size=200)
    v.build(sample_texts)
    return v


@pytest.fixture
def pos_system(vocab):
    """Pre-built POS type system."""
    from ising_spin import POSTypeSystem
    ps = POSTypeSystem(vocab_size=len(vocab), window=5)
    ps.build_from_vocabulary(vocab.word2idx, vocab.idx2word)
    ps.build_grammar_penalties(penalty_strength=60)
    return ps


@pytest.fixture
def topic_assigner(sample_texts, vocab):
    """Pre-built topic assigner."""
    from ising_spin import TopicAssigner
    ta = TopicAssigner(n_topics=4)
    ta.build(sample_texts, vocab)
    return ta


@pytest.fixture
def sequences(vocab, sample_texts):
    """Tokenized sequences."""
    seqs = []
    for text in sample_texts:
        tokens = vocab.encode(text)
        if len(tokens) > 1:
            seqs.append(tokens[:30])
    return seqs


@pytest.fixture
def word_index(sequences):
    """Pre-built word n-gram index."""
    from ising_spin import WordNgramIndex
    idx = WordNgramIndex(max_n=3, min_count=1)
    idx.build(sequences)
    return idx


@pytest.fixture
def pos_index(sequences, pos_system):
    """Pre-built POS n-gram index."""
    from ising_spin import PosNgramIndex
    from ising_spin.utils import primary_pos_tag
    word_pos_tags = {}
    for w, allowed in pos_system.allowed_types.items():
        if allowed:
            word_pos_tags[w] = primary_pos_tag(allowed)
    idx = PosNgramIndex(max_n=5, min_count=1, pos_system=pos_system)
    idx.build(sequences, word_pos_tags=word_pos_tags)
    return idx


@pytest.fixture
def topic_index(sequences, topic_assigner):
    """Pre-built topic n-gram index."""
    from ising_spin import TopicNgramIndex
    idx = TopicNgramIndex(
        max_n=3, min_count=1, n_topics=4,
        word_topics=topic_assigner.word_topics,
    )
    idx.build(sequences)
    return idx


@pytest.fixture
def document_state(vocab, pos_system, topic_assigner, sequences):
    """Pre-built document state."""
    from ising_spin import DocumentState
    ds = DocumentState(
        vocab_size=len(vocab),
        n_topics=4,
        pos_system=pos_system,
        word_topics=topic_assigner.word_topics,
    )
    ds.build(sequences)
    return ds


# ===========================================================================
# 1. EXCEPTION HIERARCHY
# ===========================================================================

class TestExceptions:
    """Test that the exception hierarchy is correct."""

    def test_base_exception(self):
        from ising_spin.exceptions import IsingSpinError
        assert issubclass(IsingSpinError, Exception)

    def test_build_errors(self):
        from ising_spin.exceptions import (
            IsingSpinError, BuildError, VocabularyBuildError,
            IndexBuildError, StateBuildError, TopicBuildError,
        )
        assert issubclass(BuildError, IsingSpinError)
        assert issubclass(VocabularyBuildError, BuildError)
        assert issubclass(IndexBuildError, BuildError)
        assert issubclass(StateBuildError, BuildError)
        assert issubclass(TopicBuildError, BuildError)

    def test_inference_errors(self):
        from ising_spin.exceptions import IsingSpinError, InferenceError, SamplingError, EnergyError
        assert issubclass(InferenceError, IsingSpinError)
        assert issubclass(SamplingError, InferenceError)
        assert issubclass(EnergyError, InferenceError)

    def test_validation_errors(self):
        from ising_spin.exceptions import (
            IsingSpinError, ValidationError, VocabularyError,
            POSValidationError, StateValidationError,
        )
        assert issubclass(ValidationError, IsingSpinError)
        assert issubclass(VocabularyError, ValidationError)
        assert issubclass(POSValidationError, ValidationError)
        assert issubclass(StateValidationError, ValidationError)


# ===========================================================================
# 2. UTILS MODULE
# ===========================================================================

class TestUtils:
    """Test shared utilities."""

    def test_tag_priority_completeness(self):
        from ising_spin.utils import TAG_PRIORITY
        from ising_spin.vocabulary.pos import N_POS
        assert len(TAG_PRIORITY) == N_POS

    def test_primary_pos_tag_closed_class(self):
        from ising_spin.utils import primary_pos_tag
        from ising_spin.vocabulary.pos import POS2IDX
        # "the" can be DET or PRON; DET has higher priority
        allowed = {POS2IDX["DET"], POS2IDX["PRON"]}
        assert primary_pos_tag(allowed) == POS2IDX["DET"]

    def test_primary_pos_tag_empty(self):
        from ising_spin.utils import primary_pos_tag
        from ising_spin.vocabulary.pos import POS2IDX
        assert primary_pos_tag(set()) == POS2IDX["X"]

    def test_get_rss_mb(self):
        from ising_spin.utils import get_rss_mb
        rss = get_rss_mb()
        assert isinstance(rss, int)
        assert rss >= 0

    def test_validate_array_ok(self):
        from ising_spin.utils import validate_array
        arr = np.array([1, 2, 3], dtype=np.int64)
        validate_array(arr, "test", dtype=np.int64, ndim=1, min_len=1)

    def test_validate_array_wrong_type(self):
        from ising_spin.utils import validate_array
        with pytest.raises(TypeError):
            validate_array([1, 2, 3], "test")

    def test_validate_array_wrong_ndim(self):
        from ising_spin.utils import validate_array
        arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
        with pytest.raises(TypeError):
            validate_array(arr, "test", ndim=1)

    def test_validate_nonempty(self):
        from ising_spin.utils import validate_nonempty
        validate_nonempty([1, 2, 3], "test")
        with pytest.raises(ValueError):
            validate_nonempty([], "test")

    def test_validate_positive(self):
        from ising_spin.utils import validate_positive
        validate_positive(1, "test")
        with pytest.raises(ValueError):
            validate_positive(0, "test")
        with pytest.raises(ValueError):
            validate_positive(-1, "test")


# ===========================================================================
# 3. VOCABULARY MODULE
# ===========================================================================

class TestVocabulary:
    """Test Vocabulary, POSTypeSystem, and TopicAssigner."""

    def test_vocab_build(self, vocab, sample_texts):
        assert vocab._built
        assert len(vocab) > 0
        assert "<UNK>" in vocab.word2idx

    def test_vocab_encode_decode(self, vocab):
        text = "the cat sat"
        ids = vocab.encode(text)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        decoded = vocab.decode(ids)
        assert "the" in decoded
        assert "cat" in decoded

    def test_vocab_unknown_words(self, vocab):
        ids = vocab.encode("xyzzy12345")
        unk_idx = vocab.word2idx["<UNK>"]
        assert all(i == unk_idx for i in ids)

    def test_pos_system_types(self, pos_system, vocab):
        assert len(pos_system.allowed_types) > 0
        # "the" should be tagged as DET
        the_idx = vocab.word2idx.get("the")
        if the_idx is not None:
            assert the_idx in pos_system.allowed_types

    def test_pos_grammar_penalty(self, pos_system):
        from ising_spin.vocabulary.pos import POS2IDX
        # DET followed by VERB should get some penalty
        penalty = pos_system.compute_grammar_penalty(
            [POS2IDX["DET"]], 0, POS2IDX["VERB"]
        )
        assert isinstance(penalty, int)
        assert penalty >= 0

    def test_topic_assigner(self, topic_assigner):
        assert topic_assigner._built
        assert topic_assigner.word_topics is not None
        assert len(topic_assigner.word_topics) > 0


# ===========================================================================
# 4. RECALL MODULE (NgramIndexBase + all three indexes)
# ===========================================================================

class TestRecall:
    """Test n-gram recall indexes and MultiScaleRecall."""

    def test_word_index_build(self, word_index):
        assert word_index._built
        assert len(word_index.index) > 0

    def test_word_index_lookup(self, word_index, sequences):
        if len(sequences) > 0 and len(sequences[0]) >= 2:
            results = word_index.lookup(sequences[0])
            assert isinstance(results, dict)

    def test_word_index_energy(self, word_index, sequences):
        if len(sequences) > 0 and len(sequences[0]) >= 2:
            candidates = np.array(sequences[0][:10], dtype=np.int64)
            energies = word_index.compute_energy(
                sequences[0], candidates, recall_scale=100
            )
            assert energies.dtype == np.int64
            assert len(energies) == len(candidates)
            # Lower energy = more likely, all should be non-negative
            assert np.all(energies >= 0)

    def test_word_index_copy(self, word_index, sequences):
        if len(sequences) > 0 and len(sequences[0]) >= 3:
            result = word_index.get_best_copy_candidate(
                context_words=sequences[0],
                min_context_length=2,
                min_confidence=0.1,
            )
            # May or may not find a candidate, but should not crash
            assert result is None or len(result) == 3

    def test_pos_index_build(self, pos_index):
        assert pos_index._built
        assert len(pos_index.index) > 0

    def test_pos_index_energy(self, pos_index, sequences):
        if len(sequences) > 0 and len(sequences[0]) >= 2:
            candidates = np.array(sequences[0][:10], dtype=np.int64)
            energies = pos_index.compute_energy(
                sequences[0], candidates, recall_scale=50
            )
            assert energies.dtype == np.int64
            assert len(energies) == len(candidates)

    def test_topic_index_build(self, topic_index):
        assert topic_index._built
        assert len(topic_index.index) > 0

    def test_topic_index_energy(self, topic_index, sequences):
        if len(sequences) > 0 and len(sequences[0]) >= 2:
            candidates = np.array(sequences[0][:10], dtype=np.int64)
            energies = topic_index.compute_energy(
                sequences[0], candidates, recall_scale=25
            )
            assert energies.dtype == np.int64
            assert len(energies) == len(candidates)

    def test_multiscale_recall(self, word_index, pos_index, topic_index, sequences):
        from ising_spin import MultiScaleRecall
        msr = MultiScaleRecall(
            word_index=word_index,
            pos_index=pos_index,
            topic_index=topic_index,
            word_scale=100,
            pos_scale=50,
            topic_scale=25,
        )
        if len(sequences) > 0 and len(sequences[0]) >= 2:
            candidates = np.array(sequences[0][:10], dtype=np.int64)
            energies = msr.compute_energy(
                sequences[0], candidates,
                interpolated=True, kn_backoff=True,
            )
            assert energies.dtype == np.int64
            assert len(energies) == len(candidates)

    def test_ngram_index_base_validation(self):
        """NgramIndexBase should reject invalid params."""
        from ising_spin.recall.base import NgramIndexBase
        from ising_spin.exceptions import ValidationError
        with pytest.raises(ValidationError):
            NgramIndexBase(max_n=0, min_count=1)
        with pytest.raises(ValidationError):
            NgramIndexBase(max_n=3, min_count=0)

    def test_empty_sequences_build(self):
        """Building from empty sequences should raise."""
        from ising_spin import WordNgramIndex
        from ising_spin.exceptions import IndexBuildError
        idx = WordNgramIndex(max_n=3, min_count=1)
        with pytest.raises(IndexBuildError):
            idx.build([])


# ===========================================================================
# 5. DOCUMENT STATE MODULE
# ===========================================================================

class TestDocumentState:
    """Test DocumentState build, update, and energy computation."""

    def test_state_build(self, document_state):
        assert document_state._built
        assert document_state.topic_word_counts is not None

    def test_state_reset(self, document_state):
        document_state.topic = 5
        document_state.mode = 3
        document_state.reset()
        assert document_state.topic == 1
        assert document_state.mode == document_state.MODE_NARRATIVE

    def test_state_update(self, document_state, vocab):
        document_state.reset()
        the_idx = vocab.word2idx.get("the", 4)
        document_state.update(the_idx, word_str="the")
        # After "the" (DET), some state might change
        sv = document_state.get_state_vector()
        assert isinstance(sv, dict)
        assert "topic" in sv
        assert "mode" in sv

    def test_state_energy(self, document_state, vocab):
        candidates = np.array([4, 5, 6, 7, 8], dtype=np.int64)
        energies = document_state.compute_energy(candidates, state_scale=100)
        assert energies.dtype == np.int64
        assert len(energies) == len(candidates)

    def test_state_unbuilt_energy(self):
        """Unbuilt state should return zero energies."""
        from ising_spin import DocumentState
        ds = DocumentState(vocab_size=100, n_topics=4)
        candidates = np.array([4, 5, 6], dtype=np.int64)
        energies = ds.compute_energy(candidates, state_scale=100)
        assert np.all(energies == 0)

    def test_state_repr(self, document_state):
        repr_str = repr(document_state)
        assert "DocumentState" in repr_str


# ===========================================================================
# 6. ENERGY MODULE
# ===========================================================================

class TestEnergyComputer:
    """Test EnergyComputer with validation and vectorized constraints."""

    def test_energy_computation(self, word_index, pos_index, topic_index,
                                document_state, pos_system, sequences):
        from ising_spin import MultiScaleRecall, EnergyComputer
        msr = MultiScaleRecall(
            word_index=word_index,
            pos_index=pos_index,
            topic_index=topic_index,
        )
        ec = EnergyComputer(
            multiscale_recall=msr,
            document_state=document_state,
            pos_system=pos_system,
        )
        if len(sequences) > 0 and len(sequences[0]) >= 2:
            candidates = np.array(sequences[0][:10], dtype=np.int64)
            energies = ec.compute_energy(
                context_words=sequences[0],
                candidate_words=candidates,
                current_type=0,
                prev_word=sequences[0][0],
                closed_class_run=0,
            )
            assert energies.dtype == np.int64
            assert len(energies) == len(candidates)

    def test_energy_validation(self, word_index, pos_index, topic_index,
                                document_state, pos_system):
        from ising_spin import MultiScaleRecall, EnergyComputer
        msr = MultiScaleRecall(
            word_index=word_index,
            pos_index=pos_index,
            topic_index=topic_index,
        )
        ec = EnergyComputer(
            multiscale_recall=msr,
            document_state=document_state,
            pos_system=pos_system,
        )
        # context_words must be a list
        with pytest.raises(TypeError):
            ec.compute_energy(
                context_words=np.array([1, 2]),
                candidate_words=np.array([3, 4], dtype=np.int64),
            )
        # candidate_words must be ndarray
        with pytest.raises(TypeError):
            ec.compute_energy(
                context_words=[1, 2],
                candidate_words=[3, 4],
            )


# ===========================================================================
# 7. SAMPLING MODULE
# ===========================================================================

class TestSampling:
    """Test IntegerBoltzmannSampler and int_log2_fine."""

    def test_sampler_init(self):
        from ising_spin import IntegerBoltzmannSampler
        sampler = IntegerBoltzmannSampler(beta=0.1, max_delta=5000)
        assert sampler.table is not None
        assert len(sampler.table) > 0

    def test_sampler_invalid_beta(self):
        from ising_spin import IntegerBoltzmannSampler
        from ising_spin.exceptions import SamplingError
        with pytest.raises(SamplingError):
            IntegerBoltzmannSampler(beta=0)

    def test_sampler_sample(self):
        from ising_spin import IntegerBoltzmannSampler
        sampler = IntegerBoltzmannSampler(beta=0.001, max_delta=5000)
        energies = np.array([100, 50, 200, 30, 500], dtype=np.int64)
        idx = sampler.sample(energies)
        assert 0 <= idx < len(energies)

    def test_sampler_sample_single(self):
        from ising_spin import IntegerBoltzmannSampler
        sampler = IntegerBoltzmannSampler(beta=0.01, max_delta=1000)
        idx = sampler.sample(np.array([100], dtype=np.int64))
        assert idx == 0

    def test_sampler_log_probabilities(self):
        from ising_spin import IntegerBoltzmannSampler
        sampler = IntegerBoltzmannSampler(beta=0.001, max_delta=5000)
        energies = np.array([100, 50, 200, 30, 500], dtype=np.int64)
        log_probs = sampler.compute_log_probabilities(energies)
        assert log_probs.dtype == np.int64
        assert len(log_probs) == len(energies)
        # Log probs should be negative (or zero)
        assert np.all(log_probs <= 0)

    def test_int_log2_fine(self):
        from ising_spin.sampling.boltzmann import int_log2_fine
        # log2(2) = 1.0 → 256
        assert int_log2_fine(2) == 256
        # log2(4) = 2.0 → 512
        assert int_log2_fine(4) == 512
        # log2(1) = 0
        assert int_log2_fine(1) == 0

    def test_int_log2_fine_precision(self):
        from ising_spin.sampling.boltzmann import int_log2_fine
        import math
        # log2(1000) ≈ 9.966 → 9.966 * 256 ≈ 2551
        result = int_log2_fine(1000)
        expected = int(math.log2(1000) * 256)
        assert abs(result - expected) <= 2  # Allow 2 units of error


# ===========================================================================
# 8. INTEGRATION: FULL PIPELINE
# ===========================================================================

class TestIntegration:
    """Integration tests — modules working together."""

    def test_model_train(self, sample_texts):
        """Full training pipeline on tiny corpus."""
        from ising_spin import IsingLMModel
        model = IsingLMModel(
            vocab_min_freq=1,
            vocab_max_size=200,
            ngram_max_n=3,
            ngram_min_count=1,
            pos_ngram_max_n=5,
            pos_ngram_min_count=1,
            n_topics=4,
            topic_ngram_max_n=3,
            topic_ngram_min_count=1,
            auto_calibrate_beta=False,
            max_seq_len=30,
        )
        model.train(texts=sample_texts, n_samples=len(sample_texts))
        assert model.vocab is not None
        assert model.word_index is not None
        assert model.generator is not None

    def test_model_generate(self, sample_texts):
        """Generation after training."""
        from ising_spin import IsingLMModel
        model = IsingLMModel(
            vocab_min_freq=1,
            vocab_max_size=200,
            ngram_max_n=3,
            ngram_min_count=1,
            pos_ngram_max_n=5,
            pos_ngram_min_count=1,
            n_topics=4,
            topic_ngram_max_n=3,
            topic_ngram_min_count=1,
            auto_calibrate_beta=False,
            max_seq_len=30,
        )
        model.train(texts=sample_texts, n_samples=len(sample_texts))
        result = model.generate(prompt="the", length=10)
        assert "text" in result
        assert "words" in result
        assert len(result["words"]) > 0

    def test_model_perplexity(self, sample_texts):
        """Perplexity computation after training."""
        from ising_spin import IsingLMModel
        model = IsingLMModel(
            vocab_min_freq=1,
            vocab_max_size=200,
            ngram_max_n=3,
            ngram_min_count=1,
            pos_ngram_max_n=5,
            pos_ngram_min_count=1,
            n_topics=4,
            topic_ngram_max_n=3,
            topic_ngram_min_count=1,
            auto_calibrate_beta=False,
            max_seq_len=30,
        )
        model.train(texts=sample_texts, n_samples=len(sample_texts))
        ppl = model.compute_perplexity(n_samples=2)
        assert isinstance(ppl, float)
        assert ppl > 0


# ===========================================================================
# 9. DETERMINISM TESTS
# ===========================================================================

class TestDeterminism:
    """Verify integer-only arithmetic is deterministic."""

    def test_int_log2_deterministic(self):
        from ising_spin.sampling.boltzmann import int_log2_fine
        # Same input → same output, always
        for x in [2, 3, 5, 10, 100, 1000, 10000]:
            results = [int_log2_fine(x) for _ in range(5)]
            assert len(set(results)) == 1, f"Non-deterministic for x={x}"

    def test_sampler_table_deterministic(self):
        from ising_spin import IntegerBoltzmannSampler
        s1 = IntegerBoltzmannSampler(beta=0.05, max_delta=1000)
        s2 = IntegerBoltzmannSampler(beta=0.05, max_delta=1000)
        assert np.array_equal(s1.table, s2.table)

    def test_state_energy_deterministic(self, document_state, vocab):
        np.random.seed(42)
        candidates = np.array([4, 5, 6, 7, 8], dtype=np.int64)
        e1 = document_state.compute_energy(candidates, state_scale=100)
        e2 = document_state.compute_energy(candidates, state_scale=100)
        assert np.array_equal(e1, e2)
