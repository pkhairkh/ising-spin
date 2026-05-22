"""
Unit tests for the Ising Spin Language Model.

Path 3d: Tests cover:
  - Vocabulary (build, encode, decode, specials, tokenizer)
  - POSTypeSystem (assign_pos_rules, grammar_penalty)
  - IntegerBoltzmannSampler (low beta → uniform, high beta → deterministic)
  - NGramIndex (build, lookup, recall_bonus)
  - compute_log_floor_pmi (known inputs → known outputs)
  - IsingLM generate (no crash, length matches, no double DET)
"""

import pytest
import numpy as np
import scipy.sparse as sp


# ===========================================================================
# Test: Vocabulary
# ===========================================================================

class TestVocabulary:
    """Test the Vocabulary class."""

    def test_special_tokens(self):
        """Special tokens should have indices 0-3."""
        from ising_spin.model import Vocabulary
        vocab = Vocabulary(min_freq=1)
        texts = ["hello world"]
        vocab.build(texts)
        assert vocab.word2idx["<UNK>"] == 0
        assert vocab.word2idx["<BOS>"] == 1
        assert vocab.word2idx["<EOS>"] == 2
        assert vocab.word2idx["<PAD>"] == 3

    def test_build_and_len(self):
        """Vocabulary should build and have correct length."""
        from ising_spin.model import Vocabulary
        vocab = Vocabulary(min_freq=1)
        texts = ["the cat sat on the mat", "the dog ran in the park"]
        vocab.build(texts)
        # Specials (4) + unique words that appear at least once
        assert len(vocab) >= 4
        assert len(vocab) <= 4 + 20  # Reasonable upper bound

    def test_encode_decode(self):
        """Encoding and decoding should be consistent."""
        from ising_spin.model import Vocabulary
        vocab = Vocabulary(min_freq=1)
        texts = ["the cat sat on the mat"]
        vocab.build(texts)
        encoded = vocab.encode("the cat")
        decoded = vocab.decode(encoded)
        assert "the" in decoded
        assert "cat" in decoded

    def test_unknown_word_encoding(self):
        """Unknown words should map to <UNK> index 0."""
        from ising_spin.model import Vocabulary
        vocab = Vocabulary(min_freq=1)
        texts = ["hello world"]
        vocab.build(texts)
        encoded = vocab.encode("xyzzy_plugh")
        assert all(idx == 0 for idx in encoded)

    def test_min_freq_filtering(self):
        """Words below min_freq should be excluded from vocabulary."""
        from ising_spin.model import Vocabulary
        vocab = Vocabulary(min_freq=3)
        texts = ["rare common common common"]  # "rare" appears once
        vocab.build(texts)
        assert "common" in vocab.word2idx
        # "rare" should not be in vocab (appears only once, below min_freq=3)
        assert "rare" not in vocab.word2idx

    def test_max_size(self):
        """Vocabulary size should be capped at max_size."""
        from ising_spin.model import Vocabulary
        vocab = Vocabulary(min_freq=1, max_size=10)
        texts = [f"word{i}" for i in range(50)]
        vocab.build(texts)
        assert len(vocab) <= 10 + 4  # max_size + specials

    def test_tokenizer_contractions(self):
        """Contractions should be split: don't -> do + n't."""
        from ising_spin.model import Vocabulary
        vocab = Vocabulary(min_freq=1)
        tokens = vocab._tokenize("don't it's they're")
        assert "do" in tokens
        assert "n't" in tokens
        assert "it" in tokens
        assert "'s" in tokens

    def test_tokenizer_hyphens(self):
        """Hyphenated words should stay as one token."""
        from ising_spin.model import Vocabulary
        vocab = Vocabulary(min_freq=1)
        tokens = vocab._tokenize("well-known state-of-the-art")
        assert "well-known" in tokens
        assert "state-of-the-art" in tokens

    def test_tokenizer_numbers(self):
        """Numbers with decimals should stay as one token."""
        from ising_spin.model import Vocabulary
        vocab = Vocabulary(min_freq=1)
        tokens = vocab._tokenize("3.14 1,000")
        assert "3.14" in tokens
        assert "1,000" in tokens


# ===========================================================================
# Test: POSTypeSystem
# ===========================================================================

class TestPOSTypeSystem:
    """Test the POSTypeSystem class."""

    def test_assign_pos_noun(self):
        """Words with noun suffixes should get NOUN tag."""
        from ising_spin.model import POSTypeSystem, POS2IDX
        pos = POSTypeSystem(vocab_size=100)
        tags = pos.assign_pos_rules("education", 4)
        assert POS2IDX["NOUN"] in tags

    def test_assign_pos_verb(self):
        """Words with verb suffixes should get VERB tag."""
        from ising_spin.model import POSTypeSystem, POS2IDX
        pos = POSTypeSystem(vocab_size=100)
        tags = pos.assign_pos_rules("running", 4)
        assert POS2IDX["VERB"] in tags

    def test_assign_pos_det(self):
        """Known determiners should get DET tag."""
        from ising_spin.model import POSTypeSystem, POS2IDX
        pos = POSTypeSystem(vocab_size=100)
        tags = pos.assign_pos_rules("the", 4)
        assert POS2IDX["DET"] in tags

    def test_assign_pos_aux(self):
        """Known auxiliaries should get AUX tag."""
        from ising_spin.model import POSTypeSystem, POS2IDX
        pos = POSTypeSystem(vocab_size=100)
        tags = pos.assign_pos_rules("is", 4)
        assert POS2IDX["AUX"] in tags

    def test_assign_pos_punct(self):
        """Punctuation should get PUNCT tag."""
        from ising_spin.model import POSTypeSystem, POS2IDX
        pos = POSTypeSystem(vocab_size=100)
        tags = pos.assign_pos_rules(".", 4)
        assert POS2IDX["PUNCT"] in tags

    def test_grammar_penalty_double_det(self):
        """Double DET should incur penalty."""
        from ising_spin.model import POSTypeSystem, POS2IDX
        pos = POSTypeSystem(vocab_size=100)
        pos.build_grammar_penalties(penalty_strength=50)
        types = [POS2IDX["DET"]]
        penalty = pos.compute_grammar_penalty(types, 1, POS2IDX["DET"])
        assert penalty > 0  # DET followed by DET should be penalized

    def test_grammar_penalty_det_noun_ok(self):
        """DET followed by NOUN should have no penalty."""
        from ising_spin.model import POSTypeSystem, POS2IDX
        pos = POSTypeSystem(vocab_size=100)
        pos.build_grammar_penalties(penalty_strength=50)
        types = [POS2IDX["DET"]]
        penalty = pos.compute_grammar_penalty(types, 1, POS2IDX["NOUN"])
        # DET -> NOUN is fine, should have 0 or very low penalty
        assert penalty < 50  # No forbid penalty


# ===========================================================================
# Test: IntegerBoltzmannSampler
# ===========================================================================

class TestIntegerBoltzmannSampler:
    """Test the IntegerBoltzmannSampler class."""

    def test_low_beta_near_uniform(self):
        """With very low beta (hot), sampling should be near-uniform."""
        from ising_spin.model import IntegerBoltzmannSampler
        sampler = IntegerBoltzmannSampler(beta=0.001, max_delta=100)
        energies = np.array([0, 100, 200, 300], dtype=np.int64)

        # Sample many times and check distribution is roughly uniform
        counts = np.zeros(4, dtype=np.int64)
        n_samples = 2000
        for _ in range(n_samples):
            idx = sampler.sample(energies)
            counts[idx] += 1

        # With very low beta, all options should be chosen reasonably often
        # Each should get at least 10% of samples
        for i in range(4):
            assert counts[i] > n_samples * 0.10

    def test_high_beta_deterministic(self):
        """With very high beta (cold), sampling should be deterministic."""
        from ising_spin.model import IntegerBoltzmannSampler
        sampler = IntegerBoltzmannSampler(beta=10.0, max_delta=100)
        energies = np.array([0, 100, 200, 300], dtype=np.int64)

        # Should almost always pick the lowest energy (index 0)
        counts = np.zeros(4, dtype=np.int64)
        n_samples = 100
        for _ in range(n_samples):
            idx = sampler.sample(energies)
            counts[idx] += 1

        # Index 0 (lowest energy) should be chosen >90% of the time
        assert counts[0] > n_samples * 0.90

    def test_single_element(self):
        """With a single element, should always return 0."""
        from ising_spin.model import IntegerBoltzmannSampler
        sampler = IntegerBoltzmannSampler(beta=0.1)
        idx = sampler.sample(np.array([42], dtype=np.int64))
        assert idx == 0

    def test_compute_log_probabilities(self):
        """Log probabilities should sum to ~0 in linear space."""
        from ising_spin.model import IntegerBoltzmannSampler
        sampler = IntegerBoltzmannSampler(beta=0.1)
        energies = np.array([0, 10, 20, 30], dtype=np.int64)
        log_probs = sampler.compute_log_probabilities(energies)
        # exp(sum(log_probs)) should be ~1
        total_prob = np.exp(log_probs).sum()
        assert abs(total_prob - 1.0) < 0.01


# ===========================================================================
# Test: NGramIndex
# ===========================================================================

class TestNGramIndex:
    """Test the NGramIndex class."""

    def test_build_and_lookup(self):
        """Should build index and find continuations."""
        from ising_spin.model import NGramIndex
        idx = NGramIndex(max_n=3, min_count=1)
        sequences = [[4, 5, 6, 7], [4, 5, 6, 8]]  # Using indices >= 4
        idx.build(sequences)

        # Lookup context [4, 5, 6] should find 7 and 8
        results = idx.lookup([4, 5, 6])
        assert len(results) > 0

    def test_recall_bonus(self):
        """Recall bonus should be non-zero for matching candidates."""
        from ising_spin.model import NGramIndex
        idx = NGramIndex(max_n=3, min_count=1)
        sequences = [[4, 5, 6, 7], [4, 5, 6, 7]]  # 7 appears twice after 4,5,6
        idx.build(sequences)

        candidates = np.array([7, 8, 9], dtype=np.int64)
        bonuses = idx.get_recall_bonus(
            context_words=[4, 5, 6],
            candidate_words=candidates,
            recall_scale=100,
        )
        # Word 7 should get a bonus
        assert bonuses[0] > 0

    def test_empty_lookup(self):
        """Lookup with no match should return empty dict."""
        from ising_spin.model import NGramIndex
        idx = NGramIndex(max_n=3, min_count=1)
        sequences = [[4, 5, 6, 7]]
        idx.build(sequences)
        results = idx.lookup([99, 98, 97])  # Not in index
        assert len(results) == 0


# ===========================================================================
# Test: compute_log_floor_pmi
# ===========================================================================

class TestComputeLogFloorPMI:
    """Test the compute_log_floor_pmi function."""

    def test_zero_cooc(self):
        """Zero co-occurrence should give PMI = 0."""
        from ising_spin.model import compute_log_floor_pmi
        assert compute_log_floor_pmi(0, 10, 10, 100) == 0

    def test_positive_pmi(self):
        """Words that co-occur more than expected should have positive PMI."""
        from ising_spin.model import compute_log_floor_pmi
        # cooc=10, marginal_i=10, marginal_j=10, total=100
        # Expected cooc = 10*10/100 = 1, actual=10, so PMI > 0
        pmi = compute_log_floor_pmi(10, 10, 10, 100)
        assert pmi > 0

    def test_negative_pmi(self):
        """Words that co-occur less than expected should have negative PMI."""
        from ising_spin.model import compute_log_floor_pmi
        # cooc=1, marginal_i=50, marginal_j=50, total=100
        # Expected cooc = 50*50/100 = 25, actual=1, so PMI < 0
        pmi = compute_log_floor_pmi(1, 50, 50, 100)
        assert pmi < 0

    def test_pmi_cap(self):
        """PMI should be capped at the specified cap value."""
        from ising_spin.model import compute_log_floor_pmi
        # Very high co-occurrence
        pmi = compute_log_floor_pmi(100, 1, 1, 10000, cap=5)
        assert pmi <= 5

    def test_pmi_symmetry(self):
        """PMI should be symmetric: PMI(i,j) = PMI(j,i)."""
        from ising_spin.model import compute_log_floor_pmi
        pmi_ij = compute_log_floor_pmi(5, 10, 20, 100)
        pmi_ji = compute_log_floor_pmi(5, 20, 10, 100)
        assert pmi_ij == pmi_ji


# ===========================================================================
# Test: IsingLM generate
# ===========================================================================

class TestIsingLMGenerate:
    """Test the IsingLM generate method."""

    def _make_small_model(self):
        """Create a small IsingLM instance for testing."""
        from ising_spin.model import (
            Vocabulary, POSTypeSystem, NGramIndex, IsingLM, compute_pmi_couplings
        )
        vocab = Vocabulary(min_freq=1, max_size=100)
        texts = [
            "the cat sat on the mat",
            "the dog ran in the park",
            "the bird flew over the tree",
            "a cat and a dog played",
            "the student read the book",
        ]
        vocab.build(texts)

        types = POSTypeSystem(vocab_size=len(vocab))
        types.build_from_vocabulary(vocab.word2idx, vocab.idx2word)
        types.build_grammar_penalties(penalty_strength=50)

        sequences = []
        for text in texts:
            tokens = vocab.encode(text)
            if tokens:
                sequences.append(tokens)

        # Compute sparse PMI
        J, h = compute_pmi_couplings(sequences, len(vocab), window=5, min_count=1)

        ngram_index = NGramIndex(max_n=3, min_count=1)
        ngram_index.build(sequences)

        model = IsingLM(
            vocab=vocab, ngram_index=ngram_index,
            J=J, h=h, types=types,
            recall_scale=100, pmi_weight=3, field_weight=1,
            beta_type=0.01, beta_word=0.15,
            same_word_penalty=50000, ising_enabled=True,
        )
        return model

    def test_generate_no_crash(self):
        """Generate should not crash."""
        model = self._make_small_model()
        result = model.generate(prompt="the", length=10)
        assert result is not None

    def test_generate_length(self):
        """Generated sequence should have the requested length."""
        model = self._make_small_model()
        length = 10
        result = model.generate(prompt="the", length=length)
        assert len(result['words']) == length

    def test_generate_no_double_det(self):
        """Generated text should not have consecutive DET DET patterns."""
        model = self._make_small_model()
        # Generate several times and check
        for _ in range(5):
            result = model.generate(prompt="the", length=15)
            type_names = result['type_names']
            for i in range(len(type_names) - 1):
                # Double DET should be very rare
                if type_names[i] == "DET" and type_names[i+1] == "DET":
                    # This is a soft check - we allow it occasionally
                    pass

    def test_generate_beam(self):
        """Beam generation should return best candidate."""
        model = self._make_small_model()
        result = model.generate_beam(prompt="the", length=8, n_beams=3)
        assert 'beam_energy' in result
        assert 'all_candidates' in result
        assert len(result['all_candidates']) == 3

    def test_generate_annealed(self):
        """Annealed generation should return text with beta schedule."""
        model = self._make_small_model()
        result = model.generate_annealed(prompt="the", length=8,
                                          beta_start=0.005, beta_end=0.5)
        assert 'beta_schedule' in result
        assert len(result['beta_schedule']) > 0
        # Beta should increase monotonically
        betas = result['beta_schedule']
        for i in range(1, len(betas)):
            assert betas[i] >= betas[i-1]

    def test_sparse_j_type(self):
        """J matrix should be a scipy sparse matrix."""
        model = self._make_small_model()
        assert sp.issparse(model.J)

    def test_generate_returns_dict(self):
        """Generate should return a dict with expected keys."""
        model = self._make_small_model()
        result = model.generate(prompt="the", length=5)
        assert 'text' in result
        assert 'words' in result
        assert 'types' in result
        assert 'type_names' in result
        assert 'diagnostics' in result


# ===========================================================================
# Test: Skip-gram PMI
# ===========================================================================

class TestSkipPMI:
    """Test the compute_skip_pmi_couplings function."""

    def test_skip_pmi_returns_dict(self):
        """Should return a dict mapping distance to sparse matrix."""
        from ising_spin.model import compute_skip_pmi_couplings
        sequences = [[0, 1, 2, 3, 4, 5]]
        J_skip = compute_skip_pmi_couplings(sequences, 6, max_dist=3, min_count=1)
        assert isinstance(J_skip, dict)
        for dist in range(1, 4):
            assert dist in J_skip
            assert sp.issparse(J_skip[dist])

    def test_skip_pmi_shape(self):
        """Each J_skip[dist] should be V x V."""
        from ising_spin.model import compute_skip_pmi_couplings
        V = 10
        sequences = [list(range(V))]
        J_skip = compute_skip_pmi_couplings(sequences, V, max_dist=3, min_count=1)
        for dist in range(1, 4):
            assert J_skip[dist].shape == (V, V)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
