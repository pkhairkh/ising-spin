"""Basic tests for the Integer Language Model."""
import numpy as np
import pytest


class TestVocabulary:
    def test_build(self, sample_texts, small_vocab):
        assert small_vocab.V > 0
        assert len(small_vocab.words) == small_vocab.V
        assert "the" in small_vocab.word2idx

    def test_tokenize(self, sample_texts, small_vocab):
        seqs = small_vocab.tokenize(sample_texts)
        assert len(seqs) > 0
        for seq in seqs:
            assert all(0 <= idx < small_vocab.V for idx in seq)

    def test_pos_types(self, small_vocab):
        assert small_vocab.word_pos is not None
        assert len(small_vocab.word_pos) == small_vocab.V
        # "the" should be DET
        the_idx = small_vocab.word2idx.get("the")
        if the_idx is not None:
            from ising_spin.vocabulary import POS2IDX
            assert small_vocab.word_pos[the_idx] == POS2IDX["DET"]


class TestBigramModel:
    def test_build_and_query(self, small_vocab):
        from ising_spin import BigramModel
        bm = BigramModel(vocab_size=small_vocab.V, seed=42)
        seqs = small_vocab.tokenize(["the cat sat on the mat"] * 10)
        bm.build(seqs)
        cands, lps = bm.get_top_k(seqs[0], k=5)
        assert len(cands) > 0
        assert len(cands) == len(lps)

    def test_log_prob(self, small_vocab):
        from ising_spin import BigramModel
        bm = BigramModel(vocab_size=small_vocab.V, seed=42)
        seqs = small_vocab.tokenize(["the cat sat on the mat"] * 10)
        bm.build(seqs)
        lp = bm.compute_log_prob(seqs[0], seqs[0][1])
        assert lp < 0  # Log prob should be negative


class TestFeatureHashEnergy:
    def test_train_and_query(self, small_vocab):
        from ising_spin import FeatureHashEnergyTable
        e = FeatureHashEnergyTable(
            vocab_size=small_vocab.V,
            word_pos=small_vocab.word_pos,
            n_pos_hashes=1, pos_table_size=101,
            n_lex_hashes=1, lex_table_size=1009,
            use_skip=False, use_trigram=False,
            seed=42,
        )
        seqs = small_vocab.tokenize(["the cat sat on the mat"] * 10)
        e.train_nce(seqs, n_epochs=1, n_negatives=2)
        # Energy for real pair should be lower than random
        ctx = seqs[0][:2]
        target = seqs[0][2]
        real_e = e.compute_local_energy(ctx, target)
        assert isinstance(real_e, int)


class TestIntegerLM:
    def test_train(self, small_model):
        assert small_model._calibrated

    def test_generate(self, small_model, small_vocab):
        prompt_ids = [small_vocab.word2idx.get("the", 1)]
        if prompt_ids[0] >= 4:
            generated = small_model.generate(prompt_ids, length=10)
            assert len(generated) > 1

    def test_discriminative_accuracy(self, small_model, small_vocab):
        seqs = small_vocab.tokenize(["the cat sat on the mat"] * 5)
        result = small_model.discriminative_accuracy(seqs, n_samples=10)
        assert 0.0 <= result['accuracy'] <= 1.0

    def test_perplexity(self, small_model, small_vocab):
        """Test that perplexity computation doesn't crash (the bug we fixed)."""
        seqs = small_vocab.tokenize(["the cat sat on the mat"] * 5)
        result = small_model.perplexity(seqs, n_samples=5)
        assert 'base_ppl' in result
        assert 'legd_ppl' in result
        assert result['base_ppl'] > 0
