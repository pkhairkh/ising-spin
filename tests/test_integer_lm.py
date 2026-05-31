"""Tests for the Integer Language Model (v80 — Dynamic Features)."""
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


class TestFeatureSpec:
    def test_lex_bigram_feature(self, small_vocab):
        from ising_spin.feature_hash_energy import LexBigramFeature
        feat = LexBigramFeature(n_hashes=1, table_size=101, eta=1, clip=50)
        seqs = small_vocab.tokenize(["the cat sat on the mat"] * 10)
        # Test batch energy
        context = seqs[0][:2]
        candidates = np.array(seqs[0][2:5], dtype=np.int64)
        e = feat.energy_batch(context, candidates, small_vocab.word_pos)
        assert len(e) == 3
        # Before training, all energies should be 0
        assert np.all(e == 0)

    def test_word_pos_bigram_feature(self, small_vocab):
        from ising_spin.feature_hash_energy import WordPosBigramFeature
        feat = WordPosBigramFeature(n_hashes=1, table_size=101, eta=1, clip=30)
        seqs = small_vocab.tokenize(["the cat sat on the mat"] * 10)
        context = seqs[0][:2]
        candidates = np.array(seqs[0][2:5], dtype=np.int64)
        e = feat.energy_batch(context, candidates, small_vocab.word_pos)
        assert len(e) == 3
        # Before training, all energies should be 0
        assert np.all(e == 0)

    def test_feature_registration(self, small_vocab):
        """Test dynamic add/remove of features."""
        from ising_spin import FeatureHashEnergyTable
        from ising_spin.feature_hash_energy import LexBigramFeature, WordPosBigramFeature

        table = FeatureHashEnergyTable(
            vocab_size=small_vocab.V,
            word_pos=small_vocab.word_pos,
        )
        assert len(table.features) == 0

        feat1 = LexBigramFeature(n_hashes=1, table_size=101)
        table.add_feature(feat1)
        assert len(table.features) == 1
        assert "lex_bi" in table.features

        feat2 = WordPosBigramFeature(n_hashes=1, table_size=101)
        table.add_feature(feat2)
        assert len(table.features) == 2

        table.remove_feature("lex_bi")
        assert len(table.features) == 1
        assert "word_pos_bi" in table.features

    def test_variable_number_of_features(self, small_vocab):
        """Test that any number of features can be added."""
        from ising_spin import FeatureHashEnergyTable
        from ising_spin.feature_hash_energy import (
            LexBigramFeature, WordPosBigramFeature, PosWordBigramFeature,
            LexSkipFeature, PosTrigramFeature, LexTrigramFeature,
        )

        # 1 feature
        table1 = FeatureHashEnergyTable(
            vocab_size=small_vocab.V, word_pos=small_vocab.word_pos,
        )
        table1.add_feature(LexBigramFeature(n_hashes=1, table_size=101))
        assert len(table1.features) == 1

        # 6 features (default)
        table6 = FeatureHashEnergyTable(
            vocab_size=small_vocab.V, word_pos=small_vocab.word_pos,
        )
        for feat in [
            LexBigramFeature(n_hashes=1, table_size=101),
            WordPosBigramFeature(n_hashes=1, table_size=101),
            PosWordBigramFeature(n_hashes=1, table_size=101),
            LexSkipFeature(n_hashes=1, table_size=101),
            PosTrigramFeature(n_hashes=1, table_size=101),
            LexTrigramFeature(n_hashes=1, table_size=101),
        ]:
            table6.add_feature(feat)
        assert len(table6.features) == 6


class TestFeatureHashEnergy:
    def test_train_and_query(self, small_vocab):
        from ising_spin import FeatureHashEnergyTable
        from ising_spin.feature_hash_energy import (
            LexBigramFeature, WordPosBigramFeature,
        )
        table = FeatureHashEnergyTable(
            vocab_size=small_vocab.V,
            word_pos=small_vocab.word_pos,
        )
        table.add_feature(LexBigramFeature(n_hashes=1, table_size=1009, eta=1, clip=50))
        table.add_feature(WordPosBigramFeature(n_hashes=1, table_size=1009, eta=1, clip=30))
        seqs = small_vocab.tokenize(["the cat sat on the mat"] * 10)
        table.train_nce(seqs, n_epochs=1, n_negatives=2)
        # Energy for real pair should be computable
        ctx = seqs[0][:2]
        target = seqs[0][2]
        real_e = table.compute_local_energy(ctx, target)
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
        """Test that perplexity computation doesn't crash."""
        seqs = small_vocab.tokenize(["the cat sat on the mat"] * 5)
        result = small_model.perplexity(seqs, n_samples=5)
        assert 'base_ppl' in result
        assert 'legd_ppl' in result
        assert result['base_ppl'] > 0

    def test_dynamic_features(self, small_vocab):
        """Test add_feature / remove_feature on IntegerLM."""
        from ising_spin import IntegerLM
        from ising_spin.feature_hash_energy import (
            LexBigramFeature, WordPosBigramFeature,
        )

        model = IntegerLM(
            vocab=small_vocab,
            features=[
                LexBigramFeature(n_hashes=1, table_size=101, eta=1, clip=50),
            ],
            top_k=20,
            seed=42,
        )
        assert model.list_features() == ["lex_bi"]

        model.add_feature(WordPosBigramFeature(n_hashes=1, table_size=101, eta=1, clip=30))
        assert model.list_features() == ["lex_bi", "word_pos_bi"]

        model.remove_feature("lex_bi")
        assert model.list_features() == ["word_pos_bi"]

    def test_default_features(self, small_vocab):
        """Test that default_features() produces a working model."""
        from ising_spin import IntegerLM
        from ising_spin.feature_hash_energy import default_features

        features = default_features(vocab_size=small_vocab.V)
        assert len(features) == 6

        model = IntegerLM(
            vocab=small_vocab,
            features=features,
            top_k=20,
            seed=42,
        )
        assert len(model.list_features()) == 6
