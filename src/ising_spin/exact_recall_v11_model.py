"""
V11 Autoregressive Ising Language Model with Exact Token Recall.

Inspired by GFST-HMB's "exact token recall from stored blocks" architecture.

INTEGER-ONLY CONSTRAINT:
  - All n-gram counts are integers
  - All recall bonuses are integers
  - All energy terms are integers
  - FP only used for final Boltzmann normalization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter

from .type_system import POSTypeSystem, N_POS, IDX2POS, POS2IDX
from .vocabulary import Vocabulary
from .pmi_couplings import PMICouplings
from .semantic_types import SemanticTypeSystem
from .dep_couplings import DependencyCouplings
from .caldera_nmf import CalderaNMF
from .autoregressive_v10_model import AutoregressiveIsingGenerator


class NGramIndex:
    """Multi-level n-gram index built from training corpus."""

    def __init__(self, max_n: int = 5, min_count: int = 1):
        self.max_n = max_n
        self.min_count = min_count
        self.index = {k: defaultdict(Counter) for k in range(1, max_n + 1)}
        self.context_totals = {k: defaultdict(int) for k in range(1, max_n + 1)}
        self.total_ngrams = Counter()
        self.total_continuations = Counter()
        self._built = False

    def build(self, sequences: List[List[int]]):
        print(f"  Building n-gram index (max_n={self.max_n})...")

        for seq in sequences:
            start = 0
            for i, w in enumerate(seq):
                if w >= 4:
                    start = i
                    break

            for t in range(start, len(seq)):
                for k in range(1, self.max_n + 1):
                    if t - k < start:
                        break
                    context = tuple(seq[t-k:t])
                    continuation = seq[t]
                    if any(w < 4 for w in context):
                        continue
                    if continuation < 4:
                        continue
                    self.index[k][context][continuation] += 1
                    self.context_totals[k][context] += 1
                    self.total_ngrams[k] += 1

        pruned = 0
        for k in range(1, self.max_n + 1):
            for context in list(self.index[k].keys()):
                low_count = [w for w, c in self.index[k][context].items()
                            if c < self.min_count]
                for w in low_count:
                    del self.index[k][context][w]
                    self.context_totals[k][context] -= 1
                    pruned += 1
                if not self.index[k][context]:
                    del self.index[k][context]
                    del self.context_totals[k][context]

        self._built = True

        for k in range(1, self.max_n + 1):
            n_contexts = len(self.index[k])
            n_conts = sum(len(v) for v in self.index[k].values())
            self.total_continuations[k] = n_conts
            print(f"    {k}-gram: {n_contexts:,} contexts, "
                  f"{n_conts:,} continuations, "
                  f"{self.total_ngrams[k]:,} total occurrences")

        print(f"    Pruned {pruned} low-count continuations (min_count={self.min_count})")

    def lookup(self, context_words: List[int], max_k: Optional[int] = None) -> Dict:
        if max_k is None:
            max_k = self.max_n
        results = {}
        for k in range(min(max_k, len(context_words)), 0, -1):
            context = tuple(context_words[-k:])
            if context in self.index[k]:
                total = self.context_totals[k][context]
                conts = self.index[k][context]
                sorted_conts = conts.most_common()
                results[k] = [(word, count, total) for word, count in sorted_conts]
        return results

    def get_recall_bonus(
        self,
        context_words: List[int],
        candidate_words: np.ndarray,
        recall_scale: int = 100,
        context_weight_factor: int = 4,
        max_k: Optional[int] = None,
        longest_only: bool = False,
    ) -> np.ndarray:
        n_candidates = len(candidate_words)
        bonuses = np.zeros(n_candidates, dtype=np.int64)
        matches = self.lookup(context_words, max_k=max_k)
        if not matches:
            return bonuses
        if longest_only and matches:
            best_k = max(matches.keys())
            matches = {best_k: matches[best_k]}
        for k, continuations in matches.items():
            context_weight = context_weight_factor ** (k - 1)
            cont_lookup = {}
            for word, count, total in continuations:
                if k >= 3:
                    bonus = count * recall_scale * context_weight
                else:
                    bonus = (count * recall_scale * context_weight) // max(1, total)
                cont_lookup[word] = int(bonus)
            for i, w in enumerate(candidate_words):
                w_int = int(w)
                if w_int in cont_lookup:
                    bonuses[i] += cont_lookup[w_int]
        return bonuses

    def get_best_copy_candidate(
        self,
        context_words: List[int],
        min_context_length: int = 3,
        min_confidence: float = 0.3,
    ) -> Optional[Tuple[int, int, int]]:
        matches = self.lookup(context_words)
        for k in sorted(matches.keys(), reverse=True):
            if k < min_context_length:
                break
            continuations = matches[k]
            if not continuations:
                continue
            best_word, best_count, total = continuations[0]
            confidence_threshold = int(min_confidence * 10)
            if best_count * 10 >= total * confidence_threshold:
                return (best_word, best_count, total)
        return None

    def get_stats(self) -> Dict:
        stats = {
            'max_n': self.max_n,
            'min_count': self.min_count,
            'built': self._built,
        }
        for k in range(1, self.max_n + 1):
            n_contexts = len(self.index[k])
            n_conts = sum(len(v) for v in self.index[k].values())
            total_occ = self.total_ngrams[k]
            avg_conts = n_conts / max(1, n_contexts)
            stats[f'{k}gram_contexts'] = n_contexts
            stats[f'{k}gram_continuations'] = n_conts
            stats[f'{k}gram_occurrences'] = total_occ
            stats[f'{k}gram_avg_continuations'] = round(avg_conts, 1)
        return stats


class ExactRecallV11Generator(AutoregressiveIsingGenerator):
    """V11 Autoregressive Ising Generator with Exact Token Recall."""

    def __init__(
        self,
        sampler,
        vocab: Vocabulary,
        ngram_index: NGramIndex,
        recall_scale: int = 100,
        context_weight_factor: int = 4,
        copy_min_context: int = 3,
        copy_min_confidence: float = 0.3,
        copy_enabled: bool = True,
        beta_type: float = 0.005,
        beta_word: float = 0.03,
        top_k_words: int = 200,
        use_idf_coupling: bool = True,
        idf_scale: int = 8,
        field_weight: float = 0.001,
        coupling_weight: int = 10,
    ):
        self.sampler = sampler
        self.vocab = vocab
        self.beta_type = beta_type
        self.beta_word = beta_word
        self.top_k_words = top_k_words
        self.use_idf_coupling = use_idf_coupling
        self.idf_scale = idf_scale
        self.field_weight = field_weight
        self.coupling_weight = coupling_weight

        self.ngram_index = ngram_index
        self.recall_scale = recall_scale
        self.context_weight_factor = context_weight_factor
        self.copy_min_context = copy_min_context
        self.copy_min_confidence = copy_min_confidence
        self.copy_enabled = copy_enabled
        self.recall_longest_only = True
        self.recall_suggests_type = True

        self.h = sampler.pmi.h
        self.vocab_size = sampler.vocab_size
        self.types = sampler.types
        self.n_types = sampler.n_types

        self.I_emit = sampler.types.I_emit
        self.emit_strength = getattr(sampler, 'emit_strength', 50)
        self.use_symmetric_emission = getattr(sampler, 'use_symmetric_emission', True)
        self.emission_bonus = sampler.emission_bonus
        self.emission_penalty = sampler.emission_penalty

        self.window = sampler.pmi.window
        self.seq_len = sampler.pmi.seq_len

        self.deps = sampler.deps
        self.dep_weight = sampler.dep_weight

        self.allowed_transitions = sampler.allowed_transitions

        self._build_type_word_index()
        self.repetition_penalty = 200
        self._compute_idf()

        self.J = sampler.J_combined * self.coupling_weight
        if use_idf_coupling and hasattr(sampler, 'J_idf') and sampler.J_idf is not None:
            if hasattr(sampler, '_build_combined_coupling_v9'):
                self.J = sampler._build_combined_coupling_v9() * self.coupling_weight
            else:
                self.J = sampler.J_combined * self.coupling_weight
        elif use_idf_coupling:
            self.J = self._apply_idf_weighting(sampler) * self.coupling_weight

        self.nmf = sampler.nmf

        self._recall_stats = {
            'total_positions': 0,
            'recall_hits': 0,
            'copy_used': 0,
            'avg_recall_bonus': 0,
            'max_recall_bonus': 0,
        }

    def _compute_word_energy_vectorized(
        self, pos: int, candidate_words: np.ndarray, word_type: int,
        fixed_words: List[int], fixed_types: List[int]
    ) -> np.ndarray:
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)

        field_vals = self.h[pos % self.seq_len, candidate_words].astype(np.float64) * self.field_weight
        energies -= field_vals.astype(np.int64)

        context_start = max(0, pos - self.window)
        context_words = fixed_words[context_start:pos]
        if len(context_words) > 0:
            context_arr = np.array(context_words, dtype=np.int64)
            coupling_block = self.J[np.ix_(candidate_words, context_arr)]
            coupling_sums = coupling_block.sum(axis=1)
            energies -= coupling_sums

        if len(context_words) > 0 and self.repetition_penalty > 0:
            context_set = set(context_words)
            for i, w in enumerate(candidate_words):
                if int(w) in context_set:
                    energies[i] += self.repetition_penalty

        if not self.use_symmetric_emission:
            emit_vals = self.I_emit[candidate_words, word_type]
            pos_mask = emit_vals > 0
            neg_mask = emit_vals < 0
            energies[pos_mask] -= emit_vals[pos_mask] * self.emission_bonus
            energies[neg_mask] += self.emission_penalty

        recall_bonuses = self.ngram_index.get_recall_bonus(
            context_words=fixed_words,
            candidate_words=candidate_words,
            recall_scale=self.recall_scale,
            context_weight_factor=self.context_weight_factor,
            longest_only=self.recall_longest_only,
        )
        energies -= recall_bonuses

        self._recall_stats['total_positions'] += 1
        if recall_bonuses.max() > 0:
            self._recall_stats['recall_hits'] += 1
            self._recall_stats['avg_recall_bonus'] += int(recall_bonuses.mean())
            self._recall_stats['max_recall_bonus'] = max(
                self._recall_stats['max_recall_bonus'],
                int(recall_bonuses.max())
            )

        return energies

    def generate(self, prompt: str, length: int = 15) -> Dict:
        prompt_idx = self.vocab.word2idx.get(prompt, None)
        if prompt_idx is None:
            prompt_idx = self.vocab.word2idx.get(prompt.lower(), None)
        if prompt_idx is None:
            prompt_idx = 4

        prompt_type = self._get_word_type(prompt_idx)
        words = [prompt_idx]
        types = [prompt_type]
        position_diagnostics = []

        for pos in range(1, length):
            recall_type_override = None
            copy_word = None
            diag_copy = False

            if self.recall_suggests_type and len(words) >= 2:
                recall_matches = self.ngram_index.lookup(words)
                if recall_matches:
                    best_k = max(recall_matches.keys())
                    best_conts = recall_matches[best_k]
                    if best_k >= 2 and best_conts:
                        best_recall_word = best_conts[0][0]
                        best_recall_count = best_conts[0][1]
                        best_recall_total = best_conts[0][2]
                        if best_recall_count * 3 >= best_recall_total:
                            recall_word_type = self._get_word_type(best_recall_word)
                            if (types[-1], recall_word_type) in self.allowed_transitions:
                                recall_type_override = recall_word_type

            if recall_type_override is not None:
                chosen_type = recall_type_override
            else:
                valid_types = self._get_valid_next_types(types[-1])
                type_energies = np.array([
                    self._compute_type_energy(pos, t, types)
                    for t in valid_types
                ], dtype=np.int64)
                type_idx = self._boltzmann_sample(type_energies, self.beta_type)
                chosen_type = valid_types[type_idx]

            if self.copy_enabled and len(words) >= self.copy_min_context:
                copy_candidate = self.ngram_index.get_best_copy_candidate(
                    context_words=words,
                    min_context_length=self.copy_min_context,
                    min_confidence=self.copy_min_confidence,
                )
                if copy_candidate is not None:
                    copy_word_idx, copy_count, copy_total = copy_candidate
                    if copy_word_idx < self.I_emit.shape[0]:
                        if int(self.I_emit[copy_word_idx, chosen_type]) > 0:
                            copy_word = copy_word_idx
                            self._recall_stats['copy_used'] += 1

            candidate_words_list = self.type_words.get(chosen_type, [])
            if not candidate_words_list:
                candidate_words_list = list(range(min(200, self.vocab_size)))
            candidate_words = np.array(candidate_words_list, dtype=np.int64)

            if len(candidate_words) > self.top_k_words:
                field_vals = self.h[pos % self.seq_len, candidate_words]
                top_k_indices = np.argsort(field_vals)[-self.top_k_words:]
                candidate_words = candidate_words[top_k_indices]

            word_energies = self._compute_word_energy_vectorized(
                pos, candidate_words, chosen_type, words, types
            )

            if copy_word is not None:
                chosen_word = copy_word
                diag_copy = True
            else:
                word_idx = self._boltzmann_sample(word_energies, self.beta_word)
                chosen_word = int(candidate_words[word_idx])

            words.append(chosen_word)
            types.append(chosen_type)

            top5_indices = np.argsort(word_energies)[:5]
            recall_matches = self.ngram_index.lookup(words[:-1])
            recall_info = {}
            for k, conts in recall_matches.items():
                for w, c, t in conts[:3]:
                    if w == chosen_word:
                        recall_info[f'{k}gram'] = f"{self.vocab.idx2word.get(w, '?')}({c}/{t})"

            diag = {
                'pos': pos,
                'chosen_type': IDX2POS.get(chosen_type, "UNK"),
                'chosen_word': self.vocab.idx2word.get(chosen_word, "<UNK>"),
                'copy_used': diag_copy,
                'top5_words': [self.vocab.idx2word.get(int(candidate_words[i]), "<UNK>")
                               for i in top5_indices],
                'top5_energies': [int(word_energies[i]) for i in top5_indices],
                'recall_matches': recall_info,
            }
            position_diagnostics.append(diag)

        text = self.vocab.decode(words)
        type_names = [IDX2POS.get(t, "UNK") for t in types]

        return {
            'text': text,
            'words': words,
            'types': types,
            'type_names': type_names,
            'diagnostics': position_diagnostics,
        }

    def get_recall_stats(self) -> Dict:
        stats = self._recall_stats.copy()
        if stats['recall_hits'] > 0:
            stats['avg_recall_bonus'] /= stats['recall_hits']
        stats['recall_hit_rate'] = (
            stats['recall_hits'] / max(1, stats['total_positions'])
        )
        stats['copy_rate'] = (
            stats['copy_used'] / max(1, stats['total_positions'])
        )
        return stats


class ExactRecallV11Model:
    """V11 Autoregressive Ising Language Model with Exact Token Recall."""

    def __init__(self, **kwargs):
        self.v11_recall_scale = kwargs.pop('recall_scale', 100)
        self.v11_context_weight_factor = kwargs.pop('context_weight_factor', 4)
        self.v11_copy_min_context = kwargs.pop('copy_min_context', 3)
        self.v11_copy_min_confidence = kwargs.pop('copy_min_confidence', 0.3)
        self.v11_copy_enabled = kwargs.pop('copy_enabled', True)
        self.v11_ngram_max_n = kwargs.pop('ngram_max_n', 5)
        self.v11_ngram_min_count = kwargs.pop('ngram_min_count', 1)

        self.v10_beta_type = kwargs.pop('beta_type', 0.005)
        self.v10_beta_word = kwargs.pop('beta_word', 0.03)
        self.v10_top_k_words = kwargs.pop('top_k_words', 200)
        self.v10_use_idf_coupling = kwargs.pop('use_idf_coupling', True)
        self.v10_idf_scale = kwargs.pop('idf_scale', 8)
        self.v10_field_weight = kwargs.pop('field_weight', 0.001)
        self.v10_coupling_weight = kwargs.pop('coupling_weight', 10)

        from .enhanced_v8_model import EnhancedV8Model
        self.v8_model = EnhancedV8Model(**kwargs)

        self.ngram_index = None
        self.v11_generator = None

    def train(self, n_samples: int = 20000):
        import time as _time
        from .data_loader import load_fineweb_edu, tokenize_texts, truncate_sequences

        print("=" * 70)
        print("V11 TRAINING: V8 Pipeline + N-Gram Index Building")
        print("=" * 70)

        self.v8_model.train(n_samples=n_samples)

        print("\n--- Building N-Gram Index for Exact Token Recall ---")

        t0 = _time.time()
        texts = load_fineweb_edu(n_samples=n_samples)
        sequences = tokenize_texts(texts, self.v8_model.vocab)
        sequences = truncate_sequences(sequences, max_len=self.v8_model.seq_len)
        print(f"  Re-loaded {len(sequences)} sequences for n-gram indexing ({_time.time()-t0:.1f}s)")

        self.ngram_index = NGramIndex(
            max_n=self.v11_ngram_max_n,
            min_count=self.v11_ngram_min_count,
        )
        self.ngram_index.build(sequences)

        self._sequences = sequences
        stats = self.ngram_index.get_stats()
        print(f"\n  N-gram index built: {stats}")

        self._build_generator()

    def _build_generator(self):
        self.v11_generator = ExactRecallV11Generator(
            sampler=self.v8_model.sampler,
            vocab=self.v8_model.vocab,
            ngram_index=self.ngram_index,
            recall_scale=self.v11_recall_scale,
            context_weight_factor=self.v11_context_weight_factor,
            copy_min_context=self.v11_copy_min_context,
            copy_min_confidence=self.v11_copy_min_confidence,
            copy_enabled=self.v11_copy_enabled,
            beta_type=self.v10_beta_type,
            beta_word=self.v10_beta_word,
            top_k_words=self.v10_top_k_words,
            use_idf_coupling=self.v10_use_idf_coupling,
            idf_scale=self.v10_idf_scale,
            field_weight=self.v10_field_weight,
            coupling_weight=self.v10_coupling_weight,
        )

    def generate_with_trace(self, prompt: str = "the", length: int = 15) -> Dict:
        if self.v11_generator is None:
            self._build_generator()
        result = self.v11_generator.generate(prompt=prompt, length=length)
        result['recall_stats'] = self.v11_generator.get_recall_stats()
        result['energy'] = 0
        result['demon_stats'] = {'demon_energy': 0, 'acceptance_rate': 1.0}
        result['marginal_stats'] = {
            'avg_consecutive_count': 0, 'max_consecutive_count': 0,
            'stuck_5plus': 0, 'flip_rate': 1.0,
        }
        return result

    def generate_raw(self, length: int = 15) -> Tuple[List[int], List[int]]:
        if self.v11_generator is None:
            self._build_generator()
        start_idx = np.random.randint(4, min(54, self.v11_generator.vocab_size))
        prompt = self.v8_model.vocab.idx2word.get(start_idx, "the")
        result = self.v11_generator.generate(prompt=prompt, length=length)
        return result['words'], result['types']

    def evaluate_grammar(self, words, types):
        return self.v8_model.evaluate_grammar(words, types)

    @property
    def vocab(self):
        return self.v8_model.vocab

    @property
    def sequences(self):
        return getattr(self, '_sequences', [])
