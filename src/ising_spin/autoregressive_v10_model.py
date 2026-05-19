"""
V10 Autoregressive Ising Language Model.

FUNDAMENTAL ARCHITECTURAL CHANGE: Instead of sampling from the joint
P(w_1,...,w_n) ~ exp(-beta * E) simultaneously (which requires complex
MCMC and suffers from mode collapse), we generate one position at a time:

    P(w_t | w_1,...,w_{t-1}) ~ exp(-beta * E_position(t, w_t | context))

This converts a hard high-dimensional optimization into L easy 1D problems.

CRITICAL CONSTRAINT: Energy computation is INTEGER-ONLY.
FP only used for final probability normalization (same compromise as V8).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from .type_system import POSTypeSystem, N_POS, IDX2POS, POS2IDX
from .vocabulary import Vocabulary
from .pmi_couplings import PMICouplings
from .semantic_types import SemanticTypeSystem
from .dep_couplings import DependencyCouplings
from .caldera_nmf import CalderaNMF
from .enhanced_v8_model import (
    EnhancedV8Sampler,
    EnhancedV8Model,
    compute_implicational_penalty,
)


class AutoregressiveIsingGenerator:
    """
    Autoregressive generator for Ising spin language models.
    """

    def __init__(
        self,
        sampler: EnhancedV8Sampler,
        vocab: Vocabulary,
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

    def _build_type_word_index(self):
        self.type_words = {}
        for t in range(self.n_types):
            col = self.I_emit[:, t]
            words = [int(i) for i in range(len(col)) if col[i] > 0]
            self.type_words[t] = words

    def _compute_idf(self):
        V = self.vocab_size
        self.idf = np.zeros(V, dtype=np.int64)
        for w in range(V):
            idf_val = int(self.h[0, w])
            self.idf[w] = max(1, idf_val)

    def _apply_idf_weighting(self, sampler) -> np.ndarray:
        V = sampler.vocab_size
        J_idf = np.zeros((V, V), dtype=np.int64)
        J_raw = sampler.pmi.J_PMI

        for w in range(V):
            idf_w = int(self.idf[w]) if w < len(self.idf) else 1
            for w_prime in range(w + 1, V):
                j_val = int(J_raw[w, w_prime])
                if j_val == 0:
                    continue
                idf_wp = int(self.idf[w_prime]) if w_prime < len(self.idf) else 1
                avg_idf = (idf_w + idf_wp) // 2
                j_idf = (j_val * avg_idf) // self.idf_scale
                J_idf[w, w_prime] = j_idf
                J_idf[w_prime, w] = j_idf

        J = sampler.pmi_weight * J_idf
        J += sampler.hebbian_weight * sampler.pmi.J_Hebb
        if sampler.sem is not None:
            J += sampler.semantic_weight * sampler.sem.J_sem

        return J

    def _get_word_type(self, word_idx: int) -> int:
        if word_idx in self.types.allowed_types and self.types.allowed_types[word_idx]:
            return max(self.types.allowed_types[word_idx],
                       key=lambda t: int(self.I_emit[word_idx, t]))
        return POS2IDX["X"]

    def _get_valid_next_types(self, prev_type: int) -> List[int]:
        valid = []
        for t in range(self.n_types):
            if (prev_type, t) in self.allowed_transitions:
                valid.append(t)
        if not valid:
            return list(range(self.n_types))
        return valid

    def _compute_type_energy(self, pos: int, type_idx: int, prev_types: List[int]) -> int:
        energy = 0
        types_for_penalty = list(prev_types) + [type_idx]
        penalty = self.types.compute_grammar_penalty(
            types_for_penalty, len(prev_types), type_idx
        )
        energy += penalty
        impl_penalty = compute_implicational_penalty(
            types_for_penalty, len(prev_types), type_idx
        )
        energy += impl_penalty
        if len(prev_types) > 0 and type_idx == prev_types[-1]:
            if type_idx not in (POS2IDX['NOUN'], POS2IDX['X'], POS2IDX['PUNCT']):
                energy += 50
        return energy

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

        return energies

    def _boltzmann_sample(self, energies: np.ndarray, beta: float) -> int:
        if len(energies) == 1:
            return 0

        e = energies.astype(np.float64)
        e_min = e.min()
        log_weights = -beta * (e - e_min)
        log_weights = np.clip(log_weights, -500, 500)
        weights = np.exp(log_weights)
        total = weights.sum()

        if total <= 0 or not np.isfinite(total):
            return np.random.randint(len(energies))

        probs = weights / total
        probs = np.maximum(probs, 0)
        probs = probs / probs.sum()

        return np.random.choice(len(energies), p=probs)

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
            valid_types = self._get_valid_next_types(types[-1])
            type_energies = np.array([
                self._compute_type_energy(pos, t, types)
                for t in valid_types
            ], dtype=np.int64)
            type_idx = self._boltzmann_sample(type_energies, self.beta_type)
            chosen_type = valid_types[type_idx]

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
            word_idx = self._boltzmann_sample(word_energies, self.beta_word)
            chosen_word = int(candidate_words[word_idx])

            words.append(chosen_word)
            types.append(chosen_type)

            top5_indices = np.argsort(word_energies)[:5]
            diag = {
                'pos': pos,
                'chosen_type': IDX2POS.get(chosen_type, "UNK"),
                'chosen_word': self.vocab.idx2word.get(chosen_word, "<UNK>"),
                'top5_words': [self.vocab.idx2word.get(int(candidate_words[i]), "<UNK>")
                               for i in top5_indices],
                'top5_energies': [int(word_energies[i]) for i in top5_indices],
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

    def generate_raw(self, length: int = 15) -> Tuple[List[int], List[int]]:
        start_idx = np.random.randint(4, min(54, self.vocab_size))
        prompt = self.vocab.idx2word.get(start_idx, "the")
        result = self.generate(prompt=prompt, length=length)
        return result['words'], result['types']
