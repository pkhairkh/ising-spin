"""
Enhanced Typed Ising-Potts Language Model v3.

Integrates all four enhancements:
  1. SpaCy-based POS tagger for accurate type couplings
  2. Dependency tree couplings (J_tree) for long-range subject-verb agreement
  3. Integer matrix factorization (J ≈ W×H) for vocabulary scaling beyond 3K
  4. Larger corpus training to densify the PMI coupling matrix

Energy function:
  E(types, words) = E_type(types)
                  + E_emit(words|types)
                  + E_lexical(words)        [factorized via NMF]
                  + E_semantic(words)
                  + E_grammar(types, words)
                  + E_dep(types, words)      [NEW: long-range dependency coupling]

All generation-path computation remains integer arithmetic only.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .vocabulary import Vocabulary
from .pmi_couplings import PMICouplings
from .type_system import POSTypeSystem, N_POS, IDX2POS, POS2IDX
from .semantic_types import SemanticTypeSystem, N_SEM, SEMANTIC_SUPERTYPES
from .spacy_tagger import SpaCyTagger
from .dep_couplings import DependencyCouplings
from .int_nmf import IntegerNMF
from .data_loader import load_fineweb_edu, tokenize_texts, truncate_sequences


class EnhancedTypedSampler:
    """
    Enhanced sampler integrating dependency couplings and NMF-factorized
    energy computation.

    Generation loop is ZERO FLOATING-POINT:
      - PMI + Hebbian + semantic coupling: NMF-factorized O(K) per position
      - Dependency coupling: sparse J_tree lookup O(nnz) per position
      - Type coupling: exact enumeration over T≈13 states
      - Grammar + agreement penalties: integer comparison and addition
      - Acceptance: precomputed integer threshold table + integer comparison
    """

    def __init__(
        self,
        pmi_couplings: PMICouplings,
        type_system: POSTypeSystem,
        semantic_system: Optional[SemanticTypeSystem],
        dep_couplings: Optional[DependencyCouplings],
        nmf: Optional[IntegerNMF],
        # Temperature schedule
        phase1_beta: int = 200,
        phase2_beta: int = 500,
        phase3_beta: int = 1000,
        total_sweeps: int = 150,
        phase1_frac: float = 0.25,
        phase2_frac: float = 0.35,
        phase3_frac: float = 0.40,
        # Weights
        pmi_weight: int = 3,
        hebbian_weight: int = 1,
        semantic_weight: int = 1,
        dep_weight: int = 2,
        # Proposals
        proposal_top_k: int = 30,
    ):
        self.pmi = pmi_couplings
        self.types = type_system
        self.sem = semantic_system
        self.deps = dep_couplings
        self.nmf = nmf

        self.vocab_size = pmi_couplings.vocab_size
        self.n_types = type_system.n_types

        self.pmi_weight = pmi_weight
        self.hebbian_weight = hebbian_weight
        self.semantic_weight = semantic_weight
        self.dep_weight = dep_weight

        # Sweep allocation
        self.sweeps_p1 = int(total_sweeps * phase1_frac)
        self.sweeps_p2 = int(total_sweeps * phase2_frac)
        self.sweeps_p3 = total_sweeps - self.sweeps_p1 - self.sweeps_p2

        # Precompute probability tables for each phase
        self.prob_table_p1 = self._build_prob_table(phase1_beta)
        self.prob_table_p2 = self._build_prob_table(phase2_beta)
        self.prob_table_p3 = self._build_prob_table(phase3_beta)

        # Build combined coupling matrix (for non-NMF path)
        if self.nmf is None or not self.nmf.fitted:
            self.J_combined = self._build_combined_coupling()
        else:
            self.J_combined = None

        # Precompute proposal sets
        self.proposal_cache = self._build_proposal_cache(proposal_top_k)

        # Precompute distributions
        self.type_cumsum_by_pos = self._build_type_distributions()
        self.emit_cumsum_by_type = self._build_emission_distributions()
        self.field_weights = self._build_field_weights()

        # Precompute J_tree neighbor cache for fast lookup
        self.tree_neighbor_cache = self._build_tree_neighbor_cache()

    def _build_prob_table(self, beta_int, max_delta_e=5000, rand_max=2**31-1):
        """Build integer threshold table for Metropolis acceptance."""
        thresholds = [0] * (2 * max_delta_e + 1)
        for delta_e in range(-max_delta_e, max_delta_e + 1):
            idx = delta_e + max_delta_e
            if delta_e <= 0:
                thresholds[idx] = rand_max
            else:
                import math
                beta = beta_int / 1000.0
                prob = math.exp(-delta_e * beta)
                threshold = int(rand_max * prob)
                thresholds[idx] = max(0, min(rand_max, threshold))
        return thresholds

    def _accept(self, delta_e, rand_val, table):
        max_delta_e = (len(table) - 1) // 2
        if delta_e <= -max_delta_e:
            return True
        if delta_e >= max_delta_e:
            return False
        return rand_val < table[delta_e + max_delta_e]

    def _build_combined_coupling(self):
        """Build combined coupling: J = alpha*J_PMI + beta*J_Hebb + gamma*J_sem."""
        J = self.pmi_weight * self.pmi.J_PMI + self.hebbian_weight * self.pmi.J_Hebb
        if self.sem is not None:
            J = J + self.semantic_weight * self.sem.J_sem
        return J

    def _build_proposal_cache(self, top_k):
        """Precompute proposal sets for each word."""
        import random
        cache = {}
        for w in range(self.vocab_size):
            candidates = set()

            # PMI neighbors
            pmi_neighbors = self.pmi.get_neighbor_words(w, top_k=top_k)
            candidates.update(pmi_neighbors)

            # Emission-compatible words
            if w in self.types.allowed_types:
                type_set = self.types.allowed_types[w]
                for t in type_set:
                    compat_words = self.types.get_allowed_words_for_type(t)
                    candidates.update(compat_words[:top_k])

            # Dependency tree neighbors (NEW)
            if self.deps is not None:
                tree_neighbors = self.deps.get_tree_neighbors(w, top_k=top_k)
                for w_idx, _ in tree_neighbors:
                    candidates.add(w_idx)

            # NMF neighbors (NEW)
            if self.nmf is not None and self.nmf.fitted:
                nmf_neighbors = self.nmf.get_top_neighbors(w, top_k=top_k)
                for w_idx, _ in nmf_neighbors:
                    candidates.add(w_idx)

            if w not in candidates:
                candidates.add(w)

            cache[w] = list(candidates)[:top_k * 3]

        return cache

    def _build_tree_neighbor_cache(self):
        """Precompute J_tree neighbors for each word."""
        if self.deps is None:
            return {}

        cache = {}
        for w in range(self.vocab_size):
            row = self.deps.J_tree[w]
            neighbors = [(int(i), int(row[i])) for i in range(self.vocab_size) if row[i] != 0]
            if neighbors:
                cache[w] = neighbors
        return cache

    def _build_type_distributions(self):
        """Precompute type proposal distributions per position."""
        distributions = {}
        for pos in range(min(self.pmi.seq_len, 50)):
            type_weights = self.types.J_type.sum(axis=1).copy()
            if pos < self.pmi.seq_len:
                h_pos = self.pmi.h[pos]
                for w in range(min(self.vocab_size, len(h_pos))):
                    if h_pos[w] > 0 and w in self.types.allowed_types:
                        for t in self.types.allowed_types[w]:
                            type_weights[t] += int(h_pos[w])
            if type_weights.sum() > 0:
                distributions[pos] = np.cumsum(type_weights)
            else:
                type_weights[:] = 1
                distributions[pos] = np.cumsum(type_weights)
        return distributions

    def _build_emission_distributions(self):
        """Precompute word emission distributions per type."""
        distributions = {}
        for t in range(self.n_types):
            col = self.types.I_emit[:, t].copy()
            h0 = self.pmi.h[0]
            col = col * np.maximum(h0, 1)
            if col.sum() > 0:
                distributions[t] = np.cumsum(col)
            else:
                col[:] = 1
                distributions[t] = np.cumsum(col)
        return distributions

    def _build_field_weights(self):
        """Precompute field-weighted distributions per position."""
        weights = {}
        for i in range(min(self.pmi.seq_len, 50)):
            h = self.pmi.h[i].copy()
            if h.sum() > 0:
                weights[i] = np.cumsum(h)
            else:
                h[:] = 1
                weights[i] = np.cumsum(h)
        return weights

    def _sample_from_cumsum(self, cumsum):
        import random
        total = int(cumsum[-1])
        if total <= 0:
            return 0
        rv = random.randint(1, total)
        idx = int(np.searchsorted(cumsum, rv))
        return min(idx, len(cumsum) - 1)

    def _sample_type(self, pos, current_type, state_types):
        """Sample type using exact enumeration with dependency agreement."""
        import random

        type_energies = np.zeros(self.n_types, dtype=np.int64)

        for t in range(self.n_types):
            energy = 0

            # Type-type coupling
            for j_offset in range(1, self.types.window + 1):
                j = pos + j_offset
                if j < len(state_types):
                    energy += int(self.types.J_type[t, state_types[j]])
                j = pos - j_offset
                if j >= 0:
                    energy += int(self.types.J_type[state_types[j], t])

            # Grammar penalty
            energy -= self.types.compute_grammar_penalty(state_types, pos, t)

            # Dependency agreement penalty (NEW)
            if self.deps is not None:
                energy -= self.deps.compute_agreement_penalty(state_types, pos, t)

            type_energies[t] = energy

        max_e = int(type_energies.max())
        type_weights = np.maximum(max_e - type_energies + 1, 0).astype(np.int64)

        total = int(type_weights.sum())
        if total <= 0:
            type_weights[:] = 1
            total = self.n_types

        cumsum = np.cumsum(type_weights)
        rv = random.randint(1, int(cumsum[-1]))
        idx = int(np.searchsorted(cumsum, rv))
        return min(idx, self.n_types - 1)

    def _compute_word_energy(
        self, pos, word, word_type, state_words, state_types, H_sum=None
    ):
        """
        Compute total energy contribution from a word at a position.

        E = E_field + E_PMI_coupling + E_type_coupling + E_emission
          + E_grammar + E_dependency

        Uses NMF-factorized PMI if available (O(K) per position).
        Otherwise uses full J_combined (O(V) per position).
        """
        energy = 0

        # 1. Field energy
        energy += int(self.pmi.h[pos % self.pmi.seq_len, word])

        # 2. PMI + Hebbian + semantic coupling
        if self.J_combined is not None:
            # Full matrix path
            for j_offset in range(1, self.pmi.window + 1):
                j = pos + j_offset
                if j < len(state_words):
                    energy += int(self.J_combined[word, state_words[j]])
                j = pos - j_offset
                if j >= 0:
                    energy += int(self.J_combined[state_words[j], word])
        elif self.nmf is not None and self.nmf.fitted and H_sum is not None:
            # NMF-factorized path: O(K) per position
            # E = W[word, :] @ H_sum
            energy += int(self.nmf.W[word] @ H_sum)

        # 3. Type coupling (Potts gating)
        for j_offset in range(1, self.types.window + 1):
            j = pos + j_offset
            if j < len(state_types):
                if word_type == state_types[j]:
                    energy += int(self.types.J_type[word_type, word_type]) // self.n_types
            j = pos - j_offset
            if j >= 0:
                if word_type == state_types[j]:
                    energy += int(self.types.J_type[word_type, word_type]) // self.n_types

        # 4. Emission energy
        if word < self.types.I_emit.shape[0] and word_type < self.types.I_emit.shape[1]:
            emit_val = int(self.types.I_emit[word, word_type])
            if emit_val > 0:
                energy += emit_val * 10
            else:
                energy -= 50

        # 5. Grammar penalty
        types_copy = list(state_types)
        types_copy[pos] = word_type
        energy -= self.types.compute_grammar_penalty(types_copy, pos, word_type)

        # 6. Dependency coupling (NEW: long-range)
        if self.deps is not None:
            # Use sparse lookup from tree_neighbor_cache
            tree_neighbors = self.tree_neighbor_cache.get(word, [])
            for j_word, j_val in tree_neighbors:
                # Check all positions for this word
                for j in range(len(state_words)):
                    if j == pos:
                        continue
                    if state_words[j] == j_word:
                        # Type-based gating
                        j_type = state_types[j]
                        type_bonus = 0
                        for dl in range(self.deps.n_dep):
                            if self.deps.J_tree_type[dl, word_type, j_type] > 0:
                                type_bonus += int(self.deps.J_tree_type[dl, word_type, j_type])
                            if self.deps.J_tree_type[dl, j_type, word_type] > 0:
                                type_bonus += int(self.deps.J_tree_type[dl, j_type, word_type])

                        if type_bonus > 0:
                            energy += j_val * self.dep_weight + type_bonus // (self.deps.n_dep * 2)
                        else:
                            # Weaker coupling without type match
                            dist = abs(pos - j)
                            if dist <= 5:
                                energy += j_val * self.dep_weight // 2

            # Dependency agreement penalty
            energy -= self.deps.compute_agreement_penalty(state_types, pos, word_type)

        return energy

    def _sample_word(self, pos, current_word, current_type, state_words,
                     state_types, prob_table, H_sum=None):
        """Propose and accept/reject a word for position pos."""
        import random

        r = random.randint(0, 99)

        if r < 30:
            if current_type in self.emit_cumsum_by_type:
                proposed = self._sample_from_cumsum(self.emit_cumsum_by_type[current_type])
            else:
                proposed = current_word
        elif r < 60:
            neighbors = self.proposal_cache.get(current_word, [current_word])
            proposed = random.choice(neighbors)
        elif r < 85:
            pos_key = pos % min(self.pmi.seq_len, 50)
            if pos_key in self.field_weights:
                proposed = self._sample_from_cumsum(self.field_weights[pos_key])
            else:
                proposed = random.randint(0, self.vocab_size - 1)
        else:
            proposed = random.randint(0, self.vocab_size - 1)

        if proposed == current_word:
            return current_word

        proposed_type = int(self.types.get_type_for_word(proposed))

        current_energy = self._compute_word_energy(
            pos, current_word, current_type, state_words, state_types, H_sum
        )
        proposed_energy = self._compute_word_energy(
            pos, proposed, proposed_type, state_words, state_types, H_sum
        )

        delta_e = proposed_energy - current_energy

        rand_val = random.randint(0, 2**31 - 2)
        if self._accept(delta_e, rand_val, prob_table):
            # Update H_sum if using NMF
            if H_sum is not None and self.nmf is not None and self.nmf.fitted:
                H_sum = self.nmf.update_H_sum(H_sum, current_word, proposed)
            return proposed, H_sum

        return current_word, H_sum

    def generate(self, length=20, prompt=None, vocab=None, verbose=False):
        """Generate text using staged annealing with all enhancements."""
        import random

        # Encode prompt
        prompt_words, prompt_types = [], []
        if prompt and vocab:
            prompt_tokens = vocab._tokenize(prompt)
            for tok in prompt_tokens:
                w_idx = vocab.word2idx.get(tok, vocab.word2idx.get("<UNK>", 0))
                prompt_words.append(w_idx)
                prompt_types.append(self.types.get_type_for_word(w_idx))

        prompt_len = len(prompt_words)

        # Initialize state
        state_words = list(prompt_words)
        state_types = list(prompt_types)

        for i in range(prompt_len, length):
            pos_key = i % min(self.pmi.seq_len, 50)
            if pos_key in self.field_weights:
                w = self._sample_from_cumsum(self.field_weights[pos_key])
            else:
                w = random.randint(0, self.vocab_size - 1)
            state_words.append(w)
            state_types.append(self.types.get_type_for_word(w))

        # Precompute H_sum for NMF path
        H_sum = None
        if self.nmf is not None and self.nmf.fitted:
            H_sum = self.nmf.compute_H_sum(state_words)

        if verbose:
            self._print_state(state_words, state_types, vocab, "Init")

        # PHASE 1: High temperature — types only
        for sweep in range(self.sweeps_p1):
            for pos in range(prompt_len, length):
                state_types[pos] = self._sample_type(
                    pos, state_types[pos], state_types
                )
            if verbose and (sweep + 1) % max(1, self.sweeps_p1 // 4) == 0:
                self._print_state(state_words, state_types, vocab, f"P1 sweep {sweep+1}")

        if verbose:
            self._print_state(state_words, state_types, vocab, "After Phase 1")

        # PHASE 2: Medium temperature — types + words
        for sweep in range(self.sweeps_p2):
            for pos in range(prompt_len, length):
                state_types[pos] = self._sample_type(
                    pos, state_types[pos], state_types
                )
                result = self._sample_word(
                    pos, state_words[pos], state_types[pos],
                    state_words, state_types, self.prob_table_p2, H_sum
                )
                if isinstance(result, tuple):
                    state_words[pos], H_sum = result
                else:
                    state_words[pos] = result
            if verbose and (sweep + 1) % max(1, self.sweeps_p2 // 4) == 0:
                self._print_state(state_words, state_types, vocab, f"P2 sweep {sweep+1}")

        if verbose:
            self._print_state(state_words, state_types, vocab, "After Phase 2")

        # PHASE 3: Low temperature — words only (types frozen)
        for sweep in range(self.sweeps_p3):
            for pos in range(prompt_len, length):
                result = self._sample_word(
                    pos, state_words[pos], state_types[pos],
                    state_words, state_types, self.prob_table_p3, H_sum
                )
                if isinstance(result, tuple):
                    state_words[pos], H_sum = result
                else:
                    state_words[pos] = result
            if verbose and (sweep + 1) % max(1, self.sweeps_p3 // 4) == 0:
                self._print_state(state_words, state_types, vocab, f"P3 sweep {sweep+1}")

        if verbose:
            self._print_state(state_words, state_types, vocab, "Final")

        return state_words, state_types

    def _print_state(self, words, types, vocab, label):
        if vocab is None:
            return
        word_str = vocab.decode(words)
        type_str = " ".join(IDX2POS.get(t, "?") for t in types[:20])
        print(f"  {label}: {word_str[:80]}")
        print(f"  {' ' * len(label)}: [{type_str}]")


class EnhancedTypedModel:
    """
    Enhanced Typed Ising-Potts Language Model v3.

    Integrates:
      1. SpaCy POS tagger for accurate type couplings
      2. Dependency tree couplings for long-range agreement
      3. Integer NMF for vocabulary scaling
      4. Larger corpus support
    """

    def __init__(
        self,
        # Vocabulary
        vocab_min_freq: int = 5,
        vocab_max_size: Optional[int] = 8000,
        # Sequence
        seq_len: int = 30,
        # Coupling
        window: int = 8,
        pmi_cap: int = 15,
        min_cooc: int = 2,
        # Weights
        pmi_weight: int = 3,
        hebbian_weight: int = 1,
        semantic_weight: int = 1,
        dep_weight: int = 2,
        # Grammar
        grammar_penalty: int = 50,
        # Annealing
        phase1_beta: int = 200,
        phase2_beta: int = 500,
        phase3_beta: int = 1000,
        total_sweeps: int = 150,
        # NMF
        use_nmf: bool = True,
        nmf_factors: int = 128,
        nmf_iterations: int = 50,
        # SpaCy
        use_spacy: bool = True,
        spacy_max_texts: Optional[int] = None,
        # Emission
        emission_weight: int = 10,
    ):
        self.vocab_min_freq = vocab_min_freq
        self.vocab_max_size = vocab_max_size
        self.seq_len = seq_len
        self.window = window
        self.pmi_cap = pmi_cap
        self.min_cooc = min_cooc
        self.pmi_weight = pmi_weight
        self.hebbian_weight = hebbian_weight
        self.semantic_weight = semantic_weight
        self.dep_weight = dep_weight
        self.grammar_penalty = grammar_penalty
        self.phase1_beta = phase1_beta
        self.phase2_beta = phase2_beta
        self.phase3_beta = phase3_beta
        self.total_sweeps = total_sweeps
        self.use_nmf = use_nmf
        self.nmf_factors = nmf_factors
        self.nmf_iterations = nmf_iterations
        self.use_spacy = use_spacy
        self.spacy_max_texts = spacy_max_texts
        self.emission_weight = emission_weight

        # Components
        self.vocab = None
        self.pmi = None
        self.types = None
        self.semantics = None
        self.spacy_tagger = None
        self.dep_couplings = None
        self.nmf_model = None
        self.sampler = None

    def train(self, n_samples=100000, verbose=True):
        """Train the enhanced model."""
        print("=" * 70)
        print("ENHANCED TYPED ISING-POTTS MODEL v3 — TRAINING")
        print("=" * 70)

        # Step 1: Load data (LARGER corpus by default)
        t0 = time.time()
        texts = load_fineweb_edu(n_samples=n_samples)
        print(f"[1/8] Data loading: {len(texts)} texts ({time.time()-t0:.1f}s)")

        # Step 2: Build vocabulary
        t0 = time.time()
        self.vocab = Vocabulary(
            min_freq=self.vocab_min_freq,
            max_size=self.vocab_max_size,
        )
        self.vocab.build(texts)
        V = len(self.vocab)
        print(f"[2/8] Vocabulary: {V} words ({time.time()-t0:.1f}s)")

        # Step 3: Tokenize
        t0 = time.time()
        sequences = tokenize_texts(texts, self.vocab)
        sequences = truncate_sequences(sequences, max_len=self.seq_len)
        print(f"[3/8] Tokenization: {len(sequences)} sequences ({time.time()-t0:.1f}s)")

        # Step 4: SpaCy POS tagging + dependency parsing (NEW)
        t0 = time.time()
        if self.use_spacy:
            self.spacy_tagger = SpaCyTagger(vocab_size=V, n_pos=N_POS)
            self.spacy_tagger.tag_corpus(
                texts, sequences,
                self.vocab.word2idx, self.vocab.idx2word,
                max_texts=self.spacy_max_texts,
            )
            print(f"[4/8] SpaCy POS + deps: "
                  f"{sum(len(v) for v in self.spacy_tagger.word_pos.values())} word-POS entries, "
                  f"{len(self.spacy_tagger.dep_edges)} dep edges ({time.time()-t0:.1f}s)")
        else:
            self.spacy_tagger = None
            print(f"[4/8] SpaCy: skipped (use_spacy=False)")

        # Step 5: PMI couplings
        t0 = time.time()
        self.pmi = PMICouplings(vocab_size=V, seq_len=self.seq_len, window=self.window)
        self.pmi.compute_from_sequences(
            sequences, min_count=self.min_cooc, pmi_cap=self.pmi_cap,
            use_hebbian=True, hebbian_weight=self.hebbian_weight,
        )
        pmi_nnz = int(np.count_nonzero(self.pmi.J_PMI))
        print(f"[5/8] PMI couplings: {pmi_nnz} non-zeros ({time.time()-t0:.1f}s)")

        # Step 6: Build type system (using SpaCy if available)
        t0 = time.time()
        self.types = POSTypeSystem(vocab_size=V, n_types=N_POS, window=self.window)

        if self.use_spacy and self.spacy_tagger is not None:
            # Use SpaCy-derived emission weights and type couplings
            self.types.I_emit = self.spacy_tagger.build_emission_weights()
            self.types.allowed_types = self.spacy_tagger.build_allowed_types()
            self.types.J_type = self.spacy_tagger.build_type_couplings(scaling=10)
        else:
            # Fallback to rule-based
            self.types.build_from_vocabulary(self.vocab.word2idx, self.vocab.idx2word)
            self.types.compute_type_couplings(sequences, self.vocab.idx2word, scaling=10)

        self.types.build_grammar_penalties(penalty_strength=self.grammar_penalty)
        self.types.precompute_type_distribution()

        n_typed = sum(1 for w in range(V) if len(self.types.allowed_types.get(w, set())) > 0)
        print(f"[6/8] POS type system: {n_typed}/{V} words typed ({time.time()-t0:.1f}s)")

        # Step 7: Dependency couplings (NEW)
        t0 = time.time()
        if self.use_spacy and self.spacy_tagger is not None:
            self.dep_couplings = DependencyCouplings(vocab_size=V, n_pos=N_POS)
            self.dep_couplings.build_from_spacy_tagger(
                self.spacy_tagger, self.vocab.idx2word,
                min_count=1, coupling_strength=3,
            )
            dep_stats = self.dep_couplings.get_dep_stats()
            print(f"[7/8] Dependency couplings: "
                  f"J_tree {dep_stats['J_tree_nnz']} nnz, "
                  f"{dep_stats['agreement_rules']} agreement rules ({time.time()-t0:.1f}s)")
        else:
            self.dep_couplings = None
            print(f"[7/8] Dependency couplings: skipped")

        # Step 7b: Semantic types
        self.semantics = SemanticTypeSystem(vocab_size=V, n_sem_types=N_SEM, compatibility_strength=3)
        self.semantics.build_from_vocabulary(self.vocab.word2idx, self.vocab.idx2word)
        self.semantics.compute_compatibility_matrix(sequences, min_cooc=2)
        self.semantics.compute_hebbian_coupling(sequences, hebbian_weight=1)

        # Step 8: Integer NMF (NEW)
        t0 = time.time()
        if self.use_nmf:
            # Combine all couplings for NMF factorization
            J_full = self.pmi_weight * self.pmi.J_PMI + self.hebbian_weight * self.pmi.J_Hebb
            J_full += self.semantic_weight * self.semantics.J_sem
            if self.dep_couplings is not None:
                J_full += self.dep_weight * self.dep_couplings.J_tree

            self.nmf_model = IntegerNMF(vocab_size=V, n_factors=self.nmf_factors)
            self.nmf_model.fit(J_full, n_iterations=self.nmf_iterations)

            mem = self.nmf_model.memory_savings()
            print(f"[8/8] Integer NMF: K={self.nmf_factors}, "
                  f"memory savings={mem['savings_pct']:.1f}% ({time.time()-t0:.1f}s)")
        else:
            self.nmf_model = None
            print(f"[8/8] Integer NMF: skipped")

        # Build sampler
        print("\nBuilding enhanced sampler...")
        t0 = time.time()
        self.sampler = EnhancedTypedSampler(
            pmi_couplings=self.pmi,
            type_system=self.types,
            semantic_system=self.semantics,
            dep_couplings=self.dep_couplings,
            nmf=self.nmf_model,
            phase1_beta=self.phase1_beta,
            phase2_beta=self.phase2_beta,
            phase3_beta=self.phase3_beta,
            total_sweeps=self.total_sweeps,
            pmi_weight=self.pmi_weight,
            hebbian_weight=self.hebbian_weight,
            semantic_weight=self.semantic_weight,
            dep_weight=self.dep_weight,
        )
        print(f"Sampler ready ({time.time()-t0:.1f}s)")

        # Summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE — v3 ENHANCED MODEL")
        print("=" * 70)
        self._print_summary()

        return self

    def _print_summary(self):
        print(f"\nModel Architecture (v3 Enhanced):")
        print(f"  Vocabulary size: {len(self.vocab)}")
        print(f"  POS types: {N_POS}")
        print(f"  Semantic types: {N_SEM}")
        print(f"  PMI coupling range: [{int(self.pmi.J_PMI.min())}, {int(self.pmi.J_PMI.max())}]")
        print(f"  PMI non-zeros: {int(np.count_nonzero(self.pmi.J_PMI))}")
        print(f"  Hebbian non-zeros: {int(np.count_nonzero(self.pmi.J_Hebb))}")
        print(f"  Semantic S non-zeros: {int(np.count_nonzero(self.semantics.S))}")
        print(f"  Grammar penalties: {len(self.types.grammar_penalties)}")
        if self.dep_couplings is not None:
            dep_stats = self.dep_couplings.get_dep_stats()
            print(f"  Dependency J_tree non-zeros: {dep_stats['J_tree_nnz']}")
            print(f"  Agreement rules: {dep_stats['agreement_rules']}")
            for label, count in sorted(dep_stats['dep_label_counts'].items()):
                if count > 0:
                    print(f"    {label}: {count}")
        if self.nmf_model is not None and self.nmf_model.fitted:
            mem = self.nmf_model.memory_savings()
            print(f"  NMF factors: {mem['n_factors_total']} (K_orig={self.nmf_factors})")
            print(f"  Memory savings: {mem['savings_pct']:.1f}%")
        print(f"  SpaCy POS: {'YES' if self.use_spacy else 'NO'}")
        print(f"  Zero FP in generation loop: YES")

    def generate(self, prompt=None, length=20, verbose=False):
        if self.sampler is None:
            raise RuntimeError("Model not trained. Call train() first.")
        words, types = self.sampler.generate(
            length=length, prompt=prompt, vocab=self.vocab, verbose=verbose
        )
        return self._decode_with_annotations(words, types)

    def generate_raw(self, prompt=None, length=20, verbose=False):
        if self.sampler is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.sampler.generate(
            length=length, prompt=prompt, vocab=self.vocab, verbose=verbose
        )

    def generate_with_trace(self, prompt=None, length=20):
        words, types = self.sampler.generate(
            length=length, prompt=prompt, vocab=self.vocab, verbose=True
        )

        energy = 0
        for i in range(length):
            energy += int(self.pmi.h[i % self.pmi.seq_len, words[i]])
        for i in range(length):
            for j_offset in range(1, self.pmi.window + 1):
                j = i + j_offset
                if j < length:
                    energy += int(self.pmi.J_PMI[words[i], words[j]])

        type_counts = {}
        for t in types:
            name = IDX2POS.get(t, "UNK")
            type_counts[name] = type_counts.get(name, 0) + 1

        sem_counts = {}
        for w in words:
            if w < len(self.semantics.word_to_sem):
                s_idx = int(self.semantics.word_to_sem[w])
                s_name = SEMANTIC_SUPERTYPES[s_idx] if s_idx < len(SEMANTIC_SUPERTYPES) else "UNK"
                sem_counts[s_name] = sem_counts.get(s_name, 0) + 1

        return {
            "text": self.vocab.decode(words),
            "types": [IDX2POS.get(t, "UNK") for t in types],
            "energy": energy,
            "type_counts": type_counts,
            "sem_counts": sem_counts,
            "words": words,
        }

    def generate_batch(self, n_samples=5, prompt=None, length=20):
        results = []
        for _ in range(n_samples):
            words, types = self.sampler.generate(
                length=length, prompt=prompt, vocab=self.vocab
            )
            results.append(self._decode_with_annotations(words, types))
        return results

    def _decode_with_annotations(self, words, types):
        parts = []
        for i, (w, t) in enumerate(zip(words, types)):
            word = self.vocab.idx2word.get(w, "<UNK>")
            pos = IDX2POS.get(t, "X")
            if word.startswith("<") and word.endswith(">"):
                continue
            parts.append(f"{word}/{pos}")
        return " ".join(parts)

    def evaluate_grammar(self, words, types):
        """Evaluate grammatical coherence."""
        metrics = {
            "det_noun": 0, "det_non_noun": 0, "aux_verb": 0,
            "adj_noun": 0, "prep_noun": 0, "double_det": 0,
            "double_prep": 0, "noun_verb": 0,
        }

        NOUN_LIKE = {POS2IDX[t] for t in ["NOUN", "PRON", "NUM"]}
        VERB_LIKE = {POS2IDX[t] for t in ["VERB", "AUX"]}

        for i, t in enumerate(types):
            if t == POS2IDX["DET"]:
                found_noun = any(
                    i+d < len(types) and types[i+d] in NOUN_LIKE
                    for d in range(1, 3)
                )
                if found_noun:
                    metrics["det_noun"] += 1
                else:
                    metrics["det_non_noun"] += 1
                if i+1 < len(types) and types[i+1] == POS2IDX["DET"]:
                    metrics["double_det"] += 1

            if t == POS2IDX["AUX"]:
                found_verb = any(
                    i+d < len(types) and types[i+d] in VERB_LIKE
                    for d in range(1, 3)
                )
                if found_verb:
                    metrics["aux_verb"] += 1

            if t == POS2IDX["ADJ"]:
                for d in range(1, 3):
                    if i+d < len(types) and types[i+d] == POS2IDX["NOUN"]:
                        metrics["adj_noun"] += 1
                        break

            if t == POS2IDX["PREP"]:
                found_noun = any(
                    i+d < len(types) and types[i+d] in NOUN_LIKE | {POS2IDX["DET"]}
                    for d in range(1, 4)
                )
                if found_noun:
                    metrics["prep_noun"] += 1
                if i+1 < len(types) and types[i+1] == POS2IDX["PREP"]:
                    metrics["double_prep"] += 1

            if t == POS2IDX["NOUN"]:
                for d in range(1, 3):
                    if i+d < len(types) and types[i+d] in VERB_LIKE:
                        metrics["noun_verb"] += 1
                        break

        return metrics

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.vocab.save(os.path.join(directory, "vocab.json"))
        self.pmi.save(os.path.join(directory, "pmi"))
        self.types.save(os.path.join(directory, "types"))
        self.semantics.save(os.path.join(directory, "semantics"))

        if self.spacy_tagger is not None:
            self.spacy_tagger.save(os.path.join(directory, "spacy"))
        if self.dep_couplings is not None:
            self.dep_couplings.save(os.path.join(directory, "deps"))
        if self.nmf_model is not None:
            self.nmf_model.save(os.path.join(directory, "nmf"))

        import json
        config = {
            "vocab_min_freq": self.vocab_min_freq,
            "vocab_max_size": self.vocab_max_size,
            "seq_len": self.seq_len,
            "window": self.window,
            "pmi_cap": self.pmi_cap,
            "min_cooc": self.min_cooc,
            "pmi_weight": self.pmi_weight,
            "hebbian_weight": self.hebbian_weight,
            "semantic_weight": self.semantic_weight,
            "dep_weight": self.dep_weight,
            "grammar_penalty": self.grammar_penalty,
            "phase1_beta": self.phase1_beta,
            "phase2_beta": self.phase2_beta,
            "phase3_beta": self.phase3_beta,
            "total_sweeps": self.total_sweeps,
            "use_nmf": self.use_nmf,
            "nmf_factors": self.nmf_factors,
            "use_spacy": self.use_spacy,
        }
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, directory):
        import json
        with open(os.path.join(directory, "config.json")) as f:
            config = json.load(f)

        model = cls(**config)
        model.vocab = Vocabulary.load(os.path.join(directory, "vocab.json"))
        model.pmi = PMICouplings.load(os.path.join(directory, "pmi"))
        model.types = POSTypeSystem.load(os.path.join(directory, "types"))
        model.semantics = SemanticTypeSystem.load(os.path.join(directory, "semantics"))

        try:
            model.spacy_tagger = SpaCyTagger.load(os.path.join(directory, "spacy"))
        except FileNotFoundError:
            model.spacy_tagger = None

        try:
            model.dep_couplings = DependencyCouplings.load(os.path.join(directory, "deps"))
        except FileNotFoundError:
            model.dep_couplings = None

        try:
            model.nmf_model = IntegerNMF.load(os.path.join(directory, "nmf"))
        except FileNotFoundError:
            model.nmf_model = None

        model.sampler = EnhancedTypedSampler(
            pmi_couplings=model.pmi,
            type_system=model.types,
            semantic_system=model.semantics,
            dep_couplings=model.dep_couplings,
            nmf=model.nmf_model,
            phase1_beta=model.phase1_beta,
            phase2_beta=model.phase2_beta,
            phase3_beta=model.phase3_beta,
            total_sweeps=model.total_sweeps,
            pmi_weight=model.pmi_weight,
            hebbian_weight=model.hebbian_weight,
            semantic_weight=model.semantic_weight,
            dep_weight=model.dep_weight,
        )

        return model
