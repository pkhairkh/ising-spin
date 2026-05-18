"""
Staged Annealing Sampler for the Typed Ising-Potts Language Model.

Implements the coupled Ising-Potts generation with:
  - Type layer (POS tags): Potts model with exact enumeration per position
  - Value layer (words): Ising-like model with Gibbs sampling
  - Grammar penalties: integer quadratic constraints
  - Staged annealing: high-T → types, mid-T → word types, low-T → specific words

Generation loop is ZERO FLOATING-POINT:
  - Energy computation: integer addition
  - Type updates: exact enumeration over T≈13 states → integer comparison
  - Word updates: propose from emission-compatible set, compute ΔE, threshold lookup
  - Acceptance: precomputed integer threshold table + integer comparison
  - Annealing: switch between precomputed threshold tables at fixed sweep counts

References:
  - Coupled Ising-Potts: Haydarov, Omirov & Rozikov (arXiv:2502.12014)
  - Spin Glass Syntax: Marcolli et al. (arXiv:1508.00504)
  - Grammar of the Ising Model: Reinhart & De las Coves (arXiv:2208.08301)
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pmi_couplings import PMICouplings
from .type_system import POSTypeSystem, N_POS, IDX2POS, POS2IDX
from .semantic_types import SemanticTypeSystem, N_SEM


class StagedAnnealingSampler:
    """
    Staged annealing Gibbs sampler for the typed Ising-Potts language model.

    State: (types, words) where types[i] ∈ {0,...,T-1}, words[i] ∈ {0,...,V-1}

    Energy:
        E(types, words) = E_type(types) + E_emit(words|types)
                        + E_lexical(words) + E_semantic(words)
                        + E_grammar(types, words)

    Generation phases:
        Phase 1 (high T): Update types only → resolve grammatical structure
        Phase 2 (mid T):  Update types + words → narrow word selection
        Phase 3 (low T):  Update words only → finalize lexical choices
    """

    def __init__(
        self,
        pmi_couplings: PMICouplings,
        type_system: POSTypeSystem,
        semantic_system: Optional[SemanticTypeSystem] = None,
        # Temperature schedule (as integer beta * 1000)
        phase1_beta: int = 200,    # high T, low beta → types
        phase2_beta: int = 500,    # mid T, mid beta → types + words
        phase3_beta: int = 1000,   # low T, high beta → words
        # Sweep allocation
        total_sweeps: int = 150,
        phase1_frac: float = 0.25,
        phase2_frac: float = 0.35,
        phase3_frac: float = 0.40,
        # Proposal parameters
        proposal_top_k: int = 30,
        pmi_weight: int = 3,
        hebbian_weight: int = 1,
        semantic_weight: int = 1,
    ):
        self.pmi = pmi_couplings
        self.types = type_system
        self.sem = semantic_system
        self.vocab_size = pmi_couplings.vocab_size
        self.n_types = type_system.n_types

        # Store weights
        self.pmi_weight = pmi_weight
        self.hebbian_weight = hebbian_weight
        self.semantic_weight = semantic_weight

        # Sweep counts per phase
        self.sweeps_p1 = int(total_sweeps * phase1_frac)
        self.sweeps_p2 = int(total_sweeps * phase2_frac)
        self.sweeps_p3 = total_sweeps - self.sweeps_p1 - self.sweeps_p2

        # Precompute probability tables for each phase
        self.prob_table_p1 = self._build_prob_table(phase1_beta)
        self.prob_table_p2 = self._build_prob_table(phase2_beta)
        self.prob_table_p3 = self._build_prob_table(phase3_beta)

        # Precompute proposal sets for each word
        self.proposal_cache = self._build_proposal_cache(proposal_top_k)

        # Precompute combined coupling matrix
        self.J_combined = self._build_combined_coupling()

        # Precompute type proposal distributions per position
        self.type_cumsum_by_pos = self._build_type_distributions()

        # Precompute word emission distributions per type
        self.emit_cumsum_by_type = self._build_emission_distributions()

        # Precompute field-weighted proposal distribution for each position
        self.field_weights = self._build_field_weights()

    def _build_prob_table(
        self, beta_int: int, max_delta_e: int = 5000, rand_max: int = 2**31 - 1
    ) -> List[int]:
        """Build integer threshold table for Metropolis acceptance."""
        thresholds = [0] * (2 * max_delta_e + 1)

        for delta_e in range(-max_delta_e, max_delta_e + 1):
            idx = delta_e + max_delta_e
            if delta_e <= 0:
                thresholds[idx] = rand_max  # always accept
            else:
                import math
                beta = beta_int / 1000.0
                prob = math.exp(-delta_e * beta)
                threshold = int(rand_max * prob)
                thresholds[idx] = max(0, min(rand_max, threshold))

        return thresholds

    def _accept(self, delta_e: int, rand_val: int, table: List[int]) -> bool:
        """Integer-only acceptance check: one lookup + one comparison."""
        max_delta_e = (len(table) - 1) // 2
        if delta_e <= -max_delta_e:
            return True
        if delta_e >= max_delta_e:
            return False
        return rand_val < table[delta_e + max_delta_e]

    def _build_proposal_cache(self, top_k: int) -> Dict[int, List[int]]:
        """Precompute top-k neighbors for each word from combined coupling."""
        cache = {}
        for w in range(self.vocab_size):
            # Get PMI neighbors
            pmi_neighbors = self.pmi.get_neighbor_words(w, top_k=top_k)
            # Also get emission-compatible words
            if w in self.types.allowed_types:
                # Get words that share at least one type with w
                type_set = self.types.allowed_types[w]
                compat_words = set()
                for t in type_set:
                    compat_words.update(self.types.get_allowed_words_for_type(t))
                # Combine PMI + type-compatible neighbors
                combined = list(set(pmi_neighbors) | compat_words)
            else:
                combined = pmi_neighbors

            if w not in combined:
                combined.append(w)
            cache[w] = combined[:top_k * 3]  # allow more proposals

        return cache

    def _build_combined_coupling(self) -> np.ndarray:
        """
        Build combined coupling: J = alpha*J_PMI + beta*J_Hebb + gamma*J_sem.

        If semantic system is available, use semantically-gated coupling.
        Otherwise, use PMI + Hebbian.
        """
        J = self.pmi_weight * self.pmi.J_PMI + self.hebbian_weight * self.pmi.J_Hebb

        if self.sem is not None:
            J = J + self.semantic_weight * self.sem.J_sem

        return J

    def _build_type_distributions(self) -> Dict[int, np.ndarray]:
        """Precompute type proposal distributions for each position."""
        distributions = {}
        for pos in range(min(self.pmi.seq_len, 50)):
            # Type distribution from J_type marginal + positional bias
            type_weights = self.types.J_type.sum(axis=1).copy()
            # Add emission-weighted type distribution at this position
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

    def _build_emission_distributions(self) -> Dict[int, np.ndarray]:
        """
        Precompute word emission distributions for each type.

        emit_cumsum_by_type[t] = cumulative sum of I_emit[:, t].
        Used to propose words conditioned on type.
        """
        distributions = {}
        for t in range(self.n_types):
            col = self.types.I_emit[:, t].copy()
            # Boost by PMI-based word frequency (from h field)
            h0 = self.pmi.h[0]  # position-0 field
            col = col * np.maximum(h0, 1)  # integer multiply

            if col.sum() > 0:
                distributions[t] = np.cumsum(col)
            else:
                # No words for this type; use uniform over vocab
                col[:] = 1
                distributions[t] = np.cumsum(col)

        return distributions

    def _build_field_weights(self) -> Dict[int, np.ndarray]:
        """Precompute field-weighted distributions for each position."""
        weights = {}
        for i in range(min(self.pmi.seq_len, 50)):
            h = self.pmi.h[i].copy()
            if h.sum() > 0:
                weights[i] = np.cumsum(h)
            else:
                h[:] = 1
                weights[i] = np.cumsum(h)
        return weights

    def _sample_from_cumsum(self, cumsum: np.ndarray) -> int:
        """Sample an index from an integer cumulative sum. Pure integer ops."""
        total = int(cumsum[-1])
        if total <= 0:
            return 0
        rv = random.randint(1, total)
        idx = int(np.searchsorted(cumsum, rv))
        return min(idx, len(cumsum) - 1)

    def _sample_type(self, pos: int, current_type: int, state_types: List[int]) -> int:
        """
        Sample a type for position pos using exact enumeration.

        For each possible type t, compute:
            E_type(t) = -sum_{j in window} J_type[t, types[j]]
                      + grammar_penalty(types, pos, t)

        Then sample proportionally to exp(-beta * E) using integer weights.
        Since T≈13, exact enumeration is fast.
        """
        # Compute energy for each type
        type_energies = np.zeros(self.n_types, dtype=np.int64)

        for t in range(self.n_types):
            # Type-type coupling energy
            energy = 0
            for j_offset in range(1, self.types.window + 1):
                j = pos + j_offset
                if j < len(state_types):
                    energy += int(self.types.J_type[t, state_types[j]])
                j = pos - j_offset
                if j >= 0:
                    energy += int(self.types.J_type[state_types[j], t])

            # Grammar penalty
            energy -= self.types.compute_grammar_penalty(state_types, pos, t)

            type_energies[t] = energy

        # Convert energies to integer weights (inverse)
        # Higher energy = less likely. Use: weight = max(0, max_E - E + 1)
        max_e = int(type_energies.max())
        type_weights = np.maximum(max_e - type_energies + 1, 0).astype(np.int64)

        # Check if current type is allowed for the word at this position
        # (If we know the word, restrict to allowed types)
        # For now, allow all types with non-zero weight

        # Sample from weights
        total = int(type_weights.sum())
        if total <= 0:
            # Uniform over all types
            type_weights[:] = 1
            total = self.n_types

        cumsum = np.cumsum(type_weights)
        rv = random.randint(1, int(cumsum[-1]))
        idx = int(np.searchsorted(cumsum, rv))
        return min(idx, self.n_types - 1)

    def _sample_word(
        self,
        pos: int,
        current_word: int,
        current_type: int,
        state_words: List[int],
        state_types: List[int],
        prob_table: List[int],
    ) -> int:
        """
        Propose and accept/reject a word for position pos.

        Propose from: emission-compatible words + PMI neighbors + field-weighted.
        Accept/reject using integer threshold lookup.
        """
        # Propose candidate word
        r = random.randint(0, 99)

        if r < 30:
            # 30%: sample from emission distribution for current type
            if current_type in self.emit_cumsum_by_type:
                proposed = self._sample_from_cumsum(
                    self.emit_cumsum_by_type[current_type]
                )
            else:
                proposed = current_word

        elif r < 60:
            # 30%: sample from proposal cache (PMI neighbors)
            neighbors = self.proposal_cache.get(current_word, [current_word])
            proposed = random.choice(neighbors)

        elif r < 85:
            # 25%: field-weighted sample at this position
            pos_key = pos % min(self.pmi.seq_len, 50)
            if pos_key in self.field_weights:
                proposed = self._sample_from_cumsum(self.field_weights[pos_key])
            else:
                proposed = random.randint(0, self.vocab_size - 1)

        else:
            # 15%: uniform random
            proposed = random.randint(0, self.vocab_size - 1)

        if proposed == current_word:
            return current_word

        # Compute energy difference (INTEGER)
        current_energy = self._compute_word_energy(
            pos, current_word, current_type, state_words, state_types
        )
        proposed_type = int(self.types.get_type_for_word(proposed))
        proposed_energy = self._compute_word_energy(
            pos, proposed, proposed_type, state_words, state_types
        )

        delta_e = proposed_energy - current_energy  # integer

        # Accept/reject using integer threshold table
        rand_val = random.randint(0, 2**31 - 2)
        if self._accept(delta_e, rand_val, prob_table):
            return proposed
        return current_word

    def _compute_word_energy(
        self,
        pos: int,
        word: int,
        word_type: int,
        state_words: List[int],
        state_types: List[int],
    ) -> int:
        """
        Compute total energy contribution from a word at a position.

        E = E_field + E_PMI_coupling + E_type_coupling + E_emission + E_grammar + E_semantic

        All integer addition.
        """
        energy = 0

        # 1. Field energy
        energy += int(self.pmi.h[pos % self.pmi.seq_len, word])

        # 2. PMI + Hebbian coupling (combined)
        for j_offset in range(1, self.pmi.window + 1):
            j = pos + j_offset
            if j < len(state_words):
                energy += int(self.J_combined[word, state_words[j]])
            j = pos - j_offset
            if j >= 0:
                energy += int(self.J_combined[state_words[j], word])

        # 3. Type coupling (Potts-like gating)
        # Word-word coupling is modulated by type agreement
        for j_offset in range(1, self.types.window + 1):
            j = pos + j_offset
            if j < len(state_types):
                if word_type == state_types[j]:
                    # Same type: strong coupling (ferromagnetic)
                    energy += int(self.types.J_type[word_type, word_type]) // self.n_types
            j = pos - j_offset
            if j >= 0:
                if word_type == state_types[j]:
                    energy += int(self.types.J_type[word_type, word_type]) // self.n_types

        # 4. Emission energy (word-type compatibility)
        if word < self.types.I_emit.shape[0] and word_type < self.types.I_emit.shape[1]:
            emit_val = int(self.types.I_emit[word, word_type])
            if emit_val > 0:
                energy += emit_val * 10  # bonus for compatible type
            else:
                energy -= 50  # penalty for incompatible type

        # 5. Grammar penalty
        types_copy = list(state_types)
        types_copy[pos] = word_type
        energy -= self.types.compute_grammar_penalty(types_copy, pos, word_type)

        return energy

    def generate(
        self,
        length: int = 20,
        prompt: Optional[str] = None,
        vocab: Optional[object] = None,
        verbose: bool = False,
    ) -> Tuple[List[int], List[int]]:
        """
        Generate text using staged annealing.

        Returns (word_state, type_state) — both lists of integers.

        Phase 1: Update types at high temperature
        Phase 2: Update types + words at medium temperature
        Phase 3: Update words only at low temperature
        """
        # Encode prompt
        prompt_words = []
        prompt_types = []
        if prompt and vocab:
            prompt_tokens = vocab._tokenize(prompt)
            for tok in prompt_tokens:
                w_idx = vocab.word2idx.get(tok, vocab.word2idx.get("<UNK>", 0))
                prompt_words.append(w_idx)
                prompt_types.append(self.types.get_type_for_word(w_idx))

        prompt_len = len(prompt_words)

        # Initialize state
        state_words = list(prompt_words) if prompt_words else []
        state_types = list(prompt_types) if prompt_types else []

        # Fill remaining positions
        for i in range(prompt_len, length):
            # Initialize word from field-weighted distribution
            pos_key = i % min(self.pmi.seq_len, 50)
            if pos_key in self.field_weights:
                w = self._sample_from_cumsum(self.field_weights[pos_key])
            else:
                w = random.randint(0, self.vocab_size - 1)
            state_words.append(w)
            state_types.append(self.types.get_type_for_word(w))

        if verbose:
            self._print_state(state_words, state_types, vocab, "Init")

        # === PHASE 1: High temperature — update types only ===
        for sweep in range(self.sweeps_p1):
            for pos in range(prompt_len, length):
                state_types[pos] = self._sample_type(
                    pos, state_types[pos], state_types
                )

            if verbose and (sweep + 1) % max(1, self.sweeps_p1 // 4) == 0:
                self._print_state(state_words, state_types, vocab,
                                  f"P1 sweep {sweep+1}")

        if verbose:
            self._print_state(state_words, state_types, vocab, "After Phase 1")

        # === PHASE 2: Medium temperature — update types + words ===
        for sweep in range(self.sweeps_p2):
            for pos in range(prompt_len, length):
                # Update type
                state_types[pos] = self._sample_type(
                    pos, state_types[pos], state_types
                )
                # Update word
                state_words[pos] = self._sample_word(
                    pos, state_words[pos], state_types[pos],
                    state_words, state_types, self.prob_table_p2
                )

            if verbose and (sweep + 1) % max(1, self.sweeps_p2 // 4) == 0:
                self._print_state(state_words, state_types, vocab,
                                  f"P2 sweep {sweep+1}")

        if verbose:
            self._print_state(state_words, state_types, vocab, "After Phase 2")

        # === PHASE 3: Low temperature — update words only (types frozen) ===
        for sweep in range(self.sweeps_p3):
            for pos in range(prompt_len, length):
                state_words[pos] = self._sample_word(
                    pos, state_words[pos], state_types[pos],
                    state_words, state_types, self.prob_table_p3
                )

            if verbose and (sweep + 1) % max(1, self.sweeps_p3 // 4) == 0:
                self._print_state(state_words, state_types, vocab,
                                  f"P3 sweep {sweep+1}")

        if verbose:
            self._print_state(state_words, state_types, vocab, "Final")

        return state_words, state_types

    def generate_multiple(
        self,
        n_samples: int = 5,
        length: int = 20,
        prompt: Optional[str] = None,
        vocab: Optional[object] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        """Generate multiple independent samples."""
        return [
            self.generate(length=length, prompt=prompt, vocab=vocab)
            for _ in range(n_samples)
        ]

    def _print_state(self, words, types, vocab, label):
        """Print current state for debugging."""
        if vocab is None:
            print(f"  {label}: words={words[:20]}, types={types[:20]}")
            return

        word_str = vocab.decode(words)
        type_str = " ".join(IDX2POS.get(t, "?") for t in types[:20])
        print(f"  {label}: {word_str[:80]}")
        print(f"  {' ' * len(label)}: [{type_str}]")
