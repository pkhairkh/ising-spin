"""
Enhanced Typed Ising-Potts Language Model v4 — Research-Backed Mitigations.

Integrates all P0+P1 fixes based on academic research:
  P0: Locally-balanced proposals (Zanella 2017, arXiv:1711.07424)
  P0: CFG-derived hard type constraints (Cranch-Marcolli-Spivak 2024, arXiv:2405.12485)
  P1: CALDERA sparse+lowrank NMF (Saha et al. 2024, arXiv:2405.18886)
  P1: Strengthened emission coupling (10x→100x weight)
  P1: Implicational couplings (Marcolli 2015, arXiv:1508.00504)

Energy function:
  E(types, words) = E_type(types)           [hard transition constraints]
                  + E_emit(words|types)      [100x strengthened]
                  + E_lexical(words)         [CALDERA: Q + L1@L2]
                  + E_semantic(words)
                  + E_grammar(types, words)  [implicational + transition]
                  + E_dep(types, words)      [long-range dependency]

All generation-path computation remains integer arithmetic only.
"""

import os
import time
import random
import math
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from .vocabulary import Vocabulary
from .pmi_couplings import PMICouplings
from .type_system import POSTypeSystem, N_POS, IDX2POS, POS2IDX
from .semantic_types import SemanticTypeSystem, N_SEM, SEMANTIC_SUPERTYPES
from .spacy_tagger import SpaCyTagger
from .dep_couplings import DependencyCouplings
from .caldera_nmf import CalderaNMF
from .data_loader import load_fineweb_edu, tokenize_texts, truncate_sequences


# =============================================================================
# Hard POS Transition Constraints (Cranch-Marcolli-Spivak 2024)
# =============================================================================

def build_allowed_pos_transitions(
    sequences: List[List[int]],
    spacy_tagger=None,
    vocab=None,
    min_count: int = 3,
) -> Set[Tuple[int, int]]:
    """
    Build the set of allowed POS bigram transitions from the corpus.
    
    This implements the CFG-derived hard constraint approach:
    only transitions that appear frequently in real text are allowed.
    Transitions not in this set get an infinite penalty in the type sampler.
    
    Returns: set of (pos_from, pos_to) tuples.
    """
    transitions = {}
    
    if spacy_tagger is not None and vocab is not None:
        # Use SpaCy-derived POS tags (more accurate)
        for word_idx in range(len(vocab)):
            pos_tags = spacy_tagger.word_pos.get(word_idx, [])
            if pos_tags:
                # Use the most common POS tag for this word
                best_pos = max(pos_tags, key=pos_tags.get)
                # We need sequence-level transitions, not word-level
                # Fall through to sequence-based approach
                pass
    
    # Sequence-based approach: use assigned POS tags
    for seq in sequences:
        prev_type = None
        for w in seq:
            # Get type from spacy or rules (simplified: use the type system)
            # For now, we build this during training with the actual type assignments
            pass
    
    # We'll build this properly during training
    return set()


def build_allowed_transitions_from_tagger(
    spacy_tagger,
    idx2word,
    sequences,
    type_system,
    min_count=3,
) -> Set[Tuple[int, int]]:
    """
    Build allowed POS transitions from SpaCy-tagged corpus.
    """
    transitions = {}
    
    for seq in sequences:
        seq_types = []
        for w in seq:
            if w in type_system.allowed_types and type_system.allowed_types[w]:
                # Get the most common POS tag for this word from SpaCy
                if spacy_tagger is not None and w in spacy_tagger.word_pos:
                    pos_counts = spacy_tagger.word_pos[w]
                    best_pos = max(pos_counts, key=pos_counts.get)
                    # Map SpaCy POS to our coarse POS
                    if best_pos in type_system.allowed_types[w]:
                        seq_types.append(best_pos)
                    else:
                        # Fallback to most likely type
                        seq_types.append(max(type_system.allowed_types[w],
                                           key=lambda t: int(type_system.I_emit[w, t])))
                else:
                    seq_types.append(max(type_system.allowed_types[w],
                                       key=lambda t: int(type_system.I_emit[w, t])))
            else:
                seq_types.append(POS2IDX["X"])
        
        # Count transitions
        for i in range(len(seq_types) - 1):
            t_pair = (seq_types[i], seq_types[i + 1])
            transitions[t_pair] = transitions.get(t_pair, 0) + 1
    
    # Also add transitions from rule-based POS bigrams
    for seq in sequences:
        prev_type = None
        for w in seq:
            if w in type_system.allowed_types and type_system.allowed_types[w]:
                t = max(type_system.allowed_types[w],
                       key=lambda t: int(type_system.I_emit[w, t]))
            else:
                t = POS2IDX["X"]
            if prev_type is not None:
                t_pair = (prev_type, t)
                transitions[t_pair] = transitions.get(t_pair, 0) + 1
            prev_type = t
    
    # Filter by minimum count
    allowed = {pair for pair, count in transitions.items() if count >= min_count}
    
    # Always allow self-transitions for content words and punctuation
    for t in range(N_POS):
        allowed.add((t, t))
    
    # Always allow transitions to/from X (unknown)
    for t in range(N_POS):
        allowed.add((t, POS2IDX["X"]))
        allowed.add((POS2IDX["X"], t))
    
    # Always allow punctuation transitions
    for t in range(N_POS):
        allowed.add((t, POS2IDX["PUNCT"]))
        allowed.add((POS2IDX["PUNCT"], t))
    
    return allowed


# =============================================================================
# Implicational Couplings (Marcolli 2015)
# =============================================================================

# If type at position i is KEY, then type at position i+offset should be in VALUES
IMPLICATION_RULES = {
    POS2IDX["DET"]: {
        +1: {POS2IDX["NOUN"], POS2IDX["ADJ"], POS2IDX["NUM"], POS2IDX["PRON"]},
    },
    POS2IDX["AUX"]: {
        +1: {POS2IDX["VERB"], POS2IDX["AUX"], POS2IDX["PART"]},
    },
    POS2IDX["PREP"]: {
        +1: {POS2IDX["DET"], POS2IDX["NOUN"], POS2IDX["ADJ"], POS2IDX["PRON"]},
    },
    POS2IDX["ADJ"]: {
        +1: {POS2IDX["NOUN"], POS2IDX["ADJ"]},
    },
    POS2IDX["PART"]: {
        +1: {POS2IDX["VERB"], POS2IDX["NOUN"], POS2IDX["ADJ"]},
    },
    POS2IDX["CONJ"]: {
        +1: {POS2IDX["DET"], POS2IDX["NOUN"], POS2IDX["VERB"], POS2IDX["ADJ"], POS2IDX["PREP"]},
    },
}

IMPLICATION_PENALTY = 300  # Strong but not infinite


def compute_implicational_penalty(
    state_types: List[int], pos: int, proposed_type: int
) -> int:
    """
    Compute implicational penalty: if proposed_type requires a certain
    type at a neighbor position, and that neighbor doesn't have it,
    add a strong penalty.
    
    Also penalize if the neighbor requires something from us that we don't satisfy.
    """
    penalty = 0
    
    # Check: does proposed_type IMPLY something about neighbors?
    if proposed_type in IMPLICATION_RULES:
        for offset, required_types in IMPLICATION_RULES[proposed_type].items():
            j = pos + offset
            if 0 <= j < len(state_types):
                if state_types[j] not in required_types:
                    penalty += IMPLICATION_PENALTY
            # Also check backward
            j = pos - offset
            if 0 <= j < len(state_types):
                if state_types[j] not in required_types:
                    penalty += IMPLICATION_PENALTY // 2  # weaker backward
    
    # Check: do neighbors IMPLY something about proposed_type?
    for offset in [-2, -1, 1, 2]:
        j = pos + offset
        if 0 <= j < len(state_types):
            neighbor_type = state_types[j]
            if neighbor_type in IMPLICATION_RULES:
                abs_offset = abs(offset)
                for req_offset, required_types in IMPLICATION_RULES[neighbor_type].items():
                    if abs_offset == req_offset:
                        # proposed_type should satisfy the implication
                        # i.e., if neighbor is at pos-1 and needs something at pos+1
                        # Check: is pos the position that the neighbor's rule points to?
                        if offset > 0 and req_offset == offset:
                            if proposed_type not in required_types:
                                penalty += IMPLICATION_PENALTY
                        elif offset < 0 and req_offset == -offset:
                            if proposed_type not in required_types:
                                penalty += IMPLICATION_PENALTY // 2
    
    return penalty


# =============================================================================
# V4 Enhanced Sampler
# =============================================================================

class EnhancedV4Sampler:
    """
    V4 Enhanced sampler with all research-backed mitigations.
    
    P0: Locally-balanced proposals — weight candidates by exp(-ΔE/2T)
    P0: Hard POS transition constraints — infinite penalty for disallowed transitions
    P1: CALDERA NMF — sparse backbone + low-rank residual
    P1: Strengthened emission — 100x bonus, 500x penalty
    P1: Implicational couplings — DET→NOUN, AUX→VERB, etc.
    """

    def __init__(
        self,
        pmi_couplings: PMICouplings,
        type_system: POSTypeSystem,
        semantic_system: Optional[SemanticTypeSystem],
        dep_couplings: Optional[DependencyCouplings],
        nmf: Optional[CalderaNMF],
        # Hard transition constraints
        allowed_transitions: Optional[Set[Tuple[int, int]]] = None,
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
        # Emission weights (STRENGTHENED)
        emission_bonus: int = 100,
        emission_penalty: int = 500,
        # Proposal parameters
        proposal_top_k: int = 50,
    ):
        self.pmi = pmi_couplings
        self.types = type_system
        self.sem = semantic_system
        self.deps = dep_couplings
        self.nmf = nmf
        self.allowed_transitions = allowed_transitions

        self.vocab_size = pmi_couplings.vocab_size
        self.n_types = type_system.n_types

        self.pmi_weight = pmi_weight
        self.hebbian_weight = hebbian_weight
        self.semantic_weight = semantic_weight
        self.dep_weight = dep_weight
        self.emission_bonus = emission_bonus
        self.emission_penalty = emission_penalty

        # Sweep allocation
        self.sweeps_p1 = int(total_sweeps * phase1_frac)
        self.sweeps_p2 = int(total_sweeps * phase2_frac)
        self.sweeps_p3 = total_sweeps - self.sweeps_p1 - self.sweeps_p2

        # Precompute probability tables for each phase
        self.prob_table_p1 = self._build_prob_table(phase1_beta)
        self.prob_table_p2 = self._build_prob_table(phase2_beta)
        self.prob_table_p3 = self._build_prob_table(phase3_beta)

        # Build combined coupling matrix (for non-CALDERA path fallback)
        self.J_combined = self._build_combined_coupling()

        # Precompute proposal sets — P0: PMI-NEIGHBOR DOMINATED
        self.proposal_cache = self._build_proposal_cache(proposal_top_k)

        # Precompute distributions
        self.type_cumsum_by_pos = self._build_type_distributions()
        self.emit_cumsum_by_type = self._build_emission_distributions()
        self.field_weights = self._build_field_weights()

        # Precompute J_tree neighbor cache
        self.tree_neighbor_cache = self._build_tree_neighbor_cache()

    def _build_prob_table(self, beta_int, max_delta_e=5000, rand_max=2**31-1):
        """Build integer threshold table for Metropolis acceptance."""
        thresholds = [0] * (2 * max_delta_e + 1)
        for delta_e in range(-max_delta_e, max_delta_e + 1):
            idx = delta_e + max_delta_e
            if delta_e <= 0:
                thresholds[idx] = rand_max
            else:
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
        """Build combined coupling for non-CALDERA path."""
        J = self.pmi_weight * self.pmi.J_PMI + self.hebbian_weight * self.pmi.J_Hebb
        if self.sem is not None:
            J = J + self.semantic_weight * self.sem.J_sem
        return J

    def _build_proposal_cache(self, top_k):
        """P0: Build proposal cache dominated by PMI neighbors."""
        cache = {}
        for w in range(self.vocab_size):
            candidates = set()

            # PRIMARY: PMI neighbors (now dominant)
            pmi_neighbors = self.pmi.get_neighbor_words(w, top_k=top_k * 2)
            candidates.update(pmi_neighbors)

            # SECONDARY: Emission-compatible words for each type
            if w in self.types.allowed_types:
                type_set = self.types.allowed_types[w]
                for t in type_set:
                    compat_words = self.types.get_allowed_words_for_type(t)
                    candidates.update(compat_words[:top_k])

            # TERTIARY: Dependency tree neighbors
            if self.deps is not None:
                tree_neighbors = self.deps.get_tree_neighbors(w, top_k=top_k)
                for w_idx, _ in tree_neighbors:
                    candidates.add(w_idx)

            # QUATERNARY: CALDERA NMF neighbors
            if self.nmf is not None and self.nmf.fitted:
                # Get top neighbors from sparse backbone Q
                q_neighbors = self.nmf.Q_rows.get(w, [])
                for col, val in q_neighbors:
                    if val != 0:
                        candidates.add(col)

            if w not in candidates:
                candidates.add(w)

            cache[w] = list(candidates)[:top_k * 4]

        return cache

    def _build_tree_neighbor_cache(self):
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
        total = int(cumsum[-1])
        if total <= 0:
            return 0
        rv = random.randint(1, total)
        idx = int(np.searchsorted(cumsum, rv))
        return min(idx, len(cumsum) - 1)

    def _sample_type(self, pos, current_type, state_types):
        """
        P0: Sample type with HARD TRANSITION CONSTRAINTS + IMPLICATIONAL COUPLINGS.
        """
        type_energies = np.zeros(self.n_types, dtype=np.int64)

        for t in range(self.n_types):
            energy = 0

            # P0: Hard transition constraint (infinite penalty for disallowed)
            if self.allowed_transitions is not None:
                # Check left neighbor
                if pos > 0:
                    if (state_types[pos - 1], t) not in self.allowed_transitions:
                        energy -= 10000  # Near-infinite penalty
                # Check right neighbor
                if pos < len(state_types) - 1:
                    if (t, state_types[pos + 1]) not in self.allowed_transitions:
                        energy -= 10000

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

            # P1: Implicational coupling penalty
            energy -= compute_implicational_penalty(state_types, pos, t)

            # Dependency agreement penalty
            if self.deps is not None:
                energy -= self.deps.compute_agreement_penalty(state_types, pos, t)

            type_energies[t] = energy

        # Sample proportional to energy (Boltzmann-like)
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
        
        P1: Uses CALDERA Q + L1@L2 for lexical coupling.
        P1: Strengthened emission coupling.
        """
        energy = 0

        # 1. Field energy
        energy += int(self.pmi.h[pos % self.pmi.seq_len, word])

        # 2. Lexical coupling: CALDERA path or full matrix path
        if self.nmf is not None and self.nmf.fitted and H_sum is not None:
            # CALDERA: Q (sparse backbone) + L1@L2 (low-rank residual)
            # Sparse Q contribution: O(nnz_per_row)
            neighbor_words = [state_words[j] for j in range(max(0, pos - self.pmi.window), 
                                                             min(len(state_words), pos + self.pmi.window + 1))
                             if j != pos]
            q_energy = self.nmf.get_sparse_energy(word, neighbor_words)
            l_energy = self.nmf.get_factorized_energy(word, H_sum)
            energy += self.pmi_weight * q_energy + l_energy
        elif self.J_combined is not None:
            # Full matrix path (fallback)
            for j_offset in range(1, self.pmi.window + 1):
                j = pos + j_offset
                if j < len(state_words):
                    energy += int(self.J_combined[word, state_words[j]])
                j = pos - j_offset
                if j >= 0:
                    energy += int(self.J_combined[state_words[j], word])

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

        # 4. P1: STRENGTHENED Emission energy
        if word < self.types.I_emit.shape[0] and word_type < self.types.I_emit.shape[1]:
            emit_val = int(self.types.I_emit[word, word_type])
            if emit_val > 0:
                energy += emit_val * self.emission_bonus  # Was 10, now 100
            else:
                energy -= self.emission_penalty  # Was 50, now 500

        # 5. Grammar penalty
        types_copy = list(state_types)
        types_copy[pos] = word_type
        energy -= self.types.compute_grammar_penalty(types_copy, pos, word_type)

        # 6. P1: Implicational coupling penalty
        energy -= compute_implicational_penalty(types_copy, pos, word_type)

        # 7. Dependency coupling
        if self.deps is not None:
            tree_neighbors = self.tree_neighbor_cache.get(word, [])
            for j_word, j_val in tree_neighbors:
                for j in range(len(state_words)):
                    if j == pos:
                        continue
                    if state_words[j] == j_word:
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
                            dist = abs(pos - j)
                            if dist <= 5:
                                energy += j_val * self.dep_weight // 2

            energy -= self.deps.compute_agreement_penalty(state_types, pos, word_type)

        return energy

    def _fast_energy(self, pos, word, word_type, state_words, state_types, H_sum=None):
        """
        Fast approximate energy for proposal weighting.
        Only computes the dominant terms: field + PMI coupling + emission.
        O(window * K) per candidate instead of O(V).
        """
        energy = 0
        # Field energy
        energy += int(self.pmi.h[pos % self.pmi.seq_len, word])

        # PMI coupling (dominant term)
        if self.J_combined is not None:
            for j_offset in range(1, self.pmi.window + 1):
                j = pos + j_offset
                if j < len(state_words):
                    energy += int(self.J_combined[word, state_words[j]])
                j = pos - j_offset
                if j >= 0:
                    energy += int(self.J_combined[state_words[j], word])

        # Strengthened emission
        if word < self.types.I_emit.shape[0] and word_type < self.types.I_emit.shape[1]:
            emit_val = int(self.types.I_emit[word, word_type])
            if emit_val > 0:
                energy += emit_val * self.emission_bonus
            else:
                energy -= self.emission_penalty

        return energy

    def _locally_balanced_proposal(
        self, pos, current_word, current_type, state_words, state_types,
        prob_table, H_sum=None
    ):
        """
        P0: Locally-balanced proposal (Zanella 2017).
        
        Uses FAST approximate energy for proposal weighting, then
        full Metropolis-Hastings correction with exact energy.
        
        Candidate set is limited to 30 for speed.
        """
        # Step 1: Gather candidates (limit to 30 for speed)
        candidates = []
        
        # PMI neighbors of current position's neighbors
        seen = {current_word}
        candidates.append(current_word)
        
        for j_offset in range(1, self.pmi.window + 1):
            j = pos + j_offset
            if j < len(state_words):
                nbrs = self.pmi.get_neighbor_words(state_words[j], top_k=10)
                for w in nbrs:
                    if w not in seen:
                        seen.add(w)
                        candidates.append(w)
                        if len(candidates) >= 30:
                            break
            if len(candidates) >= 30:
                break
            j = pos - j_offset
            if j >= 0:
                nbrs = self.pmi.get_neighbor_words(state_words[j], top_k=10)
                for w in nbrs:
                    if w not in seen:
                        seen.add(w)
                        candidates.append(w)
                        if len(candidates) >= 30:
                            break
            if len(candidates) >= 30:
                break
        
        # Emission-compatible words
        emit_words = self.types.get_allowed_words_for_type(current_type)
        for w in emit_words[:10]:
            if w not in seen:
                seen.add(w)
                candidates.append(w)
                if len(candidates) >= 30:
                    break

        # Step 2: Compute approximate ΔE for proposal weighting
        current_energy = self._fast_energy(
            pos, current_word, current_type, state_words, state_types, H_sum
        )

        # Step 3: Locally-balanced weighting: exp(-ΔE/2T)
        candidate_weights = []
        candidate_deltas = []
        max_de = (len(prob_table) - 1) // 2
        
        for w in candidates:
            proposed_type = int(self.types.get_type_for_word(w))
            proposed_energy = self._fast_energy(
                pos, w, proposed_type, state_words, state_types, H_sum
            )
            delta_e = proposed_energy - current_energy
            
            # Locally-balanced weight
            half_de = delta_e // 2
            if half_de <= -max_de:
                weight = 2**31 - 1
            elif half_de >= max_de:
                weight = 1
            else:
                weight = max(1, prob_table[half_de + max_de])
            
            candidate_weights.append(weight)
            candidate_deltas.append(delta_e)

        # Step 4: Sample proportionally
        total_weight = sum(candidate_weights)
        if total_weight <= 0:
            return current_word, current_type, H_sum

        r = random.randint(1, total_weight)
        cumsum = 0
        chosen_idx = 0
        for i, wt in enumerate(candidate_weights):
            cumsum += wt
            if r <= cumsum:
                chosen_idx = i
                break

        chosen_word = candidates[chosen_idx]
        if chosen_word == current_word:
            return current_word, current_type, H_sum
        
        # Step 5: Full Metropolis-Hastings correction with EXACT energy
        proposed_type = int(self.types.get_type_for_word(chosen_word))
        exact_current = self._compute_word_energy(
            pos, current_word, current_type, state_words, state_types, H_sum
        )
        exact_proposed = self._compute_word_energy(
            pos, chosen_word, proposed_type, state_words, state_types, H_sum
        )
        delta_e_exact = exact_proposed - exact_current

        rand_val = random.randint(0, 2**31 - 2)
        if self._accept(delta_e_exact, rand_val, prob_table):
            if H_sum is not None and self.nmf is not None and self.nmf.fitted:
                H_sum = self.nmf.update_H_sum(H_sum, current_word, chosen_word)
            return chosen_word, proposed_type, H_sum
        else:
            return current_word, current_type, H_sum

    def generate(self, length=20, prompt=None, vocab=None, verbose=False):
        """Generate text using v4 enhanced sampler with all mitigations."""
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

        # Precompute H_sum for CALDERA path
        H_sum = None
        if self.nmf is not None and self.nmf.fitted:
            H_sum = self.nmf.compute_H_sum(state_words)

        if verbose:
            self._print_state(state_words, state_types, vocab, "Init")

        # PHASE 1: High temperature — types only (with hard constraints)
        for sweep in range(self.sweeps_p1):
            for pos in range(prompt_len, length):
                state_types[pos] = self._sample_type(
                    pos, state_types[pos], state_types
                )
            if verbose and (sweep + 1) % max(1, self.sweeps_p1 // 4) == 0:
                self._print_state(state_words, state_types, vocab, f"P1 sweep {sweep+1}")

        if verbose:
            self._print_state(state_words, state_types, vocab, "After Phase 1")

        # PHASE 2: Medium temperature — types + words (locally-balanced proposals)
        for sweep in range(self.sweeps_p2):
            for pos in range(prompt_len, length):
                state_types[pos] = self._sample_type(
                    pos, state_types[pos], state_types
                )
                new_word, new_type, H_sum = self._locally_balanced_proposal(
                    pos, state_words[pos], state_types[pos],
                    state_words, state_types, self.prob_table_p2, H_sum
                )
                state_words[pos] = new_word
                state_types[pos] = new_type
            if verbose and (sweep + 1) % max(1, self.sweeps_p2 // 4) == 0:
                self._print_state(state_words, state_types, vocab, f"P2 sweep {sweep+1}")

        if verbose:
            self._print_state(state_words, state_types, vocab, "After Phase 2")

        # PHASE 3: Low temperature — words only (types frozen, strong proposals)
        for sweep in range(self.sweeps_p3):
            for pos in range(prompt_len, length):
                new_word, new_type, H_sum = self._locally_balanced_proposal(
                    pos, state_words[pos], state_types[pos],
                    state_words, state_types, self.prob_table_p3, H_sum
                )
                state_words[pos] = new_word
                # Keep type frozen in phase 3
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


# =============================================================================
# V4 Enhanced Model
# =============================================================================

class EnhancedV4Model:
    """
    V4 Enhanced Typed Ising-Potts Language Model with all research-backed mitigations.
    """

    def __init__(
        self,
        # Vocabulary
        vocab_min_freq: int = 5,
        vocab_max_size: Optional[int] = 5000,
        # Sequence
        seq_len: int = 25,
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
        # Emission (P1: STRENGTHENED)
        emission_bonus: int = 100,
        emission_penalty: int = 500,
        # Annealing
        phase1_beta: int = 200,
        phase2_beta: int = 500,
        phase3_beta: int = 1000,
        total_sweeps: int = 150,
        # CALDERA NMF
        use_caldera: bool = True,
        nmf_factors: int = 128,
        nmf_iterations: int = 50,
        nmf_n_top: int = 15,
        # SpaCy
        use_spacy: bool = True,
        spacy_max_texts: Optional[int] = None,
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
        self.emission_bonus = emission_bonus
        self.emission_penalty = emission_penalty
        self.phase1_beta = phase1_beta
        self.phase2_beta = phase2_beta
        self.phase3_beta = phase3_beta
        self.total_sweeps = total_sweeps
        self.use_caldera = use_caldera
        self.nmf_factors = nmf_factors
        self.nmf_iterations = nmf_iterations
        self.nmf_n_top = nmf_n_top
        self.use_spacy = use_spacy
        self.spacy_max_texts = spacy_max_texts

        # Components
        self.vocab = None
        self.pmi = None
        self.types = None
        self.semantics = None
        self.spacy_tagger = None
        self.dep_couplings = None
        self.nmf_model = None
        self.sampler = None
        self.allowed_transitions = None

    def train(self, n_samples=50000, verbose=True):
        """Train the v4 model with all enhancements."""
        print("=" * 70)
        print("ENHANCED TYPED ISING-POTTS MODEL v4 — TRAINING")
        print("(P0: Locally-balanced proposals, Hard transition constraints)")
        print("(P1: CALDERA NMF, Strengthened emission, Implicational couplings)")
        print("=" * 70)

        # Step 1: Load data
        t0 = time.time()
        texts = load_fineweb_edu(n_samples=n_samples)
        print(f"[1/9] Data loading: {len(texts)} texts ({time.time()-t0:.1f}s)")

        # Step 2: Build vocabulary
        t0 = time.time()
        self.vocab = Vocabulary(
            min_freq=self.vocab_min_freq,
            max_size=self.vocab_max_size,
        )
        self.vocab.build(texts)
        V = len(self.vocab)
        print(f"[2/9] Vocabulary: {V} words ({time.time()-t0:.1f}s)")

        # Step 3: Tokenize
        t0 = time.time()
        sequences = tokenize_texts(texts, self.vocab)
        sequences = truncate_sequences(sequences, max_len=self.seq_len)
        print(f"[3/9] Tokenization: {len(sequences)} sequences ({time.time()-t0:.1f}s)")

        # Step 4: SpaCy POS tagging + dependency parsing
        t0 = time.time()
        if self.use_spacy:
            self.spacy_tagger = SpaCyTagger(vocab_size=V, n_pos=N_POS)
            self.spacy_tagger.tag_corpus(
                texts, sequences,
                self.vocab.word2idx, self.vocab.idx2word,
                max_texts=self.spacy_max_texts,
            )
            print(f"[4/9] SpaCy POS + deps: "
                  f"{sum(len(v) for v in self.spacy_tagger.word_pos.values())} word-POS entries, "
                  f"{len(self.spacy_tagger.dep_edges)} dep edges ({time.time()-t0:.1f}s)")
        else:
            self.spacy_tagger = None
            print(f"[4/9] SpaCy: skipped")

        # Step 5: PMI couplings
        t0 = time.time()
        self.pmi = PMICouplings(vocab_size=V, seq_len=self.seq_len, window=self.window)
        self.pmi.compute_from_sequences(
            sequences, min_count=self.min_cooc, pmi_cap=self.pmi_cap,
            use_hebbian=True, hebbian_weight=self.hebbian_weight,
        )
        pmi_nnz = int(np.count_nonzero(self.pmi.J_PMI))
        print(f"[5/9] PMI couplings: {pmi_nnz} non-zeros ({time.time()-t0:.1f}s)")

        # Step 6: Build type system
        t0 = time.time()
        self.types = POSTypeSystem(vocab_size=V, n_types=N_POS, window=self.window)

        if self.use_spacy and self.spacy_tagger is not None:
            self.types.I_emit = self.spacy_tagger.build_emission_weights()
            self.types.allowed_types = self.spacy_tagger.build_allowed_types()
            self.types.J_type = self.spacy_tagger.build_type_couplings(scaling=10)
        else:
            self.types.build_from_vocabulary(self.vocab.word2idx, self.vocab.idx2word)
            self.types.compute_type_couplings(sequences, self.vocab.idx2word, scaling=10)

        self.types.build_grammar_penalties(penalty_strength=self.grammar_penalty)
        self.types.precompute_type_distribution()

        n_typed = sum(1 for w in range(V) if len(self.types.allowed_types.get(w, set())) > 0)
        print(f"[6/9] POS type system: {n_typed}/{V} words typed ({time.time()-t0:.1f}s)")

        # Step 7: Dependency couplings
        t0 = time.time()
        if self.use_spacy and self.spacy_tagger is not None:
            self.dep_couplings = DependencyCouplings(vocab_size=V, n_pos=N_POS)
            self.dep_couplings.build_from_spacy_tagger(
                self.spacy_tagger, self.vocab.idx2word,
                min_count=1, coupling_strength=3,
            )
            dep_stats = self.dep_couplings.get_dep_stats()
            print(f"[7/9] Dependency couplings: "
                  f"J_tree {dep_stats['J_tree_nnz']} nnz, "
                  f"{dep_stats['agreement_rules']} agreement rules ({time.time()-t0:.1f}s)")
        else:
            self.dep_couplings = None
            print(f"[7/9] Dependency couplings: skipped")

        # Step 7b: Semantic types
        self.semantics = SemanticTypeSystem(vocab_size=V, n_sem_types=N_SEM, compatibility_strength=3)
        self.semantics.build_from_vocabulary(self.vocab.word2idx, self.vocab.idx2word)
        self.semantics.compute_compatibility_matrix(sequences, min_cooc=2)
        self.semantics.compute_hebbian_coupling(sequences, hebbian_weight=1)

        # Step 8: P0: Build allowed POS transitions
        t0 = time.time()
        self.allowed_transitions = build_allowed_transitions_from_tagger(
            self.spacy_tagger, self.vocab.idx2word, sequences, self.types, min_count=3
        )
        n_allowed = len(self.allowed_transitions)
        n_total = N_POS * N_POS
        print(f"[8/9] Hard POS transitions: {n_allowed}/{n_total} allowed "
              f"({100*n_allowed/n_total:.1f}%) ({time.time()-t0:.1f}s)")

        # Step 9: P1: CALDERA NMF
        t0 = time.time()
        if self.use_caldera:
            J_full = self.pmi_weight * self.pmi.J_PMI + self.hebbian_weight * self.pmi.J_Hebb
            J_full += self.semantic_weight * self.semantics.J_sem
            if self.dep_couplings is not None:
                J_full += self.dep_weight * self.dep_couplings.J_tree

            self.nmf_model = CalderaNMF(
                vocab_size=V, n_factors=self.nmf_factors, n_top=self.nmf_n_top
            )
            self.nmf_model.fit(J_full, n_iterations=self.nmf_iterations)

            mem = self.nmf_model.memory_savings()
            # Compute quality
            J_recon = self.nmf_model.reconstruct()
            abs_err = int(np.sum(np.abs(J_full - J_recon)))
            rel_err = abs_err / max(1, int(np.sum(np.abs(J_full))))
            print(f"[9/9] CALDERA NMF: K={self.nmf_factors}, n_top={self.nmf_n_top}, "
                  f"rel_err={rel_err:.3f}, "
                  f"memory savings={mem['savings_pct']:.1f}% ({time.time()-t0:.1f}s)")
        else:
            self.nmf_model = None
            print(f"[9/9] CALDERA NMF: skipped")

        # Build sampler
        print("\nBuilding v4 enhanced sampler...")
        t0 = time.time()
        self.sampler = EnhancedV4Sampler(
            pmi_couplings=self.pmi,
            type_system=self.types,
            semantic_system=self.semantics,
            dep_couplings=self.dep_couplings,
            nmf=self.nmf_model,
            allowed_transitions=self.allowed_transitions,
            phase1_beta=self.phase1_beta,
            phase2_beta=self.phase2_beta,
            phase3_beta=self.phase3_beta,
            total_sweeps=self.total_sweeps,
            pmi_weight=self.pmi_weight,
            hebbian_weight=self.hebbian_weight,
            semantic_weight=self.semantic_weight,
            dep_weight=self.dep_weight,
            emission_bonus=self.emission_bonus,
            emission_penalty=self.emission_penalty,
        )
        print(f"Sampler ready ({time.time()-t0:.1f}s)")

        # Summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE — v4 ENHANCED MODEL")
        print("=" * 70)
        self._print_summary()

        return self

    def _print_summary(self):
        print(f"\nModel Architecture (v4 Enhanced):")
        print(f"  Vocabulary size: {len(self.vocab)}")
        print(f"  POS types: {N_POS}")
        print(f"  Hard POS transitions: {len(self.allowed_transitions)} allowed")
        print(f"  Implicational rules: {len(IMPLICATION_RULES)}")
        print(f"  Emission bonus/penalty: {self.emission_bonus}/{self.emission_penalty}")
        print(f"  PMI coupling range: [{int(self.pmi.J_PMI.min())}, {int(self.pmi.J_PMI.max())}]")
        print(f"  PMI non-zeros: {int(np.count_nonzero(self.pmi.J_PMI))}")
        if self.nmf_model is not None and self.nmf_model.fitted:
            J_full = self.pmi_weight * self.pmi.J_PMI + self.hebbian_weight * self.pmi.J_Hebb
            J_recon = self.nmf_model.reconstruct()
            abs_err = int(np.sum(np.abs(J_full - J_recon)))
            rel_err = abs_err / max(1, int(np.sum(np.abs(J_full))))
            mem = self.nmf_model.memory_savings()
            print(f"  CALDERA NMF rel_error: {rel_err:.3f}")
            print(f"  CALDERA memory savings: {mem['savings_pct']:.1f}%")
            print(f"  CALDERA Q nnz: {mem['q_nnz']}")
        if self.dep_couplings is not None:
            dep_stats = self.dep_couplings.get_dep_stats()
            print(f"  Dependency J_tree non-zeros: {dep_stats['J_tree_nnz']}")
        print(f"  Locally-balanced proposals: YES")
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

        return {
            "text": self.vocab.decode(words),
            "types": [IDX2POS.get(t, "UNK") for t in types],
            "energy": energy,
            "type_counts": type_counts,
            "words": words,
        }

    def _decode_with_annotations(self, words, types):
        parts = []
        for w, t in zip(words, types):
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

        # Save allowed transitions
        import json as json_mod
        trans_list = [list(t) for t in self.allowed_transitions] if self.allowed_transitions else []
        with open(os.path.join(directory, "allowed_transitions.json"), "w") as f:
            json_mod.dump(trans_list, f)

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
            "emission_bonus": self.emission_bonus,
            "emission_penalty": self.emission_penalty,
            "phase1_beta": self.phase1_beta,
            "phase2_beta": self.phase2_beta,
            "phase3_beta": self.phase3_beta,
            "total_sweeps": self.total_sweeps,
            "use_caldera": self.use_caldera,
            "nmf_factors": self.nmf_factors,
            "nmf_iterations": self.nmf_iterations,
            "nmf_n_top": self.nmf_n_top,
            "use_spacy": self.use_spacy,
        }
        with open(os.path.join(directory, "config.json"), "w") as f:
            json_mod.dump(config, f, indent=2)

    @classmethod
    def load(cls, directory):
        import json as json_mod
        with open(os.path.join(directory, "config.json")) as f:
            config = json_mod.load(f)

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
            model.nmf_model = CalderaNMF.load(os.path.join(directory, "nmf"))
        except FileNotFoundError:
            model.nmf_model = None

        try:
            with open(os.path.join(directory, "allowed_transitions.json")) as f:
                trans_list = json_mod.load(f)
            model.allowed_transitions = {tuple(t) for t in trans_list}
        except FileNotFoundError:
            model.allowed_transitions = None

        model.sampler = EnhancedV4Sampler(
            pmi_couplings=model.pmi,
            type_system=model.types,
            semantic_system=model.semantics,
            dep_couplings=model.dep_couplings,
            nmf=model.nmf_model,
            allowed_transitions=model.allowed_transitions,
            phase1_beta=model.phase1_beta,
            phase2_beta=model.phase2_beta,
            phase3_beta=model.phase3_beta,
            total_sweeps=model.total_sweeps,
            pmi_weight=model.pmi_weight,
            hebbian_weight=model.hebbian_weight,
            semantic_weight=model.semantic_weight,
            dep_weight=model.dep_weight,
            emission_bonus=model.emission_bonus,
            emission_penalty=model.emission_penalty,
        )

        return model
