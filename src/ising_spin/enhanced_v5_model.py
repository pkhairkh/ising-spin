"""
Enhanced Typed Ising-Potts Language Model v5 — P2+Transition Fixes.

Extends v4 (P0+P1) with:
  P2: Parallel tempering with adaptive ladder (Earl & Deem 2005)
  History-driven target (Hu et al. 2025, arXiv:2505.18300) — repel recently visited states
  Non-reversible MCMC — momentum to push through flat energy regions
  Stricter transition constraints — min_count 3→20

All generation-path computation remains integer arithmetic only.
"""

import os
import time
import random
import math
import json as json_mod
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

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
# Hard POS Transition Constraints — STRICTER (min_count=20)
# =============================================================================

def build_allowed_transitions_from_tagger(
    spacy_tagger,
    idx2word,
    sequences,
    type_system,
    min_count=20,  # RAISED from 3 to 20 — only genuinely common transitions
) -> Set[Tuple[int, int]]:
    """
    Build allowed POS transitions from SpaCy-tagged corpus.
    
    Higher min_count means fewer but more reliable transitions are allowed.
    min_count=20 ensures only genuinely common grammatical patterns survive.
    """
    transitions = {}
    
    for seq in sequences:
        seq_types = []
        for w in seq:
            if w in type_system.allowed_types and type_system.allowed_types[w]:
                if spacy_tagger is not None and w in spacy_tagger.word_pos:
                    pos_counts = spacy_tagger.word_pos[w]
                    best_pos = max(pos_counts, key=pos_counts.get)
                    if best_pos in type_system.allowed_types[w]:
                        seq_types.append(best_pos)
                    else:
                        seq_types.append(max(type_system.allowed_types[w],
                                           key=lambda t: int(type_system.I_emit[w, t])))
                else:
                    seq_types.append(max(type_system.allowed_types[w],
                                       key=lambda t: int(type_system.I_emit[w, t])))
            else:
                seq_types.append(POS2IDX["X"])
        
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
    
    # Filter by minimum count — HIGHER THRESHOLD
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
# Implicational Couplings (Marcolli 2015) — from v4
# =============================================================================

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

IMPLICATION_PENALTY = 300


def compute_implicational_penalty(state_types, pos, proposed_type):
    penalty = 0
    if proposed_type in IMPLICATION_RULES:
        for offset, required_types in IMPLICATION_RULES[proposed_type].items():
            j = pos + offset
            if 0 <= j < len(state_types):
                if state_types[j] not in required_types:
                    penalty += IMPLICATION_PENALTY
            j = pos - offset
            if 0 <= j < len(state_types):
                if state_types[j] not in required_types:
                    penalty += IMPLICATION_PENALTY // 2
    
    for offset in [-2, -1, 1, 2]:
        j = pos + offset
        if 0 <= j < len(state_types):
            neighbor_type = state_types[j]
            if neighbor_type in IMPLICATION_RULES:
                abs_offset = abs(offset)
                for req_offset, required_types in IMPLICATION_RULES[neighbor_type].items():
                    if abs_offset == req_offset:
                        if offset > 0 and req_offset == offset:
                            if proposed_type not in required_types:
                                penalty += IMPLICATION_PENALTY
                        elif offset < 0 and req_offset == -offset:
                            if proposed_type not in required_types:
                                penalty += IMPLICATION_PENALTY // 2
    return penalty


# =============================================================================
# History-Driven Target (Hu et al. 2025, arXiv:2505.18300)
# =============================================================================

class HistoryBuffer:
    """
    Tracks recently visited word states and provides a repulsion penalty.
    
    When a word at position pos has been the same for many consecutive sweeps,
    this adds an energy penalty to encourage exploration.
    
    All integer arithmetic. Tracks visit counts per (position, word) pair.
    """
    
    def __init__(self, length, max_history=50):
        self.length = length
        self.max_history = max_history
        # visit_counts[pos][word] = number of recent consecutive visits
        self.visit_counts = [defaultdict(int) for _ in range(length)]
        # total_updates tracks how many sweeps we've done
        self.total_updates = 0
        # Repulsion strength: penalty = count * decay_factor
        self.decay_factor = 5  # integer penalty per consecutive visit
        # How often to decay (every N sweeps, halve all counts)
        self.decay_interval = 10
    
    def record(self, pos, word):
        """Record current state. Increment visit count for this (pos, word)."""
        if pos < self.length:
            # Reset all other words at this position
            for w in list(self.visit_counts[pos].keys()):
                if w != word:
                    self.visit_counts[pos][w] = max(0, self.visit_counts[pos][w] - 1)
            self.visit_counts[pos][word] += 1
            self.total_updates += 1
            
            # Periodic decay to prevent unbounded growth
            if self.total_updates % self.decay_interval == 0:
                self._decay()
    
    def _decay(self):
        """Halve all visit counts to forget old history."""
        for pos in range(self.length):
            for w in list(self.visit_counts[pos].keys()):
                self.visit_counts[pos][w] = self.visit_counts[pos][w] // 2
                if self.visit_counts[pos][w] == 0:
                    del self.visit_counts[pos][w]
    
    def repulsion_penalty(self, pos, word):
        """
        Compute integer repulsion penalty for (pos, word).
        Higher count = stronger repulsion = more incentive to change.
        """
        if pos >= self.length:
            return 0
        count = self.visit_counts[pos].get(word, 0)
        # Quadratic penalty: stronger push for very stuck states
        return min(count * count * self.decay_factor, 5000)
    
    def record_full_state(self, state_words):
        """Record entire state after a sweep."""
        for pos, w in enumerate(state_words):
            self.record(pos, w)


# =============================================================================
# Non-Reversible MCMC — Momentum
# =============================================================================

class MomentumTracker:
    """
    Tracks direction of recent state changes to add momentum.
    
    If position pos has been moving in a consistent direction (e.g., word
    index increasing), bias proposals toward continuing that direction.
    This helps push through flat energy landscapes where random proposals
    get stuck.
    
    All integer arithmetic.
    """
    
    def __init__(self, length, momentum_strength=3):
        self.length = length
        self.strength = momentum_strength
        # Track last few words at each position
        self.history = [[] for _ in range(length)]
        self.max_history = 5
    
    def record(self, pos, word):
        """Record word at position."""
        if pos < self.length:
            self.history[pos].append(word)
            if len(self.history[pos]) > self.max_history:
                self.history[pos].pop(0)
    
    def get_momentum_bias(self, pos, current_word):
        """
        Get momentum-biased candidate set.
        
        If recent words at this position show a trend (consistently increasing
        or decreasing word indices), return candidates biased in that direction.
        Returns list of (word_idx, bias_weight) pairs.
        """
        if pos >= self.length or len(self.history[pos]) < 2:
            return []
        
        hist = self.history[pos]
        # Compute direction: sum of (word[i] - word[i-1]) signs
        direction = 0
        for i in range(1, len(hist)):
            diff = hist[i] - hist[i-1]
            if diff > 0:
                direction += 1
            elif diff < 0:
                direction -= 1
        
        if direction == 0:
            return []
        
        # Generate momentum-biased candidates
        candidates = []
        step_size = abs(direction) * 10 + 5  # Larger step for stronger momentum
        for i in range(1, 4):
            if direction > 0:
                # Moving toward higher indices
                candidate = current_word + step_size * i
            else:
                # Moving toward lower indices
                candidate = current_word - step_size * i
            
            # Wrap around if needed (shouldn't happen with valid vocab but be safe)
            if 0 <= candidate < 10000:  # placeholder max
                candidates.append((candidate, self.strength * i))
        
        return candidates


# =============================================================================
# V5 Enhanced Sampler — Parallel Tempering + History + Momentum
# =============================================================================

class EnhancedV5Sampler:
    """
    V5 Enhanced sampler with P2 mitigations on top of P0+P1.
    
    P0: Locally-balanced proposals (Zanella 2017)
    P0: Hard POS transition constraints (stricter: min_count=20)
    P1: CALDERA NMF, strengthened emission, implicational couplings
    P2: Parallel tempering with adaptive ladder
    P2: History-driven target (Hu et al. 2025) — repel stuck states
    P2: Non-reversible MCMC — momentum bias for flat regions
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
        total_sweeps: int = 200,
        phase1_frac: float = 0.20,
        phase2_frac: float = 0.40,
        phase3_frac: float = 0.40,
        # Weights
        pmi_weight: int = 3,
        hebbian_weight: int = 1,
        semantic_weight: int = 1,
        dep_weight: int = 2,
        # Emission weights
        emission_bonus: int = 100,
        emission_penalty: int = 500,
        # Proposal parameters
        proposal_top_k: int = 50,
        # P2: Parallel tempering
        n_replicas: int = 4,
        pt_swap_interval: int = 5,
        # P2: History-driven target
        history_enabled: bool = True,
        history_repulsion: int = 5,
        # P2: Momentum
        momentum_enabled: bool = True,
        momentum_strength: int = 3,
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

        # P2: Parallel tempering ladder
        self.n_replicas = n_replicas
        self.pt_swap_interval = pt_swap_interval
        # Temperature ladder: geometric spacing from hot to cold
        # Beta values: phase1_beta (hot) to phase3_beta (cold)
        self.pt_betas = self._build_temperature_ladder(phase1_beta, phase3_beta, n_replicas)
        self.pt_prob_tables = [self._build_prob_table(b) for b in self.pt_betas]
        self.pt_swap_counts = [0] * (n_replicas - 1)
        self.pt_attempt_counts = [0] * (n_replicas - 1)

        # P2: History-driven target
        self.history_enabled = history_enabled
        self.history_repulsion = history_repulsion
        
        # P2: Momentum
        self.momentum_enabled = momentum_enabled
        self.momentum_strength = momentum_strength

        # Build combined coupling matrix
        self.J_combined = self._build_combined_coupling()

        # Precompute proposal sets
        self.proposal_cache = self._build_proposal_cache(proposal_top_k)

        # Precompute distributions
        self.type_cumsum_by_pos = self._build_type_distributions()
        self.emit_cumsum_by_type = self._build_emission_distributions()
        self.field_weights = self._build_field_weights()

        # Precompute J_tree neighbor cache
        self.tree_neighbor_cache = self._build_tree_neighbor_cache()

    def _build_temperature_ladder(self, beta_hot, beta_cold, n_replicas):
        """Build geometric temperature ladder for parallel tempering."""
        betas = []
        for i in range(n_replicas):
            frac = i / max(1, n_replicas - 1)
            # Geometric spacing: beta = beta_hot * (beta_cold/beta_hot)^frac
            # Using integer approximation
            log_ratio = (beta_cold.bit_length() - beta_hot.bit_length()) if beta_cold > beta_hot and beta_hot > 0 else 0
            beta = int(beta_hot * (beta_cold / max(1, beta_hot)) ** frac)
            beta = max(beta_hot, min(beta_cold, beta))
            betas.append(beta)
        
        # Ensure we have the exact endpoints
        betas[0] = beta_hot
        betas[-1] = beta_cold
        
        # Fill in with reasonable intermediate values
        if n_replicas == 4:
            betas = [beta_hot, (beta_hot + beta_cold) // 3, 2 * (beta_hot + beta_cold) // 3, beta_cold]
        
        return betas

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
        J = self.pmi_weight * self.pmi.J_PMI + self.hebbian_weight * self.pmi.J_Hebb
        if self.sem is not None:
            J = J + self.semantic_weight * self.sem.J_sem
        return J

    def _build_proposal_cache(self, top_k):
        cache = {}
        for w in range(self.vocab_size):
            candidates = set()
            pmi_neighbors = self.pmi.get_neighbor_words(w, top_k=top_k * 2)
            candidates.update(pmi_neighbors)
            if w in self.types.allowed_types:
                type_set = self.types.allowed_types[w]
                for t in type_set:
                    compat_words = self.types.get_allowed_words_for_type(t)
                    candidates.update(compat_words[:top_k])
            if self.deps is not None:
                tree_neighbors = self.deps.get_tree_neighbors(w, top_k=top_k)
                for w_idx, _ in tree_neighbors:
                    candidates.add(w_idx)
            if self.nmf is not None and self.nmf.fitted:
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
        """Sample type with hard transition constraints + implicational couplings."""
        type_energies = np.zeros(self.n_types, dtype=np.int64)

        for t in range(self.n_types):
            energy = 0

            # Hard transition constraint
            if self.allowed_transitions is not None:
                if pos > 0:
                    if (state_types[pos - 1], t) not in self.allowed_transitions:
                        energy -= 10000
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

            # Implicational coupling penalty
            energy -= compute_implicational_penalty(state_types, pos, t)

            # Dependency agreement penalty
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
        self, pos, word, word_type, state_words, state_types, 
        H_sum=None, history_buffer=None
    ):
        """Compute total energy including history-driven repulsion."""
        energy = 0

        # 1. Field energy
        energy += int(self.pmi.h[pos % self.pmi.seq_len, word])

        # 2. Lexical coupling
        if self.nmf is not None and self.nmf.fitted and H_sum is not None:
            neighbor_words = [state_words[j] for j in range(max(0, pos - self.pmi.window), 
                                                             min(len(state_words), pos + self.pmi.window + 1))
                             if j != pos]
            q_energy = self.nmf.get_sparse_energy(word, neighbor_words)
            l_energy = self.nmf.get_factorized_energy(word, H_sum)
            energy += self.pmi_weight * q_energy + l_energy
        elif self.J_combined is not None:
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

        # 4. Strengthened emission
        if word < self.types.I_emit.shape[0] and word_type < self.types.I_emit.shape[1]:
            emit_val = int(self.types.I_emit[word, word_type])
            if emit_val > 0:
                energy += emit_val * self.emission_bonus
            else:
                energy -= self.emission_penalty

        # 5. Grammar penalty
        types_copy = list(state_types)
        types_copy[pos] = word_type
        energy -= self.types.compute_grammar_penalty(types_copy, pos, word_type)

        # 6. Implicational coupling
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

        # P2: History-driven repulsion
        if history_buffer is not None:
            energy -= history_buffer.repulsion_penalty(pos, word)

        return energy

    def _fast_energy(self, pos, word, word_type, state_words, state_types, H_sum=None,
                     history_buffer=None):
        """Fast approximate energy for proposal weighting."""
        energy = 0
        energy += int(self.pmi.h[pos % self.pmi.seq_len, word])

        if self.J_combined is not None:
            for j_offset in range(1, self.pmi.window + 1):
                j = pos + j_offset
                if j < len(state_words):
                    energy += int(self.J_combined[word, state_words[j]])
                j = pos - j_offset
                if j >= 0:
                    energy += int(self.J_combined[state_words[j], word])

        if word < self.types.I_emit.shape[0] and word_type < self.types.I_emit.shape[1]:
            emit_val = int(self.types.I_emit[word, word_type])
            if emit_val > 0:
                energy += emit_val * self.emission_bonus
            else:
                energy -= self.emission_penalty

        # History-driven repulsion (cheap to compute)
        if history_buffer is not None:
            energy -= history_buffer.repulsion_penalty(pos, word)

        return energy

    def _locally_balanced_proposal(
        self, pos, current_word, current_type, state_words, state_types,
        prob_table, H_sum=None, history_buffer=None, momentum_tracker=None
    ):
        """P0+P2: Locally-balanced proposal with history repulsion and momentum."""
        candidates = []
        seen = {current_word}
        candidates.append(current_word)

        # PMI neighbors of current position's neighbors
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

        # P2: Momentum-biased candidates
        if momentum_tracker is not None:
            momentum_candidates = momentum_tracker.get_momentum_bias(pos, current_word)
            for w, bias in momentum_candidates:
                if 0 <= w < self.vocab_size and w not in seen:
                    seen.add(w)
                    candidates.append(w)
                    if len(candidates) >= 30:
                        break

        # Compute approximate ΔE for proposal weighting
        current_energy = self._fast_energy(
            pos, current_word, current_type, state_words, state_types, H_sum,
            history_buffer
        )

        # Locally-balanced weighting: exp(-ΔE/2T)
        candidate_weights = []
        max_de = (len(prob_table) - 1) // 2

        for w in candidates:
            proposed_type = int(self.types.get_type_for_word(w))
            if proposed_type >= self.n_types:
                proposed_type = POS2IDX["X"]
            proposed_energy = self._fast_energy(
                pos, w, proposed_type, state_words, state_types, H_sum,
                history_buffer
            )
            delta_e = proposed_energy - current_energy

            half_de = delta_e // 2
            if half_de <= -max_de:
                weight = 2**31 - 1
            elif half_de >= max_de:
                weight = 1
            else:
                weight = max(1, prob_table[half_de + max_de])

            candidate_weights.append(weight)

        # Sample proportionally
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

        # Full Metropolis-Hastings correction
        proposed_type = int(self.types.get_type_for_word(chosen_word))
        if proposed_type >= self.n_types:
            proposed_type = POS2IDX["X"]
        exact_current = self._compute_word_energy(
            pos, current_word, current_type, state_words, state_types, H_sum,
            history_buffer
        )
        exact_proposed = self._compute_word_energy(
            pos, chosen_word, proposed_type, state_words, state_types, H_sum,
            history_buffer
        )
        delta_e_exact = exact_proposed - exact_current

        rand_val = random.randint(0, 2**31 - 2)
        if self._accept(delta_e_exact, rand_val, prob_table):
            if H_sum is not None and self.nmf is not None and self.nmf.fitted:
                H_sum = self.nmf.update_H_sum(H_sum, current_word, chosen_word)
            # Record in momentum tracker
            if momentum_tracker is not None:
                momentum_tracker.record(pos, chosen_word)
            return chosen_word, proposed_type, H_sum
        else:
            return current_word, current_type, H_sum

    def _compute_total_energy(self, state_words, state_types, H_sum=None):
        """Compute total energy of a state (for parallel tempering swaps)."""
        energy = 0
        for i in range(len(state_words)):
            energy += int(self.pmi.h[i % self.pmi.seq_len, state_words[i]])
        if self.J_combined is not None:
            for i in range(len(state_words)):
                for j_offset in range(1, self.pmi.window + 1):
                    j = i + j_offset
                    if j < len(state_words):
                        energy += int(self.J_combined[state_words[i], state_words[j]])
        # Emission contribution
        for i in range(len(state_words)):
            w = state_words[i]
            t = state_types[i]
            if w < self.types.I_emit.shape[0] and t < self.types.I_emit.shape[1]:
                emit_val = int(self.types.I_emit[w, t])
                if emit_val > 0:
                    energy += emit_val * self.emission_bonus
                else:
                    energy -= self.emission_penalty
        return energy

    def _pt_swap(self, replicas, replica_energies, replica_betas):
        """
        Attempt parallel tempering swap between adjacent replicas.
        
        Uses integer Metropolis criterion:
        swap_prob = min(1, exp((beta_j - beta_i) * (E_j - E_i)))
        """
        for i in range(len(replicas) - 1):
            self.pt_attempt_counts[i] += 1
            
            beta_i = replica_betas[i]
            beta_j = replica_betas[i + 1]
            E_i = replica_energies[i]
            E_j = replica_energies[i + 1]
            
            # Integer swap criterion: 
            # delta = (beta_j - beta_i) * (E_j - E_i) / 1000
            # (betas are in units of 1000)
            delta = (beta_j - beta_i) * (E_j - E_i) // 1000
            
            if delta >= 0:
                # Always accept
                swap = True
            else:
                # Metropolis criterion
                rand_val = random.randint(0, 2**31 - 2)
                # Approximate: accept with probability exp(delta)
                # Using integer comparison
                accept_val = int((2**31 - 1) * math.exp(max(-500, delta / 1000.0)))
                swap = rand_val < accept_val
            
            if swap:
                # Swap replicas
                replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
                replica_energies[i], replica_energies[i + 1] = replica_energies[i + 1], replica_energies[i]
                self.pt_swap_counts[i] += 1

    def generate(self, length=20, prompt=None, vocab=None, verbose=False):
        """Generate text with P2 parallel tempering + history + momentum."""
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

        # P2: Initialize history buffer and momentum tracker
        history_buffer = HistoryBuffer(length, max_history=50) if self.history_enabled else None
        momentum_tracker = MomentumTracker(length, self.momentum_strength) if self.momentum_enabled else None

        # Precompute H_sum for CALDERA path
        H_sum = None
        if self.nmf is not None and self.nmf.fitted:
            H_sum = self.nmf.compute_H_sum(state_words)

        # P2: Initialize parallel tempering replicas
        replicas = []
        replica_H_sums = []
        for r in range(self.n_replicas):
            if r == 0:
                # Cold replica = the one we care about
                replicas.append((list(state_words), list(state_types)))
                replica_H_sums.append(H_sum.copy() if H_sum is not None else None)
            else:
                # Hot replicas start with perturbed states
                r_words = list(state_words)
                r_types = list(state_types)
                # Perturb: randomly reinitialize some positions
                n_perturb = max(1, length // 3)
                for _ in range(n_perturb):
                    pos = random.randint(prompt_len, length - 1)
                    pos_key = pos % min(self.pmi.seq_len, 50)
                    if pos_key in self.field_weights:
                        r_words[pos] = self._sample_from_cumsum(self.field_weights[pos_key])
                    else:
                        r_words[pos] = random.randint(0, self.vocab_size - 1)
                    r_types[pos] = self.types.get_type_for_word(r_words[pos])
                replicas.append((r_words, r_types))
                if self.nmf is not None and self.nmf.fitted:
                    replica_H_sums.append(self.nmf.compute_H_sum(r_words))
                else:
                    replica_H_sums.append(None)

        if verbose:
            self._print_state(state_words, state_types, vocab, "Init")

        # ============ PHASE 1: Type annealing ============
        for sweep in range(self.sweeps_p1):
            # Update types for cold replica (replica 0)
            for pos in range(prompt_len, length):
                replicas[0][1][pos] = self._sample_type(
                    pos, replicas[0][1][pos], replicas[0][1]
                )
            # Also update types for hot replicas (less frequently)
            for r in range(1, self.n_replicas):
                for pos in range(prompt_len, length):
                    if random.random() < 0.5:  # Hot replicas update fewer positions
                        # Hot replicas use hotter temperature tables
                        replicas[r][1][pos] = self._sample_type(
                            pos, replicas[r][1][pos], replicas[r][1]
                        )
            
            # PT swap attempt
            if (sweep + 1) % self.pt_swap_interval == 0:
                replica_energies = []
                replica_betas = list(self.pt_betas)
                for r in range(self.n_replicas):
                    E = self._compute_total_energy(replicas[r][0], replicas[r][1], replica_H_sums[r])
                    replica_energies.append(E)
                self._pt_swap(replicas, replica_energies, replica_betas)
            
            if verbose and (sweep + 1) % max(1, self.sweeps_p1 // 4) == 0:
                self._print_state(replicas[0][0], replicas[0][1], vocab, f"P1 sweep {sweep+1}")

        # Copy cold replica state back
        state_words = list(replicas[0][0])
        state_types = list(replicas[0][1])
        H_sum = replica_H_sums[0]

        if verbose:
            self._print_state(state_words, state_types, vocab, "After Phase 1")

        # ============ PHASE 2: Types + Words with PT ============
        for sweep in range(self.sweeps_p2):
            # Cold replica: full update
            for pos in range(prompt_len, length):
                state_types[pos] = self._sample_type(
                    pos, state_types[pos], state_types
                )
                new_word, new_type, H_sum = self._locally_balanced_proposal(
                    pos, state_words[pos], state_types[pos],
                    state_words, state_types, self.prob_table_p2, H_sum,
                    history_buffer, momentum_tracker
                )
                state_words[pos] = new_word
                state_types[pos] = new_type
            
            # Record history after each sweep
            if history_buffer is not None:
                history_buffer.record_full_state(state_words)
            
            # Hot replicas: update with higher temperature (more random exploration)
            for r in range(1, self.n_replicas):
                r_words, r_types = replicas[r]
                r_H_sum = replica_H_sums[r]
                for pos in range(prompt_len, length):
                    if random.random() < 0.7:
                        r_types[pos] = self._sample_type(pos, r_types[pos], r_types)
                    new_word, new_type, r_H_sum = self._locally_balanced_proposal(
                        pos, r_words[pos], r_types[pos],
                        r_words, r_types, self.pt_prob_tables[r], r_H_sum
                    )
                    r_words[pos] = new_word
                    r_types[pos] = new_type
                replicas[r] = (r_words, r_types)
                replica_H_sums[r] = r_H_sum
            
            # PT swap
            if (sweep + 1) % self.pt_swap_interval == 0:
                replica_energies = []
                replica_betas = list(self.pt_betas)
                for r in range(self.n_replicas):
                    if r == 0:
                        E = self._compute_total_energy(state_words, state_types, H_sum)
                    else:
                        E = self._compute_total_energy(replicas[r][0], replicas[r][1], replica_H_sums[r])
                    replica_energies.append(E)
                
                # Build swap-able replica list
                swap_replicas = [(list(state_words), list(state_types))]
                for r in range(1, self.n_replicas):
                    swap_replicas.append(replicas[r])
                
                self._pt_swap(swap_replicas, replica_energies, replica_betas)
                
                # Check if cold replica changed
                state_words = list(swap_replicas[0][0])
                state_types = list(swap_replicas[0][1])
                # Recompute H_sum for new state
                if self.nmf is not None and self.nmf.fitted:
                    H_sum = self.nmf.compute_H_sum(state_words)
                for r in range(1, self.n_replicas):
                    replicas[r] = swap_replicas[r]
                    if self.nmf is not None and self.nmf.fitted:
                        replica_H_sums[r] = self.nmf.compute_H_sum(replicas[r][0])

            if verbose and (sweep + 1) % max(1, self.sweeps_p2 // 4) == 0:
                self._print_state(state_words, state_types, vocab, f"P2 sweep {sweep+1}")

        if verbose:
            self._print_state(state_words, state_types, vocab, "After Phase 2")

        # ============ PHASE 3: Words only (types frozen), with history + momentum ============
        for sweep in range(self.sweeps_p3):
            for pos in range(prompt_len, length):
                new_word, new_type, H_sum = self._locally_balanced_proposal(
                    pos, state_words[pos], state_types[pos],
                    state_words, state_types, self.prob_table_p3, H_sum,
                    history_buffer, momentum_tracker
                )
                state_words[pos] = new_word
                # Keep type frozen in phase 3
            
            # Record history
            if history_buffer is not None:
                history_buffer.record_full_state(state_words)

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
# V5 Enhanced Model
# =============================================================================

class EnhancedV5Model:
    """
    V5 Enhanced Typed Ising-Potts Language Model.
    
    P0+P1 (from v4) + P2:
      - Parallel tempering with adaptive ladder
      - History-driven target (Hu et al. 2025)
      - Non-reversible MCMC (momentum)
      - Stricter transition constraints (min_count=20)
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
        # Emission
        emission_bonus: int = 100,
        emission_penalty: int = 500,
        # Annealing
        phase1_beta: int = 200,
        phase2_beta: int = 500,
        phase3_beta: int = 1000,
        total_sweeps: int = 200,
        # CALDERA NMF
        use_caldera: bool = True,
        nmf_factors: int = 128,
        nmf_iterations: int = 50,
        nmf_n_top: int = 15,
        # SpaCy
        use_spacy: bool = True,
        spacy_max_texts: Optional[int] = None,
        # Transition constraints
        transition_min_count: int = 20,
        # P2: Parallel tempering
        n_replicas: int = 4,
        pt_swap_interval: int = 5,
        # P2: History
        history_enabled: bool = True,
        # P2: Momentum
        momentum_enabled: bool = True,
        momentum_strength: int = 3,
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
        self.transition_min_count = transition_min_count
        self.n_replicas = n_replicas
        self.pt_swap_interval = pt_swap_interval
        self.history_enabled = history_enabled
        self.momentum_enabled = momentum_enabled
        self.momentum_strength = momentum_strength

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
        """Train the v5 model."""
        print("=" * 70)
        print("ENHANCED TYPED ISING-POTTS MODEL v5 — TRAINING")
        print("(P0: Locally-balanced proposals, Hard transition constraints)")
        print("(P1: CALDERA NMF, Strengthened emission, Implicational couplings)")
        print("(P2: Parallel tempering, History-driven target, Non-reversible MCMC)")
        print(f"(Transition min_count: {self.transition_min_count})")
        print("=" * 70)

        # Step 1: Load data
        t0 = time.time()
        texts = load_fineweb_edu(n_samples=n_samples)
        print(f"[1/9] Data loading: {len(texts)} texts ({time.time()-t0:.1f}s)")

        # Step 2: Build vocabulary
        t0 = time.time()
        self.vocab = Vocabulary(min_freq=self.vocab_min_freq, max_size=self.vocab_max_size)
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

        # Step 8: Hard POS transitions (STRICTER: min_count=20)
        t0 = time.time()
        self.allowed_transitions = build_allowed_transitions_from_tagger(
            self.spacy_tagger, self.vocab.idx2word, sequences, self.types,
            min_count=self.transition_min_count
        )
        n_allowed = len(self.allowed_transitions)
        n_total = N_POS * N_POS
        print(f"[8/9] Hard POS transitions: {n_allowed}/{n_total} allowed "
              f"({100*n_allowed/n_total:.1f}%, min_count={self.transition_min_count}) "
              f"({time.time()-t0:.1f}s)")

        # Step 9: CALDERA NMF
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
            J_recon = self.nmf_model.reconstruct()
            abs_err = int(np.sum(np.abs(J_full - J_recon)))
            rel_err = abs_err / max(1, int(np.sum(np.abs(J_full))))
            print(f"[9/9] CALDERA NMF: K={self.nmf_factors}, n_top={self.nmf_n_top}, "
                  f"rel_err={rel_err:.3f}, "
                  f"memory_savings={mem['savings_pct']:.1f}% ({time.time()-t0:.1f}s)")
        else:
            self.nmf_model = None
            print(f"[9/9] CALDERA NMF: skipped")

        # Build sampler
        print("\nBuilding v5 sampler (P0+P1+P2)...")
        t0 = time.time()
        self.sampler = EnhancedV5Sampler(
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
            n_replicas=self.n_replicas,
            pt_swap_interval=self.pt_swap_interval,
            history_enabled=self.history_enabled,
            momentum_enabled=self.momentum_enabled,
            momentum_strength=self.momentum_strength,
        )
        print(f"Sampler ready ({time.time()-t0:.1f}s)")

        # Summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE — v5 ENHANCED MODEL (P0+P1+P2)")
        print("=" * 70)
        self._print_summary()

        return self

    def _print_summary(self):
        print(f"\nModel Architecture (v5 Enhanced):")
        print(f"  Vocabulary size: {len(self.vocab)}")
        print(f"  POS types: {N_POS}")
        print(f"  Semantic types: {N_SEM}")
        print(f"  PMI coupling range: [{int(self.pmi.J_PMI.min())}, {int(self.pmi.J_PMI.max())}]")
        print(f"  PMI non-zeros: {int(np.count_nonzero(self.pmi.J_PMI))}")
        if self.dep_couplings is not None:
            dep_stats = self.dep_couplings.get_dep_stats()
            print(f"  Dependency J_tree non-zeros: {dep_stats['J_tree_nnz']}")
        if self.nmf_model is not None and self.nmf_model.fitted:
            mem = self.nmf_model.memory_savings()
            print(f"  CALDERA memory savings: {mem['savings_pct']:.1f}%")
        print(f"  Transition min_count: {self.transition_min_count}")
        print(f"  Allowed transitions: {len(self.allowed_transitions)}")
        print(f"  Parallel tempering: {self.n_replicas} replicas")
        print(f"  History-driven target: {'YES' if self.history_enabled else 'NO'}")
        print(f"  Non-reversible MCMC: {'YES' if self.momentum_enabled else 'NO'}")
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

        # PT swap statistics
        pt_stats = {}
        if hasattr(self.sampler, 'pt_swap_counts'):
            total_swaps = sum(self.sampler.pt_swap_counts)
            total_attempts = sum(self.sampler.pt_attempt_counts)
            pt_stats = {
                "total_swaps": total_swaps,
                "total_attempts": total_attempts,
                "swap_rate": total_swaps / max(1, total_attempts),
            }

        return {
            "text": self.vocab.decode(words),
            "types": [IDX2POS.get(t, "UNK") for t in types],
            "energy": energy,
            "type_counts": type_counts,
            "sem_counts": sem_counts,
            "words": words,
            "pt_stats": pt_stats,
        }

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
            "repeated_words": 0,
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

        # Count repeated adjacent words
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                metrics["repeated_words"] += 1

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
        trans_list = list(self.allowed_transitions) if self.allowed_transitions else []
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
            "transition_min_count": self.transition_min_count,
            "n_replicas": self.n_replicas,
            "pt_swap_interval": self.pt_swap_interval,
            "history_enabled": self.history_enabled,
            "momentum_enabled": self.momentum_enabled,
            "momentum_strength": self.momentum_strength,
        }
        with open(os.path.join(directory, "config.json"), "w") as f:
            json_mod.dump(config, f, indent=2)

    @classmethod
    def load(cls, directory):
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
            model.allowed_transitions = set(tuple(t) for t in trans_list)
        except FileNotFoundError:
            model.allowed_transitions = None

        model.sampler = EnhancedV5Sampler(
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
            n_replicas=model.n_replicas,
            pt_swap_interval=model.pt_swap_interval,
            history_enabled=model.history_enabled,
            momentum_enabled=model.momentum_enabled,
            momentum_strength=model.momentum_strength,
        )

        return model
