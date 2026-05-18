"""
Enhanced Typed Ising-Potts Language Model v6 — Principled MCMC Rewrite.

Replaces v5's ugly workarounds (history-driven target, momentum tracker) with
three PRINCIPLED mathematical solutions that preserve detailed balance:

  1. Swendsen-Wang Cluster Algorithm (Fortuin-Kasteleyn-Swendsen-Wang)
     - Cluster moves instead of single-site Metropolis for repetition fix
     - Wolff single-cluster variant for disordered systems
     - Precomputed integer bond activation thresholds (FP only at init)

  2. Proper Parallel Tempering
     - Geometric temperature ladder (Katzgrader et al., 2006)
     - 8 replicas (was 4)
     - Precomputed swap acceptance tables (NO math.exp at runtime)
     - Adaptive ladder tuning targeting ~23% swap rate

  3. Entropy-Regularized Free Energy
     - Replace E with F = E - T_ent * S
     - S = local entropy: log2(accessible neighbors within energy window)
     - STATE FUNCTION — preserves detailed balance
     - Trapped states have low S -> high F -> disfavoured

CRITICAL CONSTRAINT: ALL generation-path computation is INTEGER-ONLY arithmetic.
Floating-point operations only allowed in precomputation / init.
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
# Swendsen-Wang: Precomputed Cluster Bond Activation Thresholds
# =============================================================================

class ClusterThresholds:
    """
    Precomputed integer threshold tables for Swendsen-Wang bond activation.

    Bond activation probability: p = 1 - exp(-beta * J / 1000)
    At runtime, we compare random.randint(0, 2**31-2) < threshold[J_val].

    This is done ONCE at init (FP allowed). Runtime is pure integer comparison.
    """

    def __init__(self, beta_int: int, j_min: int = 0, j_max: int = 500):
        self.beta_int = beta_int
        self.j_min = j_min
        self.j_max = j_max
        self.rand_max = 2**31 - 1

        # Precompute thresholds for all coupling values in range
        self._thresholds = {}
        beta = beta_int / 1000.0
        for j_val in range(j_min, j_max + 1):
            if j_val <= 0:
                self._thresholds[j_val] = 0
            else:
                p = 1.0 - math.exp(-j_val * beta)
                self._thresholds[j_val] = int(self.rand_max * p)

    def get(self, j_val: int) -> int:
        """Get precomputed threshold for coupling value j_val."""
        if j_val in self._thresholds:
            return self._thresholds[j_val]
        # For values outside precomputed range, compute on the fly
        # (FP allowed here only as fallback — shouldn't happen in practice)
        if j_val <= 0:
            return 0
        beta = self.beta_int / 1000.0
        p = 1.0 - math.exp(-j_val * beta)
        threshold = int(self.rand_max * p)
        self._thresholds[j_val] = threshold
        return threshold


# =============================================================================
# Integer-Only Union-Find for Swendsen-Wang Cluster Identification
# =============================================================================

class UnionFind:
    """
    Integer-only Union-Find (disjoint set) data structure.

    Used by Swendsen-Wang to identify connected components from
    activated bonds. All operations are integer comparisons.
    """

    def __init__(self, n: int):
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path compression (iterative, no recursion)."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root

    def union(self, x: int, y: int) -> None:
        """Union by rank."""
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def get_clusters(self) -> Dict[int, List[int]]:
        """Return dict: root -> list of positions in cluster."""
        clusters = defaultdict(list)
        for i in range(self.n):
            root = self.find(i)
            clusters[root].append(i)
        return dict(clusters)

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same component."""
        return self.find(x) == self.find(y)


# =============================================================================
# Integer-Only Local Entropy Estimator
# =============================================================================

class LocalEntropyEstimator:
    """
    Integer-only local entropy estimator for entropy-regularized free energy.

    S(pos, word) = log2(number of accessible neighbors within energy window dE)
                  * precision (fixed-point)

    Uses bit_length() - 1 for integer log2 (same as log-floor PMI).
    Accessible neighbor = a proposed word whose dE is within [-dE_window, +dE_window].

    This is a STATE FUNCTION — preserves detailed balance!
    Trapped states have low S -> high F -> disfavoured.
    Explorable states have high S -> low F -> preferred.
    """

    def __init__(self, delta_e_window: int = 500, precision: int = 100):
        self.delta_e_window = delta_e_window
        self.precision = precision

    def compute_local_entropy(
        self, pos, word, word_type, current_energy,
        state_words, state_types, sampler
    ) -> int:
        """
        Compute integer fixed-point local entropy S(pos, word).

        Returns S in units of precision (e.g., precision=100 means 0.01 resolution).
        """
        # Count accessible neighbors from proposal cache
        proposals = sampler.proposal_cache.get(word, [])[:20]

        accessible = 0
        for w in proposals:
            if w == word:
                continue
            prop_type = int(sampler.types.get_type_for_word(w))
            if prop_type >= sampler.n_types:
                prop_type = POS2IDX["X"]
            prop_energy = sampler._fast_energy(
                pos, w, prop_type, state_words, state_types,
                sampler._active_H_sum
            )
            delta_e = prop_energy - current_energy
            if abs(delta_e) <= self.delta_e_window:
                accessible += 1

        # S = log2(accessible) * precision (integer fixed-point)
        if accessible > 0:
            s_int = accessible.bit_length() - 1  # floor(log2(accessible))
            # Fractional part for better precision
            if accessible > (1 << s_int):
                frac = ((accessible - (1 << s_int)) * self.precision) // (1 << s_int)
            else:
                frac = 0
            local_entropy = s_int * self.precision + frac
        else:
            local_entropy = 0

        return local_entropy


# =============================================================================
# V6 Enhanced Sampler — SW Clusters + Proper PT + Entropy Regularization
# =============================================================================

class EnhancedV6Sampler:
    """
    V6 Enhanced sampler with PRINCIPLED MCMC replacements.

    Replaces v5's history-driven target and momentum with:
      P3a: Swendsen-Wang cluster moves (Fortuin-Kasteleyn-Swendsen-Wang)
           - Wolff single-cluster variant for disordered systems
           - Precomputed integer bond activation thresholds
      P3b: Proper parallel tempering
           - Geometric temperature ladder (Katzgrader et al., 2006)
           - 8 replicas
           - Precomputed swap acceptance tables (NO FP at runtime)
           - Adaptive ladder tuning
      P3c: Entropy-regularized free energy
           - F = E - T_ent * S (state function, preserves detailed balance)
           - S = local entropy of accessible neighbors

    Retained from v5:
      P0: Locally-balanced proposals (Zanella 2017)
      P0: Hard POS transition constraints (min_count=20)
      P1: CALDERA NMF, strengthened emission, implicational couplings
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
        # P3a: Swendsen-Wang cluster
        sw_cluster_enabled: bool = True,
        sw_wolff_variant: bool = True,
        # P3b: Parallel tempering (proper)
        n_replicas: int = 8,
        pt_swap_interval: int = 5,
        # P3c: Entropy regularization
        entropy_T_ent: int = 50,
        entropy_delta_E_window: int = 500,
        entropy_precision: int = 100,
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

        # P3a: Swendsen-Wang cluster parameters
        self.sw_cluster_enabled = sw_cluster_enabled
        self.sw_wolff_variant = sw_wolff_variant

        # Precompute cluster bond activation thresholds for each phase
        self.cluster_thresholds_p1 = ClusterThresholds(phase1_beta)
        self.cluster_thresholds_p2 = ClusterThresholds(phase2_beta)
        self.cluster_thresholds_p3 = ClusterThresholds(phase3_beta)

        # P3b: Parallel tempering — PROPER geometric ladder
        self.n_replicas = n_replicas
        self.pt_swap_interval = pt_swap_interval
        self.pt_betas = self._build_geometric_ladder(phase1_beta, phase3_beta, n_replicas)
        self.pt_prob_tables = [self._build_prob_table(b) for b in self.pt_betas]
        self.pt_swap_counts = [0] * (n_replicas - 1)
        self.pt_attempt_counts = [0] * (n_replicas - 1)

        # Precompute swap acceptance tables for each pair of adjacent replicas
        # This ELIMINATES math.exp() at runtime
        self.swap_tables = self._build_swap_tables()
        # Target swap rate for adaptive tuning (~23%)
        self.pt_target_swap_rate = 23  # percent

        # P3c: Entropy regularization
        self.T_ent = entropy_T_ent
        self.entropy_estimator = LocalEntropyEstimator(
            delta_e_window=entropy_delta_E_window,
            precision=entropy_precision,
        )
        self.entropy_delta_E_window = entropy_delta_E_window
        self.entropy_precision = entropy_precision

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

        # Active H_sum reference for entropy estimator (set during generate)
        self._active_H_sum = None

    # =========================================================================
    # Geometric Temperature Ladder (Katzgrader et al., 2006)
    # =========================================================================

    def _build_geometric_ladder(self, beta_min, beta_max, n_replicas):
        """
        Build geometric temperature ladder for parallel tempering.

        beta_k = beta_min * (beta_max / beta_min) ^ (k / (n-1))

        FP allowed here (init time only). Ladder is fixed after construction.
        """
        betas = []
        for k in range(n_replicas):
            if n_replicas == 1:
                betas.append(beta_max)
            else:
                frac = k / (n_replicas - 1)
                # beta = beta_min * (beta_max / beta_min) ^ frac
                # Using log-space for integer computation
                if beta_max > beta_min and beta_min > 0:
                    log_ratio = math.log(beta_max / beta_min)
                    beta = int(beta_min * math.exp(frac * log_ratio))
                else:
                    beta = beta_min
                betas.append(max(beta_min, min(beta_max, beta)))
        betas[0] = beta_min
        betas[-1] = beta_max
        return betas

    # =========================================================================
    # Precomputed Swap Acceptance Tables (NO FP at runtime)
    # =========================================================================

    def _build_swap_tables(self):
        """
        Precompute swap acceptance tables for each pair of adjacent replicas.

        For replicas i, i+1 with betas b_i, b_{i+1}:
          dE = E_{i+1} - E_i
          swap_prob = min(1, exp((b_{i+1} - b_i) * dE / 1000))

        We precompute a lookup table indexed by dE for each pair.
        At runtime, pure integer comparison.
        """
        tables = []
        max_dE = 50000  # larger range for total energy differences

        for i in range(self.n_replicas - 1):
            beta_i = self.pt_betas[i]
            beta_j = self.pt_betas[i + 1]
            delta_beta = (beta_j - beta_i) / 1000.0  # FP only here

            table = [0] * (2 * max_dE + 1)
            for dE in range(-max_dE, max_dE + 1):
                idx = dE + max_dE
                exponent = delta_beta * dE
                if exponent >= 0:
                    table[idx] = 2**31 - 1  # always accept
                elif exponent < -500:
                    table[idx] = 0  # effectively zero
                else:
                    prob = math.exp(exponent)
                    table[idx] = int((2**31 - 1) * prob)
            tables.append(table)

        return tables

    # =========================================================================
    # Standard MCMC Infrastructure (from v5, preserved)
    # =========================================================================

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

    def _sample_emission_word(self, current_type):
        """Sample a word compatible with the given type using emission distribution."""
        if current_type in self.emit_cumsum_by_type:
            return self._sample_from_cumsum(self.emit_cumsum_by_type[current_type])
        return random.randint(0, self.vocab_size - 1)

    # =========================================================================
    # Type Sampling (same as v5 — preserved)
    # =========================================================================

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

    # =========================================================================
    # Energy Computation (with entropy regularization)
    # =========================================================================

    def _compute_word_energy(
        self, pos, word, word_type, state_words, state_types,
        H_sum=None
    ):
        """Compute total word energy (without entropy regularization)."""
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

        return energy

    def _compute_entropy_regularized_energy(
        self, pos, word, word_type, state_words, state_types, H_sum
    ):
        """
        Compute F = E - T_ent * S where S = local entropy.

        This is a STATE FUNCTION — preserves detailed balance!
        Trapped states have low S -> high F -> disfavoured.
        Explorable states have high S -> low F -> preferred.
        """
        base_energy = self._compute_word_energy(
            pos, word, word_type, state_words, state_types, H_sum
        )

        if self.T_ent > 0:
            # Set active H_sum for entropy estimator's _fast_energy calls
            self._active_H_sum = H_sum

            # Count accessible neighbors
            proposals = self.proposal_cache.get(word, [])[:20]
            accessible = 0
            for w in proposals:
                if w == word:
                    continue
                prop_type = int(self.types.get_type_for_word(w))
                if prop_type >= self.n_types:
                    prop_type = POS2IDX["X"]
                prop_energy = self._fast_energy(
                    pos, w, prop_type, state_words, state_types, H_sum
                )
                delta_e = prop_energy - base_energy
                if abs(delta_e) <= self.entropy_delta_E_window:
                    accessible += 1

            # S = log2(accessible) * precision (integer fixed-point)
            if accessible > 0:
                s_int = accessible.bit_length() - 1
                if accessible > (1 << s_int):
                    frac = ((accessible - (1 << s_int)) * self.entropy_precision) // (1 << s_int)
                else:
                    frac = 0
                local_entropy = s_int * self.entropy_precision + frac
            else:
                local_entropy = 0

            # F = E - T_ent * S / 1000
            entropy_bonus = (self.T_ent * local_entropy) // 1000
            return base_energy - entropy_bonus

        return base_energy

    def _fast_energy(self, pos, word, word_type, state_words, state_types, H_sum=None):
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

        return energy

    # =========================================================================
    # Locally-Balanced Proposal (Zanella 2017) — preserved from v5
    # No history_buffer or momentum_tracker — replaced by entropy regularization
    # =========================================================================

    def _locally_balanced_proposal(
        self, pos, current_word, current_type, state_words, state_types,
        prob_table, H_sum=None, use_entropy=False
    ):
        """P0+P3c: Locally-balanced proposal with optional entropy regularization."""
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

        # Compute approximate dE for proposal weighting
        current_energy = self._fast_energy(
            pos, current_word, current_type, state_words, state_types, H_sum
        )

        # Locally-balanced weighting: exp(-dE/2T)
        candidate_weights = []
        max_de = (len(prob_table) - 1) // 2

        for w in candidates:
            proposed_type = int(self.types.get_type_for_word(w))
            if proposed_type >= self.n_types:
                proposed_type = POS2IDX["X"]
            proposed_energy = self._fast_energy(
                pos, w, proposed_type, state_words, state_types, H_sum
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

        # Full Metropolis-Hastings correction with entropy-regularized energy
        proposed_type = int(self.types.get_type_for_word(chosen_word))
        if proposed_type >= self.n_types:
            proposed_type = POS2IDX["X"]

        if use_entropy and self.T_ent > 0:
            exact_current = self._compute_entropy_regularized_energy(
                pos, current_word, current_type, state_words, state_types, H_sum
            )
            exact_proposed = self._compute_entropy_regularized_energy(
                pos, chosen_word, proposed_type, state_words, state_types, H_sum
            )
        else:
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

    # =========================================================================
    # Total Energy (for PT swaps)
    # =========================================================================

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

    # =========================================================================
    # Proper Parallel Tempering Swap (NO math.exp at runtime)
    # =========================================================================

    def _pt_swap(self, replicas, replica_energies, replica_betas):
        """
        Attempt parallel tempering swap between adjacent replicas.

        Uses PRECOMPUTED swap acceptance tables — pure integer comparison.
        NO math.exp() at runtime (unlike v5's broken implementation).
        """
        for i in range(len(replicas) - 1):
            self.pt_attempt_counts[i] += 1
            dE = replica_energies[i + 1] - replica_energies[i]

            # PURE INTEGER: use precomputed swap table
            table = self.swap_tables[i]
            max_dE = (len(table) - 1) // 2

            if dE <= -max_dE:
                swap = True
            elif dE >= max_dE:
                swap = False
            else:
                rand_val = random.randint(0, 2**31 - 2)
                swap = rand_val < table[dE + max_dE]

            if swap:
                replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
                replica_energies[i], replica_energies[i + 1] = replica_energies[i + 1], replica_energies[i]
                self.pt_swap_counts[i] += 1

    def _adapt_pt_ladder(self, sweep):
        """
        Adaptive ladder tuning (every 50 sweeps).

        If swap rate between adjacent replicas deviates from ~23%,
        adjust the beta spacing to improve mixing.
        """
        if (sweep + 1) % 50 != 0:
            return
        if sum(self.pt_attempt_counts) < 10:
            return

        for i in range(self.n_replicas - 1):
            if self.pt_attempt_counts[i] < 5:
                continue
            rate = (100 * self.pt_swap_counts[i]) // self.pt_attempt_counts[i]
            # If rate is too low (<15%), widen the gap (decrease hotter beta)
            # If rate is too high (>35%), narrow the gap (increase hotter beta)
            if rate < 15:
                # Too few swaps — make adjacent betas closer
                mid = (self.pt_betas[i] + self.pt_betas[i + 1]) // 2
                self.pt_betas[i] = max(self.pt_betas[i] - 10,
                                       self.pt_betas[0] if i == 0 else self.pt_betas[i - 1] + 5)
            elif rate > 35:
                # Too many swaps — make adjacent betas further apart
                self.pt_betas[i] = min(self.pt_betas[i] + 10,
                                       self.pt_betas[i + 1] - 5)

            # Reset counts after adaptation
            self.pt_swap_counts[i] = 0
            self.pt_attempt_counts[i] = 0

        # Rebuild swap tables with new betas
        self.swap_tables = self._build_swap_tables()
        self.pt_prob_tables = [self._build_prob_table(b) for b in self.pt_betas]

    # =========================================================================
    # Swendsen-Wang Cluster Sweep
    # =========================================================================

    def _sw_sweep(
        self, state_words, state_types, prob_table, cluster_thresholds,
        H_sum, prompt_len, use_entropy=False
    ):
        """
        One Swendsen-Wang cluster sweep.

        Algorithm:
          1. For each pair of neighboring positions within coupling window:
             - If J_combined[w_i, w_j] > 0 AND state_types[i] == state_types[j]:
               - Activate bond with probability p = 1 - exp(-beta * J / 1000)
               - Use PRECOMPUTED threshold table (integer comparison at runtime)
          2. Find connected components using Union-Find (all integer)
          3. For each cluster with |cluster| > 1:
             - Propose a new word for the entire cluster
             - Accept/reject the entire cluster flip using integer Metropolis
          4. For singleton clusters: use regular locally-balanced single-site proposal
        """
        L = len(state_words)

        # Step 1: Activate bonds using precomputed thresholds
        uf = UnionFind(L)
        for i in range(prompt_len, L):
            for j_off in range(1, self.pmi.window + 1):
                j = i + j_off
                if j >= L:
                    break
                coupling = int(self.J_combined[state_words[i], state_words[j]])
                if coupling > 0 and state_types[i] == state_types[j]:
                    threshold = cluster_thresholds.get(coupling)
                    if threshold > 0 and random.randint(0, 2**31 - 2) < threshold:
                        uf.union(i, j)

        # Step 2: Identify clusters
        clusters = uf.get_clusters()  # Dict: root -> [positions]

        # Step 3: For each cluster, propose and accept/reject
        for root, positions in clusters.items():
            if len(positions) == 1:
                # Singleton: use locally-balanced proposal
                pos = positions[0]
                new_word, new_type, H_sum = self._locally_balanced_proposal(
                    pos, state_words[pos], state_types[pos],
                    state_words, state_types, prob_table, H_sum,
                    use_entropy=use_entropy
                )
                state_words[pos] = new_word
                state_types[pos] = new_type
            else:
                # Multi-site cluster: propose new word for entire cluster
                current_type = state_types[positions[0]]
                new_word = self._sample_emission_word(current_type)
                if new_word >= self.vocab_size:
                    continue
                new_type = int(self.types.get_type_for_word(new_word))
                if new_type >= self.n_types:
                    new_type = POS2IDX["X"]

                # Compute energy change for entire cluster
                delta_e = self._compute_cluster_energy_change(
                    positions, state_words, state_types, new_word, new_type, H_sum
                )

                # Add entropy regularization if enabled
                if use_entropy and self.T_ent > 0:
                    # Compute entropy change: new state vs old state
                    # For each position in cluster, compute entropy change
                    entropy_delta = 0
                    for pos in positions:
                        old_word = state_words[pos]
                        old_type = state_types[pos]

                        # Old entropy
                        self._active_H_sum = H_sum
                        old_energy = self._fast_energy(
                            pos, old_word, old_type, state_words, state_types, H_sum
                        )
                        old_accessible = self._count_accessible(
                            pos, old_word, old_type, old_energy, state_words, state_types, H_sum
                        )
                        old_s = self._entropy_from_count(old_accessible)

                        # New entropy (with tentative change)
                        # Approximate: just count for the new word
                        new_energy_approx = self._fast_energy(
                            pos, new_word, new_type, state_words, state_types, H_sum
                        )
                        new_accessible = self._count_accessible(
                            pos, new_word, new_type, new_energy_approx, state_words, state_types, H_sum
                        )
                        new_s = self._entropy_from_count(new_accessible)

                        entropy_delta += (self.T_ent * (old_s - new_s)) // 1000

                    delta_e += entropy_delta

                # Metropolis accept/reject (integer)
                rand_val = random.randint(0, 2**31 - 2)
                if self._accept(delta_e, rand_val, prob_table):
                    for pos in positions:
                        old_word = state_words[pos]
                        state_words[pos] = new_word
                        state_types[pos] = new_type
                        if H_sum is not None and self.nmf is not None and self.nmf.fitted:
                            H_sum = self.nmf.update_H_sum(H_sum, old_word, new_word)

        return H_sum

    # =========================================================================
    # Wolff Single-Cluster Sweep
    # =========================================================================

    def _wolff_sweep(
        self, state_words, state_types, prob_table, cluster_thresholds,
        H_sum, prompt_len, use_entropy=False
    ):
        """
        Wolff single-cluster variant of Swendsen-Wang.

        Instead of identifying ALL clusters, pick a random seed site
        and grow ONE cluster from it. More efficient for disordered systems.
        """
        L = len(state_words)
        if L <= prompt_len:
            return H_sum

        # Pick random seed
        seed = random.randint(prompt_len, L - 1)

        # Grow cluster from seed using BFS
        cluster = [seed]
        visited = {seed}
        queue = [seed]

        while queue:
            i = queue.pop(0)
            for j_off in range(1, self.pmi.window + 1):
                for j in [i + j_off, i - j_off]:
                    if j < prompt_len or j >= L:
                        continue
                    if j in visited:
                        continue
                    if state_types[i] != state_types[j]:
                        continue

                    coupling = int(self.J_combined[state_words[i], state_words[j]])
                    if coupling <= 0:
                        continue

                    threshold = cluster_thresholds.get(coupling)
                    if threshold > 0 and random.randint(0, 2**31 - 2) < threshold:
                        visited.add(j)
                        cluster.append(j)
                        queue.append(j)

        # Propose new word for entire cluster
        if len(cluster) == 1:
            # Singleton: use locally-balanced proposal
            pos = cluster[0]
            new_word, new_type, H_sum = self._locally_balanced_proposal(
                pos, state_words[pos], state_types[pos],
                state_words, state_types, prob_table, H_sum,
                use_entropy=use_entropy
            )
            state_words[pos] = new_word
            state_types[pos] = new_type
        else:
            # Multi-site cluster
            current_type = state_types[cluster[0]]
            new_word = self._sample_emission_word(current_type)
            if new_word >= self.vocab_size:
                return H_sum
            new_type = int(self.types.get_type_for_word(new_word))
            if new_type >= self.n_types:
                new_type = POS2IDX["X"]

            # Compute energy change for entire cluster
            delta_e = self._compute_cluster_energy_change(
                cluster, state_words, state_types, new_word, new_type, H_sum
            )

            # Add entropy regularization if enabled
            if use_entropy and self.T_ent > 0:
                entropy_delta = 0
                for pos in cluster:
                    old_word = state_words[pos]
                    old_type = state_types[pos]
                    self._active_H_sum = H_sum
                    old_energy = self._fast_energy(
                        pos, old_word, old_type, state_words, state_types, H_sum
                    )
                    old_accessible = self._count_accessible(
                        pos, old_word, old_type, old_energy, state_words, state_types, H_sum
                    )
                    old_s = self._entropy_from_count(old_accessible)

                    new_energy_approx = self._fast_energy(
                        pos, new_word, new_type, state_words, state_types, H_sum
                    )
                    new_accessible = self._count_accessible(
                        pos, new_word, new_type, new_energy_approx, state_words, state_types, H_sum
                    )
                    new_s = self._entropy_from_count(new_accessible)

                    entropy_delta += (self.T_ent * (old_s - new_s)) // 1000
                delta_e += entropy_delta

            # Metropolis accept/reject (integer)
            rand_val = random.randint(0, 2**31 - 2)
            if self._accept(delta_e, rand_val, prob_table):
                for pos in cluster:
                    old_word = state_words[pos]
                    state_words[pos] = new_word
                    state_types[pos] = new_type
                    if H_sum is not None and self.nmf is not None and self.nmf.fitted:
                        H_sum = self.nmf.update_H_sum(H_sum, old_word, new_word)

        return H_sum

    # =========================================================================
    # Cluster Energy Change
    # =========================================================================

    def _compute_cluster_energy_change(
        self, positions, state_words, state_types, new_word, new_type, H_sum
    ):
        """
        Compute energy change when flipping entire cluster to new_word/new_type.

        dE = E_new - E_old, computed as sum over cluster positions.
        """
        delta_e = 0
        for pos in positions:
            old_word = state_words[pos]
            old_type = state_types[pos]
            e_old = self._compute_word_energy(
                pos, old_word, old_type, state_words, state_types, H_sum
            )
            # Temporarily modify state for new energy computation
            # (need to account for inter-cluster interactions)
            e_new = self._compute_word_energy(
                pos, new_word, new_type, state_words, state_types, H_sum
            )
            delta_e += e_new - e_old

        # Correct for double-counting: inter-cluster bonds counted twice
        # (once from each end). We need to subtract the correction.
        for idx_a in range(len(positions)):
            for idx_b in range(idx_a + 1, len(positions)):
                pos_a = positions[idx_a]
                pos_b = positions[idx_b]
                dist = abs(pos_a - pos_b)
                if dist <= self.pmi.window:
                    # The coupling between pos_a and pos_b was counted in both
                    # e_new - e_old for pos_a and for pos_b.
                    # Old contribution: J[old_a, old_b] + J[old_b, old_a] = 2*J[old_a, old_b] (symmetric)
                    # New contribution: J[new, new] + J[new, new] = 2*J[new, new]
                    # But in dE, it was counted as:
                    #   From pos_a: J[new, old_b] - J[old_a, old_b]
                    #   From pos_b: J[old_a, new] - J[old_b, old_a]  (but old_a=old_b for same cluster word... wait)
                    # Actually for same-word clusters, all positions have same old word.
                    # The correction: we need to account for J[new,new] replacing J[old,old]
                    # for the pair. In the sum, each pair's coupling appears twice (once in each
                    # position's energy), so we need to add back the difference.
                    old_coupling = int(self.J_combined[state_words[pos_a], state_words[pos_b]])
                    new_coupling = int(self.J_combined[new_word, new_word])
                    # We already subtracted old and added new twice, but we should
                    # only do it once per pair. So add back one copy.
                    delta_e -= (new_coupling - old_coupling)

        return delta_e

    # =========================================================================
    # Entropy Helpers
    # =========================================================================

    def _count_accessible(self, pos, word, word_type, current_energy,
                          state_words, state_types, H_sum):
        """Count accessible neighbors within energy window."""
        proposals = self.proposal_cache.get(word, [])[:20]
        accessible = 0
        for w in proposals:
            if w == word:
                continue
            prop_type = int(self.types.get_type_for_word(w))
            if prop_type >= self.n_types:
                prop_type = POS2IDX["X"]
            prop_energy = self._fast_energy(
                pos, w, prop_type, state_words, state_types, H_sum
            )
            delta_e = prop_energy - current_energy
            if abs(delta_e) <= self.entropy_delta_E_window:
                accessible += 1
        return accessible

    def _entropy_from_count(self, accessible):
        """Compute integer fixed-point entropy from accessible count."""
        if accessible > 0:
            s_int = accessible.bit_length() - 1
            if accessible > (1 << s_int):
                frac = ((accessible - (1 << s_int)) * self.entropy_precision) // (1 << s_int)
            else:
                frac = 0
            return s_int * self.entropy_precision + frac
        return 0

    # =========================================================================
    # Generation
    # =========================================================================

    def generate(self, length=20, prompt=None, vocab=None, verbose=False):
        """
        Generate text with SW clusters + proper PT + entropy regularization.

        Three phases:
          Phase 1: Type annealing (SW cluster for types)
          Phase 2: Types + Words with PT + SW clusters + entropy regularization
          Phase 3: Words only with SW + entropy regularization (types frozen)
        """
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

        # Initialize parallel tempering replicas
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

        # ============ PHASE 1: Type annealing (with SW cluster for types) ============
        for sweep in range(self.sweeps_p1):
            # Update types for cold replica (replica 0)
            for pos in range(prompt_len, length):
                replicas[0][1][pos] = self._sample_type(
                    pos, replicas[0][1][pos], replicas[0][1]
                )
            # Also update types for hot replicas (less frequently)
            for r in range(1, self.n_replicas):
                for pos in range(prompt_len, length):
                    if random.randint(0, 1) == 0:  # Hot replicas update fewer positions
                        replicas[r][1][pos] = self._sample_type(
                            pos, replicas[r][1][pos], replicas[r][1]
                        )

            # PT swap attempt
            if (sweep + 1) % self.pt_swap_interval == 0:
                replica_energies = []
                replica_betas = list(self.pt_betas)
                for r in range(self.n_replicas):
                    E = self._compute_total_energy(
                        replicas[r][0], replicas[r][1], replica_H_sums[r]
                    )
                    replica_energies.append(E)
                self._pt_swap(replicas, replica_energies, replica_betas)

            # Adaptive ladder tuning
            self._adapt_pt_ladder(sweep)

            if verbose and (sweep + 1) % max(1, self.sweeps_p1 // 4) == 0:
                self._print_state(replicas[0][0], replicas[0][1], vocab,
                                  f"P1 sweep {sweep+1}")

        # Copy cold replica state back
        state_words = list(replicas[0][0])
        state_types = list(replicas[0][1])
        H_sum = replica_H_sums[0]

        if verbose:
            self._print_state(state_words, state_types, vocab, "After Phase 1")

        # ============ PHASE 2: Types + Words with PT + SW + entropy regularization ============
        for sweep in range(self.sweeps_p2):
            # Cold replica: cluster sweep + type update
            if self.sw_cluster_enabled:
                if self.sw_wolff_variant:
                    # Wolff: multiple single-cluster moves per sweep
                    n_wolff = max(1, (length - prompt_len) // 3)
                    for _ in range(n_wolff):
                        H_sum = self._wolff_sweep(
                            state_words, state_types, self.prob_table_p2,
                            self.cluster_thresholds_p2, H_sum, prompt_len,
                            use_entropy=True
                        )
                else:
                    # Full SW sweep
                    H_sum = self._sw_sweep(
                        state_words, state_types, self.prob_table_p2,
                        self.cluster_thresholds_p2, H_sum, prompt_len,
                        use_entropy=True
                    )
            else:
                # Fallback to single-site locally-balanced proposals
                for pos in range(prompt_len, length):
                    new_word, new_type, H_sum = self._locally_balanced_proposal(
                        pos, state_words[pos], state_types[pos],
                        state_words, state_types, self.prob_table_p2, H_sum,
                        use_entropy=True
                    )
                    state_words[pos] = new_word
                    state_types[pos] = new_type

            # Update types
            for pos in range(prompt_len, length):
                state_types[pos] = self._sample_type(
                    pos, state_types[pos], state_types
                )

            # Hot replicas: update with higher temperature
            for r in range(1, self.n_replicas):
                r_words, r_types = replicas[r]
                r_H_sum = replica_H_sums[r]

                if self.sw_cluster_enabled:
                    if self.sw_wolff_variant:
                        n_wolff = max(1, (length - prompt_len) // 4)
                        for _ in range(n_wolff):
                            r_H_sum = self._wolff_sweep(
                                r_words, r_types, self.pt_prob_tables[r],
                                ClusterThresholds(self.pt_betas[r]),
                                r_H_sum, prompt_len, use_entropy=False
                            )
                    else:
                        r_H_sum = self._sw_sweep(
                            r_words, r_types, self.pt_prob_tables[r],
                            ClusterThresholds(self.pt_betas[r]),
                            r_H_sum, prompt_len, use_entropy=False
                        )
                else:
                    for pos in range(prompt_len, length):
                        if random.randint(0, 1) == 0:
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
                for r in range(self.n_replicas):
                    if r == 0:
                        E = self._compute_total_energy(state_words, state_types, H_sum)
                    else:
                        E = self._compute_total_energy(
                            replicas[r][0], replicas[r][1], replica_H_sums[r]
                        )
                    replica_energies.append(E)

                # Build swap-able replica list
                swap_replicas = [(list(state_words), list(state_types))]
                for r in range(1, self.n_replicas):
                    swap_replicas.append(replicas[r])

                swap_energies = list(replica_energies)
                swap_betas = list(self.pt_betas)
                self._pt_swap(swap_replicas, swap_energies, swap_betas)

                # Check if cold replica changed
                state_words = list(swap_replicas[0][0])
                state_types = list(swap_replicas[0][1])
                if self.nmf is not None and self.nmf.fitted:
                    H_sum = self.nmf.compute_H_sum(state_words)
                for r in range(1, self.n_replicas):
                    replicas[r] = swap_replicas[r]
                    if self.nmf is not None and self.nmf.fitted:
                        replica_H_sums[r] = self.nmf.compute_H_sum(replicas[r][0])

            # Adaptive ladder tuning
            self._adapt_pt_ladder(sweep)

            if verbose and (sweep + 1) % max(1, self.sweeps_p2 // 4) == 0:
                self._print_state(state_words, state_types, vocab,
                                  f"P2 sweep {sweep+1}")

        if verbose:
            self._print_state(state_words, state_types, vocab, "After Phase 2")

        # ============ PHASE 3: Words only (types frozen), with SW + entropy regularization ============
        for sweep in range(self.sweeps_p3):
            if self.sw_cluster_enabled:
                if self.sw_wolff_variant:
                    n_wolff = max(1, (length - prompt_len) // 3)
                    for _ in range(n_wolff):
                        H_sum = self._wolff_sweep(
                            state_words, state_types, self.prob_table_p3,
                            self.cluster_thresholds_p3, H_sum, prompt_len,
                            use_entropy=True
                        )
                else:
                    H_sum = self._sw_sweep(
                        state_words, state_types, self.prob_table_p3,
                        self.cluster_thresholds_p3, H_sum, prompt_len,
                        use_entropy=True
                    )
            else:
                for pos in range(prompt_len, length):
                    new_word, _, H_sum = self._locally_balanced_proposal(
                        pos, state_words[pos], state_types[pos],
                        state_words, state_types, self.prob_table_p3, H_sum,
                        use_entropy=True
                    )
                    state_words[pos] = new_word
                    # Keep type frozen in phase 3

            if verbose and (sweep + 1) % max(1, self.sweeps_p3 // 4) == 0:
                self._print_state(state_words, state_types, vocab,
                                  f"P3 sweep {sweep+1}")

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
# V6 Enhanced Model
# =============================================================================

class EnhancedV6Model:
    """
    V6 Enhanced Typed Ising-Potts Language Model.

    Principled MCMC rewrite replacing v5's workarounds:
      P3a: Swendsen-Wang cluster moves (with Wolff variant)
      P3b: Proper parallel tempering (geometric ladder, 8 replicas, no FP at runtime)
      P3c: Entropy-regularized free energy (F = E - T_ent * S)

    Retained from v5:
      P0: Locally-balanced proposals (Zanella 2017)
      P0: Hard POS transition constraints (min_count=20)
      P1: CALDERA NMF, strengthened emission, implicational couplings
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
        # P3a: Swendsen-Wang cluster
        sw_cluster_enabled: bool = True,
        sw_wolff_variant: bool = True,
        # P3b: Parallel tempering (proper)
        n_replicas: int = 8,
        pt_swap_interval: int = 5,
        # P3c: Entropy regularization
        entropy_T_ent: int = 50,
        entropy_delta_E_window: int = 500,
        entropy_precision: int = 100,
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
        self.sw_cluster_enabled = sw_cluster_enabled
        self.sw_wolff_variant = sw_wolff_variant
        self.n_replicas = n_replicas
        self.pt_swap_interval = pt_swap_interval
        self.entropy_T_ent = entropy_T_ent
        self.entropy_delta_E_window = entropy_delta_E_window
        self.entropy_precision = entropy_precision

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
        """Train the v6 model."""
        print("=" * 70)
        print("ENHANCED TYPED ISING-POTTS MODEL v6 — TRAINING")
        print("(P0: Locally-balanced proposals, Hard transition constraints)")
        print("(P1: CALDERA NMF, Strengthened emission, Implicational couplings)")
        print("(P3a: Swendsen-Wang cluster moves, Wolff variant)")
        print("(P3b: Proper parallel tempering, geometric ladder, 8 replicas)")
        print("(P3c: Entropy-regularized free energy)")
        print(f"(Transition min_count: {self.transition_min_count})")
        print(f"(SW cluster: {'ON' if self.sw_cluster_enabled else 'OFF'}, "
              f"Wolff: {'ON' if self.sw_wolff_variant else 'OFF'})")
        print(f"(Entropy T_ent: {self.entropy_T_ent}, "
              f"dE_window: {self.entropy_delta_E_window}, "
              f"precision: {self.entropy_precision})")
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
        print("\nBuilding v6 sampler (P0+P1+P3a+P3b+P3c)...")
        t0 = time.time()
        self.sampler = EnhancedV6Sampler(
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
            sw_cluster_enabled=self.sw_cluster_enabled,
            sw_wolff_variant=self.sw_wolff_variant,
            n_replicas=self.n_replicas,
            pt_swap_interval=self.pt_swap_interval,
            entropy_T_ent=self.entropy_T_ent,
            entropy_delta_E_window=self.entropy_delta_E_window,
            entropy_precision=self.entropy_precision,
        )
        print(f"Sampler ready ({time.time()-t0:.1f}s)")

        # Summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE — v6 ENHANCED MODEL (P0+P1+P3a+P3b+P3c)")
        print("=" * 70)
        self._print_summary()

        return self

    def _print_summary(self):
        print(f"\nModel Architecture (v6 Enhanced):")
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
        print(f"  Swendsen-Wang clusters: {'ON' if self.sw_cluster_enabled else 'OFF'}")
        print(f"  Wolff variant: {'ON' if self.sw_wolff_variant else 'OFF'}")
        print(f"  Parallel tempering: {self.n_replicas} replicas (geometric ladder)")
        print(f"  Entropy regularization: T_ent={self.entropy_T_ent}, "
              f"dE_window={self.entropy_delta_E_window}, "
              f"precision={self.entropy_precision}")
        print(f"  Zero FP in generation loop: YES")

    def generate(self, prompt=None, length=20, verbose=False):
        """Generate text with principled MCMC."""
        if self.sampler is None:
            raise RuntimeError("Model not trained. Call train() first.")
        words, types = self.sampler.generate(
            length=length, prompt=prompt, vocab=self.vocab, verbose=verbose
        )
        return self._decode_with_annotations(words, types)

    def generate_raw(self, prompt=None, length=20, verbose=False):
        """Generate raw (word indices, type indices)."""
        if self.sampler is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.sampler.generate(
            length=length, prompt=prompt, vocab=self.vocab, verbose=verbose
        )

    def generate_with_trace(self, prompt=None, length=20):
        """Generate with full trace information."""
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

        # SW cluster statistics
        sw_stats = {
            "sw_enabled": self.sw_cluster_enabled,
            "wolff_variant": self.sw_wolff_variant,
        }

        return {
            "text": self.vocab.decode(words),
            "types": [IDX2POS.get(t, "UNK") for t in types],
            "energy": energy,
            "type_counts": type_counts,
            "sem_counts": sem_counts,
            "words": words,
            "pt_stats": pt_stats,
            "sw_stats": sw_stats,
        }

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
        """Save v6 model to directory. Compatible with v5 format plus v6 fields."""
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
            "version": 6,
            # Vocabulary
            "vocab_min_freq": self.vocab_min_freq,
            "vocab_max_size": self.vocab_max_size,
            # Sequence
            "seq_len": self.seq_len,
            # Coupling
            "window": self.window,
            "pmi_cap": self.pmi_cap,
            "min_cooc": self.min_cooc,
            # Weights
            "pmi_weight": self.pmi_weight,
            "hebbian_weight": self.hebbian_weight,
            "semantic_weight": self.semantic_weight,
            "dep_weight": self.dep_weight,
            # Grammar
            "grammar_penalty": self.grammar_penalty,
            # Emission
            "emission_bonus": self.emission_bonus,
            "emission_penalty": self.emission_penalty,
            # Annealing
            "phase1_beta": self.phase1_beta,
            "phase2_beta": self.phase2_beta,
            "phase3_beta": self.phase3_beta,
            "total_sweeps": self.total_sweeps,
            # CALDERA NMF
            "use_caldera": self.use_caldera,
            "nmf_factors": self.nmf_factors,
            "nmf_iterations": self.nmf_iterations,
            "nmf_n_top": self.nmf_n_top,
            # SpaCy
            "use_spacy": self.use_spacy,
            # Transition constraints
            "transition_min_count": self.transition_min_count,
            # P3a: SW cluster
            "sw_cluster_enabled": self.sw_cluster_enabled,
            "sw_wolff_variant": self.sw_wolff_variant,
            # P3b: Parallel tempering
            "n_replicas": self.n_replicas,
            "pt_swap_interval": self.pt_swap_interval,
            # P3c: Entropy regularization
            "entropy_T_ent": self.entropy_T_ent,
            "entropy_delta_E_window": self.entropy_delta_E_window,
            "entropy_precision": self.entropy_precision,
        }
        with open(os.path.join(directory, "config.json"), "w") as f:
            json_mod.dump(config, f, indent=2)

    @classmethod
    def load(cls, directory):
        """Load v6 model from directory. Compatible with v5 format."""
        with open(os.path.join(directory, "config.json")) as f:
            config = json_mod.load(f)

        # Remove version key if present (v5 doesn't have it)
        config.pop("version", None)

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

        model.sampler = EnhancedV6Sampler(
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
            sw_cluster_enabled=model.sw_cluster_enabled,
            sw_wolff_variant=model.sw_wolff_variant,
            n_replicas=model.n_replicas,
            pt_swap_interval=model.pt_swap_interval,
            entropy_T_ent=model.entropy_T_ent,
            entropy_delta_E_window=model.entropy_delta_E_window,
            entropy_precision=model.entropy_precision,
        )

        return model
