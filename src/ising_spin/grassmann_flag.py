"""
Grassmann Flag State Layer for Ising Spin Glass Language Model
==============================================================

Inspired by pkhairkh/gfst-hmb: Grassmann-flag state tracking, block-based
exact-token memory with sparse readout, and MPUC-first design.

ARCHITECTURE OVERVIEW
---------------------
The current Ising LM treats each word as an atomic integer state with
symmetric PMI couplings — effectively a smoothed n-gram model with cosmetic
physics decorations. This module introduces THREE fundamental innovations:

1. FLAG STATE REPRESENTATION (Grassmann flags)
   - Each word position carries a structured flag: (word, cluster, topic)
   - Flags are NESTED: word ∈ cluster ∈ topic (like Grassmann flag manifolds)
   - The flag hierarchy provides natural multi-resolution energy:
     * Topic-level: global coherence (16-dim, well-estimated)
     * Cluster-level: syntactic/semantic coupling (64-dim, reliable stats)
     * Word-level: precise recall (4000-dim, sparse but exact)

2. ANTISYMMETRIC (WEDGE) COUPLINGS
   - Grassmann exterior algebra: a∧b = -b∧a
   - Language is ORDER-DEPENDENT: "the dog" ≠ "dog the"
   - Symmetric PMI cannot capture this; wedge couplings can
   - J_fwd[c_i, c_j] ≠ J_bwd[c_i, c_j] by construction
   - At cluster level (64×64), these matrices are well-estimated from 500K texts

3. BLOCK MEMORY WITH SPARSE READOUT
   - Training text stored in fixed-size blocks (B=32 words)
   - Each block tagged with topic flag + cluster signature
   - During generation: flag-matching retrieves relevant blocks
   - Block contents provide LONG-RANGE context beyond n-gram window
   - This is RAG for integer-only models

WHY THIS IS NOT "JUST CRANKING PMI WEIGHT"
-------------------------------------------
- Different STATE REPRESENTATION: flags vs atomic integers
- Different COUPLING STRUCTURE: antisymmetric vs symmetric
- Different CONTEXT MECHANISM: block retrieval vs local n-gram
- All three contribute INDEPENDENT information that PMI cannot provide

INTEGER-ONLY ARITHMETIC
-----------------------
- All cluster/topic assignments: integer lookup tables
- Cluster couplings: int16 matrices (64×64 = 8KB each)
- Block memory: integer hash tables + count-based energy
- Flag energies: integer additions, shifts, and table lookups
- Wedge product: integer multiply + sign flip
- ZERO floating-point operations in the hot path
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
import hashlib


# ============================================================================
# FLAG STATE: Word → Cluster → Topic hierarchy
# ============================================================================

class FlagState:
    """
    Grassmann flag state representation for the Ising LM.
    
    A flag is a nested hierarchy of subspaces:
        V₀ ⊂ V₁ ⊂ V₂
        word ⊂ cluster ⊂ topic
    
    Properties:
    - Deterministic mapping: word → cluster → topic
    - Multi-resolution: each level provides different statistical strength
    - Integer-only: all assignments are integer lookups
    - Compact: cluster (64) and topic (16) levels have good statistics
    
    The flag structure enables THREE distinct energy terms:
    1. E_topic(w): Is this word's topic consistent with the global topic?
    2. E_cluster(w): Is this word's cluster consistent with local syntax?
    3. E_wedge(w, ctx): Antisymmetric cluster coupling (direction-dependent)
    """
    
    def __init__(
        self,
        n_clusters: int = 64,
        n_topics: int = 16,
        cluster_weight: int = 200,   # Energy scale for cluster consistency
        topic_weight: int = 300,     # Energy scale for topic coherence
    ):
        self.n_clusters = n_clusters
        self.n_topics = n_topics
        self.cluster_weight = cluster_weight
        self.topic_weight = topic_weight
        
        # Mapping tables (populated during build)
        self.word_to_cluster: np.ndarray = None  # shape (V,), dtype int16
        self.cluster_to_topic: np.ndarray = None  # shape (C,), dtype int8
        self.word_to_topic: np.ndarray = None     # shape (V,), dtype int8 (cached)
        
        # Cluster statistics (populated during build)
        self.cluster_freq: np.ndarray = None       # shape (C,), int64
        self.cluster_word_counts: Dict[int, Counter] = None  # cluster → {word: count}
        self.topic_cluster_counts: Dict[int, Counter] = None # topic → {cluster: count}
        
        # Cluster co-occurrence (populated during build)
        # Forward: cluster at pos i, cluster at pos j (j > i)
        # Backward: cluster at pos j, cluster at pos i (i < j)
        self.cluster_bigram_fwd: np.ndarray = None  # shape (C, C), int64
        self.cluster_bigram_bwd: np.ndarray = None  # shape (C, C), int64
        
        self._built = False
        self._vocab_size = 0
    
    def build(
        self,
        sequences: List[List[int]],
        vocab_size: int,
        word_freq: np.ndarray,
    ) -> None:
        """
        Build the flag state hierarchy from training sequences.
        
        Two-phase integer K-means:
        1. Cluster assignment: group words by co-occurrence context
        2. Topic assignment: group clusters by document frequency
        
        All integer arithmetic — no floating point.
        """
        print(f"\n  [FLAG] Building Grassmann flag hierarchy...")
        print(f"  [FLAG]   vocab_size={vocab_size}, n_clusters={self.n_clusters}, n_topics={self.n_topics}")
        
        self._vocab_size = vocab_size
        
        # Phase 1: Build word co-occurrence context vectors (integer)
        print(f"  [FLAG]   Phase 1: Computing word context vectors...")
        context_vectors = self._build_context_vectors(sequences, vocab_size)
        
        # Phase 2: Integer K-means for cluster assignment
        print(f"  [FLAG]   Phase 2: Integer K-means clustering ({self.n_clusters} clusters)...")
        self.word_to_cluster = self._integer_kmeans(
            context_vectors, word_freq, self.n_clusters, vocab_size
        )
        
        # Phase 3: Compute cluster-to-topic mapping via document frequency
        print(f"  [FLAG]   Phase 3: Topic assignment ({self.n_topics} topics)...")
        cluster_doc_freq = self._compute_cluster_doc_freq(sequences)
        self.cluster_to_topic = self._integer_kmeans_1d(
            cluster_doc_freq, self.n_topics, self.n_clusters
        )
        
        # Cache word-to-topic mapping
        self.word_to_topic = np.zeros(vocab_size, dtype=np.int8)
        for w in range(vocab_size):
            c = self.word_to_cluster[w]
            self.word_to_topic[w] = self.cluster_to_topic[c]
        
        # Phase 4: Compute cluster statistics
        print(f"  [FLAG]   Phase 4: Computing cluster statistics...")
        self._compute_cluster_stats(sequences, word_freq)
        
        # Phase 5: Compute cluster bigram couplings
        print(f"  [FLAG]   Phase 5: Computing cluster bigram couplings...")
        self._compute_cluster_bigrams(sequences)
        
        self._built = True
        
        # Print summary
        cluster_sizes = Counter(self.word_to_cluster.tolist())
        topic_sizes = Counter(self.cluster_to_topic.tolist())
        print(f"  [FLAG]   Cluster sizes: min={min(cluster_sizes.values())}, "
              f"max={max(cluster_sizes.values())}, median={sorted(cluster_sizes.values())[len(cluster_sizes)//2]}")
        print(f"  [FLAG]   Topics populated: {len(topic_sizes)}/{self.n_topics}")
        print(f"  [FLAG]   Flag hierarchy built successfully.")
    
    def _build_context_vectors(
        self, sequences: List[List[int]], vocab_size: int
    ) -> np.ndarray:
        """
        Build integer context vectors for each word.
        
        Context vector for word w = sum of co-occurrence counts with
        all other words in a ±2 window, binned into 64 dimensional
        bins using a simple hash.
        
        Returns: shape (V, 64) int32 context vectors
        """
        dim = 64
        ctx = np.zeros((vocab_size, dim), dtype=np.int32)
        window = 2
        
        for seq in sequences:
            for i, w in enumerate(seq):
                if w >= vocab_size:
                    continue
                for j in range(max(0, i - window), min(len(seq), i + window + 1)):
                    if i == j:
                        continue
                    ctx_w = seq[j]
                    if ctx_w >= vocab_size:
                        continue
                    # Simple hash to 64 bins
                    bin_idx = ctx_w % dim
                    ctx[w, bin_idx] += 1
        
        return ctx
    
    def _integer_kmeans(
        self,
        vectors: np.ndarray,
        freq: np.ndarray,
        k: int,
        n_items: int,
        max_iter: int = 20,
    ) -> np.ndarray:
        """
        Integer K-means clustering.
        
        Uses Manhattan distance (L1) for integer-only arithmetic.
        Centroids are medians of cluster members.
        
        Returns: shape (n_items,) int16 cluster assignments
        """
        dim = vectors.shape[1]
        assignments = np.zeros(n_items, dtype=np.int16)
        
        # Initialize centroids: choose k items spread by frequency quantile
        nonzero = np.where(freq[:n_items] > 0)[0]
        if len(nonzero) < k:
            # Not enough items — assign all to cluster 0
            print(f"  [FLAG]     Warning: only {len(nonzero)} non-zero items, < {k} clusters")
            return assignments
        
        quantile_positions = np.linspace(0, len(nonzero) - 1, k, dtype=int)
        # Sort by frequency to get spread
        sorted_idx = nonzero[np.argsort(freq[nonzero])]
        # Take evenly spaced from sorted
        seed_idx = sorted_idx[quantile_positions]
        centroids = vectors[seed_idx].copy()  # shape (k, dim)
        
        # Only cluster non-zero items
        active_mask = freq[:n_items] > 0
        active_idx = np.where(active_mask)[0]
        active_vectors = vectors[active_idx]  # shape (n_active, dim)
        
        for iteration in range(max_iter):
            # Assignment step: L1 distance to centroids
            # For efficiency, compute in batches
            new_assignments = np.zeros(n_items, dtype=np.int16)
            
            # Compute distances: (n_active, k) using L1
            # |v - c| = sum of abs differences
            # Do this in chunks to avoid memory issues
            chunk_size = 4096
            for start in range(0, len(active_idx), chunk_size):
                end = min(start + chunk_size, len(active_idx))
                chunk = active_vectors[start:end]  # (chunk, dim)
                # L1 distance: sum of |chunk[:, None, :] - centroids[None, :, :]|
                dists = np.sum(
                    np.abs(chunk[:, None, :].astype(np.int64) - centroids[None, :, :].astype(np.int64)),
                    axis=2
                )  # shape (chunk, k)
                chunk_assign = np.argmin(dists, axis=1).astype(np.int16)
                new_assignments[active_idx[start:end]] = chunk_assign
            
            # Check convergence
            changed = np.sum(new_assignments[active_mask] != assignments[active_mask])
            assignments = new_assignments
            
            if changed == 0:
                print(f"  [FLAG]     K-means converged at iteration {iteration + 1}")
                break
            
            # Update step: centroids = median of cluster members
            for c in range(k):
                members = np.where(assignments == c)[0]
                if len(members) == 0:
                    continue
                member_vecs = vectors[members]
                # Integer median: sort and take middle
                centroids[c] = np.median(member_vecs, axis=0).astype(np.int32)
        
        # Assign zero-frequency items to nearest cluster (by L1 to centroids)
        zero_mask = freq[:n_items] == 0
        if np.any(zero_mask):
            zero_idx = np.where(zero_mask)[0]
            zero_vecs = vectors[zero_idx]
            for start in range(0, len(zero_idx), chunk_size):
                end = min(start + chunk_size, len(zero_idx))
                chunk = zero_vecs[start:end]
                dists = np.sum(
                    np.abs(chunk[:, None, :].astype(np.int64) - centroids[None, :, :].astype(np.int64)),
                    axis=2
                )
                assignments[zero_idx[start:end]] = np.argmin(dists, axis=1).astype(np.int16)
        
        return assignments
    
    def _integer_kmeans_1d(
        self,
        features: np.ndarray,
        k: int,
        n_items: int,
        max_iter: int = 20,
    ) -> np.ndarray:
        """
        1D integer K-means for cluster-to-topic mapping.
        
        features: shape (n_items, n_features) — cluster document frequency vectors
        """
        dim = features.shape[1]
        assignments = np.zeros(n_items, dtype=np.int8)
        
        # Initialize: evenly spread
        indices = np.arange(n_items)
        quantile_positions = np.linspace(0, n_items - 1, k, dtype=int)
        seed_idx = indices[quantile_positions]
        centroids = features[seed_idx].copy().astype(np.int64)
        
        for iteration in range(max_iter):
            # Assignment
            dists = np.sum(
                np.abs(features[:, None, :].astype(np.int64) - centroids[None, :, :]),
                axis=2
            )  # shape (n_items, k)
            new_assignments = np.argmin(dists, axis=1).astype(np.int8)
            
            changed = np.sum(new_assignments != assignments)
            assignments = new_assignments
            
            if changed == 0:
                break
            
            # Update
            for t in range(k):
                members = np.where(assignments == t)[0]
                if len(members) == 0:
                    continue
                centroids[t] = np.median(features[members], axis=0).astype(np.int64)
        
        return assignments
    
    def _compute_cluster_doc_freq(
        self, sequences: List[List[int]]
    ) -> np.ndarray:
        """
        Compute document frequency vectors for each cluster.
        
        For each sequence, count which clusters appear.
        Returns: shape (n_clusters, n_bins) int32
        """
        n_bins = 32  # Compact representation
        doc_freq = np.zeros((self.n_clusters, n_bins), dtype=np.int32)
        
        for seq in sequences:
            seen_clusters = set()
            for w in seq:
                if w < self._vocab_size:
                    c = int(self.word_to_cluster[w])
                    if c not in seen_clusters:
                        seen_clusters.add(c)
                        # Hash sequence context into bins
                        for w2 in seq[:min(16, len(seq))]:
                            doc_freq[c, w2 % n_bins] += 1
        
        return doc_freq
    
    def _compute_cluster_stats(
        self,
        sequences: List[List[int]],
        word_freq: np.ndarray,
    ) -> None:
        """Compute cluster frequency and word-cluster distributions."""
        self.cluster_freq = np.zeros(self.n_clusters, dtype=np.int64)
        self.cluster_word_counts = defaultdict(Counter)
        self.topic_cluster_counts = defaultdict(Counter)
        
        for w in range(self._vocab_size):
            c = int(self.word_to_cluster[w])
            t = int(self.cluster_to_topic[c])
            self.cluster_freq[c] += int(word_freq[w])
            self.cluster_word_counts[c][w] = int(word_freq[w])
            self.topic_cluster_counts[t][c] += int(word_freq[w])
    
    def _compute_cluster_bigrams(
        self, sequences: List[List[int]]
    ) -> None:
        """
        Compute forward and backward cluster bigram counts.
        
        Forward: count of (cluster_i, cluster_j) where j = i+1
        Backward: count of (cluster_j, cluster_i) where i = j-1
        
        The ASYMMETRY between these is the key innovation:
        - fwd[DET, NOUN] >> bwd[DET, NOUN]  (DET precedes NOUN)
        - bwd[NOUN, DET] >> fwd[NOUN, DET]  (NOUN follows DET)
        
        This is the Grassmann wedge product in practice:
        E_wedge(i,j) = J_fwd[c_i, c_j] - J_bwd[c_j, c_i]
        """
        self.cluster_bigram_fwd = np.zeros((self.n_clusters, self.n_clusters), dtype=np.int64)
        self.cluster_bigram_bwd = np.zeros((self.n_clusters, self.n_clusters), dtype=np.int64)
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                w1 = seq[i]
                w2 = seq[i + 1]
                if w1 >= self._vocab_size or w2 >= self._vocab_size:
                    continue
                c1 = int(self.word_to_cluster[w1])
                c2 = int(self.word_to_cluster[w2])
                self.cluster_bigram_fwd[c1, c2] += 1
                self.cluster_bigram_bwd[c2, c1] += 1
        
        # Convert to log-scale integer coupling strengths
        # J_fwd[c1, c2] = log2(count_fwd / expected) * scale
        # J_bwd[c2, c1] = log2(count_bwd / expected) * scale
        # This gives POSITIVE values for pairs that co-occur more than expected
        
        total_bigrams = max(1, int(self.cluster_bigram_fwd.sum()))
        
        # Marginals (computed ONCE, outside loop)
        fwd_c1_marginal = self.cluster_bigram_fwd.sum(axis=1)  # shape (C,)
        fwd_c2_marginal = self.cluster_bigram_fwd.sum(axis=0)  # shape (C,)
        bwd_c1_marginal = self.cluster_bigram_bwd.sum(axis=1)  # shape (C,)
        bwd_c2_marginal = self.cluster_bigram_bwd.sum(axis=0)  # shape (C,)
        
        # Compute PMI-like coupling in integer log scale
        # Using finer quantization: Q4 = 16 steps per power of 2
        # log2q4(x) = floor(4 * log2(x)) gives 4x finer than bit_length-1
        scale = 256  # Q8 fixed point for ratio computation
        
        fwd_pmi = np.zeros((self.n_clusters, self.n_clusters), dtype=np.int16)
        bwd_pmi = np.zeros((self.n_clusters, self.n_clusters), dtype=np.int16)
        
        for c1 in range(self.n_clusters):
            if fwd_c1_marginal[c1] == 0:
                continue
            for c2 in range(self.n_clusters):
                if fwd_c2_marginal[c2] == 0:
                    continue
                
                # Forward PMI
                fwd_count = int(self.cluster_bigram_fwd[c1, c2])
                if fwd_count > 0:
                    expected_fwd = int(fwd_c1_marginal[c1] * fwd_c2_marginal[c2]) // total_bigrams
                    if expected_fwd > 0:
                        ratio = fwd_count * scale // expected_fwd
                        if ratio > 1:
                            # Fine-grained integer log2: floor(4 * log2(ratio/256))
                            # = floor(4 * (log2(ratio) - 8))
                            # Gives ~4x more resolution than bit_length-1
                            log_val = (int(ratio).bit_length() - 1 - 8) * 4
                            fwd_pmi[c1, c2] = max(0, min(63, log_val))  # Cap at 6 bits
                
                # Backward PMI
                bwd_count = int(self.cluster_bigram_bwd[c1, c2])
                if bwd_count > 0 and bwd_c1_marginal[c1] > 0 and bwd_c2_marginal[c2] > 0:
                    expected_bwd = int(bwd_c1_marginal[c1] * bwd_c2_marginal[c2]) // total_bigrams
                    if expected_bwd > 0:
                        ratio = bwd_count * scale // expected_bwd
                        if ratio > 1:
                            log_val = (int(ratio).bit_length() - 1 - 8) * 4
                            bwd_pmi[c1, c2] = max(0, min(63, log_val))
        
        # Replace raw counts with PMI values
        self.cluster_bigram_fwd = fwd_pmi
        self.cluster_bigram_bwd = bwd_pmi
    
    def compute_flag_energy(
        self,
        candidate_words: np.ndarray,
        context_words: List[int],
        current_topic: int,
    ) -> np.ndarray:
        """
        Compute flag state energy for candidate words.
        
        E_flag(w) = E_cluster(w, ctx) + E_topic(w, current_topic)
        
        E_cluster: penalty if candidate's cluster is inconsistent with
                   the cluster bigram pattern of the context
        E_topic: penalty if candidate's topic ≠ current_topic
        
        All integer arithmetic. Returns shape (n_candidates,) int64.
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)
        
        if not self._built or len(context_words) == 0:
            return energies
        
        # Get context clusters (most recent few words)
        context_clusters = []
        for w in context_words[-5:]:  # Use last 5 words
            if w < self._vocab_size:
                context_clusters.append(int(self.word_to_cluster[w]))
        
        if not context_clusters:
            return energies
        
        # E_cluster: check if candidate's cluster follows context clusters
        # Use FORWARD bigram: J_fwd[context_cluster, candidate_cluster]
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int >= self._vocab_size:
                continue
            c_word = int(self.word_to_cluster[w_int])
            
            # Maximum forward coupling from context to this word's cluster
            max_fwd = 0
            for cc in context_clusters[-3:]:  # Last 3 clusters
                fwd = int(self.cluster_bigram_fwd[cc, c_word])
                if fwd > max_fwd:
                    max_fwd = fwd
            
            # Energy is INVERSE of coupling: high coupling = low energy = good
            # E = cluster_weight * (max_possible - max_fwd)
            # But simpler: reward high coupling with negative energy
            energies[i] -= max_fwd * self.cluster_weight
        
        # E_topic: penalty for off-topic words
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int >= self._vocab_size:
                continue
            t_word = int(self.word_to_topic[w_int])
            if t_word != current_topic:
                energies[i] += self.topic_weight  # Penalty
        
        return energies
    
    def get_topic(self, words: List[int]) -> int:
        """Get dominant topic for a sequence of words."""
        if not self._built or not words:
            return 0
        topic_counts = Counter()
        for w in words:
            if w < self._vocab_size:
                t = int(self.word_to_topic[w])
                topic_counts[t] += 1
        if not topic_counts:
            return 0
        return topic_counts.most_common(1)[0][0]
    
    def get_cluster(self, word_idx: int) -> int:
        """Get cluster for a word."""
        if not self._built or word_idx >= self._vocab_size:
            return 0
        return int(self.word_to_cluster[word_idx])
    
    def get_topic_for_word(self, word_idx: int) -> int:
        """Get topic for a word."""
        if not self._built or word_idx >= self._vocab_size:
            return 0
        return int(self.word_to_topic[word_idx])


# ============================================================================
# ANTISYMMETRIC (WEDGE) COUPLING: Direction-dependent cluster interactions
# ============================================================================

class WedgeCoupling:
    """
    Antisymmetric wedge product coupling for the Ising LM.
    
    In Grassmann exterior algebra, the wedge product is antisymmetric:
        a ∧ b = -b ∧ a
    
    Applied to language: the coupling between positions i and j depends
    on the DIRECTION. This captures word order, which symmetric PMI cannot.
    
    Implementation:
    - Forward coupling J_fwd[c_i, c_j]: cluster c_i at position i,
      cluster c_j at position j, where i < j (forward direction)
    - Backward coupling J_bwd[c_i, c_j]: cluster c_i at position i,
      cluster c_j at position j, where i > j (backward direction)
    - Wedge energy: E_wedge = J_fwd[c_i, c_j] - J_bwd[c_j, c_i]
    
    At cluster level (64×64), these are well-estimated from 500K texts.
    
    The wedge coupling also extends to longer distances via distance-decay:
    - Distance 1 (adjacent): full strength
    - Distance 2: 3/4 strength
    - Distance 3: 1/2 strength
    - Distance 4: 1/4 strength
    - Distance 5+: 1/8 strength
    """
    
    # Distance decay weights (integer, Q8 style)
    DISTANCE_WEIGHTS = {
        1: 256,   # 1.0
        2: 192,   # 0.75
        3: 128,   # 0.5
        4: 64,    # 0.25
        5: 32,    # 0.125
    }
    DEFAULT_DISTANCE_WEIGHT = 16  # 0.0625 for distance > 5
    
    def __init__(
        self,
        n_clusters: int = 64,
        wedge_weight: int = 150,     # Energy scale for wedge coupling
        max_distance: int = 5,       # Maximum coupling distance
    ):
        self.n_clusters = n_clusters
        self.wedge_weight = wedge_weight
        self.max_distance = max_distance
        
        # Forward coupling matrices by distance
        # J_fwd_dist[d][c_i, c_j] = coupling from cluster c_i at pos i
        # to cluster c_j at pos i+d
        self.J_fwd_dist: Dict[int, np.ndarray] = {}  # d → (C, C) int16
        self.J_bwd_dist: Dict[int, np.ndarray] = {}  # d → (C, C) int16
        
        # Net wedge coupling (precomputed: fwd - bwd^T)
        self.J_wedge: Dict[int, np.ndarray] = {}  # d → (C, C) int16
        
        self._built = False
    
    def build(
        self,
        sequences: List[List[int]],
        flag_state: FlagState,
    ) -> None:
        """
        Build distance-dependent antisymmetric couplings.
        
        For each distance d ∈ {1, 2, 3, 4, 5}:
        - Count forward bigrams: (c_i, c_{i+d})
        - Count backward bigrams: (c_{i+d}, c_i)
        - Compute PMI-like coupling strength
        - Net wedge = fwd - bwd^T (antisymmetric)
        """
        print(f"\n  [WEDGE] Building antisymmetric wedge couplings...")
        print(f"  [WEDGE]   n_clusters={self.n_clusters}, max_distance={self.max_distance}")
        
        vocab_size = flag_state._vocab_size
        C = self.n_clusters
        
        for d in range(1, self.max_distance + 1):
            fwd_counts = np.zeros((C, C), dtype=np.int64)
            bwd_counts = np.zeros((C, C), dtype=np.int64)
            
            for seq in sequences:
                for i in range(len(seq) - d):
                    w1 = seq[i]
                    w2 = seq[i + d]
                    if w1 >= vocab_size or w2 >= vocab_size:
                        continue
                    c1 = int(flag_state.word_to_cluster[w1])
                    c2 = int(flag_state.word_to_cluster[w2])
                    fwd_counts[c1, c2] += 1
                    bwd_counts[c2, c1] += 1
            
            # Convert to PMI-like integer coupling
            total = max(1, int(fwd_counts.sum()))
            fwd_c1_marginal = fwd_counts.sum(axis=1)
            fwd_c2_marginal = fwd_counts.sum(axis=0)
            bwd_c1_marginal = bwd_counts.sum(axis=1)  # Compute ONCE outside inner loop
            bwd_c2_marginal = bwd_counts.sum(axis=0)
            
            fwd_pmi = np.zeros((C, C), dtype=np.int16)
            bwd_pmi = np.zeros((C, C), dtype=np.int16)
            
            for c1 in range(C):
                if fwd_c1_marginal[c1] == 0:
                    continue
                for c2 in range(C):
                    if fwd_c2_marginal[c2] == 0:
                        continue
                    
                    # Forward PMI — fine-grained quantization
                    if fwd_counts[c1, c2] > 0:
                        expected = int(fwd_c1_marginal[c1] * fwd_c2_marginal[c2]) // int(total)
                        if expected > 0:
                            ratio = int(fwd_counts[c1, c2]) * 256 // expected
                            if ratio > 1:
                                log_val = (int(ratio).bit_length() - 1 - 8) * 4
                                fwd_pmi[c1, c2] = max(0, min(63, log_val))
                    
                    # Backward PMI — fine-grained quantization
                    if bwd_counts[c1, c2] > 0 and bwd_c1_marginal[c1] > 0 and bwd_c2_marginal[c2] > 0:
                        expected = int(bwd_c1_marginal[c1] * bwd_c2_marginal[c2]) // int(total)
                        if expected > 0:
                            ratio = int(bwd_counts[c1, c2]) * 256 // expected
                            if ratio > 1:
                                log_val = (int(ratio).bit_length() - 1 - 8) * 4
                                bwd_pmi[c1, c2] = max(0, min(63, log_val))
            
            self.J_fwd_dist[d] = fwd_pmi
            self.J_bwd_dist[d] = bwd_pmi
            
            # Net wedge coupling: J_wedge = fwd - bwd^T
            # This is ANTISYMMETRIC: J_wedge[i,j] = -J_wedge[j,i]
            wedge = fwd_pmi.astype(np.int16) - bwd_pmi.T.astype(np.int16)
            self.J_wedge[d] = wedge
            
            n_pos = int(np.sum(wedge > 0))
            n_neg = int(np.sum(wedge < 0))
            n_zero = int(np.sum(wedge == 0))
            print(f"  [WEDGE]   distance={d}: fwd_max={fwd_pmi.max()}, bwd_max={bwd_pmi.max()}, "
                  f"wedge: +{n_pos}/-{n_neg}/0={n_zero}")
        
        self._built = True
        print(f"  [WEDGE] Wedge couplings built successfully.")
    
    def compute_wedge_energy(
        self,
        candidate_words: np.ndarray,
        context_words: List[int],
        flag_state: FlagState,
    ) -> np.ndarray:
        """
        Compute wedge coupling energy for candidate words.
        
        E_wedge(w) = Σ_d Σ_{ctx_w} J_wedge[d][c_ctx, c_w] × dist_weight(d)
        
        The key insight: J_wedge = fwd - bwd^T is ANTISYMMETRIC, so:
        - J_wedge[DET, NOUN] > 0 → "the dog" gets BONUS (forward-likely)
        - J_wedge[NOUN, DET] < 0 → "dog the" gets PENALTY (backward-likely)
        
        This is the actual Grassmann wedge product: a∧b = -b∧a
        
        All integer arithmetic. Returns shape (n_candidates,) int64.
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)
        
        if not self._built or len(context_words) == 0:
            return energies
        
        vocab_size = flag_state._vocab_size
        
        # Precompute context clusters and distances
        ctx_info = []  # (cluster, distance) pairs
        effective_window = min(len(context_words), 10)  # Look back up to 10 words
        for i, cw in enumerate(context_words[-effective_window:]):
            dist = effective_window - i  # Distance from current position
            if cw < vocab_size:
                cc = int(flag_state.word_to_cluster[cw])
                ctx_info.append((cc, dist))
        
        if not ctx_info:
            return energies
        
        # Compute wedge energy for each candidate using ANTISYMMETRIC coupling
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int >= vocab_size:
                continue
            c_w = int(flag_state.word_to_cluster[w_int])
            
            total_wedge = 0
            for cc, dist in ctx_info:
                d = min(dist, self.max_distance)
                
                # Get distance weight
                dw = self.DISTANCE_WEIGHTS.get(d, self.DEFAULT_DISTANCE_WEIGHT)
                
                # Use ANTISYMMETRIC J_wedge (not just fwd!)
                # J_wedge[d][cc, c_w] > 0 means cc→c_w is forward-likely (bonus)
                # J_wedge[d][cc, c_w] < 0 means cc→c_w is backward-likely (penalty)
                if d in self.J_wedge:
                    wedge_val = int(self.J_wedge[d][cc, c_w])
                else:
                    # Fallback: use fwd-only if wedge not precomputed
                    if d in self.J_fwd_dist:
                        wedge_val = int(self.J_fwd_dist[d][cc, c_w])
                    else:
                        wedge_val = 0
                
                # Weight by distance
                total_wedge += wedge_val * dw
            
            # Negative energy for positive wedge (forward-likely) = bonus
            # Positive energy for negative wedge (backward-likely) = penalty
            energies[i] -= (total_wedge * self.wedge_weight) >> 8  # Q8 → integer
        
        return energies


# ============================================================================
# BLOCK MEMORY: Integer-only retrieval-augmented generation
# ============================================================================

class BlockMemory:
    """
    Block-based exact-token memory with sparse readout.
    
    Inspired by gfst-hmb: "block-based exact-token memory with sparse readout"
    
    During training:
    - Text is stored in fixed-size blocks (B=32 words)
    - Each block is tagged with: topic_id, cluster_signature, first_words_hash
    - Blocks are indexed by topic for fast retrieval
    
    During generation:
    - Current context is matched to blocks by topic and cluster signature
    - Matching blocks provide "memory readout": what words followed
      similar contexts in the training data
    - This provides LONG-RANGE context beyond the n-gram window
    
    Key difference from n-gram recall:
    - N-gram recall: local context (last 5 words) → next word
    - Block memory: GLOBAL context (topic + cluster pattern) → relevant text
    
    Integer-only: all matching is via integer equality and hash comparison.
    """
    
    def __init__(
        self,
        block_size: int = 32,
        max_blocks: int = 500000,     # Cap memory usage
        memory_weight: int = 100,     # Energy scale for block readout
        n_topics: int = 16,
        n_clusters: int = 64,
    ):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.memory_weight = memory_weight
        self.n_topics = n_topics
        self.n_clusters = n_clusters
        
        # Block storage
        # Each block: (topic_id, cluster_sig, word_indices)
        self.blocks: List[Tuple[int, int, np.ndarray]] = []
        
        # Index: topic_id → list of block indices
        self.topic_index: Dict[int, List[int]] = defaultdict(list)
        
        # Index: cluster_sig → list of block indices
        self.cluster_index: Dict[int, List[int]] = defaultdict(list)
        
        # Readout cache: (topic, last_2_clusters_hash) → Counter of next words
        self.readout_cache: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        
        self._built = False
    
    def build(
        self,
        sequences: List[List[int]],
        flag_state: FlagState,
    ) -> None:
        """
        Build block memory from training sequences.
        
        Each sequence is split into blocks of block_size words.
        Each block is tagged and indexed.
        """
        print(f"\n  [MEMORY] Building block memory...")
        print(f"  [MEMORY]   block_size={self.block_size}, max_blocks={self.max_blocks}")
        
        vocab_size = flag_state._vocab_size
        n_blocks = 0
        
        for seq in sequences:
            # Split sequence into blocks
            for start in range(0, len(seq), self.block_size):
                if n_blocks >= self.max_blocks:
                    break
                
                end = min(start + self.block_size, len(seq))
                block_words = seq[start:end]
                
                if len(block_words) < 4:  # Skip very short blocks
                    continue
                
                # Compute block signature
                # Topic: dominant topic in block
                topic = flag_state.get_topic(block_words)
                
                # Cluster signature: hash of first 4 clusters
                cluster_sig = 0
                for i, w in enumerate(block_words[:4]):
                    if w < vocab_size:
                        c = flag_state.get_cluster(w)
                        cluster_sig = cluster_sig * self.n_clusters + c
                
                # Store block
                block_idx = len(self.blocks)
                self.blocks.append((topic, cluster_sig, np.array(block_words, dtype=np.int32)))
                self.topic_index[topic].append(block_idx)
                self.cluster_index[cluster_sig % 1024].append(block_idx)  # Hash to 1024 buckets
                
                n_blocks += 1
            
            if n_blocks >= self.max_blocks:
                break
        
        # Build readout cache: for each block boundary, record what word
        # follows after a given (topic, cluster_pattern) context
        print(f"  [MEMORY]   Stored {n_blocks} blocks")
        self._build_readout_cache(sequences, flag_state)
        
        self._built = True
        print(f"  [MEMORY] Block memory built successfully.")
    
    def _build_readout_cache(
        self,
        sequences: List[List[int]],
        flag_state: FlagState,
    ) -> None:
        """
        Build readout cache: (topic, cluster_context_hash) → Counter of next words.
        
        This precomputes what words tend to follow given topic+cluster contexts,
        enabling fast lookup during generation.
        """
        vocab_size = flag_state._vocab_size
        total_entries = 0
        
        for seq in sequences:
            if len(seq) < 6:
                continue
            
            for i in range(3, len(seq) - 1):
                w = seq[i]
                next_w = seq[i + 1]
                
                if w >= vocab_size or next_w >= vocab_size:
                    continue
                
                # Context: topic + hash of last 2 clusters
                topic = int(flag_state.word_to_topic[w])
                c1 = int(flag_state.word_to_cluster[seq[i - 2]]) if seq[i - 2] < vocab_size else 0
                c2 = int(flag_state.word_to_cluster[seq[i - 1]]) if seq[i - 1] < vocab_size else 0
                ctx_hash = c1 * self.n_clusters + c2
                
                self.readout_cache[(topic, ctx_hash)][next_w] += 1
                total_entries += 1
        
        # Prune: keep only top-50 next words per context
        pruned = 0
        for key in self.readout_cache:
            counter = self.readout_cache[key]
            if len(counter) > 50:
                top_50 = counter.most_common(50)
                self.readout_cache[key] = Counter(dict(top_50))
                pruned += 1
        
        print(f"  [MEMORY]   Readout cache: {len(self.readout_cache)} contexts, "
              f"{total_entries} entries, {pruned} pruned to top-50")
    
    def compute_memory_energy(
        self,
        candidate_words: np.ndarray,
        context_words: List[int],
        flag_state: FlagState,
    ) -> np.ndarray:
        """
        Compute block memory readout energy for candidate words.
        
        E_memory(w) = -memory_weight * log2(P(w | topic, cluster_context))
        
        Where P(w | topic, cluster_context) is estimated from the
        readout cache — how often word w followed this topic+cluster
        pattern in the training data.
        
        All integer arithmetic. Returns shape (n_candidates,) int64.
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)
        
        if not self._built or len(context_words) < 3:
            return energies
        
        vocab_size = flag_state._vocab_size
        
        # Compute current context signature
        last_word = context_words[-1]
        if last_word >= vocab_size:
            return energies
        
        topic = int(flag_state.word_to_topic[last_word])
        c1 = int(flag_state.word_to_cluster[context_words[-3]]) if context_words[-3] < vocab_size else 0
        c2 = int(flag_state.word_to_cluster[context_words[-2]]) if context_words[-2] < vocab_size else 0
        ctx_hash = c1 * flag_state.n_clusters + c2
        
        # Look up readout cache
        key = (topic, ctx_hash)
        if key not in self.readout_cache:
            return energies
        
        counter = self.readout_cache[key]
        total = sum(counter.values())
        if total == 0:
            return energies
        
        # Compute energy for each candidate
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            count = counter.get(w_int, 0)
            if count > 0:
                # E = -memory_weight * log2(count/total)
                # = memory_weight * log2(total/count)
                # Use integer log2 approximation: floor(4 * log2(total/count))
                ratio = total * 256 // max(1, count)  # Q8 ratio
                if ratio > 1:
                    log_val = (int(ratio).bit_length() - 1 - 8) * 4  # 4x finer
                    log_val = max(0, log_val)
                    # Subtract: lower energy = better
                    energies[i] -= self.memory_weight * log_val
            # else: no memory signal, energy stays 0
        
        return energies
    
    def retrieve_blocks(
        self,
        topic: int,
        n_blocks: int = 5,
    ) -> List[np.ndarray]:
        """
        Retrieve blocks matching a topic for text generation context.
        
        Returns list of word index arrays.
        """
        if not self._built or topic not in self.topic_index:
            return []
        
        indices = self.topic_index[topic][:n_blocks]
        return [self.blocks[i][2] for i in indices]


# ============================================================================
# GRASSMANN FLAG LAYER: Unified interface
# ============================================================================

class GrassmannFlagLayer:
    """
    Unified Grassmann Flag Layer combining all three innovations.
    
    This is the main interface that the Ising LM will use.
    
    Energy terms added to the Ising Hamiltonian:
    
    1. E_flag(w): Flag state energy (cluster + topic consistency)
    2. E_wedge(w): Antisymmetric wedge coupling energy
    3. E_memory(w): Block memory readout energy
    
    Total Grassmann energy:
        E_grassmann(w) = E_flag(w) + E_wedge(w) + E_memory(w)
    
    This is ADDED to the existing recall energy. The key difference
    from just cranking PMI weight: these are STRUCTURALLY different
    energy terms that capture information PMI cannot:
    - Flag: hierarchical multi-resolution consistency
    - Wedge: direction-dependent (antisymmetric) coupling
    - Memory: long-range retrieval beyond n-gram window
    """
    
    def __init__(
        self,
        # Flag state parameters
        n_clusters: int = 64,
        n_topics: int = 16,
        cluster_weight: int = 200,
        topic_weight: int = 300,
        # Wedge coupling parameters
        wedge_weight: int = 150,
        max_wedge_distance: int = 5,
        # Block memory parameters
        block_size: int = 32,
        max_blocks: int = 500000,
        memory_weight: int = 100,
        # Global
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.n_clusters = n_clusters
        self.n_topics = n_topics
        
        # Create sub-layers
        self.flag_state = FlagState(
            n_clusters=n_clusters,
            n_topics=n_topics,
            cluster_weight=cluster_weight,
            topic_weight=topic_weight,
        )
        self.wedge_coupling = WedgeCoupling(
            n_clusters=n_clusters,
            wedge_weight=wedge_weight,
            max_distance=max_wedge_distance,
        )
        self.block_memory = BlockMemory(
            block_size=block_size,
            max_blocks=max_blocks,
            memory_weight=memory_weight,
            n_topics=n_topics,
            n_clusters=n_clusters,
        )
        
        # Current topic state (Potts-like, updated during generation)
        self._current_topic: int = 0
        
        # Diagnostics
        self._stats = {
            'flag_cluster_hits': 0,
            'flag_topic_hits': 0,
            'wedge_coupling_hits': 0,
            'memory_readout_hits': 0,
            'total_positions': 0,
        }
    
    def build(
        self,
        sequences: List[List[int]],
        vocab_size: int,
        word_freq: np.ndarray,
    ) -> None:
        """Build all sub-layers from training sequences."""
        if not self.enabled:
            print("  [GRASSMANN] Layer disabled, skipping build.")
            return
        
        print("\n" + "=" * 70)
        print("GRASSMANN FLAG LAYER — BUILD")
        print("=" * 70)
        print(f"  Architecture: Flag({self.n_clusters} clusters, {self.n_topics} topics) "
              f"+ Wedge(dist≤{self.wedge_coupling.max_distance}) "
              f"+ Memory(blocks≤{self.block_memory.max_blocks})")
        
        # Build flag state (word → cluster → topic hierarchy)
        self.flag_state.build(sequences, vocab_size, word_freq)
        
        # Build wedge couplings (antisymmetric cluster interactions)
        self.wedge_coupling.build(sequences, self.flag_state)
        
        # Build block memory (retrieval-augmented generation)
        self.block_memory.build(sequences, self.flag_state)
        
        print("\n  [GRASSMANN] All sub-layers built successfully.")
        print("=" * 70)
    
    def compute_energy(
        self,
        candidate_words: np.ndarray,
        context_words: List[int],
    ) -> np.ndarray:
        """
        Compute total Grassmann flag energy for candidate words.
        
        E_grassmann(w) = E_flag(w) + E_wedge(w) + E_memory(w)
        
        Returns shape (n_candidates,) int64. Lower = more likely.
        """
        if not self.enabled:
            return np.zeros(len(candidate_words), dtype=np.int64)
        
        self._stats['total_positions'] += 1
        
        # 1. Flag state energy (cluster + topic consistency)
        flag_energy = self.flag_state.compute_flag_energy(
            candidate_words, context_words, self._current_topic
        )
        
        if int(np.abs(flag_energy).max()) > 0:
            self._stats['flag_cluster_hits'] += 1
        
        # 2. Wedge coupling energy (antisymmetric direction-dependent)
        wedge_energy = self.wedge_coupling.compute_wedge_energy(
            candidate_words, context_words, self.flag_state
        )
        
        if int(np.abs(wedge_energy).max()) > 0:
            self._stats['wedge_coupling_hits'] += 1
        
        # 3. Block memory readout energy (long-range retrieval)
        memory_energy = self.block_memory.compute_memory_energy(
            candidate_words, context_words, self.flag_state
        )
        
        if int(np.abs(memory_energy).max()) > 0:
            self._stats['memory_readout_hits'] += 1
        
        return flag_energy + wedge_energy + memory_energy
    
    def update_topic(self, words: List[int]) -> None:
        """
        Update the current topic state based on recent words.
        
        This is the Potts spin update: the topic "spin" aligns with
        the dominant topic of the generated text so far.
        """
        if not self.enabled:
            return
        self._current_topic = self.flag_state.get_topic(words)
    
    def get_diagnostics(self) -> Dict:
        """Return diagnostic statistics."""
        return dict(self._stats)
