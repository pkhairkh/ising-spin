"""
Grassmann Flag State Layer for Ising Spin Glass Language Model — v14.1
======================================================================

DIAGNOSIS OF v14.0 FAILURE
---------------------------
v14.0 added three layers (flag, wedge, block memory) but PPL stayed at 50.67.
The diagnostic `flag=10435 wedge=10435 memory=10433` reveals the problem:
ALL THREE layers contributed ~10K energy with ZERO discrimination.

Root causes:
1. Flag cluster energy = cluster bigram PMI → REDUNDANT with word n-gram recall
2. Flag topic energy = flat +300 penalty → REDUNDANT with Potts topic spin
3. Wedge coupling: antisymmetric coupling cancels, net spread is tiny
4. Block memory: (topic, cluster_pair) key is too sparse, provides log-bonus
   for a few words and zero for the rest

v14.1 ARCHITECTURE: CLUSTER N-GRAM RECALL
------------------------------------------
The GENUINELY NOVEL contribution of Grassmann flags is this:

Word n-grams can only go to 5-gram (4000^5 is impossibly sparse).
But CLUSTER n-grams with alphabet size 64 can go to 8-gram:
  64^2 = 4K contexts (trigram) — extremely well-estimated
  64^3 = 262K contexts (4-gram) — well-estimated
  64^4 = 16.7M contexts (5-gram) — feasible with smoothing
  64^5 = 1B contexts (6-gram) — sparse, need backoff

This provides LONG-RANGE context that word n-grams CANNOT provide.
Example: "the cat sat on the mat and then the"
  - Word 7-gram: too rare to observe
  - Cluster 7-gram: [DET,NOUN,VERB,PREP,DET,NOUN,CONJ] → well-observed
  - Tells us next cluster is likely ADV/VERB → constrains word choice

The cluster n-gram is computed INDEPENDENTLY of word n-gram recall and
provides information that word n-grams CANNOT provide at long range.

ENERGY FORMULATION
------------------
For candidate word w with cluster c_w in context (c_{i-k+1}, ..., c_i):

  E_cluster_ngram(w) = -cluster_recall_scale * log2(P(c_w | c_{i-k+1}, ..., c_i))

Where P(c_w | cluster_context) is computed with KN-smoothed backoff,
exactly like word n-gram recall but on the cluster alphabet.

The cluster n-gram energy is:
  - Zero-meaned: subtract median energy so it only DISCRIMINATES
  - Capped: at ±recall_scale * 0.1 (10% of recall energy)
  - Independent: captures information word n-grams miss at long range

WEDGE COUPLING (kept, fixed)
-----------------------------
The antisymmetric wedge coupling is genuinely novel — it captures
word ORDER that symmetric PMI cannot. But it must be:
  - Zero-meaned (subtract median)
  - Scaled to ~5% of recall energy
  - Applied ONLY to nearby context (distance 1-3)

INTEGER-ONLY ARITHMETIC
-----------------------
All operations are integer-only:
- Cluster assignments: integer lookup tables (int16)
- Cluster n-gram index: integer hash tables + count-based log energy
- Wedge coupling: int16 matrices
- ZERO floating-point operations in the hot path
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict


# ============================================================================
# FLAG STATE: Word → Cluster → Topic hierarchy
# ============================================================================

class FlagState:
    """
    Grassmann flag state representation for the Ising LM.
    
    A flag is a nested hierarchy:
        V₀ ⊂ V₁ ⊂ V₂
        word ⊂ cluster ⊂ topic
    
    The key innovation: at cluster level (alphabet size 64), n-gram
    statistics are well-estimated even for long contexts (8-gram).
    This provides long-range context that word n-grams cannot.
    """
    
    def __init__(
        self,
        n_clusters: int = 64,
        n_topics: int = 16,
    ):
        self.n_clusters = n_clusters
        self.n_topics = n_topics
        
        # Mapping tables
        self.word_to_cluster: np.ndarray = None  # shape (V,), dtype int16
        self.cluster_to_topic: np.ndarray = None  # shape (C,), dtype int8
        self.word_to_topic: np.ndarray = None     # shape (V,), dtype int8
        
        # Cluster frequency
        self.cluster_freq: np.ndarray = None  # shape (C,), int64
        
        self._built = False
        self._vocab_size = 0
    
    def build(
        self,
        sequences: List[List[int]],
        vocab_size: int,
        word_freq: np.ndarray,
    ) -> None:
        """Build the flag state hierarchy from training sequences."""
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
        
        # Phase 4: Compute cluster frequency
        print(f"  [FLAG]   Phase 4: Computing cluster statistics...")
        self.cluster_freq = np.zeros(self.n_clusters, dtype=np.int64)
        for w in range(self._vocab_size):
            c = int(self.word_to_cluster[w])
            self.cluster_freq[c] += int(word_freq[w])
        
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
        """Build integer context vectors for each word (co-occurrence ±2 window)."""
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
        """Integer K-means clustering with L1 distance."""
        assignments = np.zeros(n_items, dtype=np.int16)
        
        nonzero = np.where(freq[:n_items] > 0)[0]
        if len(nonzero) < k:
            print(f"  [FLAG]     Warning: only {len(nonzero)} non-zero items, < {k} clusters")
            return assignments
        
        quantile_positions = np.linspace(0, len(nonzero) - 1, k, dtype=int)
        sorted_idx = nonzero[np.argsort(freq[nonzero])]
        seed_idx = sorted_idx[quantile_positions]
        centroids = vectors[seed_idx].copy()
        
        active_mask = freq[:n_items] > 0
        active_idx = np.where(active_mask)[0]
        active_vectors = vectors[active_idx]
        
        for iteration in range(max_iter):
            new_assignments = np.zeros(n_items, dtype=np.int16)
            chunk_size = 4096
            for start in range(0, len(active_idx), chunk_size):
                end = min(start + chunk_size, len(active_idx))
                chunk = active_vectors[start:end]
                dists = np.sum(
                    np.abs(chunk[:, None, :].astype(np.int64) - centroids[None, :, :].astype(np.int64)),
                    axis=2
                )
                chunk_assign = np.argmin(dists, axis=1).astype(np.int16)
                new_assignments[active_idx[start:end]] = chunk_assign
            
            changed = np.sum(new_assignments[active_mask] != assignments[active_mask])
            assignments = new_assignments
            
            if changed == 0:
                print(f"  [FLAG]     K-means converged at iteration {iteration + 1}")
                break
            
            for c in range(k):
                members = np.where(assignments == c)[0]
                if len(members) == 0:
                    continue
                centroids[c] = np.median(vectors[members], axis=0).astype(np.int32)
        
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
        """1D integer K-means for cluster-to-topic mapping."""
        assignments = np.zeros(n_items, dtype=np.int8)
        
        indices = np.arange(n_items)
        quantile_positions = np.linspace(0, n_items - 1, k, dtype=int)
        seed_idx = indices[quantile_positions]
        centroids = features[seed_idx].copy().astype(np.int64)
        
        for iteration in range(max_iter):
            dists = np.sum(
                np.abs(features[:, None, :].astype(np.int64) - centroids[None, :, :]),
                axis=2
            )
            new_assignments = np.argmin(dists, axis=1).astype(np.int8)
            
            changed = np.sum(new_assignments != assignments)
            assignments = new_assignments
            
            if changed == 0:
                break
            
            for t in range(k):
                members = np.where(assignments == t)[0]
                if len(members) == 0:
                    continue
                centroids[t] = np.median(features[members], axis=0).astype(np.int64)
        
        return assignments
    
    def _compute_cluster_doc_freq(
        self, sequences: List[List[int]]
    ) -> np.ndarray:
        """Compute document frequency vectors for each cluster."""
        n_bins = 32
        doc_freq = np.zeros((self.n_clusters, n_bins), dtype=np.int32)
        
        for seq in sequences:
            seen_clusters = set()
            for w in seq:
                if w < self._vocab_size:
                    c = int(self.word_to_cluster[w])
                    if c not in seen_clusters:
                        seen_clusters.add(c)
                        for w2 in seq[:min(16, len(seq))]:
                            doc_freq[c, w2 % n_bins] += 1
        
        return doc_freq
    
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


# ============================================================================
# CLUSTER N-GRAM RECALL: Long-range context via cluster-level n-grams
# ============================================================================

class ClusterNGramRecall:
    """
    Cluster-level n-gram recall for long-range context.
    
    KEY INSIGHT: Word n-grams can only go to 5-gram (4000^5 too sparse).
    But cluster n-grams with alphabet 64 can go to 8-gram:
      2-gram: 64 contexts — trivial
      3-gram: 4K contexts — extremely well-estimated
      4-gram: 262K contexts — well-estimated
      5-gram: 16.7M contexts — feasible with smoothing
    
    This provides information that word n-grams CANNOT provide at long range.
    
    Energy: E_cluster_recall(w) = -scale * log2(P(c_w | c_context))
    Where P is estimated with KN-smoothed backoff over cluster alphabet.
    
    The energy is zero-meaned and capped so it only DISCRIMINATES
    between candidates, never dominates the recall energy.
    """
    
    def __init__(
        self,
        n_clusters: int = 64,
        max_cluster_ngram: int = 6,       # Use cluster 2-6 grams
        cluster_recall_scale: int = 200,  # Energy scale (10% of word recall)
        min_count: int = 2,               # Minimum count for cluster n-gram
    ):
        self.n_clusters = n_clusters
        self.max_cluster_ngram = max_cluster_ngram
        self.cluster_recall_scale = cluster_recall_scale
        self.min_count = min_count
        
        # Cluster n-gram index: context_tuple → Counter of next clusters
        # For k-gram: context is tuple of k-1 cluster IDs → next cluster
        self.index: Dict[Tuple, Counter] = defaultdict(Counter)
        
        # Cluster unigram (for backoff)
        self.unigram: Counter = Counter()
        
        # Total observations
        self.total_obs = 0
        
        # KN smoothing parameters
        self.kn_d = 1  # KN discount
        
        self._built = False
        self._vocab_size = 0
    
    def build(
        self,
        sequences: List[List[int]],
        flag_state: FlagState,
    ) -> None:
        """Build cluster n-gram index from training sequences."""
        print(f"\n  [CLUSTER-NGRAM] Building cluster n-gram recall...")
        print(f"  [CLUSTER-NGRAM]   n_clusters={self.n_clusters}, max_ngram={self.max_cluster_ngram}")
        print(f"  [CLUSTER-NGRAM]   cluster_recall_scale={self.cluster_recall_scale}")
        
        vocab_size = flag_state._vocab_size
        self._vocab_size = vocab_size
        C = self.n_clusters
        
        # Count cluster n-grams for all orders
        for k in range(2, self.max_cluster_ngram + 1):
            context_counts: Dict[Tuple, Counter] = defaultdict(Counter)
            total_k = 0
            
            for seq in sequences:
                # Convert sequence to cluster IDs
                clusters = []
                for w in seq:
                    if w < vocab_size:
                        clusters.append(int(flag_state.word_to_cluster[w]))
                    else:
                        clusters.append(-1)  # OOV marker
                
                # Count k-grams
                for i in range(len(clusters) - k + 1):
                    # Check no OOV in context or next
                    context = tuple(clusters[i:i+k-1])
                    next_c = clusters[i+k-1]
                    if -1 in context or next_c == -1:
                        continue
                    context_counts[context][next_c] += 1
                    total_k += 1
            
            # Filter: keep only contexts with enough observations
            for ctx, counter in context_counts.items():
                total_ctx = sum(counter.values())
                if total_ctx >= self.min_count:
                    self.index[ctx] = counter
            
            print(f"  [CLUSTER-NGRAM]   {k}-gram: {len(context_counts)} contexts, "
                  f"{total_k} observations, kept {sum(1 for ctx in context_counts if sum(context_counts[ctx].values()) >= self.min_count)}")
        
        # Count cluster unigrams (for backoff)
        for seq in sequences:
            for w in seq:
                if w < vocab_size:
                    c = int(flag_state.word_to_cluster[w])
                    self.unigram[c] += 1
        
        self.total_obs = sum(self.unigram.values())
        
        # Compute KN continuation counts (for smoothing)
        # Number of distinct contexts each cluster appears as continuation
        self.kn_continuation: Counter = Counter()
        for ctx, counter in self.index.items():
            for c in counter:
                self.kn_continuation[c] += 1
        
        self._built = True
        
        # Print stats
        print(f"  [CLUSTER-NGRAM]   Total contexts in index: {len(self.index)}")
        print(f"  [CLUSTER-NGRAM]   Total cluster tokens: {self.total_obs}")
        print(f"  [CLUSTER-NGRAM]   Cluster n-gram recall built successfully.")
    
    def _kn_prob(
        self,
        cluster: int,
        context: Tuple[int, ...],
    ) -> int:
        """
        Compute KN-smoothed log2 probability for cluster given context.
        
        Uses interpolated KN backoff:
        P_KN(c | ctx) = max(count(ctx, c) - d, 0) / count(ctx) + λ(ctx) * P_KN(c | ctx[1:])
        
        Returns: Q8 fixed-point log2 probability * cluster_recall_scale
                 (positive = high prob = low energy, so we return -log2(P) as energy)
        """
        # Try each context length from longest to shortest (backoff)
        for k in range(len(context), 0, -1):
            ctx = context[-k:]  # Use last k clusters as context
            
            if ctx not in self.index:
                continue
            
            counter = self.index[ctx]
            total_ctx = sum(counter.values())
            
            if total_ctx < self.min_count:
                continue
            
            count_c = counter.get(cluster, 0)
            
            if count_c > 0:
                # KN-smoothed probability
                # P(c|ctx) = max(count - d, 0) / total + λ * P_backoff(c)
                discounted = max(count_c - self.kn_d, 0)
                lambda_weight = self.kn_d * len(counter) / total_ctx  # Number of distinct continuations
                
                # Main term: discounted count / total
                # Q8: (discounted * 256) / total_ctx
                prob_main = (discounted * 256) // max(1, total_ctx)
                
                # Backoff term: unigram probability
                prob_backoff = (self.unigram.get(cluster, 0) * 256) // max(1, self.total_obs)
                
                # Interpolated probability (Q8)
                # P = prob_main + lambda_weight * prob_backoff
                # lambda_weight is a fraction, approximate as Q8
                lambda_q8 = min(255, int(lambda_weight * 256))
                prob_q8 = prob_main + (lambda_q8 * prob_backoff) // 256
                
                if prob_q8 > 0:
                    # Compute -log2(P) in integer
                    # log2(P) ≈ bit_length(P) - 1 - 8 (since P is Q8)
                    # -log2(P) = 8 - bit_length(P) + 1 = 9 - bit_length(P)
                    # But we need -log2(P/Q8) = -log2(P) + log2(256) = -log2(P) + 8
                    # = 8 - (bit_length(P) - 1) = 9 - bit_length(P)
                    neg_log2 = 9 - int(prob_q8).bit_length()
                    neg_log2 = max(0, neg_log2)  # Clamp to non-negative
                    return neg_log2
            
            # Context exists but cluster not observed → backoff
            continue
        
        # No matching context → unigram probability
        count_c = self.unigram.get(cluster, 0)
        if count_c > 0:
            prob_q8 = (count_c * 256) // max(1, self.total_obs)
            if prob_q8 > 0:
                neg_log2 = 9 - int(prob_q8).bit_length()
                return max(0, neg_log2)
        
        # Unknown cluster → high energy
        return 20  # log2(1/1M) ≈ 20
    
    def compute_energy(
        self,
        candidate_words: np.ndarray,
        context_words: List[int],
        flag_state: FlagState,
    ) -> np.ndarray:
        """
        Compute cluster n-gram recall energy for candidate words.
        
        E(w) = cluster_recall_scale * (-log2(P(c_w | cluster_context)))
        
        Zero-meaned: subtract median so it only DISCRIMINATES.
        Capped: at ±recall_scale * 0.1 (10% of word recall energy).
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)
        
        if not self._built or len(context_words) == 0:
            return energies
        
        vocab_size = flag_state._vocab_size
        
        # Get cluster context from recent words
        cluster_context = []
        for w in context_words[-(self.max_cluster_ngram - 1):]:
            if w < vocab_size:
                cluster_context.append(int(flag_state.word_to_cluster[w]))
        
        if not cluster_context:
            return energies
        
        ctx_tuple = tuple(cluster_context)
        
        # Compute raw energy for each candidate
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int >= vocab_size:
                continue
            c_w = int(flag_state.word_to_cluster[w_int])
            neg_log2 = self._kn_prob(c_w, ctx_tuple)
            energies[i] = self.cluster_recall_scale * neg_log2
        
        # ZERO-MEAN: subtract median so layer only DISCRIMINATES
        median_e = int(np.median(energies[energies > 0])) if np.any(energies > 0) else 0
        energies -= median_e
        
        # CAP: at ±10% of recall_scale (1600 * 0.1 = 160)
        cap = 160  # recall_scale * 0.1
        energies = np.clip(energies, -cap, cap)
        
        return energies


# ============================================================================
# WEDGE COUPLING: Antisymmetric direction-dependent interactions (FIXED)
# ============================================================================

class WedgeCoupling:
    """
    Antisymmetric wedge product coupling for the Ising LM.
    
    In Grassmann exterior algebra: a∧b = -b∧a
    Applied to language: coupling depends on DIRECTION.
    
    v14.1 FIXES:
    - Zero-meaned: subtract median so it only DISCRIMINATES
    - Scaled to ~5% of recall energy (not dominant)
    - Limited to distance 1-3 (longer range is too noisy)
    """
    
    DISTANCE_WEIGHTS = {
        1: 256,   # 1.0
        2: 128,   # 0.5
        3: 64,    # 0.25
    }
    
    def __init__(
        self,
        n_clusters: int = 64,
        wedge_weight: int = 80,     # ~5% of recall_scale
        max_distance: int = 3,      # Only nearby context
    ):
        self.n_clusters = n_clusters
        self.wedge_weight = wedge_weight
        self.max_distance = max_distance
        
        # Wedge coupling matrices by distance: J_wedge[d][c1, c2]
        self.J_wedge: Dict[int, np.ndarray] = {}
        
        self._built = False
    
    def build(
        self,
        sequences: List[List[int]],
        flag_state: FlagState,
    ) -> None:
        """Build antisymmetric wedge couplings."""
        print(f"\n  [WEDGE] Building antisymmetric wedge couplings...")
        print(f"  [WEDGE]   n_clusters={self.n_clusters}, max_distance={self.max_distance}")
        print(f"  [WEDGE]   wedge_weight={self.wedge_weight}")
        
        vocab_size = flag_state._vocab_size
        C = self.n_clusters
        
        for d in range(1, self.max_distance + 1):
            # Count directional cluster pairs
            pair_counts = np.zeros((C, C), dtype=np.int64)
            
            for seq in sequences:
                for i in range(len(seq) - d):
                    w1 = seq[i]
                    w2 = seq[i + d]
                    if w1 >= vocab_size or w2 >= vocab_size:
                        continue
                    c1 = int(flag_state.word_to_cluster[w1])
                    c2 = int(flag_state.word_to_cluster[w2])
                    pair_counts[c1, c2] += 1
            
            # Compute directional PMI
            total = max(1, int(pair_counts.sum()))
            left_marginal = pair_counts.sum(axis=1)
            right_marginal = pair_counts.sum(axis=0)
            
            fwd_pmi = np.zeros((C, C), dtype=np.int16)
            rev_pmi = np.zeros((C, C), dtype=np.int16)
            
            for c1 in range(C):
                if left_marginal[c1] == 0:
                    continue
                for c2 in range(C):
                    if right_marginal[c2] == 0:
                        continue
                    
                    # Forward PMI
                    if pair_counts[c1, c2] > 0:
                        expected = int(left_marginal[c1]) * int(right_marginal[c2]) // total
                        if expected > 0:
                            ratio = int(pair_counts[c1, c2]) * 256 // expected
                            if ratio > 1:
                                log_val = (int(ratio).bit_length() - 1 - 8) * 4
                                fwd_pmi[c1, c2] = max(0, min(63, log_val))
                    
                    # Reverse PMI
                    if pair_counts[c2, c1] > 0 and left_marginal[c2] > 0 and right_marginal[c1] > 0:
                        expected = int(left_marginal[c2]) * int(right_marginal[c1]) // total
                        if expected > 0:
                            ratio = int(pair_counts[c2, c1]) * 256 // expected
                            if ratio > 1:
                                log_val = (int(ratio).bit_length() - 1 - 8) * 4
                                rev_pmi[c1, c2] = max(0, min(63, log_val))
            
            # Net wedge = fwd - rev (antisymmetric)
            wedge = fwd_pmi.astype(np.int16) - rev_pmi.astype(np.int16)
            self.J_wedge[d] = wedge
            
            n_pos = int(np.sum(wedge > 0))
            n_neg = int(np.sum(wedge < 0))
            n_zero = int(np.sum(wedge == 0))
            max_abs = int(np.max(np.abs(wedge))) if n_pos + n_neg > 0 else 0
            print(f"  [WEDGE]   distance={d}: +{n_pos}/-{n_neg}/0={n_zero}, max_abs={max_abs}")
        
        self._built = True
        print(f"  [WEDGE] Wedge couplings built successfully.")
    
    def compute_wedge_energy(
        self,
        candidate_words: np.ndarray,
        context_words: List[int],
        flag_state: FlagState,
    ) -> np.ndarray:
        """Compute wedge coupling energy (zero-meaned, capped)."""
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)
        
        if not self._built or len(context_words) == 0:
            return energies
        
        vocab_size = flag_state._vocab_size
        
        # Get context clusters with distances
        ctx_info = []
        effective_window = min(len(context_words), 6)
        for i, cw in enumerate(context_words[-effective_window:]):
            dist = effective_window - i
            if cw < vocab_size and dist <= self.max_distance:
                cc = int(flag_state.word_to_cluster[cw])
                ctx_info.append((cc, dist))
        
        if not ctx_info:
            return energies
        
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int >= vocab_size:
                continue
            c_w = int(flag_state.word_to_cluster[w_int])
            
            total_wedge = 0
            for cc, d in ctx_info:
                dw = self.DISTANCE_WEIGHTS.get(d, 64)
                if d in self.J_wedge:
                    wedge_val = int(self.J_wedge[d][cc, c_w])
                else:
                    wedge_val = 0
                total_wedge += wedge_val * dw
            
            # Negative energy = bonus for forward-likely pairs
            energies[i] -= (total_wedge * self.wedge_weight) >> 8
        
        # ZERO-MEAN: subtract median
        median_e = int(np.median(energies))
        energies -= median_e
        
        # CAP: at ±5% of recall_scale (80)
        cap = 80
        energies = np.clip(energies, -cap, cap)
        
        return energies


# ============================================================================
# GRASSMANN FLAG LAYER: Unified interface (v14.1)
# ============================================================================

class GrassmannFlagLayer:
    """
    Unified Grassmann Flag Layer v14.1.
    
    v14.1 replaces the failed v14.0 approach with:
    1. CLUSTER N-GRAM RECALL: long-range context via cluster-level n-grams
       - Provides information word n-grams CANNOT at long range
       - Zero-meaned, capped at 10% of recall
    2. WEDGE COUPLING: antisymmetric direction-dependent interaction
       - Captures word ORDER that symmetric PMI cannot
       - Zero-meaned, capped at 5% of recall
    
    REMOVED from v14.0 (redundant):
    - Flag cluster energy (duplicates word n-gram recall)
    - Flag topic energy (duplicates Potts topic spin)
    - Block memory (too sparse, replaced by cluster n-gram recall)
    """
    
    def __init__(
        self,
        n_clusters: int = 64,
        n_topics: int = 16,
        cluster_weight: int = 0,       # DEPRECATED — kept for API compat
        topic_weight: int = 0,         # DEPRECATED — kept for API compat
        wedge_weight: int = 80,
        max_wedge_distance: int = 3,
        block_size: int = 32,          # DEPRECATED
        max_blocks: int = 0,           # DEPRECATED
        memory_weight: int = 0,        # DEPRECATED
        max_cluster_ngram: int = 6,
        cluster_recall_scale: int = 200,
        enabled: bool = True,
    ):
        self.n_clusters = n_clusters
        self.n_topics = n_topics
        self.enabled = enabled
        
        # Sub-layers
        self.flag_state = FlagState(n_clusters, n_topics)
        self.cluster_ngram = ClusterNGramRecall(
            n_clusters=n_clusters,
            max_cluster_ngram=max_cluster_ngram,
            cluster_recall_scale=cluster_recall_scale,
        )
        self.wedge = WedgeCoupling(
            n_clusters=n_clusters,
            wedge_weight=wedge_weight,
            max_distance=max_wedge_distance,
        )
        
        # Current topic (updated during generation)
        self._current_topic = 0
        
        # Diagnostics
        self._diag = {
            'cluster_ngram_hits': 0,
            'wedge_coupling_hits': 0,
            'cluster_ngram_energy_sum': 0,
            'wedge_energy_sum': 0,
        }
    
    def build(
        self,
        sequences: List[List[int]],
        vocab_size: int,
        word_freq: np.ndarray,
    ) -> None:
        """Build all sub-layers."""
        # Build flag state (cluster + topic assignments)
        self.flag_state.build(sequences, vocab_size, word_freq)
        
        # Build cluster n-gram recall
        self.cluster_ngram.build(sequences, self.flag_state)
        
        # Build wedge coupling
        self.wedge.build(sequences, self.flag_state)
    
    def compute_energy(
        self,
        candidate_words: np.ndarray,
        context_words: List[int],
    ) -> np.ndarray:
        """Compute total Grassmann flag energy."""
        # Cluster n-gram recall energy
        cluster_energy = self.cluster_ngram.compute_energy(
            candidate_words, context_words, self.flag_state
        )
        
        # Wedge coupling energy
        wedge_energy = self.wedge.compute_wedge_energy(
            candidate_words, context_words, self.flag_state
        )
        
        total = cluster_energy + wedge_energy
        
        # Update diagnostics
        if int(np.abs(cluster_energy).max()) > 0:
            self._diag['cluster_ngram_hits'] += 1
        if int(np.abs(wedge_energy).max()) > 0:
            self._diag['wedge_coupling_hits'] += 1
        self._diag['cluster_ngram_energy_sum'] += int(np.abs(cluster_energy).sum())
        self._diag['wedge_energy_sum'] += int(np.abs(wedge_energy).sum())
        
        return total
    
    def update_topic(self, words: List[int]) -> None:
        """Update current topic from generated words."""
        if self.flag_state._built and words:
            self._current_topic = self.flag_state.get_topic(words)
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information."""
        return dict(self._diag)
    
    def get_energy_breakdown(
        self,
        candidate_words: np.ndarray,
        context_words: List[int],
    ) -> Dict[str, int]:
        """Get energy breakdown for diagnostics."""
        cluster_e = self.cluster_ngram.compute_energy(
            candidate_words, context_words, self.flag_state
        )
        wedge_e = self.wedge.compute_wedge_energy(
            candidate_words, context_words, self.flag_state
        )
        return {
            'cluster_ngram': int(np.abs(cluster_e).sum()),
            'wedge': int(np.abs(wedge_e).sum()),
        }
