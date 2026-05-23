"""
Context Accumulator Field (CAF) for Ising Spin Glass Language Model — v15
==========================================================================

THE FUNDAMENTAL PROBLEM WITH v14 AND EARLIER
---------------------------------------------
All non-recall layers are capped at 5-10% of recall_scale.
  recall:      ±32000  (100%)
  cluster_ngram: ±160  (0.5%)
  wedge:         ±80   (0.25%)
  topic:         ±400  (1.25%)

These layers can NEVER override recall's ranking. They are structurally
prevented from changing which word the model selects.

v15 ARCHITECTURE: CONTEXT ACCUMULATOR FIELD
--------------------------------------------
KEY INSIGHT: Instead of computing per-cluster energies that are capped
at tiny values, we compute a PER-WORD energy field that:

1. Uses the ENTIRE context window (50+ tokens), not just the last 3-5
2. Sums contributions from ALL active clusters → can reach ±1600+ energy
3. Is estimated from cluster→word co-occurrence at training time
4. Provides INDEPENDENT discriminative signal (not redundant with recall)

THREE NEW COMPONENTS:

A. CLUSTER HISTOGRAM ACCUMULATOR
   - Maintains integer histogram H[64] over last 50 tokens
   - Exponential decay: older tokens contribute less
   - Updated incrementally: O(1) per token
   - Captures "what topics/themes are active right now"

B. HISTOGRAM-TO-WORD (H2W) COUPLING MATRIX
   - J_h2w[64][V]: maps cluster activation → per-word energy bonus
   - Computed as: field[w] = Σ_c H[c] × J_h2w[c][w]
   - With H[c]≈16 for 5 active clusters and J≈50: field ≈ 4000
   - THIS COMPETES WITH RECALL (range ±32000)
   - Estimated from training: for each word, what clusters appeared in
     the preceding 50-token window?

C. CLUSTER 3-SPIN COUPLINGS
   - J3[64][64][64]: when clusters A,B both active, cluster C gets bonus
   - Words in cluster C inherit this bonus
   - 64³ = 262K entries — very manageable
   - Compositional: captures "A + B → C" patterns that n-grams miss
   - Estimated from training: count cluster triples in windows

D. ADAPTIVE RECALL CONFIDENCE
   - When n-gram match is strong (5-gram, high count): recall_scale stays at 100%
   - When n-gram match is weak (1-gram backoff, low count): recall_scale drops
   - This lets the accumulator field MATTER when recall is uncertain
   - recall_confidence = min(1.0, ngram_level/5 × min(1.0, count/10))

ENERGY FORMULATION (v15)
-------------------------
E(w) = E_recall(w) × confidence(w)
     + E_accumulator(w) × accumulator_weight
     + E_3spin(w) × 3spin_weight
     + E_cluster_ngram(w)    [kept from v14.1]
     + E_wedge(w)            [kept from v14.1]
     + E_topic(w)            [kept from v8.2]
     + small terms + hard constraints

With accumulator_weight = 800 (50% of recall_scale), the accumulator
CAN override recall for specific contexts where it has strong signal.

INTEGER-ONLY ARITHMETIC
-----------------------
All operations are integer-only:
- Cluster histogram: int16 accumulators with right-shift decay
- H2W coupling: int16 matrix, int32 multiply-accumulate
- 3-Spin: int16 tensor, sparse lookup
- Recall confidence: Q8 fixed-point (0-256)
- ZERO floating-point operations in the hot path
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict


# ============================================================================
# CLUSTER HISTOGRAM ACCUMULATOR
# ============================================================================

class ClusterHistogramAccumulator:
    """
    Running cluster histogram over the last ~50 tokens.
    
    Maintains an integer vector H[64] that tracks which clusters
    are currently "active" in the context. Recent tokens contribute
    more than old ones (exponential decay via periodic right-shift).
    
    This is the integer-only equivalent of a hidden state in a neural LM:
    it encodes the entire past context in a compact 64-dimensional vector.
    """
    
    def __init__(
        self,
        n_clusters: int = 64,
        decay_interval: int = 10,   # Every 10 tokens, right-shift histogram
        increment: int = 16,        # How much to add for current cluster
    ):
        self.n_clusters = n_clusters
        self.decay_interval = decay_interval
        self.increment = increment
        
        # Histogram state
        self.H = np.zeros(n_clusters, dtype=np.int32)
        self._token_count = 0
        self._recent_clusters = []  # Last 50 cluster IDs for 3-spin lookup
    
    def reset(self):
        """Reset accumulator state."""
        self.H = np.zeros(self.n_clusters, dtype=np.int32)
        self._token_count = 0
        self._recent_clusters = []
    
    def update(self, cluster_id: int):
        """
        Update accumulator with a new token's cluster.
        
        - Add increment to the current cluster
        - Every decay_interval tokens, right-shift all clusters (decay)
        - Track recent clusters for 3-spin lookup
        """
        if cluster_id < 0 or cluster_id >= self.n_clusters:
            return
        
        self.H[cluster_id] += self.increment
        self._token_count += 1
        self._recent_clusters.append(cluster_id)
        
        # Keep only last 50 clusters
        if len(self._recent_clusters) > 50:
            self._recent_clusters = self._recent_clusters[-50:]
        
        # Periodic decay
        if self._token_count % self.decay_interval == 0:
            self.H >>= 1  # Right-shift = halve all values
    
    def get_histogram(self) -> np.ndarray:
        """Get current histogram (read-only)."""
        return self.H
    
    def get_active_clusters(self, threshold: int = 4) -> List[Tuple[int, int]]:
        """Get clusters with activation > threshold, sorted by activation."""
        active = []
        for c in range(self.n_clusters):
            if self.H[c] > threshold:
                active.append((c, int(self.H[c])))
        active.sort(key=lambda x: -x[1])
        return active
    
    def get_recent_clusters(self, n: int = 10) -> List[int]:
        """Get the n most recent cluster IDs."""
        return self._recent_clusters[-n:] if self._recent_clusters else []


# ============================================================================
# HISTOGRAM-TO-WORD (H2W) COUPLING MATRIX
# ============================================================================

class H2WCoupling:
    """
    Maps cluster histogram activation → per-word energy field.
    
    J_h2w[c, w] captures: "when cluster c is active in the broad context,
    word w is more/less likely to appear next."
    
    KEY DIFFERENCE FROM PMI:
    - PMI: word→word coupling (4000×4000 = 16M entries, mostly noise)
    - H2W: cluster→word coupling (64×4000 = 256K entries, well-estimated)
    
    KEY DIFFERENCE FROM CLUSTER N-GRAM:
    - Cluster n-gram: sequential pattern (c1, c2, c3 → c4)
    - H2W: bag-of-clusters pattern (active={c1,c3,c7} → w)
    - They capture DIFFERENT information!
    
    The field for word w is:
        field[w] = Σ_c H[c] × J_h2w[c, w]
    
    With ~5-10 active clusters at H[c]≈8-16 and J_h2w≈50-100,
    the total field can reach ±2000-4000 — competing with recall.
    """
    
    def __init__(
        self,
        n_clusters: int = 64,
        vocab_size: int = 4000,
        accumulator_weight: int = 800,   # 50% of recall_scale
        context_window: int = 50,        # Look-back window for training
        min_count: int = 5,              # Minimum co-occurrence count
    ):
        self.n_clusters = n_clusters
        self.vocab_size = vocab_size
        self.accumulator_weight = accumulator_weight
        self.context_window = context_window
        self.min_count = min_count
        
        # J_h2w[c, w]: cluster→word coupling matrix
        # Stored as int16 to save memory: 64×4000 = 256K entries × 2 bytes = 512KB
        self.J_h2w: Optional[np.ndarray] = None  # shape (C, V), dtype int16
        
        # Precomputed per-cluster word rankings for fast lookup
        self._top_words_per_cluster: Optional[Dict[int, List[Tuple[int, int]]]] = None
        
        self._built = False
    
    def build(
        self,
        sequences: List[List[int]],
        flag_state,  # FlagState from grassmann_flag.py
    ) -> None:
        """
        Build H2W coupling matrix from training sequences.
        
        For each word occurrence, compute the cluster histogram in the
        preceding window. Then estimate J_h2w[c, w] as the log-ratio
        of observed co-occurrence to expected (like PMI but cluster→word).
        """
        print(f"\n  [H2W] Building Histogram-to-Word coupling matrix...")
        print(f"  [H2W]   n_clusters={self.n_clusters}, vocab_size={self.vocab_size}")
        print(f"  [H2W]   accumulator_weight={self.accumulator_weight}")
        print(f"  [H2W]   context_window={self.context_window}")
        
        vocab_size = flag_state._vocab_size
        C = self.n_clusters
        V = min(self.vocab_size, vocab_size)
        
        # Phase 1: Count cluster→word co-occurrences
        # For each word w, count how often each cluster c appeared in the
        # preceding context_window tokens
        print(f"  [H2W]   Phase 1: Counting cluster→word co-occurrences...")
        
        # Co-occurrence counts: pair_count[c, w] = times cluster c appeared
        # in the context window before word w
        pair_count = np.zeros((C, V), dtype=np.int32)
        cluster_total = np.zeros(C, dtype=np.int64)  # Total cluster occurrences
        word_total = np.zeros(V, dtype=np.int64)      # Total word occurrences
        total_windows = 0
        
        for seq in sequences:
            if len(seq) < 2:
                continue
            
            # Convert to cluster IDs
            clusters = []
            for w in seq:
                if w < vocab_size:
                    clusters.append(int(flag_state.word_to_cluster[w]))
                else:
                    clusters.append(-1)
            
            # Slide window and count
            for i in range(len(seq)):
                w = seq[i]
                if w >= V:
                    continue
                
                word_total[w] += 1
                
                # Look at preceding context_window tokens
                start = max(0, i - self.context_window)
                seen_clusters = set()
                for j in range(start, i):
                    c = clusters[j]
                    if c >= 0 and c not in seen_clusters:
                        # Only count each cluster ONCE per window (binary)
                        pair_count[c, w] += 1
                        seen_clusters.add(c)
                
                total_windows += 1
        
        # Count total cluster occurrences in context windows
        for c in range(C):
            cluster_total[c] = int(pair_count[c].sum())
        
        print(f"  [H2W]   Total windows: {total_windows:,}")
        print(f"  [H2W]   Non-zero pairs: {int(np.count_nonzero(pair_count)):,}")
        
        # Phase 2: Compute cluster→word PMI
        # PMI(c, w) = log2(P(c,w) / (P(c) × P(w)))
        # In integer: PMI ≈ bit_length(observed/expected) - 1
        print(f"  [H2W]   Phase 2: Computing cluster→word PMI...")
        
        self.J_h2w = np.zeros((C, V), dtype=np.int16)
        
        if total_windows == 0:
            print(f"  [H2W]   WARNING: No windows found, H2W disabled")
            self._built = True
            return
        
        for c in range(C):
            if cluster_total[c] == 0:
                continue
            p_c = int(cluster_total[c])  # × 256 / total_windows in Q8
            
            for w in range(V):
                if word_total[w] == 0 or pair_count[c, w] < self.min_count:
                    continue
                
                # PMI(c, w) = log2(P(c,w) / (P(c) × P(w)))
                # P(c,w) ≈ pair_count[c,w] / total_windows
                # P(c) ≈ cluster_total[c] / total_windows
                # P(w) ≈ word_total[w] / total_windows
                # PMI = log2(pair_count * total_windows / (cluster_total * word_total))
                
                expected = max(1, int(cluster_total[c]) * int(word_total[w]) // total_windows)
                if expected == 0:
                    continue
                
                ratio = int(pair_count[c, w]) * 256 // expected  # Q8 ratio
                
                if ratio > 256:  # PMI > 0
                    # log2(ratio/256) ≈ bit_length(ratio) - 1 - 8
                    pmi_val = int(ratio).bit_length() - 1 - 8
                    # Scale: PMI can be 0-8, map to int16 range
                    # Use 8× multiplier so small PMI differences matter
                    scaled = min(127, pmi_val * 8)
                    self.J_h2w[c, w] = scaled
                # NEGATIVE PMI: DISCARD entirely
                # Only keeping POSITIVE couplings means the accumulator
                # can only PROMOTE words, never penalize. This avoids
                # the problem of many small negative entries adding up
                # and distorting the energy landscape.
        
        # Phase 3: Sparsify — keep only top-K positive entries per cluster
        # This ensures the accumulator field is SPARSE and DISCRIMINATIVE
        # rather than adding noise from many small entries
        TOP_K = 100  # Keep top 100 positive words per cluster
        print(f"  [H2W]   Phase 3: Sparsifying (top-{TOP_K} per cluster)...")
        for c in range(C):
            # Find positive entries
            pos_mask = self.J_h2w[c] > 0
            n_pos = int(np.sum(pos_mask))
            if n_pos > TOP_K:
                # Keep only top-K
                threshold = int(np.sort(self.J_h2w[c][pos_mask])[-TOP_K])
                self.J_h2w[c][self.J_h2w[c] < threshold] = 0
            # Zero out all negative entries (already zero, but be explicit)
            self.J_h2w[c][self.J_h2w[c] < 0] = 0
        
        # Phase 4: Build top-words index for diagnostics
        self._top_words_per_cluster = {}
        for c in range(C):
            top = []
            sorted_words = np.argsort(-self.J_h2w[c])
            for w in sorted_words[:20]:
                if self.J_h2w[c, w] > 0:
                    top.append((int(w), int(self.J_h2w[c, w])))
            if top:
                self._top_words_per_cluster[c] = top
        
        # Statistics
        n_pos = int(np.sum(self.J_h2w > 0))
        n_neg = int(np.sum(self.J_h2w < 0))
        max_val = int(np.max(self.J_h2w)) if n_pos > 0 else 0
        min_val = int(np.min(self.J_h2w)) if n_neg > 0 else 0
        print(f"  [H2W]   Matrix stats: +{n_pos}/-{n_neg} non-zero, "
              f"range=[{min_val}, {max_val}]")
        print(f"  [H2W]   Memory: {self.J_h2w.nbytes / 1024:.0f} KB")
        
        self._built = True
        print(f"  [H2W]   H2W coupling matrix built successfully.")
    
    def compute_field(
        self,
        candidate_words: np.ndarray,
        histogram: np.ndarray,
    ) -> np.ndarray:
        """
        Compute H2W energy field for candidate words.
        
        field[w] = Σ_c H[c] × J_h2w[c, w]
        
        With H[c]≈8-16 for active clusters and J≈50-100,
        and 5-10 active clusters: field ≈ ±2000-4000
        
        This competes with recall (range ±32000).
        Scaled by accumulator_weight/256 (Q8 division).
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)
        
        if not self._built or self.J_h2w is None:
            return energies
        
        # Get active clusters (H[c] > 0)
        active_mask = histogram > 0
        if not np.any(active_mask):
            return energies
        
        active_clusters = np.where(active_mask)[0]
        active_H = histogram[active_clusters].astype(np.int64)
        
        # Compute field for each candidate word
        # field[w] = Σ_c H[c] × J_h2w[c, w]
        # This is a matrix-vector product: active_H @ J_h2w[active, candidates]
        
        # Clamp candidate indices to J_h2w bounds (some candidates may be OOV)
        V = self.J_h2w.shape[1]
        safe_candidates = np.clip(candidate_words, 0, V - 1)
        
        # Sub-select rows for active clusters
        J_sub = self.J_h2w[active_clusters][:, safe_candidates].astype(np.int64)
        
        # Weighted sum: (n_active,) @ (n_active, n_candidates) → (n_candidates,)
        energies = active_H @ J_sub
        
        # Scale by accumulator_weight (Q8: divide by 256)
        # This converts from J_h2w × H units to energy units
        energies = (energies * self.accumulator_weight) >> 8
        
        # Cap: ±50% of recall_scale (800 for recall_scale=1600)
        # This prevents the accumulator from completely dominating
        cap = 800
        energies = np.clip(energies, -cap, cap)
        
        return energies
    
    def get_top_words_for_cluster(self, cluster_id: int, n: int = 10) -> List[Tuple[int, int]]:
        """Get top words associated with a cluster (for diagnostics)."""
        if self._top_words_per_cluster is None:
            return []
        return self._top_words_per_cluster.get(cluster_id, [])[:n]


# ============================================================================
# CLUSTER 3-SPIN COUPLINGS
# ============================================================================

class Cluster3SpinCoupling:
    """
    Three-body cluster couplings for compositional predictions.
    
    When clusters A and B are both active in the context, words in
    cluster C get an energy bonus if the triple (A, B, C) was frequently
    observed in training.
    
    This is COMPOSITIONAL: it captures "A + B → C" patterns that:
    - N-grams can't capture (too sparse at word level)
    - Pairwise PMI can't capture (doesn't model 3-body interactions)
    - Cluster n-grams can't capture (sequential, not set-based)
    
    Example: If cluster A = {science, research, experiment}
             And cluster B = {policy, government, regulation}
             Then cluster C = {regulate, mandate, comply} gets bonus
             Because "science" + "policy" → "regulate" is compositional.
    
    Storage: 64³ = 262K entries × 2 bytes = 524 KB (very manageable)
    But we use SPARSE storage: only store triples with count > min_count
    """
    
    def __init__(
        self,
        n_clusters: int = 64,
        spin3_weight: int = 400,      # 25% of recall_scale
        window_size: int = 20,        # Look for triples within this window
        min_count: int = 3,           # Minimum triple count
        max_triples: int = 500000,    # Maximum stored triples
    ):
        self.n_clusters = n_clusters
        self.spin3_weight = spin3_weight
        self.window_size = window_size
        self.min_count = min_count
        self.max_triples = max_triples
        
        # Sparse triple storage: (c1, c2) → {c3: count}
        # We store (c1, c2) as a combined key: c1 * n_clusters + c2
        self.triples: Dict[int, Dict[int, int]] = defaultdict(dict)
        
        # Precomputed energy lookup: (c1, c2) → {c3: energy}
        self.triple_energy: Dict[int, Dict[int, int]] = defaultdict(dict)
        
        # Per-cluster base frequency (for PMI-like normalization)
        self.cluster_freq: Optional[np.ndarray] = None
        
        self._built = False
    
    def build(
        self,
        sequences: List[List[int]],
        flag_state,  # FlagState from grassmann_flag.py
    ) -> None:
        """
        Build cluster 3-spin couplings from training sequences.
        
        Scan windows of size `window_size` and count how often
        each cluster triple (c1, c2, c3) appears together.
        Then compute a PMI-like score for each triple.
        """
        print(f"\n  [3SPIN] Building cluster 3-spin couplings...")
        print(f"  [3SPIN]   n_clusters={self.n_clusters}, spin3_weight={self.spin3_weight}")
        print(f"  [3SPIN]   window_size={self.window_size}, min_count={self.min_count}")
        
        vocab_size = flag_state._vocab_size
        C = self.n_clusters
        
        # Phase 1: Count cluster frequencies
        print(f"  [3SPIN]   Phase 1: Counting cluster frequencies...")
        self.cluster_freq = np.zeros(C, dtype=np.int64)
        for seq in sequences:
            for w in seq:
                if w < vocab_size:
                    c = int(flag_state.word_to_cluster[w])
                    self.cluster_freq[c] += 1
        
        total_tokens = max(1, int(self.cluster_freq.sum()))
        
        # Phase 2: Count triples in windows
        print(f"  [3SPIN]   Phase 2: Counting cluster triples in windows...")
        raw_triples: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        for seq_idx, seq in enumerate(sequences):
            if (seq_idx + 1) % 100000 == 0:
                print(f"  [3SPIN]     Processing sequence {seq_idx+1}...")
            
            # Convert to cluster IDs
            clusters = []
            for w in seq:
                if w < vocab_size:
                    clusters.append(int(flag_state.word_to_cluster[w]))
            
            if len(clusters) < 3:
                continue
            
            # Slide window
            for i in range(len(clusters)):
                window_start = max(0, i - self.window_size)
                # For each token, look at pairs of clusters in the preceding window
                c3 = clusters[i]
                
                # Get unique clusters in the preceding window
                preceding = list(set(clusters[window_start:i]))
                
                # Count all pairs (c1, c2) with c1 < c2
                for j_idx in range(len(preceding)):
                    for k_idx in range(j_idx + 1, len(preceding)):
                        c1, c2 = preceding[j_idx], preceding[k_idx]
                        # Normalize order: smaller first
                        if c1 > c2:
                            c1, c2 = c2, c1
                        key = c1 * C + c2
                        raw_triples[key][c3] += 1
        
        # Phase 3: Filter and compute PMI
        print(f"  [3SPIN]   Phase 3: Filtering and computing PMI...")
        
        n_stored = 0
        for key, c3_counts in raw_triples.items():
            c1 = key // C
            c2 = key % C
            
            for c3, count in c3_counts.items():
                if count < self.min_count:
                    continue
                
                # Compute PMI-like score
                # P(c1,c2,c3) ≈ count / total_windows
                # P(c1) × P(c2) × P(c3) ≈ freq(c1)*freq(c2)*freq(c3) / total³
                # PMI3 = log2(count × total² / (freq(c1)*freq(c2)*freq(c3)))
                
                p_c1 = max(1, int(self.cluster_freq[c1]))
                p_c2 = max(1, int(self.cluster_freq[c2]))
                p_c3 = max(1, int(self.cluster_freq[c3]))
                
                expected = max(1, p_c1 * p_c2 * p_c3 // (total_tokens * total_tokens))
                if expected == 0:
                    continue
                
                # PMI-like ratio
                ratio = count * total_tokens // expected
                
                if ratio > 2:  # PMI > 1 bit
                    # Scale to int8 range: PMI of 1-8 bits → 4-32
                    pmi_bits = int(ratio).bit_length() - 1
                    energy_val = min(32, pmi_bits * 4)
                    self.triple_energy[key][c3] = energy_val
                    n_stored += 1
                
                if n_stored >= self.max_triples:
                    break
            
            if n_stored >= self.max_triples:
                print(f"  [3SPIN]   Hit max_triples cap ({self.max_triples})")
                break
        
        # Statistics
        n_pairs = len(self.triple_energy)
        avg_targets = np.mean([len(v) for v in self.triple_energy.values()]) if n_pairs > 0 else 0
        print(f"  [3SPIN]   Stored {n_stored} triples from {n_pairs} cluster pairs")
        print(f"  [3SPIN]   Average targets per pair: {avg_targets:.1f}")
        
        self._built = True
        print(f"  [3SPIN]   Cluster 3-spin couplings built successfully.")
    
    def compute_energy(
        self,
        candidate_words: np.ndarray,
        context_clusters: List[int],
        flag_state,  # FlagState for word→cluster mapping
    ) -> np.ndarray:
        """
        Compute 3-spin coupling energy for candidate words.
        
        For each candidate word w in cluster c_w:
        E_3spin(w) = -spin3_weight × Σ_{(c1,c2) in context} PMI3(c1, c2, c_w)
        
        The negative sign means: words in clusters that co-occur with
        context cluster pairs get an energy BONUS (lower energy = more likely).
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)
        
        if not self._built or not context_clusters:
            return energies
        
        vocab_size = flag_state._vocab_size
        C = self.n_clusters
        
        # Get unique clusters from recent context (deduplicate)
        recent_set = list(set(context_clusters[-20:]))  # Last 20, unique
        
        if len(recent_set) < 2:
            return energies
        
        # Precompute all (c1, c2) pairs from context
        active_pairs = []
        for i in range(len(recent_set)):
            for j in range(i + 1, len(recent_set)):
                c1, c2 = recent_set[i], recent_set[j]
                if c1 > c2:
                    c1, c2 = c2, c1
                key = c1 * C + c2
                if key in self.triple_energy:
                    active_pairs.append(key)
        
        if not active_pairs:
            return energies
        
        # For each candidate word, sum up 3-spin energy
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int >= vocab_size:
                continue
            c_w = int(flag_state.word_to_cluster[w_int])
            
            total_spin3 = 0
            for key in active_pairs:
                if c_w in self.triple_energy[key]:
                    total_spin3 += self.triple_energy[key][c_w]
            
            # Negative energy = bonus (lower energy = more likely)
            energies[i] -= total_spin3 * self.spin3_weight // 32  # Normalize
        
        # Cap: ±25% of recall_scale (400 for recall_scale=1600)
        cap = 400
        energies = np.clip(energies, -cap, cap)
        
        return energies


# ============================================================================
# RECALL CONFIDENCE ESTIMATOR
# ============================================================================

class RecallConfidenceEstimator:
    """
    Estimate how confident the n-gram recall should be.
    
    When the n-gram match is strong (5-gram, high count), recall_scale
    stays at 100%. When the match is weak (backoff to 1-gram, low count),
    recall_scale is reduced, allowing other layers to matter.
    
    This is the KEY MECHANISM that lets the accumulator field compete:
    - Strong n-gram match → recall dominates (correct for common patterns)
    - Weak n-gram match → accumulator helps (needed for rare patterns)
    
    Confidence formula:
        confidence = ngram_level_factor × count_factor
        
        ngram_level_factor = ngram_match_level / max_ngram_level
        count_factor = min(1.0, sqrt(count / min_confident_count))
    
    In integer (Q8):
        confidence_q8 = (level * 256 // max_level) × min(256, isqrt(count * 256 // min_count)) // 256
    """
    
    def __init__(
        self,
        max_ngram_level: int = 5,
        min_confident_count: int = 10,
        min_confidence_q8: int = 128,   # Minimum 50% confidence (128/256)
    ):
        self.max_ngram_level = max_ngram_level
        self.min_confident_count = min_confident_count
        self.min_confidence_q8 = min_confidence_q8
    
    def compute_confidence(
        self,
        ngram_level: int,  # What n-gram level matched (1=unigram, 5=5gram)
        count: int,         # Count of the matched n-gram
    ) -> int:
        """
        Compute recall confidence as Q8 fixed-point (0-256).
        256 = full confidence, 128 = 50%, 64 = 25%.
        """
        # Level factor: 5-gram = 256, 4-gram = 204, ..., 1-gram = 51
        if self.max_ngram_level > 0:
            level_factor = (ngram_level * 256) // self.max_ngram_level
        else:
            level_factor = 51  # Default to 1-gram level
        
        # Count factor: reaches 256 at min_confident_count, grows slowly after
        if count <= 0:
            count_factor = 32  # Very low confidence for unseen
        elif count >= self.min_confident_count:
            # Integer sqrt approximation for gradual growth
            count_factor = min(256, 128 + int(self._isqrt(count * 64 // self.min_confident_count)))
        else:
            count_factor = (count * 256) // self.min_confident_count
        
        # Combined confidence (Q8)
        confidence = (level_factor * count_factor) >> 8
        
        # Enforce minimum (don't let recall go below 50%)
        confidence = max(self.min_confidence_q8, min(256, confidence))
        
        return confidence
    
    def _isqrt(self, n: int) -> int:
        """Integer square root."""
        if n <= 0:
            return 0
        if n < 4:
            return 1
        x = n
        y = (x + 1) // 2
        while y < x:
            x = y
            y = (x + n // x) // 2
        return x


# ============================================================================
# UNIFIED CONTEXT ACCUMULATOR LAYER (v15)
# ============================================================================

class ContextAccumulatorLayer:
    """
    Unified v15 Context Accumulator Layer.
    
    THREE new mechanisms that can actually compete with recall:
    1. Cluster Histogram Accumulator + H2W Field (50% of recall_scale)
    2. Cluster 3-Spin Couplings (25% of recall_scale)
    3. Recall Confidence Scaling (reduces recall when uncertain)
    
    Plus the v14.1 components kept:
    - Cluster n-gram recall (10% of recall_scale)
    - Wedge coupling (5% of recall_scale)
    
    ENERGY FORMULATION (v15):
    E(w) = E_recall(w) × confidence
         + E_accumulator(w)  [scale: 800, cap ±800]
         + E_3spin(w)        [scale: 400, cap ±400]
         + E_cluster_ngram(w) [scale: 200, cap ±160]
         + E_wedge(w)        [scale: 80, cap ±80]
         + E_topic(w)        [scale: 400]
    
    The accumulator field can SHIFT the ranking because:
    - With confidence=128 (50%) for weak recall: recall energy is halved
    - Accumulator adds ±800 on top
    - Total swing: ~1600 (from -800 to +800), vs recall range ~1600 (halved)
    - This means the accumulator CAN override recall for uncertain contexts
    """
    
    def __init__(
        self,
        n_clusters: int = 64,
        n_topics: int = 16,
        # v15 new parameters
        accumulator_weight: int = 800,
        context_window: int = 50,
        decay_interval: int = 10,
        histogram_increment: int = 16,
        spin3_weight: int = 400,
        spin3_window: int = 20,
        spin3_min_count: int = 3,
        # Confidence parameters
        confidence_min_count: int = 10,
        min_confidence_q8: int = 128,
        # v14.1 parameters (kept)
        wedge_weight: int = 80,
        max_wedge_distance: int = 3,
        max_cluster_ngram: int = 6,
        cluster_recall_scale: int = 200,
        # General
        enabled: bool = True,
    ):
        self.n_clusters = n_clusters
        self.n_topics = n_topics
        self.enabled = enabled
        
        # v15 components
        self.accumulator = ClusterHistogramAccumulator(
            n_clusters=n_clusters,
            decay_interval=decay_interval,
            increment=histogram_increment,
        )
        self.h2w = H2WCoupling(
            n_clusters=n_clusters,
            accumulator_weight=accumulator_weight,
            context_window=context_window,
        )
        self.spin3 = Cluster3SpinCoupling(
            n_clusters=n_clusters,
            spin3_weight=spin3_weight,
            window_size=spin3_window,
            min_count=spin3_min_count,
        )
        self.confidence_estimator = RecallConfidenceEstimator(
            min_confident_count=confidence_min_count,
            min_confidence_q8=min_confidence_q8,
        )
        
        # v14.1 components (kept)
        # Import here to avoid circular imports
        from .grassmann_flag import FlagState, ClusterNGramRecall, WedgeCoupling
        
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
        
        # Diagnostics
        self._diag = {
            'h2w_hits': 0,
            'spin3_hits': 0,
            'cluster_ngram_hits': 0,
            'wedge_hits': 0,
            'h2w_energy_sum': 0,
            'spin3_energy_sum': 0,
            'confidence_sum': 0,
            'confidence_count': 0,
        }
    
    def build(
        self,
        sequences: List[List[int]],
        vocab_size: int,
        word_freq: np.ndarray,
    ) -> None:
        """Build all sub-layers."""
        print(f"\n  [v15] Building Context Accumulator Layer...")
        print(f"  [v15]   n_clusters={self.n_clusters}, n_topics={self.n_topics}")
        print(f"  [v15]   accumulator_weight={self.h2w.accumulator_weight}")
        print(f"  [v15]   spin3_weight={self.spin3.spin3_weight}")
        
        # Build flag state (cluster + topic assignments)
        self.flag_state.build(sequences, vocab_size, word_freq)
        
        # Build v15 new components
        self.h2w.build(sequences, self.flag_state)
        self.spin3.build(sequences, self.flag_state)
        
        # Build v14.1 components
        self.cluster_ngram.build(sequences, self.flag_state)
        self.wedge.build(sequences, self.flag_state)
        
        print(f"  [v15] Context Accumulator Layer built successfully.")
    
    def compute_energy(
        self,
        candidate_words: np.ndarray,
        context_words: List[int],
        recall_ngram_level: int = 5,
        recall_ngram_count: int = 100,
    ) -> np.ndarray:
        """
        Compute total Context Accumulator energy for candidate words.
        
        Returns energy contributions from:
        1. H2W accumulator field
        2. Cluster 3-spin couplings
        3. Cluster n-gram recall (v14.1)
        4. Wedge coupling (v14.1)
        
        Also returns recall confidence (Q8) for the caller to apply.
        """
        # Compute recall confidence
        confidence = self.confidence_estimator.compute_confidence(
            recall_ngram_level, recall_ngram_count
        )
        
        # H2W accumulator field
        h2w_energy = self.h2w.compute_field(
            candidate_words, self.accumulator.get_histogram()
        )
        
        # 3-spin couplings
        recent_clusters = self.accumulator.get_recent_clusters(20)
        spin3_energy = self.spin3.compute_energy(
            candidate_words, recent_clusters, self.flag_state
        )
        
        # Cluster n-gram recall (v14.1)
        cluster_ngram_energy = self.cluster_ngram.compute_energy(
            candidate_words, context_words, self.flag_state
        )
        
        # Wedge coupling (v14.1)
        wedge_energy = self.wedge.compute_wedge_energy(
            candidate_words, context_words, self.flag_state
        )
        
        total = h2w_energy + spin3_energy + cluster_ngram_energy + wedge_energy
        
        # Update diagnostics
        if int(np.abs(h2w_energy).max()) > 0:
            self._diag['h2w_hits'] += 1
        if int(np.abs(spin3_energy).max()) > 0:
            self._diag['spin3_hits'] += 1
        if int(np.abs(cluster_ngram_energy).max()) > 0:
            self._diag['cluster_ngram_hits'] += 1
        if int(np.abs(wedge_energy).max()) > 0:
            self._diag['wedge_hits'] += 1
        self._diag['h2w_energy_sum'] += int(np.abs(h2w_energy).sum())
        self._diag['spin3_energy_sum'] += int(np.abs(spin3_energy).sum())
        self._diag['confidence_sum'] += confidence
        self._diag['confidence_count'] += 1
        
        return total, confidence
    
    def update_context(self, word_idx: int):
        """Update accumulator state with a new generated word."""
        if self.flag_state._built and word_idx < self.flag_state._vocab_size:
            c = int(self.flag_state.word_to_cluster[word_idx])
            self.accumulator.update(c)
    
    def reset(self):
        """Reset accumulator state (call at start of generation)."""
        self.accumulator.reset()
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information."""
        diag = dict(self._diag)
        if diag['confidence_count'] > 0:
            diag['avg_confidence'] = diag['confidence_sum'] / diag['confidence_count']
        return diag
    
    def get_energy_breakdown(
        self,
        candidate_words: np.ndarray,
        context_words: List[int],
    ) -> Dict[str, int]:
        """Get energy breakdown for diagnostics."""
        h2w_e = self.h2w.compute_field(
            candidate_words, self.accumulator.get_histogram()
        )
        recent_clusters = self.accumulator.get_recent_clusters(20)
        spin3_e = self.spin3.compute_energy(
            candidate_words, recent_clusters, self.flag_state
        )
        cn_e = self.cluster_ngram.compute_energy(
            candidate_words, context_words, self.flag_state
        )
        wedge_e = self.wedge.compute_wedge_energy(
            candidate_words, context_words, self.flag_state
        )
        return {
            'h2w': int(np.abs(h2w_e).sum()),
            'spin3': int(np.abs(spin3_e).sum()),
            'cluster_ngram': int(np.abs(cn_e).sum()),
            'wedge': int(np.abs(wedge_e).sum()),
        }
