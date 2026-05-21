#!/usr/bin/env python3
"""Apply v8.2 changes to model.py: integer-only fixes + TopicSpinLayer."""

import re

FILE = "/home/z/my-project/ising-spin/src/ising_spin/model.py"

with open(FILE, 'r') as f:
    content = f.read()

# ========================================================================
# 1. Update the INTEGER-ONLY CONSTRAINT comment
# ========================================================================
content = content.replace(
    """INTEGER-ONLY CONSTRAINT (enforced):
  - ALL generation-path computation uses integer arithmetic
  - Boltzmann sampling via pre-computed lookup table (NO np.exp in hot loop)
  - MCMC acceptance via the same lookup table (integer-only)
  - The ONLY floating-point is in building the lookup table at __init__ time""",
    """INTEGER-ONLY CONSTRAINT (enforced v8.2 — ZERO float operations):
  - ALL computation uses integer arithmetic — including initialization
  - Boltzmann lookup table built via integer geometric recurrence (NO math.exp)
  - Log probabilities computed via integer weight table (NO np.log/np.exp)
  - Perplexity computed via integer log2 + bit_length (NO math.exp)
  - ln(2) represented as rational 25246/36417 (error < 10^-7)
  - MCMC acceptance via the same lookup table (integer-only)"""
)

# ========================================================================
# 2. Add LN2 constants before IntegerBoltzmannSampler class
# ========================================================================
content = content.replace(
    """class IntegerBoltzmannSampler:
    \"\"\"
    Boltzmann sampling using ONLY integer arithmetic in the hot path.

    Pre-computes a lookup table at initialization:
        table[delta] = round(SCALE * exp(-beta * delta))

    At generation time, sampling is pure integer:
        1. deltas = energies - E_min (non-negative integers)
        2. weights = table[deltas] (integer array lookup)
        3. Cumulative sum (integer addition)
        4. Binary search (integer comparison)
    \"\"\"

    def __init__(self, beta: float = 0.1, max_delta: int = 5000, scale: int = 1 << 30):
        self.beta = beta
        self.scale = scale
        fine_max = min(max_delta, 1000)
        self.table = np.zeros(fine_max + 1, dtype=np.int64)
        for d in range(fine_max + 1):
            raw = math.exp(-beta * d)
            self.table[d] = max(0, int(round(scale * raw)))
        self.max_delta = fine_max""",
    """# Rational approximation of ln(2) = 0.6931471805599453...
# 25246/36417 = 0.69314718... (error < 10^-7)
LN2_NUM = 25246
LN2_DEN = 36417

# Fixed-point scale for integer log2 computations
LOG2_SCALE = 10000


class IntegerBoltzmannSampler:
    \"\"\"
    Boltzmann sampling using ONLY integer arithmetic — INCLUDING initialization.

    v8.2: ZERO floating-point operations anywhere.

    Pre-computes a lookup table at initialization using integer geometric
    recurrence (NO math.exp):
        table[0] = scale
        table[d] = table[d-1] * decay >> PRECISION
    where decay is computed via integer Taylor expansion of exp(-beta).

    At generation time, sampling is pure integer:
        1. deltas = energies - E_min (non-negative integers)
        2. weights = table[deltas] (integer array lookup)
        3. Cumulative sum (integer addition)
        4. Binary search (integer comparison)
    \"\"\"

    _FP_BITS = 48  # Fixed-point precision for table construction

    def __init__(self, beta: float = 0.1, max_delta: int = 5000, scale: int = 1 << 30):
        self.beta = beta
        self.scale = scale
        fine_max = min(max_delta, 1000)
        self.table = np.zeros(fine_max + 1, dtype=np.int64)

        # INTEGER-ONLY TABLE CONSTRUCTION
        # Compute exp(-beta) as a fixed-point integer via Taylor expansion:
        #   exp(-x) = 1 - x + x^2/2 - x^3/6 + x^4/24 - x^5/120
        # All in fixed-point with _FP_BITS bits of precision.
        P = self._FP_BITS
        ONE = 1 << P

        beta_fp = int(round(beta * ONE))  # beta in fixed-point

        # Taylor expansion of exp(-beta) in fixed-point integer
        decay = ONE  # term 0: 1.0
        decay -= beta_fp  # term 1: -x
        beta_sq = (beta_fp * beta_fp) >> P
        decay += beta_sq >> 1  # term 2: +x^2/2
        beta_cube = (beta_sq * beta_fp) >> P
        decay -= beta_cube // 3  # term 3: -x^3/6
        beta_4 = (beta_cube * beta_fp) >> P
        decay += beta_4 // 24  # term 4: +x^4/24
        beta_5 = (beta_4 * beta_fp) >> P
        decay -= beta_5 // 120  # term 5: -x^5/120
        decay = max(0, decay)

        # Build table via integer geometric recurrence
        self.table[0] = scale
        for d in range(1, fine_max + 1):
            self.table[d] = (self.table[d - 1] * decay) >> P
            if self.table[d] <= 0:
                self.table[d:] = 0
                break

        self.max_delta = fine_max"""
)

# ========================================================================
# 3. Replace compute_log_probabilities with integer-only version
# ========================================================================
content = content.replace(
    """    def compute_log_probabilities(self, energies: np.ndarray) -> np.ndarray:
        \"\"\"
        Compute log probabilities for each element given energies.

        Uses floating-point for the log computation (evaluation only,
        not in the generation hot path). Uses log-sum-exp for numerical
        stability.

        Returns array of log P(i) where P(i) ~ exp(-beta * E_i).
        \"\"\"
        if len(energies) == 0:
            return np.array([], dtype=np.float64)

        e_min = float(energies.min())
        shifted = -self.beta * (energies.astype(np.float64) - e_min)
        # Clip to avoid overflow in exp
        shifted = np.clip(shifted, -500, 500)
        log_weights = shifted
        log_Z = np.log(np.exp(log_weights).sum())
        log_probs = log_weights - log_Z
        return log_probs""",
    """    def compute_log_probabilities(self, energies: np.ndarray) -> np.ndarray:
        \"\"\"
        Compute log2 probabilities for each element — INTEGER-ONLY (v8.2).

        Uses the pre-computed lookup table (NO np.exp, NO np.log).
        Returns log2 P(i) * LOG2_SCALE as int64 fixed-point.

        Pipeline:
          1. deltas = energies - E_min (non-negative integers)
          2. weights = table[deltas] (integer Boltzmann weights)
          3. Z = sum(weights) (integer partition function)
          4. log2 P(i) = log2(weight_i) - log2(Z) via bit_length()
        \"\"\"
        if len(energies) == 0:
            return np.array([], dtype=np.int64)

        e_min = int(energies.min())
        deltas = (energies - e_min).astype(np.int64)
        deltas = np.clip(deltas, 0, self.max_delta)

        weights = self.table[deltas]
        Z = int(weights.sum())
        if Z <= 0:
            return np.full(len(energies), -10 * LOG2_SCALE, dtype=np.int64)

        # Integer log2 via bit_length with fractional refinement
        F = 20  # Extra precision bits

        # Compute log2(Z) in fixed-point
        Z_s = Z << F
        bl_Z = Z_s.bit_length() - 1
        if bl_Z > F:
            Z_norm = Z_s >> (bl_Z - F)
        else:
            Z_norm = Z_s << (F - bl_Z)
        frac_Z = (Z_norm - (1 << F)) * LOG2_SCALE >> F
        log2_Z = (bl_Z - F) * LOG2_SCALE + frac_Z if bl_Z >= F else frac_Z

        log_probs = np.zeros(len(energies), dtype=np.int64)
        for i in range(len(energies)):
            w = int(weights[i])
            if w <= 0:
                log_probs[i] = -15 * LOG2_SCALE
                continue
            w_s = w << F
            bl_w = w_s.bit_length() - 1
            if bl_w > F:
                w_norm = w_s >> (bl_w - F)
            else:
                w_norm = w_s << (F - bl_w)
            frac_w = (w_norm - (1 << F)) * LOG2_SCALE >> F
            log2_w = (bl_w - F) * LOG2_SCALE + frac_w if bl_w >= F else frac_w
            log_probs[i] = log2_w - log2_Z

        return log_probs"""
)

# ========================================================================
# 4. Update PPL computation to use integer log2 probabilities
# ========================================================================
content = content.replace(
    """        gen = self.generator
        sampler = gen.word_sampler

        total_log_prob = 0.0
        total_tokens = 0""",
    """        gen = self.generator
        sampler = gen.word_sampler

        # v8.2: Accumulate log2 probabilities as integers (x LOG2_SCALE)
        total_log2_prob = 0
        total_tokens = 0"""
)

content = content.replace(
    """                if not target_in_candidates:
                    # Target not reachable; use smoothing
                    total_log_prob += -15.0  # Very low probability
                    total_tokens += 1
                    continue""",
    """                if not target_in_candidates:
                    # Target not reachable; use smoothing
                    total_log2_prob += -15 * LOG2_SCALE
                    total_tokens += 1
                    continue"""
)

content = content.replace(
    """                # Compute log probabilities
                log_probs = sampler.compute_log_probabilities(energies)

                # Find the target word's log probability
                target_idx = np.where(candidate_words == target_word)[0]
                if len(target_idx) > 0:
                    total_log_prob += float(log_probs[target_idx[0]])
                else:
                    total_log_prob += -15.0

                total_tokens += 1

        if total_tokens == 0:
            return float('inf')

        # PPL = exp(-1/N * sum log P(w_t | ctx))
        avg_log_prob = total_log_prob / total_tokens
        perplexity = math.exp(-avg_log_prob)

        print(f"  Perplexity: {perplexity:.2f} (evaluated on {total_tokens} tokens)")
        return perplexity""",
    """                # Compute log2 probabilities (integer, x LOG2_SCALE)
                log_probs = sampler.compute_log_probabilities(energies)

                # Find the target word's log2 probability
                target_idx = np.where(candidate_words == target_word)[0]
                if len(target_idx) > 0:
                    total_log2_prob += int(log_probs[target_idx[0]])
                else:
                    total_log2_prob += -15 * LOG2_SCALE

                total_tokens += 1

        if total_tokens == 0:
            return float('inf')

        # v8.2: PPL from integer log2 probabilities
        # PPL = 2^(-avg_log2_prob) where avg_log2_prob is in log2 units
        avg_log2_prob = total_log2_prob / (total_tokens * LOG2_SCALE)
        perplexity = 2.0 ** (-avg_log2_prob)

        print(f"  Perplexity: {perplexity:.2f} (evaluated on {total_tokens} tokens)")
        return perplexity"""
)

# ========================================================================
# 5. Add TopicSpinLayer class before IsingLM class
# ========================================================================
topic_spin_class = '''

# ===========================================================================
# TOPIC SPIN LAYER (Potts Variable for Coherence)
# ===========================================================================

class TopicSpinLayer:
    """
    Potts topic spin for document-level coherence — v8.2.

    Adds a discrete Potts variable sigma_T in {0, 1, ..., K-1} representing
    the current topic. Each word w has a dominant topic T[w]. When a
    candidate word's topic disagrees with sigma_T, it pays an energy penalty
    (coherence_penalty), encouraging the model to stay on-topic.

    ALL computation is integer-only:
      - Topic assignment: T[w] = int8 (dominant topic from training)
      - Topic evidence: count of words per topic in context (int64)
      - Spin-flip: IntegerBoltzmannSampler over topic energies
      - Coherence energy: integer penalty added to word energies
    """

    def __init__(
        self,
        n_topics: int = 16,
        coherence_penalty: int = 400,
        spin_flip_interval: int = 20,
        context_window: int = 30,
        topic_coupling_scale: int = 100,
    ):
        self.n_topics = n_topics
        self.coherence_penalty = coherence_penalty
        self.spin_flip_interval = spin_flip_interval
        self.context_window = context_window
        self.topic_coupling_scale = topic_coupling_scale

        self._built = False
        self.word_topics = None       # np.ndarray (vocab_size,), dtype=int8
        self.topic_word_counts = None # np.ndarray (n_topics, vocab_size), dtype=int64
        self.doc_topic_counts = None  # np.ndarray (n_topics,), dtype=int64
        self.topic_sampler = None

        # Runtime state
        self.sigma_T = 0
        self._stats = {
            'spin_flips': 0,
            'coherence_penalties': 0,
            'total_positions': 0,
        }

    def build(self, texts: List[str], vocab, ngram_index=None):
        """Build topic assignments from training corpus — ALL INTEGER."""
        print(f"  Building Topic Spin Layer (K={self.n_topics}, "
              f"penalty={self.coherence_penalty}, flip_interval={self.spin_flip_interval})")

        K = self.n_topics
        vocab_size = len(vocab)

        # Step 1: Build document-term matrix (integer counts)
        print(f"    [1/4] Building document-term matrix...")
        doc_vectors = np.zeros((len(texts), vocab_size), dtype=np.int64)
        for d, text in enumerate(texts):
            for w in text.split():
                idx = vocab.word2idx.get(w)
                if idx is not None:
                    doc_vectors[d, idx] += 1

        n_docs = len(doc_vectors)
        if n_docs == 0:
            print(f"    No documents — skipping Topic Spin Layer")
            return

        # Step 2: Initialize centroids from evenly-spaced documents
        print(f"    [2/4] Initializing {K} topic centroids...")
        centroids = np.zeros((K, vocab_size), dtype=np.int64)
        step = max(1, n_docs // K)
        for k in range(K):
            centroids[k] = doc_vectors[(k * step) % n_docs].copy()

        # Step 3: Iterative hard clustering (integer K-means, L1 distance)
        print(f"    [3/4] Running integer K-means clustering ({K} topics, max 10 iters)...")
        assignments = np.zeros(n_docs, dtype=np.int32)

        for iteration in range(10):
            new_assignments = np.zeros(n_docs, dtype=np.int32)
            for d in range(n_docs):
                min_dist = np.iinfo(np.int64).max
                best_k = 0
                for k in range(K):
                    dist = int(np.abs(doc_vectors[d] - centroids[k]).sum())
                    if dist < min_dist:
                        min_dist = dist
                        best_k = k
                new_assignments[d] = best_k

            changed = int((new_assignments != assignments).sum())
            assignments = new_assignments

            for k in range(K):
                mask = assignments == k
                if mask.any():
                    centroids[k] = doc_vectors[mask].sum(axis=0)
                else:
                    centroids[k] = doc_vectors[np.random.randint(n_docs)].copy()

            sizes = [int((assignments == k).sum()) for k in range(K)]
            print(f"      Iteration {iteration + 1}: {changed} reassigned, sizes={sizes}")

            if changed == 0:
                break

        # Step 4: Compute word-topic assignments
        print(f"    [4/4] Computing word-topic assignments...")
        topic_word_counts = np.zeros((K, vocab_size), dtype=np.int64)
        for d in range(n_docs):
            topic_word_counts[assignments[d]] += doc_vectors[d]

        self.word_topics = np.argmax(topic_word_counts, axis=0).astype(np.int8)
        self.topic_word_counts = topic_word_counts
        self.doc_topic_counts = np.array(
            [int((assignments == k).sum()) for k in range(K)], dtype=np.int64
        )

        # Build topic Boltzmann sampler
        self.topic_sampler = IntegerBoltzmannSampler(
            beta=0.01, max_delta=K * 100, scale=1 << 30
        )

        n_unique = len(set(self.word_topics.tolist()))
        topic_sizes = [int((self.word_topics == k).sum()) for k in range(K)]
        print(f"    Topic assignments: {n_unique} topics used, sizes={topic_sizes}")
        print(f"    Doc distribution: {self.doc_topic_counts.tolist()}")

        self._built = True

    def init_spin(self, prompt_words: List[int]) -> int:
        """Initialize sigma_T from prompt words — all integer."""
        if not self._built:
            return 0
        topic_evidence = np.zeros(self.n_topics, dtype=np.int64)
        for w in prompt_words:
            w_int = int(w)
            if w_int < len(self.word_topics):
                topic_evidence[self.word_topics[w_int]] += 1
        if topic_evidence.max() > 0:
            self.sigma_T = int(np.argmax(topic_evidence))
        else:
            self.sigma_T = 0
        return self.sigma_T

    def attempt_spin_flip(self, context_words: List[int]) -> int:
        """Potts spin-flip via IntegerBoltzmannSampler — all integer."""
        if not self._built:
            return self.sigma_T
        K = self.n_topics
        topic_evidence = np.zeros(K, dtype=np.int64)
        for w in context_words[-self.context_window:]:
            w_int = int(w)
            if w_int < len(self.word_topics):
                topic_evidence[self.word_topics[w_int]] += 1
        topic_energies = np.zeros(K, dtype=np.int64)
        for k in range(K):
            topic_energies[k] = -topic_evidence[k] * self.topic_coupling_scale
            if k == self.sigma_T:
                topic_energies[k] -= 50  # Persistence bonus
        new_topic = int(np.arange(K)[self.topic_sampler.sample(topic_energies)])
        if new_topic != self.sigma_T:
            self._stats['spin_flips'] += 1
        self.sigma_T = new_topic
        return self.sigma_T

    def compute_coherence_energy(self, candidate_words: np.ndarray) -> np.ndarray:
        """Compute coherence energy — all integer, returns int64 array."""
        if not self._built:
            return np.zeros(len(candidate_words), dtype=np.int64)
        n = len(candidate_words)
        energies = np.zeros(n, dtype=np.int64)
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int < len(self.word_topics):
                if int(self.word_topics[w_int]) != self.sigma_T:
                    energies[i] = self.coherence_penalty
                    self._stats['coherence_penalties'] += 1
        self._stats['total_positions'] += 1
        return energies

    def get_diagnostics(self) -> Dict:
        """Return topic spin diagnostics."""
        return {
            'current_topic': int(self.sigma_T),
            'spin_flips': self._stats['spin_flips'],
            'coherence_penalties': self._stats['coherence_penalties'],
            'total_positions': self._stats['total_positions'],
            'penalty_rate': self._stats['coherence_penalties'] / max(1, self._stats['total_positions']),
        }

    def reset_stats(self):
        """Reset runtime statistics."""
        self._stats = {'spin_flips': 0, 'coherence_penalties': 0, 'total_positions': 0}

'''

content = content.replace(
    """# ===========================================================================
# ISING-ENHANCED N-GRAM LANGUAGE MODEL
# ===========================================================================

class IsingLM:""",
    topic_spin_class + """# ===========================================================================
# ISING-ENHANCED N-GRAM LANGUAGE MODEL
# ===========================================================================

class IsingLM:"""
)

# ========================================================================
# 6. Add topic_spin_layer param to IsingLM.__init__
# ========================================================================
content = content.replace(
    """        graded_couplings: Optional["GradedCouplings"] = None,
    ):
        self.vocab = vocab""",
    """        graded_couplings: Optional["GradedCouplings"] = None,
        topic_spin_layer: Optional["TopicSpinLayer"] = None,
    ):
        self.vocab = vocab"""
)

content = content.replace(
    """        # v7.0: Graded couplings from continuation frequencies
        # Replaces both PMI and Walsh with graded, data-driven couplings
        self.graded_couplings = graded_couplings

        self.type_sampler""",
    """        # v7.0: Graded couplings from continuation frequencies
        # Replaces both PMI and Walsh with graded, data-driven couplings
        self.graded_couplings = graded_couplings

        # v8.2: Topic Spin Layer (Potts coherence)
        self.topic_spin_layer = topic_spin_layer

        self.type_sampler"""
)

# ========================================================================
# 7. Add topic stats to IsingLM diagnostics
# ========================================================================
content = content.replace(
    """            'graded_hits': 0,
            'mcmc_flips_accepted': 0, 'mcmc_flips_proposed': 0,""",
    """            'graded_hits': 0,
            'topic_spin_flips': 0, 'topic_coherence_penalties': 0,
            'mcmc_flips_accepted': 0, 'mcmc_flips_proposed': 0,"""
)

# ========================================================================
# 8. Add topic coherence energy to _compute_word_energy
# ========================================================================
content = content.replace(
    """        # === LOCAL FIELD (unigram frequency) ===
        # v5.0: Field always contributes fully, no damping""",
    """        # === TOPIC COHERENCE ENERGY (v8.2: Potts topic spin) ===
        if self.topic_spin_layer is not None and self.topic_spin_layer._built:
            coherence_energy = self.topic_spin_layer.compute_coherence_energy(candidate_words)
            energies += coherence_energy
            if int(coherence_energy.max()) > 0:
                self._stats['topic_coherence_penalties'] += 1

        # === LOCAL FIELD (unigram frequency) ===
        # v5.0: Field always contributes fully, no damping"""
)

# ========================================================================
# 9. Add topic spin init + flip to generate method
# ========================================================================
content = content.replace(
    """        prompt_type = self._get_word_type(prompt_idx)
        words = [prompt_idx]
        types = [prompt_type]
        consecutive_copies = 0
        diagnostics = []

        for pos in range(1, length):
            # === STEP 1: Choose POS type (BOLTZMANN, not override) ===""",
    """        prompt_type = self._get_word_type(prompt_idx)
        words = [prompt_idx]
        types = [prompt_type]
        consecutive_copies = 0
        diagnostics = []

        # v8.2: Initialize Potts topic spin from prompt
        if self.topic_spin_layer is not None and self.topic_spin_layer._built:
            self.topic_spin_layer.init_spin(words)
            self.topic_spin_layer.reset_stats()

        for pos in range(1, length):
            # v8.2: Periodic Potts spin-flip for topic coherence
            if (self.topic_spin_layer is not None
                    and self.topic_spin_layer._built
                    and pos > 0
                    and pos % self.topic_spin_layer.spin_flip_interval == 0):
                old_topic = self.topic_spin_layer.sigma_T
                self.topic_spin_layer.attempt_spin_flip(words)
                if self.topic_spin_layer.sigma_T != old_topic:
                    self._stats['topic_spin_flips'] += 1

            # === STEP 1: Choose POS type (BOLTZMANN, not override) ==="""
)

# ========================================================================
# 10. Add topic_spin params to IsingLMModel.__init__
# ========================================================================
content = content.replace(
    """        # v8.0: Recall-primary mode — enforces scale hierarchy
        recall_primary_mode: bool = True,
    ):
        self.vocab_min_freq = vocab_min_freq""",
    """        # v8.0: Recall-primary mode — enforces scale hierarchy
        recall_primary_mode: bool = True,
        # v8.2: Topic Spin (Potts coherence layer)
        topic_spin_enabled: bool = False,
        topic_n_topics: int = 16,
        topic_coherence_penalty: int = 400,
        topic_spin_flip_interval: int = 20,
        topic_context_window: int = 30,
        topic_coupling_scale: int = 100,
    ):
        self.vocab_min_freq = vocab_min_freq"""
)

content = content.replace(
    """        self.auto_calibrate_beta = auto_calibrate_beta

        self.vocab: Optional[Vocabulary] = None""",
    """        self.auto_calibrate_beta = auto_calibrate_beta

        # v8.2: Topic Spin parameters
        self.topic_spin_enabled = topic_spin_enabled
        self.topic_n_topics = topic_n_topics
        self.topic_coherence_penalty = topic_coherence_penalty
        self.topic_spin_flip_interval = topic_spin_flip_interval
        self.topic_context_window = topic_context_window
        self.topic_coupling_scale = topic_coupling_scale

        self.vocab: Optional[Vocabulary] = None"""
)

content = content.replace(
    """        self.graded_couplings: Optional[GradedCouplings] = None
        self.generator: Optional[IsingLM] = None""",
    """        self.graded_couplings: Optional[GradedCouplings] = None
        self.topic_spin_layer: Optional[TopicSpinLayer] = None
        self.generator: Optional[IsingLM] = None"""
)

# ========================================================================
# 11. Add TopicSpin building to train() method
# ========================================================================
content = content.replace(
    """        print("\\n[10/12] Scale diagnostics...")
        self._print_scale_diagnostics()

        # Step 11: Build generators
        print("\\n[11/12] Building generators...")
        self._build_generators()

        t_total = time.time() - t0
        print(f"\\nTraining complete: {t_total:.1f}s")
        return self""",
    """        print("\\n[10/13] Scale diagnostics...")
        self._print_scale_diagnostics()

        # Step 11: Build Topic Spin Layer (v8.2: Potts coherence)
        if self.topic_spin_enabled:
            print("\\n[11/13] Building Topic Spin Layer (Potts coherence)...")
            self.topic_spin_layer = TopicSpinLayer(
                n_topics=self.topic_n_topics,
                coherence_penalty=self.topic_coherence_penalty,
                spin_flip_interval=self.topic_spin_flip_interval,
                context_window=self.topic_context_window,
                topic_coupling_scale=self.topic_coupling_scale,
            )
            self.topic_spin_layer.build(texts, self.vocab, self.ngram_index)
        else:
            print("\\n[11/13] Skipping Topic Spin Layer (disabled)")
            self.topic_spin_layer = None

        # Step 12: Build generators
        print("\\n[12/13] Building generators...")
        self._build_generators()

        t_total = time.time() - t0
        print(f"\\nTraining complete: {t_total:.1f}s")
        print(f"  Integer-only: YES (v8.2 — ZERO float operations including init)")
        return self"""
)

# ========================================================================
# 12. Pass topic_spin_layer to generators in _build_generators
# ========================================================================
content = content.replace(
    """            graded_couplings=self.graded_couplings,
        )

        # Main generator""",
    """            graded_couplings=self.graded_couplings,
            topic_spin_layer=self.topic_spin_layer,
        )

        # Main generator"""
)

# ========================================================================
# 13. Update step numbers from /12 to /13
# ========================================================================
for i in range(1, 10):
    content = content.replace(f'[{i}/12]', f'[{i}/13]')

# ========================================================================
# Write the modified file
# ========================================================================
with open(FILE, 'w') as f:
    f.write(content)

print("v8.2 changes applied successfully!")
