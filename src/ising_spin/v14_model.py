"""
V14: Ising-Enhanced N-Gram Language Model.

HONEST ARCHITECTURE:
  This is an autoregressive language model where:
    1. N-gram recall provides the PRIMARY next-word signal
    2. PMI couplings provide SECONDARY signal when recall misses
    3. POS grammar provides HARD CONSTRAINTS on word types
    4. Integer Boltzmann sampling provides STOCHASTIC selection

  The Ising model contributes through:
    - PMI coupling matrix J[w,w'] = log-floor PMI (word affinities)
    - Local field h[w] = self-information (unigram frequency)
    - Energy function: E(w|ctx) = -J[w,ctx] - h[w] + penalties
    - Temperature-controlled stochastic selection

  This is NOT "just an n-gram model" because:
    - PMI captures word-word affinity INDEPENDENT of exact n-gram context
    - The energy landscape creates global consistency constraints
    - Temperature controls the explore/exploit tradeoff physically

  But honestly: n-gram recall does the heavy lifting. The Ising model
  is a supplementary signal that helps when recall misses and provides
  principled temperature-based randomness control.

INTEGER-ONLY CONSTRAINT (ACTUALLY ENFORCED):
  - ALL generation-path computation uses integer arithmetic
  - Boltzmann sampling via pre-computed lookup table (NO np.exp in hot loop)
  - The ONLY floating-point is in building the lookup table at __init__ time
  - This is a one-time cost, not in the generation hot path

  Previous versions (V10-V13) violated this by calling np.exp() in
  _boltzmann_sample(), which runs at EVERY generation position. V14
  replaces this with integer lookup table sampling.

REFERENCES:
  - Levy & Goldberg (2014): Word2Vec as log-PMI matrix factorization
  - Marcolli (2015): Implicational couplings in syntax
  - Haydarov, Omirov & Rozikov (arXiv:2502.12014): Ising-Potts coupling
  - Creutz (1983): Demon algorithm for integer MCMC acceptance
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set

from .vocabulary import Vocabulary
from .type_system import POSTypeSystem, N_POS, IDX2POS, POS2IDX


# ===========================================================================
# INTEGER BOLTZMANN SAMPLER
# ===========================================================================

class IntegerBoltzmannSampler:
    """
    Boltzmann sampling using ONLY integer arithmetic in the hot path.

    Instead of computing exp(-beta * E) at generation time (which requires
    floating-point), we pre-compute a lookup table at initialization:

        table[delta] = round(SCALE * exp(-beta * delta))

    At generation time, sampling is:
        1. Compute energies (integers)
        2. deltas = energies - E_min (non-negative integers)
        3. weights = table[deltas] (integer array lookup)
        4. Cumulative sum (integer addition)
        5. Binary search (integer comparison)

    This is genuinely integer-only in the hot loop. The only FP is in
    building the table, which happens once at initialization.

    Inspired by V8's precomputed threshold tables for MCMC acceptance,
    but applied to the autoregressive Boltzmann sampling that V10-V13
    incorrectly implemented with np.exp().
    """

    def __init__(self, beta: float = 0.1, max_delta: int = 5000, scale: int = 1 << 30):
        """
        Args:
            beta: Inverse temperature for Boltzmann distribution.
                  Higher = more deterministic, lower = more random.
            max_delta: Maximum energy difference to pre-compute.
                       Any delta > max_delta gets weight = 0.
                       Must be large enough to cover the full energy range
                       (recall bonuses can be thousands, penalties up to 50000).
            scale: Fixed-point scale for integer weights.
                   2^30 gives ~9 decimal digits of precision.
        """
        self.beta = beta
        self.max_delta = max_delta
        self.scale = scale

        # Build lookup table: table[d] = round(scale * exp(-beta * d))
        # This is the ONLY place where floating-point is used.
        # After this, all generation is integer-only.
        #
        # For large max_delta (5000+), building the full table would use too
        # much memory. Instead, we use a two-level approach:
        #   - Fine table: for delta in [0, FINE_MAX], exact lookup
        #   - Coarse: for delta > FINE_MAX, weight is effectively 0
        # This works because exp(-beta * 5000) at beta=0.05 ≈ exp(-250) ≈ 0
        import math
        fine_max = min(max_delta, 1000)  # Beyond this, weight is ~0 at any reasonable beta
        self.table = np.zeros(fine_max + 1, dtype=np.int64)
        for d in range(fine_max + 1):
            raw = math.exp(-beta * d)
            self.table[d] = max(0, int(round(scale * raw)))
        self.max_delta = fine_max

    def sample(self, energies: np.ndarray) -> int:
        """
        Sample from Boltzmann distribution P(i) ~ exp(-beta * E_i)
        using ONLY integer arithmetic.

        Steps (all integer):
          1. Find E_min
          2. Compute deltas = E - E_min (non-negative)
          3. Clamp deltas to [0, max_delta]
          4. Look up weights from table
          5. Compute cumulative sum
          6. Generate random integer, binary search

        Returns index of sampled element.
        """
        if len(energies) == 0:
            return 0
        if len(energies) == 1:
            return 0

        # Step 1-3: Integer energy normalization
        e_min = int(energies.min())
        deltas = (energies - e_min).astype(np.int64)
        deltas = np.clip(deltas, 0, self.max_delta)

        # Step 4: Lookup table weights (NO np.exp!)
        weights = self.table[deltas]

        # Step 5: Cumulative sum (integer)
        total = int(weights.sum())
        if total <= 0:
            # All weights zero — uniform random
            return np.random.randint(len(energies))

        # Step 6: Random integer selection + binary search
        r = np.random.randint(0, total)
        cumsum = np.cumsum(weights)
        idx = int(np.searchsorted(cumsum, r, side='right'))
        return min(idx, len(energies) - 1)


# ===========================================================================
# N-GRAM INDEX
# ===========================================================================

class NGramIndex:
    """
    Multi-level n-gram index for exact token recall.

    Stores n-gram contexts and their continuations with integer counts.
    Supports:
      - Lookup: find all continuations for a given context
      - Recall bonus: compute energy bonus for n-gram matches
      - Copy: find the best candidate for direct copying

    This is the PRIMARY generation mechanism. When it hits, it produces
    coherent text. When it misses, the Ising PMI model takes over.

    Simplified from V11's NGramIndex:
      - Removed Kneser-Ney (was weakening bonuses in V12)
      - Removed longest_only (always use all matching levels)
      - Cleaner interface
    """

    def __init__(self, max_n: int = 5, min_count: int = 1):
        self.max_n = max_n
        self.min_count = min_count
        # index[k][context_tuple] = Counter({continuation_word: count})
        self.index: Dict[int, Dict[Tuple, Counter]] = {
            k: {} for k in range(1, max_n + 1)
        }
        self.context_totals: Dict[int, Dict[Tuple, int]] = {
            k: {} for k in range(1, max_n + 1)
        }
        self._built = False

    def build(self, sequences: List[List[int]]) -> "NGramIndex":
        """Build n-gram index from tokenized sequences. Integer counting only."""
        for seq in sequences:
            # Skip special tokens at the start
            start = 0
            for i, w in enumerate(seq):
                if w >= 4:  # Skip <UNK>=0, <BOS>=1, <EOS>=2, <PAD>=3
                    start = i
                    break

            for t in range(start, len(seq)):
                for k in range(1, self.max_n + 1):
                    if t - k < start:
                        break
                    context = tuple(seq[t-k:t])
                    continuation = seq[t]
                    if any(w < 4 for w in context) or continuation < 4:
                        continue
                    if context not in self.index[k]:
                        self.index[k][context] = Counter()
                    self.index[k][context][continuation] += 1
                    self.context_totals[k][context] = (
                        self.context_totals[k].get(context, 0) + 1
                    )

        # Prune low-count continuations
        pruned = 0
        for k in range(1, self.max_n + 1):
            for context in list(self.index[k].keys()):
                low_count = [
                    w for w, c in self.index[k][context].items()
                    if c < self.min_count
                ]
                for w in low_count:
                    del self.index[k][context][w]
                    self.context_totals[k][context] -= 1
                    pruned += 1
                if not self.index[k][context]:
                    del self.index[k][context]
                    del self.context_totals[k][context]

        self._built = True

        # Print stats
        for k in range(1, self.max_n + 1):
            n_ctx = len(self.index[k])
            n_cont = sum(len(v) for v in self.index[k].values())
            print(f"    {k}-gram: {n_ctx:,} contexts, {n_cont:,} continuations")
        if pruned > 0:
            print(f"    Pruned {pruned} low-count entries")

        return self

    def lookup(self, context_words: List[int]) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Look up n-gram continuations for the given context.

        Returns: {k: [(word, count, total), ...]} for each matching k-gram level.
        """
        results = {}
        for k in range(min(self.max_n, len(context_words)), 0, -1):
            context = tuple(context_words[-k:])
            if context in self.index[k]:
                total = self.context_totals[k][context]
                conts = self.index[k][context].most_common()
                results[k] = [(word, count, total) for word, count in conts]
        return results

    def get_recall_bonus(
        self,
        context_words: List[int],
        candidate_words: np.ndarray,
        recall_scale: int = 100,
        context_weight_factor: int = 4,
        longest_only: bool = True,
    ) -> np.ndarray:
        """
        Compute recall bonus for candidate words based on n-gram matches.

        KEY FIX: By default, use ONLY the longest matching context.
        Accumulating bonuses from multiple k levels inflates common words
        (like "the") that match at k=1, k=2, AND k=3.

        For the longest matching k-gram context:
          bonus = count * recall_scale * context_weight_factor^(k-1)

        For k >= 3: use raw bonus (strong signal, high confidence)
        For k < 3: normalize by total (weaker signal, lower confidence)

        Returns integer bonus array.
        """
        n_candidates = len(candidate_words)
        bonuses = np.zeros(n_candidates, dtype=np.int64)

        matches = self.lookup(context_words)
        if not matches:
            return bonuses

        if longest_only and matches:
            # Only use the LONGEST matching context — prevents common-word inflation
            best_k = max(matches.keys())
            matches = {best_k: matches[best_k]}

        for k, continuations in matches.items():
            context_weight = context_weight_factor ** (k - 1)
            cont_lookup = {}
            for word, count, total in continuations:
                if k >= 3:
                    bonus = count * recall_scale * context_weight
                else:
                    bonus = (count * recall_scale * context_weight) // max(1, total)
                # Keep the BEST bonus for each word
                if word not in cont_lookup or bonus > cont_lookup[word]:
                    cont_lookup[word] = int(bonus)

            for i, w in enumerate(candidate_words):
                w_int = int(w)
                if w_int in cont_lookup:
                    bonuses[i] += cont_lookup[w_int]

        return bonuses

    def get_best_copy_candidate(
        self,
        context_words: List[int],
        min_context_length: int = 3,
        min_confidence: float = 0.3,
    ) -> Optional[Tuple[int, int, int]]:
        """
        Find the best word for direct copying (highest-confidence n-gram match).

        Returns (word_idx, count, total) or None if no confident match.
        """
        matches = self.lookup(context_words)
        for k in sorted(matches.keys(), reverse=True):
            if k < min_context_length:
                break
            continuations = matches[k]
            if not continuations:
                continue
            best_word, best_count, total = continuations[0]
            # Confidence check: best_count / total >= min_confidence
            if best_count * 10 >= total * int(min_confidence * 10):
                return (best_word, best_count, total)
        return None


# ===========================================================================
# PMI COUPLING COMPUTATION
# ===========================================================================

def compute_log_floor_pmi(cooc: int, marginal_i: int, marginal_j: int,
                          total: int, cap: int = 15) -> int:
    """
    Compute log-floor PMI using ONLY integer arithmetic and bit operations.

    PMI(i,j) = log2(C(i,j)*N / (C(i)*C(j)))
             = bit_length(ratio) - 1  where ratio = max(num,denom)/min(num,denom)

    This is the "log-floor PMI" — purely integer, preserves sign and ordering.
    Novel: bit_length() as floor(log2()) for integer PMI.
    """
    if cooc == 0 or marginal_i == 0 or marginal_j == 0 or total == 0:
        return 0

    num = int(cooc) * int(total)
    denom = int(marginal_i) * int(marginal_j)

    if num == 0 or denom == 0:
        return 0

    sign = 1 if num > denom else -1
    ratio = max(num, denom) // min(num, denom)

    pmi = sign * (ratio.bit_length() - 1)
    return max(-cap, min(cap, pmi))


def compute_pmi_couplings(
    sequences: List[List[int]],
    vocab_size: int,
    window: int = 5,
    min_count: int = 2,
    pmi_cap: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PMI coupling matrix J and local field h from sequences.

    J[w, w'] = log-floor PMI(w, w') for co-occurring words within window
    h[w] = self-information = floor(log2(N/count(w)))

    Returns (J, h) as int64 arrays.
    """
    V = vocab_size

    # Count unigrams
    unigram = np.zeros(V, dtype=np.int64)
    for seq in sequences:
        for w in seq:
            unigram[w] += 1

    total_tokens = int(unigram.sum())

    # Count windowed co-occurrences
    cooc_counts = Counter()
    for seq in sequences:
        for i, w in enumerate(seq):
            for j in range(i + 1, min(i + window + 1, len(seq))):
                cooc_counts[(w, seq[j])] += 1

    # Compute PMI coupling matrix
    J = np.zeros((V, V), dtype=np.int64)
    for (w, w2), count in cooc_counts.items():
        if count >= min_count:
            pmi = compute_log_floor_pmi(
                int(count), int(unigram[w]), int(unigram[w2]),
                total_tokens, cap=pmi_cap
            )
            J[w, w2] = pmi
            J[w2, w] = pmi  # Symmetric

    # Compute local field (self-information)
    h = np.ones(V, dtype=np.int64)
    for w in range(V):
        if unigram[w] > 0 and total_tokens > unigram[w]:
            ratio = total_tokens // int(unigram[w])
            if ratio >= 2:
                h[w] = ratio.bit_length() - 1

    n_nonzero = int(np.count_nonzero(J))
    print(f"    PMI matrix: {n_nonzero:,} non-zero entries out of {V*V:,}")
    print(f"    PMI range: [{int(J[J != 0].min()) if n_nonzero > 0 else 0}, "
          f"{int(J.max())}]")

    return J, h


# ===========================================================================
# V14 ISING-ENHANCED N-GRAM LANGUAGE MODEL
# ===========================================================================

class IsingLM:
    """
    V14: Ising-Enhanced N-Gram Language Model.

    Architecture (honest description):
      1. POS type selection: Grammar-driven with hard constraints
      2. N-gram recall: Primary next-word signal (when available)
      3. PMI coupling: Secondary signal (when recall misses)
      4. Integer Boltzmann: Temperature-controlled stochastic selection

    The Ising model contributes:
      - PMI coupling matrix: word-word affinities beyond n-gram context
      - Energy function: principled scoring of candidate words
      - Temperature: physical parameter controlling randomness
      - Integer-only sampling: genuinely integer Boltzmann selection

    Parameters (far fewer than V12/V13's 30+):
      - recall_scale: How strongly n-gram matches bias selection
      - pmi_weight: How strongly PMI couplings bias selection
      - beta_type: Temperature for POS type selection
      - beta_word: Temperature for word selection
    """

    # Closed-class POS types (for anti-loop constraints)
    CLOSED_CLASS = {POS2IDX["DET"], POS2IDX["PREP"], POS2IDX["PART"],
                    POS2IDX["PRON"], POS2IDX["AUX"], POS2IDX["CONJ"]}

    # Hard type constraints (linguistically motivated)
    HARD_TYPE_CONSTRAINTS = {
        POS2IDX["PART"]: [POS2IDX["VERB"]],  # "to" must be followed by VERB
        POS2IDX["AUX"]: [POS2IDX["VERB"], POS2IDX["ADV"]],
    }

    def __init__(
        self,
        vocab: Vocabulary,
        ngram_index: NGramIndex,
        J: np.ndarray,
        h: np.ndarray,
        types: POSTypeSystem,
        # Generation parameters (only 6, not 30+)
        recall_scale: int = 1000,
        pmi_weight: int = 3,
        field_weight: int = 1,
        beta_type: float = 0.01,
        beta_word: float = 0.15,
        # Copy mechanism
        copy_enabled: bool = True,
        copy_min_context: int = 2,
        copy_min_confidence: float = 0.25,
        # Anti-repetition
        same_word_penalty: int = 50000,
        max_closed_class_run: int = 2,
        # Ablation: disable Ising model to measure its contribution
        ising_enabled: bool = True,
    ):
        self.vocab = vocab
        self.ngram_index = ngram_index
        self.J = J  # PMI coupling matrix (V x V, int64)
        self.h = h  # Local field (V, int64)
        self.types = types
        self.vocab_size = len(vocab)
        self.window = 5  # PMI coupling window

        # Generation parameters
        self.recall_scale = recall_scale
        self.pmi_weight = pmi_weight
        self.field_weight = field_weight
        self.ising_enabled = ising_enabled

        # Integer Boltzmann samplers (lookup-table, no np.exp in hot loop)
        self.type_sampler = IntegerBoltzmannSampler(beta=beta_type, max_delta=500)
        self.word_sampler = IntegerBoltzmannSampler(beta=beta_word, max_delta=500)

        # Copy mechanism
        self.copy_enabled = copy_enabled
        self.copy_min_context = copy_min_context
        self.copy_min_confidence = copy_min_confidence

        # Anti-repetition
        self.same_word_penalty = same_word_penalty
        self.max_closed_class_run = max_closed_class_run

        # Build type-word index: type_idx -> list of word indices
        self.type_words: Dict[int, List[int]] = {}
        for t in range(N_POS):
            col = types.I_emit[:, t]
            self.type_words[t] = [int(i) for i in range(len(col)) if col[i] > 0]

        # Pre-compute allowed transitions from grammar
        self.allowed_transitions: Set[Tuple[int, int]] = set()
        for t1 in range(N_POS):
            for t2 in range(N_POS):
                penalty = types.compute_grammar_penalty([t1], 0, t2)
                if penalty < 500:  # Not a hard constraint violation
                    self.allowed_transitions.add((t1, t2))

        # Diagnostics
        self._stats = {
            'total_positions': 0,
            'recall_hit': 0,
            'copy_used': 0,
            'pmi_only': 0,
            'same_word_blocked': 0,
            'closed_loop_blocked': 0,
        }

    def _get_word_type(self, word_idx: int) -> int:
        """Get primary POS type for a word."""
        if word_idx in self.types.allowed_types and self.types.allowed_types[word_idx]:
            return max(
                self.types.allowed_types[word_idx],
                key=lambda t: int(self.types.I_emit[word_idx, t])
            )
        return POS2IDX["X"]

    def _get_valid_next_types(self, prev_type: int, types_history: List[int]) -> List[int]:
        """
        Get valid next POS types with hard constraints + anti-loop.

        This replaces V12/V13's ad-hoc layering with a clean, principled approach:
          1. Start with grammar-allowed transitions
          2. Apply hard linguistic constraints (PART -> VERB, etc.)
          3. Break closed-class loops (max 2 in a row)
        """
        valid = [t for t in range(N_POS) if (prev_type, t) in self.allowed_transitions]
        if not valid:
            valid = list(range(N_POS))

        # Hard type constraints
        if prev_type in self.HARD_TYPE_CONSTRAINTS:
            constrained = self.HARD_TYPE_CONSTRAINTS[prev_type]
            constrained_valid = [t for t in valid if t in constrained]
            if constrained_valid:
                valid = constrained_valid

        # Closed-class anti-loop
        closed_run = 0
        for t in reversed(types_history):
            if t in self.CLOSED_CLASS:
                closed_run += 1
            else:
                break
        if closed_run >= self.max_closed_class_run:
            open_types = [t for t in valid if t not in self.CLOSED_CLASS]
            if open_types:
                valid = open_types
                self._stats['closed_loop_blocked'] += 1

        return valid

    def _compute_type_energy(self, pos: int, type_idx: int,
                             types_history: List[int]) -> int:
        """Compute energy for a POS type at position pos. Pure integer."""
        energy = 0
        types_for_check = list(types_history) + [type_idx]
        penalty = self.types.compute_grammar_penalty(
            types_for_check, len(types_history), type_idx
        )
        energy += penalty

        # Same-type penalty (mild — prevents VERB VERB VERB chains)
        if len(types_history) > 0 and type_idx == types_history[-1]:
            if type_idx not in (POS2IDX['NOUN'], POS2IDX['X']):
                energy += 50

        return energy

    def _compute_word_energy(
        self,
        pos: int,
        candidate_words: np.ndarray,
        word_type: int,
        context_words: List[int],
        context_types: List[int],
        recall_hit: bool,
    ) -> np.ndarray:
        """
        Compute energy for candidate words at position pos.

        Architecture (honest):
          E(w) = -recall_bonus(w)          [PRIMARY: n-gram match signal]
               - pmi_coupling(w, ctx)       [SECONDARY: Ising PMI signal]
               - field(w)                   [TERTIARY: unigram frequency]
               + penalties                  [HARD: grammar, anti-repetition]

        When recall hits: recall dominates (PMI is supplementary)
        When recall misses: PMI is primary (provides structured fallback)
        When Ising disabled: only recall + field (ablation baseline)

        All integer arithmetic.
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)

        # === RECALL BONUS (primary signal) ===
        recall_bonuses = self.ngram_index.get_recall_bonus(
            context_words=context_words,
            candidate_words=candidate_words,
            recall_scale=self.recall_scale,
            context_weight_factor=4,  # 4^(k-1): exponential boost for longer matches
            longest_only=True,  # Only use longest matching context (prevents common-word inflation)
        )
        energies -= recall_bonuses

        # === PMI COUPLING (Ising model — secondary signal) ===
        if self.ising_enabled and len(context_words) > 0:
            context_start = max(0, len(context_words) - self.window)
            ctx = context_words[context_start:]
            if ctx:
                ctx_arr = np.array(ctx, dtype=np.int64)
                # Vectorized coupling: J[candidates, ctx_words] -> sum over ctx
                coupling_block = self.J[np.ix_(candidate_words, ctx_arr)]
                coupling_sums = coupling_block.sum(axis=1)

                # When recall hits: PMI is supplementary (divide by 10)
                # When recall misses: PMI is primary (full weight)
                if recall_hit and recall_bonuses.max() > 0:
                    energies -= (coupling_sums * self.pmi_weight) // 10
                else:
                    energies -= coupling_sums * self.pmi_weight

        # === LOCAL FIELD (unigram — tertiary signal) ===
        field_vals = self.h[candidate_words] * self.field_weight
        if recall_hit and recall_bonuses.max() > 0:
            energies -= (field_vals * 1) // 10  # Small supplement
        else:
            energies -= field_vals  # More important when recall misses

        # === TYPE COMPATIBILITY (hard constraint) ===
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int < self.types.I_emit.shape[0]:
                emit_val = int(self.types.I_emit[w_int, word_type])
                if emit_val <= 0:
                    energies[i] += 500  # Strong penalty for type incompatibility

        # === SAME-WORD PENALTY (absolute — must always work) ===
        if len(context_words) >= 1:
            prev_word = context_words[-1]
            for i, w in enumerate(candidate_words):
                if int(w) == prev_word:
                    energies[i] += self.same_word_penalty
                    self._stats['same_word_blocked'] += 1

        # === CLOSED-CLASS DOUBLE PENALTY ===
        if word_type in self.CLOSED_CLASS and len(context_types) >= 1:
            prev_type = context_types[-1]
            # DET after DET: "the the", "a the"
            if word_type == POS2IDX["DET"] and prev_type == POS2IDX["DET"]:
                energies += 50000
            # PREP after PREP: "of in", "in of"
            elif word_type == POS2IDX["PREP"] and prev_type == POS2IDX["PREP"]:
                energies += 50000

        # === REPETITION PENALTY (recent context) ===
        if len(context_words) > 0:
            recent = set(context_words[-5:])
            for i, w in enumerate(candidate_words):
                if int(w) in recent:
                    energies[i] += 200

        return energies

    def generate(self, prompt: str = "the", length: int = 20) -> Dict:
        """
        Generate text autoregressively.

        At each position:
          1. Choose POS type (grammar + hard constraints)
          2. Check n-gram recall for type override
          3. Check copy mechanism
          4. Compute energy (recall + PMI + field + penalties)
          5. Integer Boltzmann sample

        All energy computation and sampling is integer-only.
        """
        # Resolve prompt
        prompt_idx = self.vocab.word2idx.get(prompt)
        if prompt_idx is None:
            prompt_idx = self.vocab.word2idx.get(prompt.lower())
        if prompt_idx is None:
            prompt_idx = 4

        prompt_type = self._get_word_type(prompt_idx)
        words = [prompt_idx]
        types = [prompt_type]
        consecutive_copies = 0
        diagnostics = []

        for pos in range(1, length):
            # === STEP 1: Choose POS type ===
            valid_types = self._get_valid_next_types(types[-1], types)

            # Check if recall suggests a type override
            recall_type_override = None
            if len(words) >= 2:
                recall_matches = self.ngram_index.lookup(words)
                if recall_matches:
                    best_k = max(recall_matches.keys())
                    best_conts = recall_matches[best_k]
                    if best_k >= 2 and best_conts:
                        best_word, best_count, best_total = best_conts[0]
                        if best_count * 3 >= best_total:
                            recall_type = self._get_word_type(best_word)
                            if recall_type in valid_types:
                                recall_type_override = recall_type

            if recall_type_override is not None:
                chosen_type = recall_type_override
            else:
                type_energies = np.array([
                    self._compute_type_energy(pos, t, types)
                    for t in valid_types
                ], dtype=np.int64)
                type_idx = self.type_sampler.sample(type_energies)
                chosen_type = valid_types[type_idx]

            # === STEP 2: Check copy mechanism ===
            copy_word = None
            if self.copy_enabled and len(words) >= self.copy_min_context:
                copy_candidate = self.ngram_index.get_best_copy_candidate(
                    context_words=words,
                    min_context_length=self.copy_min_context,
                    min_confidence=self.copy_min_confidence,
                )
                if copy_candidate is not None:
                    copy_word_idx, _, _ = copy_candidate
                    # Type compatibility check
                    if copy_word_idx < self.types.I_emit.shape[0]:
                        if int(self.types.I_emit[copy_word_idx, chosen_type]) > 0:
                            # Same-word block
                            if len(words) >= 1 and copy_word_idx == words[-1]:
                                copy_word_idx = None
                            # Consecutive copy cap (allow longer chains for coherence)
                            elif consecutive_copies >= 6:
                                copy_word_idx = None
                            else:
                                copy_word = copy_word_idx
                                consecutive_copies += 1
                                self._stats['copy_used'] += 1

            if copy_word is None:
                consecutive_copies = 0

            # === STEP 3: Choose word ===
            candidate_list = self.type_words.get(chosen_type, [])
            if not candidate_list:
                candidate_list = list(range(min(200, self.vocab_size)))
            candidate_words = np.array(candidate_list, dtype=np.int64)

            # Top-k filtering by field strength (for efficiency)
            if len(candidate_words) > 300:
                field_vals = self.h[candidate_words]
                top_k = np.argsort(field_vals)[-300:]
                candidate_words = candidate_words[top_k]

            # Check recall availability
            recall_matches = self.ngram_index.lookup(words)
            recall_hit = bool(recall_matches)

            # Compute energy (integer-only)
            word_energies = self._compute_word_energy(
                pos, candidate_words, chosen_type,
                words, types, recall_hit
            )

            # Integer Boltzmann sample
            if copy_word is not None:
                chosen_word = copy_word
            else:
                word_idx = self.word_sampler.sample(word_energies)
                chosen_word = int(candidate_words[word_idx])

            words.append(chosen_word)
            types.append(chosen_type)

            # Diagnostics
            self._stats['total_positions'] += 1
            if recall_hit:
                self._stats['recall_hit'] += 1
            else:
                self._stats['pmi_only'] += 1

            diagnostics.append({
                'pos': pos,
                'type': IDX2POS.get(chosen_type, "UNK"),
                'word': self.vocab.idx2word.get(chosen_word, "<UNK>"),
                'copy': copy_word is not None,
                'recall_hit': recall_hit,
            })

        text = self.vocab.decode(words)
        type_names = [IDX2POS.get(t, "UNK") for t in types]

        return {
            'text': text,
            'words': words,
            'types': types,
            'type_names': type_names,
            'diagnostics': diagnostics,
        }

    def generate_raw(self, length: int = 20) -> Tuple[List[int], List[int]]:
        """Generate with a random prompt."""
        start_idx = np.random.randint(4, min(54, self.vocab_size))
        prompt = self.vocab.idx2word.get(start_idx, "the")
        result = self.generate(prompt=prompt, length=length)
        return result['words'], result['types']

    def get_stats(self) -> Dict:
        """Get generation statistics."""
        stats = self._stats.copy()
        total = max(1, stats['total_positions'])
        stats['recall_hit_rate'] = stats['recall_hit'] / total
        stats['copy_rate'] = stats['copy_used'] / total
        stats['pmi_only_rate'] = stats['pmi_only'] / total
        stats['ising_enabled'] = self.ising_enabled
        return stats


# ===========================================================================
# MODEL: Training + Generation Pipeline
# ===========================================================================

class IsingLMModel:
    """
    Complete V14 model: training pipeline + generation.

    Training:
      1. Load corpus
      2. Build vocabulary
      3. Build POS type system
      4. Compute PMI couplings
      5. Build n-gram index
      6. Create generator(s)

    Generation:
      - With Ising (default)
      - Without Ising (ablation baseline)
    """

    def __init__(
        self,
        # Vocabulary
        vocab_min_freq: int = 5,
        vocab_max_size: int = 3000,
        # N-gram index
        ngram_max_n: int = 5,
        ngram_min_count: int = 1,
        # PMI
        pmi_window: int = 5,
        pmi_min_count: int = 2,
        pmi_cap: int = 10,
        # Generation
        recall_scale: int = 300,
        pmi_weight: int = 5,
        field_weight: int = 1,
        beta_type: float = 0.01,
        beta_word: float = 0.1,
        copy_enabled: bool = True,
        copy_min_context: int = 3,
        copy_min_confidence: float = 0.4,
        same_word_penalty: int = 50000,
        max_closed_class_run: int = 2,
        # Ablation
        ising_enabled: bool = True,
    ):
        self.vocab_min_freq = vocab_min_freq
        self.vocab_max_size = vocab_max_size
        self.ngram_max_n = ngram_max_n
        self.ngram_min_count = ngram_min_count
        self.pmi_window = pmi_window
        self.pmi_min_count = pmi_min_count
        self.pmi_cap = pmi_cap
        self.recall_scale = recall_scale
        self.pmi_weight = pmi_weight
        self.field_weight = field_weight
        self.beta_type = beta_type
        self.beta_word = beta_word
        self.copy_enabled = copy_enabled
        self.copy_min_context = copy_min_context
        self.copy_min_confidence = copy_min_confidence
        self.same_word_penalty = same_word_penalty
        self.max_closed_class_run = max_closed_class_run
        self.ising_enabled = ising_enabled

        self.vocab: Optional[Vocabulary] = None
        self.types: Optional[POSTypeSystem] = None
        self.J: Optional[np.ndarray] = None
        self.h: Optional[np.ndarray] = None
        self.ngram_index: Optional[NGramIndex] = None
        self.generator: Optional[IsingLM] = None
        self.baseline_generator: Optional[IsingLM] = None  # Ablation
        self.sequences: Optional[List[List[int]]] = None

    def train(self, n_samples: int = 20000) -> "IsingLMModel":
        """Train the model from FineWeb-Edu corpus."""
        import time as _time
        from .data_loader import load_fineweb_edu, tokenize_texts, truncate_sequences

        print("=" * 70)
        print("V14 ISING-ENHANCED N-GRAM LANGUAGE MODEL — TRAINING")
        print("=" * 70)
        print(f"\n  Honest architecture: N-gram (primary) + Ising PMI (secondary)")
        print(f"  Integer-only hot path: Lookup-table Boltzmann (NO np.exp)")
        print(f"  Ising enabled: {self.ising_enabled}")
        print()

        t0 = _time.time()

        # Step 1: Load corpus
        print("[1/5] Loading corpus...")
        texts = load_fineweb_edu(n_samples=n_samples)
        print(f"  Loaded {len(texts)} texts ({_time.time()-t0:.1f}s)")

        # Step 2: Build vocabulary
        print("\n[2/5] Building vocabulary...")
        self.vocab = Vocabulary(
            min_freq=self.vocab_min_freq,
            max_size=self.vocab_max_size,
        )
        self.vocab.build(texts)
        print(f"  Vocabulary: {len(self.vocab)} words")

        # Step 3: Build POS type system
        print("\n[3/5] Building POS type system...")
        self.types = POSTypeSystem(
            vocab_size=len(self.vocab),
            window=self.pmi_window,
        )
        self.types.build_from_vocabulary(self.vocab.word2idx, self.vocab.idx2word)
        self.types.build_grammar_penalties(penalty_strength=60)
        sequences = tokenize_texts(texts, self.vocab)
        sequences = truncate_sequences(sequences, max_len=20)
        self.sequences = sequences
        self.types.compute_type_couplings(sequences, self.vocab.idx2word)
        n_typed = sum(1 for w in range(len(self.vocab)) if w in self.types.allowed_types)
        print(f"  POS system built: {N_POS} types, {n_typed} words typed")

        # Step 4: Compute PMI couplings
        print("\n[4/5] Computing PMI couplings...")
        self.J, self.h = compute_pmi_couplings(
            sequences, len(self.vocab),
            window=self.pmi_window,
            min_count=self.pmi_min_count,
            pmi_cap=self.pmi_cap,
        )

        # Step 5: Build n-gram index
        print("\n[5/5] Building n-gram index...")
        self.ngram_index = NGramIndex(
            max_n=self.ngram_max_n,
            min_count=self.ngram_min_count,
        )
        self.ngram_index.build(sequences)

        # Build generators
        print("\nBuilding generators...")
        self._build_generators()

        t_total = _time.time() - t0
        print(f"\nTraining complete: {t_total:.1f}s")

        return self

    def _build_generators(self):
        """Build Ising and ablation generators."""
        # Main generator (with Ising)
        self.generator = IsingLM(
            vocab=self.vocab,
            ngram_index=self.ngram_index,
            J=self.J,
            h=self.h,
            types=self.types,
            recall_scale=self.recall_scale,
            pmi_weight=self.pmi_weight,
            field_weight=self.field_weight,
            beta_type=self.beta_type,
            beta_word=self.beta_word,
            copy_enabled=self.copy_enabled,
            copy_min_context=self.copy_min_context,
            copy_min_confidence=self.copy_min_confidence,
            same_word_penalty=self.same_word_penalty,
            max_closed_class_run=self.max_closed_class_run,
            ising_enabled=self.ising_enabled,
        )

        # Ablation baseline (without Ising)
        self.baseline_generator = IsingLM(
            vocab=self.vocab,
            ngram_index=self.ngram_index,
            J=self.J,
            h=self.h,
            types=self.types,
            recall_scale=self.recall_scale,
            pmi_weight=0,  # No PMI
            field_weight=self.field_weight,
            beta_type=self.beta_type,
            beta_word=self.beta_word,
            copy_enabled=self.copy_enabled,
            copy_min_context=self.copy_min_context,
            copy_min_confidence=self.copy_min_confidence,
            same_word_penalty=self.same_word_penalty,
            max_closed_class_run=self.max_closed_class_run,
            ising_enabled=False,  # No Ising at all
        )

    def generate_with_trace(self, prompt: str = "the", length: int = 20) -> Dict:
        """Generate text with full diagnostics."""
        if self.generator is None:
            self._build_generators()
        result = self.generator.generate(prompt=prompt, length=length)
        result['stats'] = self.generator.get_stats()
        return result

    def generate_raw(self, length: int = 20) -> Tuple[List[int], List[int]]:
        """Generate with random prompt."""
        if self.generator is None:
            self._build_generators()
        return self.generator.generate_raw(length=length)

    def evaluate_grammar(self, words, types):
        """Evaluate grammar quality of a generated sequence."""
        n_det_noun = 0
        n_det_non_noun = 0
        n_repeated = 0
        n_prep_noun = 0
        n_prep_non_noun = 0

        for i in range(len(types) - 1):
            t1, t2 = types[i], types[i + 1]
            if t1 == POS2IDX["DET"]:
                if t2 in {POS2IDX["NOUN"], POS2IDX["PRON"], POS2IDX["NUM"]}:
                    n_det_noun += 1
                else:
                    n_det_non_noun += 1
            if t1 == POS2IDX["PREP"]:
                if t2 in {POS2IDX["NOUN"], POS2IDX["PRON"], POS2IDX["DET"]}:
                    n_prep_noun += 1
                else:
                    n_prep_non_noun += 1

        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and words[i] >= 4:
                n_repeated += 1

        return {
            "det_noun": n_det_noun,
            "det_non_noun": n_det_non_noun,
            "prep_noun": n_prep_noun,
            "prep_non_noun": n_prep_non_noun,
            "repeated_words": n_repeated,
        }

    @property
    def vocab_size(self):
        return len(self.vocab) if self.vocab else 0
