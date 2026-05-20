"""
V12 Coherent Ising Language Model — Fixes for V11's Three Pathologies.

V11 was the breakthrough: exact n-gram recall (GFST-HMB-inspired) produced
coherent text for the first time. But three specific pathologies remain:

PATHOLOGY 1: POS-Recall Conflicts → Fragment Artifacts
  "to thanks" — recall finds "thanks" after "to" but grammar requires VERB after PART
  "of the of the" — recall echoes closed-class loops without structural breaks
  CAUSE: recall_suggests_type overrides grammar, function-word loops not blocked

PATHOLOGY 2: Verbatim Echoing → Stitching Gaps
  Copy mechanism produces exact training-data phrases but boundaries are jarring
  CAUSE: hard copy/generate switch with no smoothing at segment boundaries

PATHOLOGY 3: Weak Fallback → Random Words
  When no recall match exists, PMI coupling is essentially a bigram model
  CAUSE: no backoff hierarchy, no interpolation, unigram too weak

V12 FIXES:

F1: TYPE-COMPATIBLE RECALL + FUNCTION-WORD ANTI-LOOP
  - Recall candidates are FILTERED by the chosen POS type (not the other way around)
  - If recall suggests a word incompatible with the grammar-chosen type, either:
    (a) find the word under a compatible type alias, or
    (b) skip this recall candidate and use the next-best
  - Function-word anti-loop: detect closed-class repetitions and inject a
    structural break (force an open-class type after 2+ closed-class words)
  - "of the of the" prevention: same-word-at-distance-2 penalty extended to
    same-POS-at-distance-2 for closed-class types (DET, PREP, PART)

F2: BRIDGE STITCHING + COPY-FADE
  - At copy→generate boundaries, boost PMI coupling to the LAST 2 words of
    the copied segment (bridge context weighting)
  - Copy-fade: instead of hard copy/generate, use a smooth transition:
    - First position after copy ends: recall_bonus reduced by 50%
    - Second position: recall_bonus reduced by 25%
    - Third position: full recall_bonus again
  - Overlap context: when a copy segment ends, check if a DIFFERENT n-gram
    context overlaps with the copy's tail — if so, prefer that bridge

F3: KNESER-NEY BACKOFF + INTERPOLATION + CONTINUATION MODEL
  - Implement proper Kneser-Ney-style backoff in NGramIndex:
    - 5-gram match → use 5-gram continuation counts
    - 4-gram match → use 4-gram continuation counts
    - ... down to unigram
  - Key insight from Kneser-Ney: use CONTINUATION COUNTS (how many DIFFERENT
    contexts a word follows), not raw occurrence counts, for lower-order models
  - Interpolation: P(w|ctx) = λ₁·P_recall(w|ctx) + λ₂·P_PMI(w|ctx) + λ₃·P_unigram(w)
    - λ₁, λ₂, λ₃ adapt based on recall confidence
    - When recall hit: λ₁=0.7, λ₂=0.2, λ₃=0.1
    - When no recall: λ₁=0, λ₂=0.6, λ₃=0.4
  - Enhanced unigram: use field h[0, w] weighted by type compatibility

INTEGER-ONLY CONSTRAINT PRESERVED:
  - All n-gram counts, continuation counts, backoff weights: integers
  - All energy terms: integers
  - Interpolation weights: rational approximations (7/10, 2/10, 1/10)
  - FP only for final Boltzmann normalization (same compromise as V8-V11)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter

from .type_system import POSTypeSystem, N_POS, IDX2POS, POS2IDX
from .vocabulary import Vocabulary
from .pmi_couplings import PMICouplings
from .semantic_types import SemanticTypeSystem
from .dep_couplings import DependencyCouplings
from .caldera_nmf import CalderaNMF
from .exact_recall_v11_model import NGramIndex, ExactRecallV11Generator


# =============================================================================
# F3: Enhanced N-Gram Index with Kneser-Ney Backoff + Continuation Counts
# =============================================================================

class KneserNeyNGramIndex(NGramIndex):
    """
    Enhanced n-gram index with Kneser-Ney-style backoff and continuation counts.
    
    KEY INNOVATION OVER V11's NGramIndex:
    - Continuation counts: for lower-order backoff, count how many DISTINCT
      left-contexts a word follows (not raw occurrence count). This is the
      core Kneser-Ney insight that makes the backoff distribution meaningful.
    - Proper backoff: if no 5-gram match, fall back to 4-gram, then 3-gram,
      etc., with each level using continuation counts.
    - Discount: apply absolute discounting (d=1) for smoothed probability
      estimation, redistributing mass to lower-order models.
    
    All integer arithmetic. All counts are integers.
    """
    
    def __init__(self, max_n: int = 5, min_count: int = 1, discount: int = 1):
        """
        Args:
            max_n: Maximum n-gram length
            min_count: Minimum continuation count to store
            discount: Absolute discount for Kneser-Ney (integer, typically 0 or 1)
        """
        super().__init__(max_n=max_n, min_count=min_count)
        self.discount = discount
        
        # Continuation counts: for each word w, count the number of DISTINCT
        # contexts that w follows. This is the Kneser-Ney lower-order estimate.
        # continuation_count[k][w] = number of distinct contexts of length k
        #   that have w as a continuation
        self.continuation_count = {k: Counter() for k in range(1, max_n + 1)}
        
        # Total distinct contexts at each level (for normalization)
        self.total_distinct_contexts = {k: 0 for k in range(1, max_n + 1)}
        
        # Number of unique words that follow each context (for alpha computation)
        self.context_unique_continuations = {k: Counter() for k in range(1, max_n + 1)}
    
    def build(self, sequences: List[List[int]]):
        """
        Build the n-gram index with continuation counts.
        """
        # First, build the standard index using parent class
        super().build(sequences)
        
        # Now compute continuation counts (Kneser-Ney lower-order distribution)
        print("  Computing Kneser-Ney continuation counts...")
        
        for k in range(1, self.max_n + 1):
            # For each context of length k, track which words follow it
            # continuation_count[k][w] = number of distinct contexts where w appears
            for context, continuations in self.index[k].items():
                unique_words = set(continuations.keys())
                self.context_unique_continuations[k][context] = len(unique_words)
                
                for w in unique_words:
                    self.continuation_count[k][w] += 1
            
            self.total_distinct_contexts[k] = len(self.index[k])
            
            n_cont_words = len(self.continuation_count[k])
            print(f"    {k}-gram continuation: {n_cont_words:,} words with "
                  f"distinct contexts, {self.total_distinct_contexts[k]:,} contexts")
    
    def get_backoff_bonus(
        self,
        context_words: List[int],
        candidate_words: np.ndarray,
        recall_scale: int = 100,
        context_weight_factor: int = 4,
    ) -> np.ndarray:
        """
        Compute Kneser-Ney backoff bonus for candidate words.
        
        This replaces V11's simple recall bonus with a proper backoff hierarchy:
        - If 5-gram context matches: use 5-gram continuation (highest order)
        - Else if 4-gram matches: use 4-gram continuation
        - ... down to unigram
        
        At each level, the bonus is based on CONTINUATION COUNTS (not raw counts),
        which is the Kneser-Ney insight: P_KN(w) ∝ |{ctx : w follows ctx}|
        
        All integer arithmetic. Returns integer bonus array.
        """
        n_candidates = len(candidate_words)
        bonuses = np.zeros(n_candidates, dtype=np.int64)
        
        # Find the longest matching context
        matches = self.lookup(context_words)
        
        if matches:
            # Use only the LONGEST matching context (Kneser-Ney highest order)
            best_k = max(matches.keys())
            continuations = matches[best_k]
            total = self.context_totals[best_k][tuple(context_words[-best_k:])]
            
            # Apply discounting: for each continuation word, bonus = max(count - d, 0)
            # Plus a backoff weight proportional to the number of unique continuations
            d = self.discount
            
            # Build lookup
            cont_lookup = {}
            for word, count, _ in continuations:
                # Discounted count
                discounted = max(count - d, 0)
                # Context weight for exponential boost
                context_weight = context_weight_factor ** (best_k - 1)
                # Primary bonus from highest-order match
                primary_bonus = discounted * recall_scale * context_weight // max(1, total)
                # Backoff bonus: redistribute mass via continuation counts
                # |{unique continuations}| * discount / total * P_KN_lower(w)
                n_unique = self.context_unique_continuations[best_k].get(
                    tuple(context_words[-best_k:]), 1)
                backoff_weight = n_unique * d // max(1, total)
                
                cont_lookup[word] = int(primary_bonus + backoff_weight * recall_scale // 10)
            
            # Apply bonuses
            for i, w in enumerate(candidate_words):
                w_int = int(w)
                if w_int in cont_lookup:
                    bonuses[i] += cont_lookup[w_int]
        else:
            # No context match at all — use Kneser-Ney unigram (continuation count)
            # P_KN(w) = continuation_count[1][w] / total_distinct_contexts[1]
            total_ctx = max(1, self.total_distinct_contexts.get(1, 1))
            for i, w in enumerate(candidate_words):
                w_int = int(w)
                cont = self.continuation_count[1].get(w_int, 0)
                if cont > 0:
                    bonuses[i] += cont * recall_scale // (total_ctx * 2)
        
        return bonuses
    
    def get_continuation_probability(self, word_idx: int, order: int = 1) -> int:
        """
        Get the Kneser-Ney continuation probability (as integer * scale).
        
        P_KN(w) = continuation_count[order][w] / total_distinct_contexts[order]
        
        Returns integer (probability * 10000 for precision).
        """
        total = max(1, self.total_distinct_contexts.get(order, 1))
        cont = self.continuation_count[order].get(word_idx, 0)
        return (cont * 10000) // total


# =============================================================================
# F1+F2: Enhanced Generator with All Three Fixes
# =============================================================================

class CoherentV12Generator(ExactRecallV11Generator):
    """
    V12 Generator with fixes for V11's three pathologies.
    
    Inherits V11's exact recall mechanism and adds:
    
    F1: TYPE-COMPATIBLE RECALL
      - Recall candidates filtered by grammar-chosen POS type
      - Function-word anti-loop (max 2 closed-class words in a row)
      - Same-POS-at-distance-2 penalty for DET/PREP/PART
    
    F2: BRIDGE STITCHING + COPY-FADE
      - Boosted PMI coupling at copy→generate boundaries
      - Copy-fade: gradual transition from copy to generate
      - Overlap context: prefer n-grams that bridge copied segment tails
    
    F3: KNESER-NEY BACKOFF + INTERPOLATION
      - KneserNeyNGramIndex for proper backoff hierarchy
      - Interpolation: adaptive λ weights based on recall confidence
      - Enhanced unigram with type compatibility
    """
    
    def __init__(
        self,
        sampler,
        vocab: Vocabulary,
        ngram_index: NGramIndex,  # Can be KneserNeyNGramIndex or plain NGramIndex
        # V12-specific parameters
        max_closed_class_run: int = 2,     # Max consecutive closed-class words
        closed_class_loop_penalty: int = 300,  # Penalty for DET/PREP/PART loops
        copy_fade_strength: float = 0.5,   # How much to reduce recall bonus at boundaries
        bridge_context_boost: int = 3,      # Boost PMI coupling at copy→generate boundary
        # Interpolation weights (as integers, scaled by 10)
        lambda_recall_hit: int = 7,        # 0.7 for recall when hit
        lambda_pmi_hit: int = 2,           # 0.2 for PMI when recall hit
        lambda_unigram_hit: int = 1,       # 0.1 for unigram when recall hit
        lambda_pmi_miss: int = 6,          # 0.6 for PMI when no recall
        lambda_unigram_miss: int = 4,      # 0.4 for unigram when no recall
        # Recall parameters (inherited from V11)
        recall_scale: int = 100,
        context_weight_factor: int = 4,
        copy_min_context: int = 3,
        copy_min_confidence: float = 0.3,
        copy_enabled: bool = True,
        # Temperature parameters
        beta_type: float = 0.005,
        beta_word: float = 0.03,
        # Candidate filtering
        top_k_words: int = 200,
        # IDF weighting
        use_idf_coupling: bool = True,
        idf_scale: int = 8,
        # Field weight
        field_weight: float = 0.001,
        # Coupling weight
        coupling_weight: int = 10,
    ):
        # Initialize using V11's constructor (which handles all the base setup)
        super().__init__(
            sampler=sampler,
            vocab=vocab,
            ngram_index=ngram_index,
            recall_scale=recall_scale,
            context_weight_factor=context_weight_factor,
            copy_min_context=copy_min_context,
            copy_min_confidence=copy_min_confidence,
            copy_enabled=copy_enabled,
            beta_type=beta_type,
            beta_word=beta_word,
            top_k_words=top_k_words,
            use_idf_coupling=use_idf_coupling,
            idf_scale=idf_scale,
            field_weight=field_weight,
            coupling_weight=coupling_weight,
        )
        
        # V12-specific parameters
        self.max_closed_class_run = max_closed_class_run
        self.closed_class_loop_penalty = closed_class_loop_penalty
        self.copy_fade_strength = copy_fade_strength
        self.bridge_context_boost = bridge_context_boost
        
        # Interpolation weights (scaled by 10, must sum to 10)
        self.lambda_recall_hit = lambda_recall_hit
        self.lambda_pmi_hit = lambda_pmi_hit
        self.lambda_unigram_hit = lambda_unigram_hit
        self.lambda_pmi_miss = lambda_pmi_miss
        self.lambda_unigram_miss = lambda_unigram_miss
        
        # Check if we have KneserNeyNGramIndex
        self.use_kneser_ney = isinstance(ngram_index, KneserNeyNGramIndex)
        
        # V12 diagnostics
        self._v12_stats = {
            'total_positions': 0,
            'recall_hit': 0,
            'recall_miss': 0,
            'type_repair': 0,        # times recall was filtered by type
            'closed_loop_blocked': 0, # times closed-class loop was blocked
            'copy_fade_used': 0,     # times copy-fade was applied
            'bridge_used': 0,        # times bridge context was boosted
            'backoff_used': 0,       # times Kneser-Ney backoff was used
            'copy_loop_broken': 0,   # times copy loop was detected and broken
            'same_word_blocked': 0,  # times same-word repetition was blocked
        }
        
        # Track copy segment boundaries for stitching
        self._last_copy_end = -10   # position where last copy segment ended
        self._copy_run_length = 0   # current consecutive copy positions
        
        # Closed-class POS types
        self._closed_class_types = {
            POS2IDX["DET"], POS2IDX["PREP"], POS2IDX["PART"],
            POS2IDX["PRON"], POS2IDX["AUX"], POS2IDX["CONJ"],
        }
        
        # F1+: Hard type constraints after specific function words
        # "to" (PART) MUST be followed by VERB (not AUX — "to is" is unidiomatic)
        # "of" (PREP) MUST be followed by DET or NOUN-LIKE
        self._hard_type_constraints = {
            POS2IDX["PART"]: [POS2IDX["VERB"]],                   # "to" → verb ONLY
            POS2IDX["AUX"]: [POS2IDX["VERB"], POS2IDX["ADV"]],   # aux → verb/adv
        }
        
        # F2+: Copy loop detection — track recent copied words
        self._recent_copied_words = []  # last N words that were copied
        self._max_copy_loop_length = 4  # detect loops of this length or shorter
        self._max_same_copy_phrase = 3   # max times same phrase can repeat
        
        # F2++: Recall diversity — penalize recall patterns that have been used too often
        self._used_ngram_contexts = Counter()  # track how many times each n-gram context was followed
        self._recall_diversity_penalty = 500    # energy penalty for reusing a context too many times
        self._recall_diversity_threshold = 1     # after this many uses, start penalizing
        
        # F1++: Same-word penalty (independent of type)
        # Catches "the the" even when types are DET and PRON
        self._same_word_penalty = 500  # energy penalty for same word as previous
    
    # =========================================================================
    # F1: Type-Compatible Recall
    # =========================================================================
    
    def _count_closed_class_run(self, types: List[int]) -> int:
        """Count consecutive closed-class types at the END of the type list."""
        run = 0
        for t in reversed(types):
            if t in self._closed_class_types:
                run += 1
            else:
                break
        return run
    
    def _get_valid_next_types_v12(self, prev_type: int, types: List[int], 
                                       words: List[int] = None) -> List[int]:
        """
        Get valid next POS types with F1+: function-word anti-loop + hard constraints.
        
        If we've had max_closed_class_run consecutive closed-class words,
        FORCE an open-class type (NOUN, VERB, ADJ, ADV) to break the loop.
        
        F1+: Apply hard type constraints after specific function words:
        - "to" (PART) → MUST be VERB or AUX
        - aux → MUST be VERB or ADV
        """
        # Get base valid types from grammar
        valid = self._get_valid_next_types(prev_type)
        
        # F1+: Hard type constraints after specific function words
        if prev_type in self._hard_type_constraints:
            constrained = self._hard_type_constraints[prev_type]
            # Intersect with grammar-allowed types
            constrained_valid = [t for t in valid if t in constrained]
            if constrained_valid:
                valid = constrained_valid
                self._v12_stats['type_repair'] += 1
        
        # Check if we're in a closed-class run
        closed_run = self._count_closed_class_run(types)
        
        if closed_run >= self.max_closed_class_run:
            # Force open-class: filter valid types to open-class only
            open_types = [t for t in valid if t not in self._closed_class_types]
            if open_types:
                valid = open_types
                self._v12_stats['closed_loop_blocked'] += 1
        
        return valid
    
    def _filter_recall_by_type(
        self, 
        recall_matches: Dict[int, List[Tuple[int, int, int]]],
        chosen_type: int,
    ) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        F1: Filter recall candidates to only include type-compatible words.
        
        If recall suggests "thanks" (NOUN/VERB) but grammar chose VERB,
        keep "thanks" as VERB but remove it as NOUN.
        If recall suggests "research" (NOUN) but grammar chose PREP,
        remove "research" from recall — it can't be a preposition.
        """
        filtered = {}
        
        for k, continuations in recall_matches.items():
            filtered_conts = []
            for word, count, total in continuations:
                # Check if this word can have the chosen type
                if word < self.I_emit.shape[0]:
                    emit_val = self.I_emit[word, chosen_type]
                    if emit_val > 0:
                        # Word is compatible with chosen type — keep it
                        filtered_conts.append((word, count, total))
                    # else: word is incompatible with chosen type — filter it out
                else:
                    # Unknown word — keep it (rare case)
                    filtered_conts.append((word, count, total))
            
            if filtered_conts:
                filtered[k] = filtered_conts
        
        return filtered
    
    def _get_word_type_compat(self, word_idx: int, chosen_type: int) -> bool:
        """Check if a word is compatible with a given POS type."""
        if word_idx >= self.I_emit.shape[0]:
            return True  # unknown word, assume compatible
        return int(self.I_emit[word_idx, chosen_type]) > 0
    
    # =========================================================================
    # F2+: Copy Loop Detection
    # =========================================================================
    
    def _detect_copy_loop(self, candidate_word: int, words: List[int]) -> bool:
        """
        F2+: Detect if adding candidate_word would create a repetitive loop.
        
        A copy loop occurs when the same phrase repeats:
        "to maintain a christmas tree it is important to maintain a christmas tree"
        
        Detection: check if the recent word sequence (last N words + candidate)
        appears as a repeated pattern. Specifically:
        1. Check if candidate_word matches a word from the recent sequence
           at a position that would complete a repeated phrase
        2. Check if any phrase of length 2-4 has already appeared 3+ times
        
        Returns True if a loop is detected.
        """
        if len(words) < 4:
            return False
        
        # Check 1: Simple same-word repetition
        # If candidate word appeared in the last 2 positions, likely a loop
        if len(words) >= 2 and candidate_word == words[-1] == words[-2]:
            return True
        
        # Check 2: Phrase repetition — look for repeated phrases of length 2-4
        # If the last 2-4 words + candidate match an earlier sequence, it's a loop
        all_words = words + [candidate_word]
        
        for phrase_len in range(2, min(5, len(all_words) // 2 + 1)):
            # Current phrase (last phrase_len words including candidate)
            current_phrase = tuple(all_words[-phrase_len:])
            
            # Count occurrences of this phrase in the full sequence
            count = 0
            for i in range(len(all_words) - phrase_len + 1):
                candidate_phrase = tuple(all_words[i:i+phrase_len])
                if candidate_phrase == current_phrase:
                    count += 1
            
            # If this phrase appears more than max_same_copy_phrase times, it's a loop
            if count > self._max_same_copy_phrase:
                return True
        
        # Check 3: Copy run too long — if we've been copying for many steps,
        # there's a risk of entering a loop. Break after 8 consecutive copies.
        if self._copy_run_length >= 8:
            return True
        
        return False
    
    # =========================================================================
    # F2: Bridge Stitching + Copy-Fade
    # =========================================================================
    
    def _apply_copy_fade(self, recall_bonuses: np.ndarray, pos: int) -> np.ndarray:
        """
        F2: Apply copy-fade at copy→generate boundaries.
        
        When transitioning from a copied segment to a generated segment,
        gradually reduce the recall bonus to smooth the transition.
        
        Copy-fade: after a copy segment ends, the next 2 positions get
        reduced recall bonus so the model smoothly transitions from
        "copy mode" to "generate mode".
        """
        dist_from_copy_end = pos - self._last_copy_end
        
        if dist_from_copy_end == 1:
            # First position after copy: 50% recall bonus
            recall_bonuses = (recall_bonuses * self.copy_fade_strength).astype(np.int64)
            self._v12_stats['copy_fade_used'] += 1
        elif dist_from_copy_end == 2:
            # Second position after copy: 75% recall bonus
            recall_bonuses = (recall_bonuses * (1 - self.copy_fade_strength / 2)).astype(np.int64)
            self._v12_stats['copy_fade_used'] += 1
        
        return recall_bonuses
    
    def _compute_bridge_coupling(
        self,
        pos: int,
        candidate_words: np.ndarray,
        fixed_words: List[int],
    ) -> np.ndarray:
        """
        F2: Compute boosted PMI coupling at copy→generate boundaries.
        
        At the boundary between a copied segment and a generated segment,
        boost the PMI coupling to the last 2-3 words of the copied segment.
        This creates a "bridge" that connects the two segments more smoothly.
        """
        n_candidates = len(candidate_words)
        bridge_bonus = np.zeros(n_candidates, dtype=np.int64)
        
        dist_from_copy_end = pos - self._last_copy_end
        if dist_from_copy_end <= 2 and dist_from_copy_end > 0:
            # We're near a copy boundary — boost coupling to the copy's tail
            # Use the last 3 words of the copy segment as bridge context
            bridge_start = max(0, self._last_copy_end - 2)
            bridge_end = self._last_copy_end + 1
            bridge_words = fixed_words[bridge_start:bridge_end]
            
            if bridge_words:
                bridge_arr = np.array(bridge_words, dtype=np.int64)
                # Boosted coupling: normal coupling * bridge_context_boost
                coupling_block = self.J[np.ix_(candidate_words, bridge_arr)]
                coupling_sums = coupling_block.sum(axis=1)
                bridge_bonus = (coupling_sums * self.bridge_context_boost).astype(np.int64)
                self._v12_stats['bridge_used'] += 1
        
        return bridge_bonus
    
    # =========================================================================
    # F3: Interpolation + Enhanced Fallback
    # =========================================================================
    
    def _compute_interpolated_energy(
        self,
        pos: int,
        candidate_words: np.ndarray,
        word_type: int,
        fixed_words: List[int],
        fixed_types: List[int],
        recall_hit: bool,
    ) -> np.ndarray:
        """
        F3: Compute energy using interpolation of recall, PMI, and unigram.
        
        Instead of just adding recall_bonus to PMI energy (V11's approach),
        we use proper interpolation with adaptive weights:
        
        E(w) = -λ₁·recall_bonus(w) - λ₂·PMI_coupling(w) - λ₃·field(w) + penalties
        
        When recall hits: λ₁=7/10, λ₂=2/10, λ₃=1/10
        When recall misses: λ₁=0, λ₂=6/10, λ₃=4/10
        
        This ensures that:
        - When recall is available, it dominates but PMI/unigram provide diversity
        - When recall misses, PMI and unigram provide a meaningful fallback
        - The unigram term (field) provides base frequency information
        """
        n_candidates = len(candidate_words)
        
        # === Component 1: Recall bonus ===
        if self.use_kneser_ney:
            # Use Kneser-Ney backoff for more graceful fallback
            recall_bonuses = self.ngram_index.get_backoff_bonus(
                context_words=fixed_words,
                candidate_words=candidate_words,
                recall_scale=self.recall_scale,
                context_weight_factor=self.context_weight_factor,
            )
        else:
            # Standard V11 recall
            recall_bonuses = self.ngram_index.get_recall_bonus(
                context_words=fixed_words,
                candidate_words=candidate_words,
                recall_scale=self.recall_scale,
                context_weight_factor=self.context_weight_factor,
                longest_only=self.recall_longest_only,
            )
        
        # F1: Filter recall by type compatibility
        # Check which candidates are type-compatible
        type_compat = np.ones(n_candidates, dtype=np.int64)
        for i, w in enumerate(candidate_words):
            if not self._get_word_type_compat(int(w), word_type):
                type_compat[i] = 0
                self._v12_stats['type_repair'] += 1
        
        # Zero out recall bonuses for type-incompatible words
        recall_bonuses *= type_compat
        
        # F1++: ABSOLUTE same-word block — zero out recall bonus for the
        # word that was just generated, regardless of type assignment
        # This prevents "the the" (DET→PRON) or any other same-word repetition
        if len(fixed_words) >= 1:
            prev_word = fixed_words[-1]
            for i, w in enumerate(candidate_words):
                if int(w) == prev_word:
                    recall_bonuses[i] = 0  # ABSOLUTE: no recall bonus for same word
                    self._v12_stats['same_word_blocked'] += 1
        
        # F2: Apply copy-fade
        recall_bonuses = self._apply_copy_fade(recall_bonuses, pos)
        
        # F2++: Recall diversity penalty — penalize overused n-gram contexts
        # If a specific n-gram context has been followed too many times,
        # reduce the recall bonus for its continuations to encourage diversity
        if len(fixed_words) >= 2:
            # Check last 2-4 word context
            for ctx_len in range(min(4, len(fixed_words)), 1, -1):
                ctx_key = tuple(fixed_words[-ctx_len:])
                use_count = self._used_ngram_contexts.get(ctx_key, 0)
                if use_count > self._recall_diversity_threshold:
                    # Reduce recall bonus proportionally to overuse
                    diversity_factor = max(1, use_count - self._recall_diversity_threshold)
                    # Each extra use reduces bonus by recall_diversity_penalty
                    recall_bonuses = np.maximum(
                        0, 
                        recall_bonuses - diversity_factor * self._recall_diversity_penalty
                    ).astype(np.int64)
                    break
        
        # === Component 2: PMI coupling ===
        pmi_energies = np.zeros(n_candidates, dtype=np.int64)
        
        # Standard PMI coupling to left context
        context_start = max(0, pos - self.window)
        context_words = fixed_words[context_start:pos]
        if len(context_words) > 0:
            context_arr = np.array(context_words, dtype=np.int64)
            coupling_block = self.J[np.ix_(candidate_words, context_arr)]
            coupling_sums = coupling_block.sum(axis=1)
            pmi_energies -= coupling_sums
        
        # F2: Bridge coupling at copy boundaries
        bridge_bonus = self._compute_bridge_coupling(pos, candidate_words, fixed_words)
        pmi_energies -= bridge_bonus
        
        # === Component 3: Unigram (field) ===
        unigram_energies = np.zeros(n_candidates, dtype=np.int64)
        field_vals = self.h[pos % self.seq_len, candidate_words].astype(np.float64) * self.field_weight
        unigram_energies -= field_vals.astype(np.int64)
        
        # Enhanced unigram: weight by type compatibility
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int < self.I_emit.shape[0]:
                emit_strength = int(self.I_emit[w_int, word_type])
                if emit_strength > 0:
                    # Boost unigram for type-compatible words
                    unigram_energies[i] -= emit_strength * 5  # small bonus
        
        # === Interpolation ===
        if recall_hit and recall_bonuses.max() > 0:
            # Recall hit: λ_recall=7/10, λ_pmi=2/10, λ_unigram=1/10
            # E(w) = -(7*recall + 2*pmi + 1*unigram) / 10
            # Note: all components are already negative (bonus), so we add them
            energies = -(
                self.lambda_recall_hit * recall_bonuses +
                self.lambda_pmi_hit * (-pmi_energies) +  # negate: pmi_energies is negative bonus
                self.lambda_unigram_hit * (-unigram_energies)  # same
            ) // 10
        else:
            # No recall: λ_pmi=6/10, λ_unigram=4/10
            energies = -(
                self.lambda_pmi_miss * (-pmi_energies) +
                self.lambda_unigram_miss * (-unigram_energies)
            ) // 10
        
        # === Add penalties (not interpolated — always apply) ===
        
        # Repetition penalty — ENHANCED: also penalize same-word at distance 2
        if len(fixed_words) > 0 and self.repetition_penalty > 0:
            context_set = set(context_words) if len(context_words) > 0 else set()
            # Also check words at distance 2 (catches "the ... the")
            recent_words = set(fixed_words[max(0, len(fixed_words)-5):])
            for i, w in enumerate(candidate_words):
                w_int = int(w)
                # Immediate context: strong penalty
                if w_int in context_set:
                    energies[i] += self.repetition_penalty
                # Recent context (distance 2-5): moderate penalty
                elif w_int in recent_words:
                    energies[i] += self.repetition_penalty // 2
                    self._v12_stats['same_word_blocked'] += 1
        
        # F1++: Same-word penalty (independent of type)
        # Catches "the the" even when first "the" is DET and second is PRON
        # Must be STRONG — recall bonuses can be 10K+, so we need a big penalty
        if len(fixed_words) >= 1 and self._same_word_penalty > 0:
            prev_word = fixed_words[-1]
            for i, w in enumerate(candidate_words):
                if int(w) == prev_word:
                    # Base penalty + proportional to max recall bonus in this position
                    # This ensures the penalty can overcome even strong recall
                    max_recall = max(1, int(recall_bonuses.max())) if recall_bonuses.max() > 0 else 1
                    energies[i] += self._same_word_penalty + max_recall // 2
        
        # F1: Closed-class same-POS-at-distance-2 penalty
        # Prevents "of the of the", "in the in the", "the the" patterns
        if word_type in self._closed_class_types and len(fixed_types) >= 2:
            for d in range(1, min(4, len(fixed_types) + 1)):
                if fixed_types[-d] == word_type and word_type in {
                    POS2IDX["DET"], POS2IDX["PREP"], POS2IDX["PART"]
                }:
                    # Scale penalty by proximity (closer = stronger)
                    energies += self.closed_class_loop_penalty // d
                    break
            
            # Extra penalty for DET→DET immediate adjacency ("the the")
            if word_type == POS2IDX["DET"] and len(fixed_types) >= 1:
                if fixed_types[-1] == POS2IDX["DET"]:
                    energies += self.closed_class_loop_penalty * 3
        
        # Emission compatibility (non-symmetric mode)
        if not self.use_symmetric_emission:
            emit_vals = self.I_emit[candidate_words, word_type]
            pos_mask = emit_vals > 0
            neg_mask = emit_vals < 0
            energies[pos_mask] -= emit_vals[pos_mask] * self.emission_bonus
            energies[neg_mask] += self.emission_penalty
        
        # Track recall diagnostics
        self._recall_stats['total_positions'] += 1
        if recall_bonuses.max() > 0:
            self._recall_stats['recall_hits'] += 1
            self._recall_stats['avg_recall_bonus'] += int(recall_bonuses.mean())
            self._recall_stats['max_recall_bonus'] = max(
                self._recall_stats['max_recall_bonus'],
                int(recall_bonuses.max())
            )
        
        return energies
    
    # =========================================================================
    # V12 Generation: The Main Loop
    # =========================================================================
    
    def generate(
        self, prompt: str, length: int = 15
    ) -> Dict:
        """
        Generate text with all V12 fixes applied.
        
        At each position:
          1. Choose POS type (grammar-driven + F1: closed-class anti-loop)
          2. Look up recall candidates (F1: type-compatible filter)
          3. Check copy mechanism (F2: with copy-fade)
          4. Compute interpolated energy (F3: recall + PMI + unigram)
          5. Sample from Boltzmann distribution
        """
        # Resolve prompt word
        prompt_idx = self.vocab.word2idx.get(prompt, None)
        if prompt_idx is None:
            prompt_idx = self.vocab.word2idx.get(prompt.lower(), None)
        if prompt_idx is None:
            prompt_idx = 4  # usually "the"

        prompt_type = self._get_word_type(prompt_idx)

        words = [prompt_idx]
        types = [prompt_type]

        # Reset copy tracking
        self._last_copy_end = -10
        self._copy_run_length = 0
        self._recent_copied_words = []
        self._used_ngram_contexts = Counter()

        position_diagnostics = []

        for pos in range(1, length):
            # === Step 1: Choose POS type (with F1+ anti-loop + hard constraints) ===
            valid_types = self._get_valid_next_types_v12(types[-1], types, words)
            
            # Check if recall suggests a type (with F1+: constrained by hard type rules)
            recall_type_override = None
            copy_word = None
            diag_copy = False
            
            if self.recall_suggests_type and len(words) >= 2:
                recall_matches = self.ngram_index.lookup(words)
                if recall_matches:
                    best_k = max(recall_matches.keys())
                    best_conts = recall_matches[best_k]
                    if best_k >= 2 and best_conts:
                        best_recall_word = best_conts[0][0]
                        best_recall_count = best_conts[0][1]
                        best_recall_total = best_conts[0][2]
                        
                        if best_recall_count * 3 >= best_recall_total:
                            recall_word_type = self._get_word_type(best_recall_word)
                            # F1+: Only override type if it's in the valid set
                            # AND respects hard constraints (e.g. PART→VERB)
                            if recall_word_type in valid_types:
                                recall_type_override = recall_word_type
            
            if recall_type_override is not None:
                chosen_type = recall_type_override
            else:
                type_energies = np.array([
                    self._compute_type_energy(pos, t, types)
                    for t in valid_types
                ], dtype=np.int64)
                type_idx = self._boltzmann_sample(type_energies, self.beta_type)
                chosen_type = valid_types[type_idx]

            # === Step 2: Check for exact recall copy (with F1: type check + F2+: loop detection) ===
            if self.copy_enabled and len(words) >= self.copy_min_context:
                copy_candidate = self.ngram_index.get_best_copy_candidate(
                    context_words=words,
                    min_context_length=self.copy_min_context,
                    min_confidence=self.copy_min_confidence,
                )
                if copy_candidate is not None:
                    copy_word_idx, copy_count, copy_total = copy_candidate
                    # F2+: Copy loop detection — check if we'd repeat a phrase
                    loop_detected = self._detect_copy_loop(copy_word_idx, words)
                    # F1: Check type compatibility
                    type_ok = self._get_word_type_compat(copy_word_idx, chosen_type)
                    if type_ok and not loop_detected:
                        copy_word = copy_word_idx
                        self._recall_stats['copy_used'] += 1
                        self._copy_run_length += 1
                        self._recent_copied_words.append(copy_word_idx)
                    elif loop_detected:
                        self._v12_stats['copy_loop_broken'] += 1
                        # Don't use copy — force generation instead
                        copy_word = None
            
            # === Step 3: Choose word ===
            candidate_words_list = self.type_words.get(chosen_type, [])
            if not candidate_words_list:
                candidate_words_list = list(range(min(200, self.vocab_size)))
            candidate_words = np.array(candidate_words_list, dtype=np.int64)
            
            if len(candidate_words) > self.top_k_words:
                field_vals = self.h[pos % self.seq_len, candidate_words]
                top_k_indices = np.argsort(field_vals)[-self.top_k_words:]
                candidate_words = candidate_words[top_k_indices]
            
            # Check recall availability for interpolation weight selection
            recall_matches_raw = self.ngram_index.lookup(words)
            recall_hit = bool(recall_matches_raw)
            
            # F1: Filter recall by type compatibility
            if recall_matches_raw:
                recall_matches_filtered = self._filter_recall_by_type(
                    recall_matches_raw, chosen_type
                )
                recall_hit = bool(recall_matches_filtered)
            
            # Compute interpolated energy (F3: with recall/PMI/unigram interpolation)
            word_energies = self._compute_interpolated_energy(
                pos, candidate_words, chosen_type, words, types, recall_hit
            )
            
            if copy_word is not None:
                # COPY: Use the exact recalled continuation
                chosen_word = copy_word
                diag_copy = True
                # Track copy segment boundary
                self._last_copy_end = pos
            else:
                # GENERATE: Sample from Boltzmann with interpolated energy
                word_idx = self._boltzmann_sample(word_energies, self.beta_word)
                chosen_word = int(candidate_words[word_idx])
                
                # Track copy→generate boundary
                if self._copy_run_length > 0:
                    # We just ended a copy segment
                    self._last_copy_end = pos - 1
                    self._copy_run_length = 0
                
                # Not a copy — clear recent copied words slowly
                if self._recent_copied_words and len(self._recent_copied_words) > 10:
                    self._recent_copied_words = self._recent_copied_words[-5:]

            words.append(chosen_word)
            types.append(chosen_type)
            
            # Track n-gram context usage for diversity penalty
            if len(words) >= 3:
                for ctx_len in range(2, min(5, len(words))):
                    ctx_key = tuple(words[-ctx_len-1:-1])  # context before the chosen word
                    self._used_ngram_contexts[ctx_key] += 1

            # Diagnostics
            top5_indices = np.argsort(word_energies)[:5]
            
            recall_info = {}
            for k, conts in recall_matches_raw.items():
                for w, c, t in conts[:3]:
                    if w == chosen_word:
                        recall_info[f'{k}gram'] = f"{self.vocab.idx2word.get(w, '?')}({c}/{t})"
            
            diag = {
                'pos': pos,
                'chosen_type': IDX2POS.get(chosen_type, "UNK"),
                'chosen_word': self.vocab.idx2word.get(chosen_word, "<UNK>"),
                'copy_used': diag_copy,
                'recall_hit': recall_hit,
                'top5_words': [self.vocab.idx2word.get(int(candidate_words[i]), "<UNK>")
                               for i in top5_indices],
                'top5_energies': [int(word_energies[i]) for i in top5_indices],
                'recall_matches': recall_info,
            }
            position_diagnostics.append(diag)
            
            # Track V12 stats
            self._v12_stats['total_positions'] += 1
            if recall_hit:
                self._v12_stats['recall_hit'] += 1
            else:
                self._v12_stats['recall_miss'] += 1

        # Decode
        text = self.vocab.decode(words)
        type_names = [IDX2POS.get(t, "UNK") for t in types]

        return {
            'text': text,
            'words': words,
            'types': types,
            'type_names': type_names,
            'diagnostics': position_diagnostics,
        }
    
    def get_v12_stats(self) -> Dict:
        """Get V12-specific statistics."""
        stats = self._v12_stats.copy()
        total = max(1, stats['total_positions'])
        stats['recall_hit_rate'] = stats['recall_hit'] / total
        stats['type_repair_rate'] = stats['type_repair'] / total
        stats['closed_loop_block_rate'] = stats['closed_loop_blocked'] / total
        stats['copy_fade_rate'] = stats['copy_fade_used'] / total
        stats['bridge_rate'] = stats['bridge_used'] / total
        stats['copy_loop_break_rate'] = stats['copy_loop_broken'] / total
        stats['same_word_block_rate'] = stats['same_word_blocked'] / total
        return stats


# =============================================================================
# V12 Model: Training + Generation
# =============================================================================

class CoherentV12Model:
    """
    V12 Coherent Ising Language Model.
    
    Combines V11's exact recall with three targeted fixes:
    
    F1: Type-Compatible Recall + Function-Word Anti-Loop
      - Fixes "to thanks" (POS/recall conflict)
      - Fixes "of the of the" (closed-class loops)
    
    F2: Bridge Stitching + Copy-Fade
      - Smooth transitions between copied and generated segments
      - No more jarring boundaries
    
    F3: Kneser-Ney Backoff + Interpolation + Enhanced Fallback
      - Graceful fallback when no recall match
      - Continuation counts for better lower-order estimates
      - Adaptive interpolation weights
    
    INTEGER-ONLY CONSTRAINT PRESERVED.
    """
    
    def __init__(self, **kwargs):
        # Extract V12-specific parameters
        self.v12_max_closed_class_run = kwargs.pop('max_closed_class_run', 2)
        self.v12_closed_class_loop_penalty = kwargs.pop('closed_class_loop_penalty', 300)
        self.v12_copy_fade_strength = kwargs.pop('copy_fade_strength', 0.5)
        self.v12_bridge_context_boost = kwargs.pop('bridge_context_boost', 3)
        self.v12_lambda_recall_hit = kwargs.pop('lambda_recall_hit', 7)
        self.v12_lambda_pmi_hit = kwargs.pop('lambda_pmi_hit', 2)
        self.v12_lambda_unigram_hit = kwargs.pop('lambda_unigram_hit', 1)
        self.v12_lambda_pmi_miss = kwargs.pop('lambda_pmi_miss', 6)
        self.v12_lambda_unigram_miss = kwargs.pop('lambda_unigram_miss', 4)
        self.v12_use_kneser_ney = kwargs.pop('use_kneser_ney', True)
        
        # V11 parameters
        self.v11_recall_scale = kwargs.pop('recall_scale', 100)
        self.v11_context_weight_factor = kwargs.pop('context_weight_factor', 4)
        self.v11_copy_min_context = kwargs.pop('copy_min_context', 3)
        self.v11_copy_min_confidence = kwargs.pop('copy_min_confidence', 0.3)
        self.v11_copy_enabled = kwargs.pop('copy_enabled', True)
        self.v11_ngram_max_n = kwargs.pop('ngram_max_n', 5)
        self.v11_ngram_min_count = kwargs.pop('ngram_min_count', 1)
        
        # V10 autoregressive params
        self.v10_beta_type = kwargs.pop('beta_type', 0.005)
        self.v10_beta_word = kwargs.pop('beta_word', 0.03)
        self.v10_top_k_words = kwargs.pop('top_k_words', 200)
        self.v10_use_idf_coupling = kwargs.pop('use_idf_coupling', True)
        self.v10_idf_scale = kwargs.pop('idf_scale', 8)
        self.v10_field_weight = kwargs.pop('field_weight', 0.001)
        self.v10_coupling_weight = kwargs.pop('coupling_weight', 10)
        
        # Import and create V8 model for training infrastructure
        from .enhanced_v8_model import EnhancedV8Model
        self.v8_model = EnhancedV8Model(**kwargs)
        
        # Will be built during training
        self.ngram_index = None
        self.v12_generator = None
    
    def train(self, n_samples: int = 20000):
        """Train the model: V8 pipeline + N-gram index building."""
        import time as _time
        from .data_loader import load_fineweb_edu, tokenize_texts, truncate_sequences
        
        print("=" * 70)
        print("V12 TRAINING: V8 Pipeline + KN/N-gram Index Building")
        print("=" * 70)
        
        # Step 1: Train V8 model
        self.v8_model.train(n_samples=n_samples)
        
        # Step 2: Build N-gram index
        print("\n--- Building N-Gram Index for Exact Token Recall ---")
        
        t0 = _time.time()
        texts = load_fineweb_edu(n_samples=n_samples)
        sequences = tokenize_texts(texts, self.v8_model.vocab)
        sequences = truncate_sequences(sequences, max_len=self.v8_model.seq_len)
        print(f"  Re-loaded {len(sequences)} sequences for n-gram indexing ({_time.time()-t0:.1f}s)")
        
        # Build appropriate index type
        if self.v12_use_kneser_ney:
            print("  Using KneserNeyNGramIndex (with continuation counts)")
            self.ngram_index = KneserNeyNGramIndex(
                max_n=self.v11_ngram_max_n,
                min_count=self.v11_ngram_min_count,
                discount=1,
            )
        else:
            self.ngram_index = NGramIndex(
                max_n=self.v11_ngram_max_n,
                min_count=self.v11_ngram_min_count,
            )
        
        self.ngram_index.build(sequences)
        self._sequences = sequences
        
        stats = self.ngram_index.get_stats()
        print(f"\n  N-gram index built: {stats}")
        
        # Step 3: Build the V12 generator
        self._build_generator()
    
    def _build_generator(self):
        """Build the V12 generator from trained components."""
        self.v12_generator = CoherentV12Generator(
            sampler=self.v8_model.sampler,
            vocab=self.v8_model.vocab,
            ngram_index=self.ngram_index,
            # V12-specific
            max_closed_class_run=self.v12_max_closed_class_run,
            closed_class_loop_penalty=self.v12_closed_class_loop_penalty,
            copy_fade_strength=self.v12_copy_fade_strength,
            bridge_context_boost=self.v12_bridge_context_boost,
            lambda_recall_hit=self.v12_lambda_recall_hit,
            lambda_pmi_hit=self.v12_lambda_pmi_hit,
            lambda_unigram_hit=self.v12_lambda_unigram_hit,
            lambda_pmi_miss=self.v12_lambda_pmi_miss,
            lambda_unigram_miss=self.v12_lambda_unigram_miss,
            # Recall parameters
            recall_scale=self.v11_recall_scale,
            context_weight_factor=self.v11_context_weight_factor,
            copy_min_context=self.v11_copy_min_context,
            copy_min_confidence=self.v11_copy_min_confidence,
            copy_enabled=self.v11_copy_enabled,
            # V10 parameters
            beta_type=self.v10_beta_type,
            beta_word=self.v10_beta_word,
            top_k_words=self.v10_top_k_words,
            use_idf_coupling=self.v10_use_idf_coupling,
            idf_scale=self.v10_idf_scale,
            field_weight=self.v10_field_weight,
            coupling_weight=self.v10_coupling_weight,
        )
    
    def generate_with_trace(self, prompt: str = "the", length: int = 15) -> Dict:
        """Generate text with full diagnostics."""
        if self.v12_generator is None:
            self._build_generator()
        
        result = self.v12_generator.generate(prompt=prompt, length=length)
        
        # Add recall + V12 stats
        result['recall_stats'] = self.v12_generator.get_recall_stats()
        result['v12_stats'] = self.v12_generator.get_v12_stats()
        
        # V8-compatible fields
        result['energy'] = 0
        result['demon_stats'] = {'demon_energy': 0, 'acceptance_rate': 1.0}
        result['marginal_stats'] = {
            'avg_consecutive_count': 0,
            'max_consecutive_count': 0,
            'stuck_5plus': 0,
            'flip_rate': 1.0,
        }
        
        return result
    
    def generate_raw(self, length: int = 15) -> Tuple[List[int], List[int]]:
        """Generate raw word/type arrays for evaluation."""
        if self.v12_generator is None:
            self._build_generator()
        
        start_idx = np.random.randint(4, min(54, self.v12_generator.vocab_size))
        prompt = self.v8_model.vocab.idx2word.get(start_idx, "the")
        result = self.v12_generator.generate(prompt=prompt, length=length)
        return result['words'], result['types']
    
    def evaluate_grammar(self, words, types):
        """Delegate to V8's grammar evaluation."""
        return self.v8_model.evaluate_grammar(words, types)
    
    @property
    def vocab(self):
        return self.v8_model.vocab
    
    @property
    def sequences(self):
        return getattr(self, '_sequences', [])
