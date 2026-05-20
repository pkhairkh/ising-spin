"""
V13 Fixed Ising Language Model — Regression Fixes for V12.

V12 had the RIGHT ideas but WRONG implementations that caused severe regression:
  - Interpolation diluted recall from 100% to 70%, adding noisy PMI
  - Copy mechanism at 60% rate bypassed ALL V12 fixes
  - Kneser-Ney divided by total, weakening bonuses for common contexts
  - min_context=2 made copy far too aggressive
  - Copy loop detection missed cross-sentence repetitions

V13 FIXES (each directly addresses a V12 regression):

R1: RESTORE FULL RECALL STRENGTH
  - Recall bonus goes back to 100% (not 70% via interpolation)
  - PMI/unigram are added as SUPPLEMENTARY bonuses, not interpolated
  - When recall hits: E = -recall_bonus - 0.3*PMI - 0.1*unigram + penalties
  - When recall misses: E = -PMI_coupling - 0.5*unigram + penalties
  - This gives recall full weight while still benefiting from PMI diversity

R2: COPY MECHANISM REFORM
  - Raise copy_min_context from 2 to 3 (2-gram matches are too common)
  - Apply type-compatibility check to copy candidates (V12 missed this!)
  - Apply same-word block to copy candidates (V12 missed this too!)
  - Cap copy rate: after 3 consecutive copies, force a generation step
  - Better loop detection: hash-based phrase tracking for O(1) lookup

R3: KNESER-NEY BONUS SCALING FIX
  - For high-order matches (3+), use raw count * scale * weight (like V11)
  - Kneser-Ney continuation counts only used for BACKOFF (no match at all)
  - This preserves V11's strong recall signal while still having graceful fallback

R4: IMPROVED ANTI-REPETITION
  - Global phrase memory: track all generated n-grams, not just copy loops
  - Soft penalty for n-grams seen before (not just copy loops)
  - Stronger same-word penalty: 2× max recall bonus (guaranteed to overcome)

INTEGER-ONLY CONSTRAINT PRESERVED.
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
# R3: Fixed N-Gram Index — V11-style strong recall + KN fallback
# =============================================================================

class FixedNGramIndex(NGramIndex):
    """
    N-gram index that uses V11's strong recall for matched contexts
    and Kneser-Ney continuation counts ONLY for fallback (no context match).
    
    KEY INSIGHT: V12's KneserNeyNGramIndex applied discount/total division
    to ALL matches, weakening the bonus. V13 uses V11's raw count * scale
    for matched contexts (strong signal) and KN continuation counts only
    when there's no match at all (graceful fallback).
    """
    
    def __init__(self, max_n: int = 5, min_count: int = 1, discount: int = 1):
        super().__init__(max_n=max_n, min_count=min_count)
        self.discount = discount
        
        # Continuation counts for KN fallback
        self.continuation_count = {k: Counter() for k in range(1, max_n + 1)}
        self.total_distinct_contexts = {k: 0 for k in range(1, max_n + 1)}
    
    def build(self, sequences: List[List[int]]):
        # Build standard index
        super().build(sequences)
        
        # Compute continuation counts for fallback only
        print("  Computing continuation counts for KN fallback...")
        
        for k in range(1, self.max_n + 1):
            for context, continuations in self.index[k].items():
                unique_words = set(continuations.keys())
                for w in unique_words:
                    self.continuation_count[k][w] += 1
            
            self.total_distinct_contexts[k] = len(self.index[k])
            n_cont = len(self.continuation_count[k])
            print(f"    {k}-gram continuation: {n_cont:,} words with distinct contexts")
    
    def get_recall_bonus_with_fallback(
        self,
        context_words: List[int],
        candidate_words: np.ndarray,
        recall_scale: int = 100,
        context_weight_factor: int = 4,
    ) -> Tuple[np.ndarray, bool]:
        """
        V11-style strong recall for matched contexts + KN fallback for misses.
        
        Returns (bonuses, recall_hit) tuple.
        
        For MATCHED contexts: uses V11's raw formula (count * scale * weight^k)
        For NO MATCH: uses KN continuation counts for graceful fallback
        """
        n_candidates = len(candidate_words)
        bonuses = np.zeros(n_candidates, dtype=np.int64)
        
        # Try V11-style recall first
        matches = self.lookup(context_words)
        recall_hit = bool(matches)
        
        if matches:
            # V11 formula: raw count * recall_scale * context_weight^(k-1)
            # For k>=3, don't divide by total (strong signal)
            # For k<3, divide by total (normalized — less reliable)
            for k, continuations in matches.items():
                context_weight = context_weight_factor ** (k - 1)
                cont_lookup = {}
                for word, count, total in continuations:
                    if k >= 3:
                        # High-order: strong raw bonus (V11 style)
                        bonus = count * recall_scale * context_weight
                    else:
                        # Low-order: normalized bonus
                        bonus = (count * recall_scale * context_weight) // max(1, total)
                    cont_lookup[word] = int(bonus)
                
                for i, w in enumerate(candidate_words):
                    w_int = int(w)
                    if w_int in cont_lookup:
                        bonuses[i] += cont_lookup[w_int]
        else:
            # NO context match at all — use KN continuation count fallback
            # P_KN(w) = continuation_count[1][w] / total_distinct_contexts[1]
            # Scale by recall_scale/2 (fallback is weaker than direct match)
            total_ctx = max(1, self.total_distinct_contexts.get(1, 1))
            for i, w in enumerate(candidate_words):
                w_int = int(w)
                cont = self.continuation_count[1].get(w_int, 0)
                if cont > 0:
                    bonuses[i] += (cont * recall_scale) // (total_ctx * 2)
        
        return bonuses, recall_hit


# =============================================================================
# V13 Generator: Fixed Recall + Copy Reform
# =============================================================================

class FixedV13Generator(ExactRecallV11Generator):
    """
    V13 Generator that fixes V12's regressions.
    
    R1: Full recall strength (no interpolation dilution)
    R2: Copy mechanism reform (type check, same-word block, rate cap)
    R3: V11-style recall + KN fallback
    R4: Improved anti-repetition
    """
    
    def __init__(
        self,
        sampler,
        vocab: Vocabulary,
        ngram_index: NGramIndex,
        # R1: Recall strength
        pmi_supplement_weight: int = 3,     # PMI as supplement (out of 10)
        unigram_supplement_weight: int = 1,  # Unigram as supplement (out of 10)
        # R2: Copy reform
        max_consecutive_copies: int = 3,     # Cap on consecutive copy steps
        copy_type_check: bool = True,        # Check type compat for copy
        copy_same_word_block: bool = True,   # Block same-word copy
        # R4: Anti-repetition
        same_word_penalty_strength: int = 2, # Multiplier for max_recall in penalty
        ngram_repetition_penalty: int = 200, # Penalty for repeating n-grams
        # F1 inherited from V12
        max_closed_class_run: int = 2,
        closed_class_loop_penalty: int = 300,
        # Recall parameters
        recall_scale: int = 100,
        context_weight_factor: int = 4,
        copy_min_context: int = 3,
        copy_min_confidence: float = 0.3,
        copy_enabled: bool = True,
        # Temperature
        beta_type: float = 0.005,
        beta_word: float = 0.03,
        # Candidate filtering
        top_k_words: int = 200,
        # IDF
        use_idf_coupling: bool = True,
        idf_scale: int = 8,
        # Field
        field_weight: float = 0.001,
        # Coupling
        coupling_weight: int = 10,
    ):
        # Initialize V11 base
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
        
        # R1: Supplement weights (not interpolation — recall at 100%)
        self.pmi_supplement_weight = pmi_supplement_weight
        self.unigram_supplement_weight = unigram_supplement_weight
        
        # R2: Copy reform
        self.max_consecutive_copies = max_consecutive_copies
        self.copy_type_check = copy_type_check
        self.copy_same_word_block = copy_same_word_block
        
        # R4: Anti-repetition
        self.same_word_penalty_strength = same_word_penalty_strength
        self.ngram_repetition_penalty = ngram_repetition_penalty
        
        # R5: Recall bonus cap (prevent single n-gram from overwhelming)
        self.max_recall_bonus = 2000  # Cap per-position recall bonus
        
        # R6: Common word discount DISABLED — hurts coherence
        # Function words like "the", "of" NEED strong recall for grammatical
        # sequences. Instead use phrase-level anti-repetition (R4 n-gram tracking)
        self._common_word_discount = {}  # Disabled: no discount
        
        # F1: Closed-class anti-loop (from V12)
        self.max_closed_class_run = max_closed_class_run
        self.closed_class_loop_penalty = closed_class_loop_penalty
        self._closed_class_types = {
            POS2IDX["DET"], POS2IDX["PREP"], POS2IDX["PART"],
            POS2IDX["PRON"], POS2IDX["AUX"], POS2IDX["CONJ"],
        }
        self._hard_type_constraints = {
            POS2IDX["PART"]: [POS2IDX["VERB"]],
            POS2IDX["AUX"]: [POS2IDX["VERB"], POS2IDX["ADV"]],
        }
        
        # Check index type
        self.use_fixed_index = isinstance(ngram_index, FixedNGramIndex)
        
        # R6: Common word discount disabled (hurts coherence)
        
        # V13 diagnostics
        self._v13_stats = {
            'total_positions': 0,
            'recall_hit': 0,
            'recall_miss': 0,
            'copy_used': 0,
            'copy_blocked_type': 0,
            'copy_blocked_same_word': 0,
            'copy_blocked_rate_cap': 0,
            'copy_blocked_loop': 0,
            'type_repair': 0,
            'closed_loop_blocked': 0,
            'same_word_blocked': 0,
            'ngram_rep_penalty': 0,
        }
        
        # Runtime state
        self._consecutive_copies = 0
        self._generated_ngrams = defaultdict(int)  # track generated n-grams for R4
    
    # =========================================================================
    # F1: Type-Compatible Valid Types (from V12, but fixed)
    # =========================================================================
    
    def _get_word_type_compat(self, word_idx: int, chosen_type: int) -> bool:
        """Check if a word is compatible with a given POS type."""
        if word_idx >= self.I_emit.shape[0]:
            return True
        return int(self.I_emit[word_idx, chosen_type]) > 0
    
    def _compute_common_word_discount(self):
        """
        R6: Compute frequency-weighted discount for very common words.
        
        Words that appear as continuations in many different n-gram contexts
        (like "the", "of", "a") get their recall bonuses discounted because
        they're too easy to recall — they dominate generation.
        
        Discount = 1 / log2(n_contexts + 1) for words above threshold.
        Stored as integer (discount * 100) for integer arithmetic.
        """
        import math
        
        # Count how many distinct contexts each word appears in
        word_context_count = Counter()
        for k in range(1, self.ngram_index.max_n + 1):
            for context, continuations in self.ngram_index.index[k].items():
                for w in continuations:
                    word_context_count[w] += 1
        
        # Compute discount for words above threshold
        for w, count in word_context_count.items():
            if count > self._common_word_threshold:
                # Discount: 1 / log2(count + 1)
                # Very common words (count=1000+): discount ≈ 10%  (stored as 10)
                # Moderately common (count=100): discount ≈ 15% (stored as 15)
                # Slightly common (count=50): discount ≈ 18% (stored as 18)
                discount_pct = int(100 / math.log2(count + 1))
                discount_pct = max(10, min(50, discount_pct))  # Clamp to 10-50%
                self._common_word_discount[w] = discount_pct
        
        n_discounted = len(self._common_word_discount)
        if n_discounted > 0:
            top_discounted = word_context_count.most_common(10)
            print(f"  Common word discount: {n_discounted} words discounted")
            for w, count in top_discounted[:5]:
                if w in self._common_word_discount:
                    w_name = self.vocab.idx2word.get(w, '?')
                    d = self._common_word_discount[w]
                    print(f"    '{w_name}': {count} contexts → {d}% of full bonus")
    
    def _count_closed_class_run(self, types: List[int]) -> int:
        run = 0
        for t in reversed(types):
            if t in self._closed_class_types:
                run += 1
            else:
                break
        return run
    
    def _get_valid_next_types_v13(self, prev_type: int, types: List[int]) -> List[int]:
        """Get valid types with hard constraints + closed-class anti-loop."""
        valid = self._get_valid_next_types(prev_type)
        
        # Hard type constraints
        if prev_type in self._hard_type_constraints:
            constrained = self._hard_type_constraints[prev_type]
            constrained_valid = [t for t in valid if t in constrained]
            if constrained_valid:
                valid = constrained_valid
                self._v13_stats['type_repair'] += 1
        
        # Closed-class anti-loop
        closed_run = self._count_closed_class_run(types)
        if closed_run >= self.max_closed_class_run:
            open_types = [t for t in valid if t not in self._closed_class_types]
            if open_types:
                valid = open_types
                self._v13_stats['closed_loop_blocked'] += 1
        
        return valid
    
    # =========================================================================
    # R2: Copy Loop Detection (improved)
    # =========================================================================
    
    def _detect_copy_loop(self, candidate_word: int, words: List[int]) -> bool:
        """Improved copy loop detection using n-gram tracking."""
        # Same-word immediate repetition
        if len(words) >= 1 and candidate_word == words[-1]:
            return True
        
        # Check if the 2-gram (words[-1], candidate_word) has been used too many times
        if len(words) >= 1:
            bigram = (words[-1], candidate_word)
            if self._generated_ngrams.get(bigram, 0) >= 3:
                return True
        
        # Check 3-gram repetition
        if len(words) >= 2:
            trigram = (words[-2], words[-1], candidate_word)
            if self._generated_ngrams.get(trigram, 0) >= 2:
                return True
        
        # Consecutive copy cap
        if self._consecutive_copies >= self.max_consecutive_copies:
            return True
        
        return False
    
    # =========================================================================
    # R1+R3+R4: Fixed Energy Computation
    # =========================================================================
    
    def _compute_word_energy_v13(
        self,
        pos: int,
        candidate_words: np.ndarray,
        word_type: int,
        fixed_words: List[int],
        fixed_types: List[int],
        recall_hit: bool,
    ) -> np.ndarray:
        """
        R1: Full recall + supplementary PMI/unigram (no interpolation dilution).
        R3: V11-style strong recall for matches, KN fallback for misses.
        R4: Anti-repetition via n-gram tracking.
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)
        
        # === R3: Recall bonus (FULL STRENGTH, not interpolated) ===
        if self.use_fixed_index:
            recall_bonuses, actual_recall_hit = self.ngram_index.get_recall_bonus_with_fallback(
                context_words=fixed_words,
                candidate_words=candidate_words,
                recall_scale=self.recall_scale,
                context_weight_factor=self.context_weight_factor,
            )
        else:
            # Use standard V11 recall
            recall_bonuses = self.ngram_index.get_recall_bonus(
                context_words=fixed_words,
                candidate_words=candidate_words,
                recall_scale=self.recall_scale,
                context_weight_factor=self.context_weight_factor,
                longest_only=True,
            )
            actual_recall_hit = recall_bonuses.max() > 0
        
        # R1: Apply recall at FULL strength (not diluted by interpolation)
        # R5: But CAP the bonus to prevent single n-gram from overwhelming
        capped_bonuses = np.minimum(recall_bonuses, self.max_recall_bonus)
        
        # R6: Apply frequency-weighted discount for very common words
        # Words like "the", "of" get only 10-20% of their recall bonus
        if self._common_word_discount:
            for i, w in enumerate(candidate_words):
                w_int = int(w)
                discount_pct = self._common_word_discount.get(w_int, 100)
                if discount_pct < 100:
                    capped_bonuses[i] = capped_bonuses[i] * discount_pct // 100
        
        energies -= capped_bonuses
        
        # === R1: PMI as SUPPLEMENT (not replacing recall) ===
        # When recall hits: small PMI supplement for diversity
        # When recall misses: PMI is the primary signal
        context_start = max(0, pos - self.window)
        context_words_list = fixed_words[context_start:pos]
        if len(context_words_list) > 0:
            context_arr = np.array(context_words_list, dtype=np.int64)
            coupling_block = self.J[np.ix_(candidate_words, context_arr)]
            coupling_sums = coupling_block.sum(axis=1)
            
            if actual_recall_hit:
                # PMI as supplement: divide by 10 (weak supplement)
                energies -= (coupling_sums * self.pmi_supplement_weight) // 10
            else:
                # No recall: PMI is primary
                energies -= coupling_sums
        
        # === R1: Unigram as supplement ===
        field_vals = self.h[pos % self.seq_len, candidate_words].astype(np.float64) * self.field_weight
        if actual_recall_hit:
            # Small unigram supplement
            energies -= (field_vals * self.unigram_supplement_weight).astype(np.int64) // 10
        else:
            # No recall: unigram is secondary signal
            energies -= field_vals.astype(np.int64)
        
        # === Type compatibility filter ===
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int < self.I_emit.shape[0]:
                emit_val = int(self.I_emit[w_int, word_type])
                if emit_val <= 0:
                    # Type-incompatible: heavy penalty (like V11 emission penalty)
                    energies[i] += self.emission_penalty
        
        # === R4: Same-word penalty (ABSOLUTE — must always work) ===
        if len(fixed_words) >= 1:
            prev_word = fixed_words[-1]
            for i, w in enumerate(candidate_words):
                if int(w) == prev_word:
                    # ABSOLUTE penalty: large enough to overcome ANY recall bonus
                    # Plus proportional component for safety
                    max_b = max(1, int(capped_bonuses.max()) if capped_bonuses.max() > 0 else 1)
                    energies[i] += 50000 + self.same_word_penalty_strength * max_b
                    self._v13_stats['same_word_blocked'] += 1
        
        # === F1: Closed-class same-POS-at-distance penalty ===
        if word_type in self._closed_class_types and len(fixed_types) >= 1:
            # Same closed-class type as previous
            if fixed_types[-1] == word_type and word_type in {
                POS2IDX["DET"], POS2IDX["PREP"], POS2IDX["PART"]
            }:
                energies += self.closed_class_loop_penalty
        
        # === HARD BLOCK: "of of", "the the", "the a", "a the" etc. ===
        # Double determiners/double prepositions are NEVER grammatical
        if len(fixed_words) >= 1 and word_type in self._closed_class_types:
            prev_word_idx = fixed_words[-1]
            prev_type = fixed_types[-1] if fixed_types else -1
            
            for i, w in enumerate(candidate_words):
                w_int = int(w)
                # Same closed-class word as previous = hard block
                if w_int == prev_word_idx and prev_type in self._closed_class_types:
                    energies[i] += 100000  # Near-infinite penalty
                
                # DET→DET hard block: "the a", "a the", "this that" etc
                if word_type == POS2IDX["DET"] and prev_type == POS2IDX["DET"]:
                    energies[i] += 50000  # Very strong penalty
                
                # PREP→PREP hard block: "of in", "in of" etc
                if word_type == POS2IDX["PREP"] and prev_type == POS2IDX["PREP"]:
                    energies[i] += 50000
        
        # === R4: N-gram repetition penalty (STRONGER) ===
        if len(fixed_words) >= 1:
            for i, w in enumerate(candidate_words):
                w_int = int(w)
                # Bigram penalty
                bigram = (fixed_words[-1], w_int)
                rep_count = self._generated_ngrams.get(bigram, 0)
                if rep_count >= 2:
                    energies[i] += self.ngram_repetition_penalty * (rep_count - 1)
                    self._v13_stats['ngram_rep_penalty'] += 1
                # Trigram penalty (stronger signal)
                if len(fixed_words) >= 2:
                    trigram = (fixed_words[-2], fixed_words[-1], w_int)
                    tri_rep = self._generated_ngrams.get(trigram, 0)
                    if tri_rep >= 1:
                        energies[i] += self.ngram_repetition_penalty * tri_rep
                        self._v13_stats['ngram_rep_penalty'] += 1
        
        # === Standard repetition penalty ===
        if len(context_words_list) > 0 and self.repetition_penalty > 0:
            context_set = set(context_words_list)
            for i, w in enumerate(candidate_words):
                if int(w) in context_set:
                    energies[i] += self.repetition_penalty
        
        # === Emission compatibility (non-symmetric mode) ===
        if not self.use_symmetric_emission:
            emit_vals = self.I_emit[candidate_words, word_type]
            pos_mask = emit_vals > 0
            neg_mask = emit_vals < 0
            energies[pos_mask] -= emit_vals[pos_mask] * self.emission_bonus
            energies[neg_mask] += self.emission_penalty
        
        # Diagnostics
        self._recall_stats['total_positions'] += 1
        if actual_recall_hit:
            self._recall_stats['recall_hits'] += 1
            self._recall_stats['avg_recall_bonus'] += int(recall_bonuses.mean())
            self._recall_stats['max_recall_bonus'] = max(
                self._recall_stats['max_recall_bonus'],
                int(recall_bonuses.max())
            )
        
        return energies
    
    # =========================================================================
    # V13 Generation: Main Loop
    # =========================================================================
    
    def generate(
        self, prompt: str, length: int = 15
    ) -> Dict:
        """Generate text with V13 fixes."""
        prompt_idx = self.vocab.word2idx.get(prompt, None)
        if prompt_idx is None:
            prompt_idx = self.vocab.word2idx.get(prompt.lower(), None)
        if prompt_idx is None:
            prompt_idx = 4

        prompt_type = self._get_word_type(prompt_idx)
        words = [prompt_idx]
        types = [prompt_type]
        
        # Reset runtime state
        self._consecutive_copies = 0
        self._generated_ngrams = defaultdict(int)
        
        position_diagnostics = []

        for pos in range(1, length):
            # === Step 1: Choose POS type ===
            valid_types = self._get_valid_next_types_v13(types[-1], types)
            
            # Check recall for type override
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

            # === Step 2: Check copy (with R2: type check, same-word block, rate cap) ===
            if self.copy_enabled and len(words) >= self.copy_min_context:
                copy_candidate = self.ngram_index.get_best_copy_candidate(
                    context_words=words,
                    min_context_length=self.copy_min_context,
                    min_confidence=self.copy_min_confidence,
                )
                if copy_candidate is not None:
                    copy_word_idx, copy_count, copy_total = copy_candidate
                    
                    # R2: Same-word block for copy
                    if self.copy_same_word_block and len(words) >= 1 and copy_word_idx == words[-1]:
                        self._v13_stats['copy_blocked_same_word'] += 1
                        copy_word_idx = None  # Block it
                    
                    # R2: Type compatibility check for copy
                    if copy_word_idx is not None and self.copy_type_check:
                        if not self._get_word_type_compat(copy_word_idx, chosen_type):
                            self._v13_stats['copy_blocked_type'] += 1
                            copy_word_idx = None  # Block it
                    
                    # R2: Loop detection
                    if copy_word_idx is not None and self._detect_copy_loop(copy_word_idx, words):
                        self._v13_stats['copy_blocked_loop'] += 1
                        copy_word_idx = None  # Block it
                    
                    # R2: Rate cap
                    if copy_word_idx is not None and self._consecutive_copies >= self.max_consecutive_copies:
                        self._v13_stats['copy_blocked_rate_cap'] += 1
                        copy_word_idx = None  # Force generation
                    
                    if copy_word_idx is not None:
                        copy_word = copy_word_idx
                        self._recall_stats['copy_used'] += 1
                        self._consecutive_copies += 1
            
            # === Step 3: Choose word ===
            candidate_words_list = self.type_words.get(chosen_type, [])
            if not candidate_words_list:
                candidate_words_list = list(range(min(200, self.vocab_size)))
            candidate_words = np.array(candidate_words_list, dtype=np.int64)
            
            if len(candidate_words) > self.top_k_words:
                field_vals = self.h[pos % self.seq_len, candidate_words]
                top_k_indices = np.argsort(field_vals)[-self.top_k_words:]
                candidate_words = candidate_words[top_k_indices]
            
            # Check recall availability
            recall_matches_raw = self.ngram_index.lookup(words)
            recall_hit = bool(recall_matches_raw)
            
            # R1+R3+R4: Fixed energy computation
            word_energies = self._compute_word_energy_v13(
                pos, candidate_words, chosen_type, words, types, recall_hit
            )
            
            if copy_word is not None:
                chosen_word = copy_word
                diag_copy = True
            else:
                word_idx = self._boltzmann_sample(word_energies, self.beta_word)
                chosen_word = int(candidate_words[word_idx])
                self._consecutive_copies = 0  # Reset copy counter on generation

            words.append(chosen_word)
            types.append(chosen_type)
            
            # R4: Track generated n-grams for repetition detection
            for n in range(2, min(5, len(words))):
                ngram = tuple(words[-n:])
                self._generated_ngrams[ngram] += 1

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
            
            # Track V13 stats
            self._v13_stats['total_positions'] += 1
            if recall_hit:
                self._v13_stats['recall_hit'] += 1
            else:
                self._v13_stats['recall_miss'] += 1
            if diag_copy:
                self._v13_stats['copy_used'] += 1

        text = self.vocab.decode(words)
        type_names = [IDX2POS.get(t, "UNK") for t in types]

        return {
            'text': text,
            'words': words,
            'types': types,
            'type_names': type_names,
            'diagnostics': position_diagnostics,
        }
    
    def get_v13_stats(self) -> Dict:
        stats = self._v13_stats.copy()
        total = max(1, stats['total_positions'])
        stats['recall_hit_rate'] = stats['recall_hit'] / total
        stats['copy_rate'] = stats['copy_used'] / total
        stats['type_repair_rate'] = stats['type_repair'] / total
        stats['closed_loop_block_rate'] = stats['closed_loop_blocked'] / total
        stats['same_word_block_rate'] = stats['same_word_blocked'] / total
        return stats


# =============================================================================
# V13 Model: Training + Generation
# =============================================================================

class FixedV13Model:
    """V13 Fixed Ising Language Model."""
    
    def __init__(self, **kwargs):
        # Extract V13-specific parameters
        self.v13_pmi_supplement_weight = kwargs.pop('pmi_supplement_weight', 3)
        self.v13_unigram_supplement_weight = kwargs.pop('unigram_supplement_weight', 1)
        self.v13_max_consecutive_copies = kwargs.pop('max_consecutive_copies', 3)
        self.v13_copy_type_check = kwargs.pop('copy_type_check', True)
        self.v13_copy_same_word_block = kwargs.pop('copy_same_word_block', True)
        self.v13_same_word_penalty_strength = kwargs.pop('same_word_penalty_strength', 2)
        self.v13_ngram_repetition_penalty = kwargs.pop('ngram_repetition_penalty', 200)
        self.v13_max_closed_class_run = kwargs.pop('max_closed_class_run', 2)
        self.v13_closed_class_loop_penalty = kwargs.pop('closed_class_loop_penalty', 300)
        self.v13_use_fixed_index = kwargs.pop('use_fixed_index', True)
        
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
        
        self.ngram_index = None
        self.v13_generator = None
    
    def train(self, n_samples: int = 20000):
        """Train the model."""
        import time as _time
        from .data_loader import load_fineweb_edu, tokenize_texts, truncate_sequences
        
        print("=" * 70)
        print("V13 TRAINING: V8 Pipeline + Fixed N-gram Index Building")
        print("=" * 70)
        
        # Step 1: Train V8 model
        self.v8_model.train(n_samples=n_samples)
        
        # Step 2: Build N-gram index
        print("\n--- Building Fixed N-Gram Index ---")
        
        t0 = _time.time()
        texts = load_fineweb_edu(n_samples=n_samples)
        sequences = tokenize_texts(texts, self.v8_model.vocab)
        sequences = truncate_sequences(sequences, max_len=self.v8_model.seq_len)
        print(f"  Re-loaded {len(sequences)} sequences ({_time.time()-t0:.1f}s)")
        
        if self.v13_use_fixed_index:
            print("  Using FixedNGramIndex (V11 recall + KN fallback)")
            self.ngram_index = FixedNGramIndex(
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
        
        # Step 3: Build generator
        self._build_generator()
    
    def _build_generator(self):
        self.v13_generator = FixedV13Generator(
            sampler=self.v8_model.sampler,
            vocab=self.v8_model.vocab,
            ngram_index=self.ngram_index,
            # V13-specific
            pmi_supplement_weight=self.v13_pmi_supplement_weight,
            unigram_supplement_weight=self.v13_unigram_supplement_weight,
            max_consecutive_copies=self.v13_max_consecutive_copies,
            copy_type_check=self.v13_copy_type_check,
            copy_same_word_block=self.v13_copy_same_word_block,
            same_word_penalty_strength=self.v13_same_word_penalty_strength,
            ngram_repetition_penalty=self.v13_ngram_repetition_penalty,
            max_closed_class_run=self.v13_max_closed_class_run,
            closed_class_loop_penalty=self.v13_closed_class_loop_penalty,
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
        if self.v13_generator is None:
            self._build_generator()
        result = self.v13_generator.generate(prompt=prompt, length=length)
        result['recall_stats'] = self.v13_generator.get_recall_stats()
        result['v13_stats'] = self.v13_generator.get_v13_stats()
        result['energy'] = 0
        result['demon_stats'] = {'demon_energy': 0, 'acceptance_rate': 1.0}
        result['marginal_stats'] = {
            'avg_consecutive_count': 0, 'max_consecutive_count': 0,
            'stuck_5plus': 0, 'flip_rate': 1.0,
        }
        return result
    
    def generate_raw(self, length: int = 15) -> Tuple[List[int], List[int]]:
        if self.v13_generator is None:
            self._build_generator()
        start_idx = np.random.randint(4, min(54, self.v13_generator.vocab_size))
        prompt = self.v8_model.vocab.idx2word.get(start_idx, "the")
        result = self.v13_generator.generate(prompt=prompt, length=length)
        return result['words'], result['types']
    
    def evaluate_grammar(self, words, types):
        return self.v8_model.evaluate_grammar(words, types)
    
    @property
    def vocab(self):
        return self.v8_model.vocab
    
    @property
    def sequences(self):
        return getattr(self, '_sequences', [])
