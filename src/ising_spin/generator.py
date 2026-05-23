"""
Ising Spin Glass Language Model v17 — Text Generator

Generation loop (per position):
  1. Choose POS type via Boltzmann sampling (grammar + recall bias)
  2. Get candidate words for that type
  3. Check copy mechanism (high-confidence n-gram recall)
  4. Compute energy via EnergyComputer (multi-scale recall + state + constraints)
  5. Boltzmann sample word
  6. Update document state (deterministic rules)

Also provides perplexity computation using integer-only Boltzmann arithmetic.
"""

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .vocabulary import Vocabulary, POSTypeSystem
from .vocabulary.pos import POS2IDX, IDX2POS, N_POS, CLOSED_CLASS, NOUN_LIKE
from .recall import WordNgramIndex, MultiScaleRecall
from .state import DocumentState
from .energy import EnergyComputer
from .sampling import IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE


class IsingLMGenerator:
    """
    v17 Text generator using Multi-Scale Recall + Document State.

    Much simpler than v1-v16 generator because:
    - NO PMI couplings, knowledge layer, category layer, logic layer
    - NO Walsh, graded couplings, Grassmann, context accumulator, long-range
    - ALL energy comes from EnergyComputer (which uses MultiScaleRecall + DocumentState)
    - Document state is updated per-word (deterministic rules)
    - Copy mechanism preserved (it's a form of recall)

    The generation loop is:
      1. Choose POS type (Boltzmann from type energy landscape)
      2. Check copy mechanism
      3. Get candidate words for chosen type
      4. Compute energy via EnergyComputer
      5. Boltzmann sample word
      6. Update document state
    """

    CLOSED_CLASS_IDS = {POS2IDX["DET"], POS2IDX["PREP"], POS2IDX["PART"],
                        POS2IDX["PRON"], POS2IDX["AUX"], POS2IDX["CONJ"]}

    HARD_TYPE_CONSTRAINTS = {
        POS2IDX["PART"]: [POS2IDX["VERB"]],
        POS2IDX["AUX"]: [POS2IDX["VERB"], POS2IDX["ADV"]],
    }

    def __init__(
        self,
        vocab: Vocabulary,
        pos_system: POSTypeSystem,
        multiscale_recall: MultiScaleRecall,
        document_state: DocumentState,
        energy_computer: EnergyComputer,
        word_sampler: IntegerBoltzmannSampler,
        type_sampler: IntegerBoltzmannSampler,
        word_index: Optional[WordNgramIndex] = None,
        copy_enabled: bool = True,
        copy_min_context: int = 3,
        copy_min_confidence: float = 0.4,
        same_word_penalty: int = 200,
        max_closed_class_run: int = 2,
        interpolated: bool = True,
        kn_backoff: bool = True,
        recall_scale: int = 1600,
        pos_recall_scale: int = 800,
        topic_recall_scale: int = 400,
        state_scale: int = 200,
    ):
        self.vocab = vocab
        self.pos_system = pos_system
        self.multiscale_recall = multiscale_recall
        self.document_state = document_state
        self.energy_computer = energy_computer
        self.word_sampler = word_sampler
        self.type_sampler = type_sampler
        self.word_index = word_index  # for copy mechanism
        self.vocab_size = len(vocab)

        self.copy_enabled = copy_enabled
        self.copy_min_context = copy_min_context
        self.copy_min_confidence = copy_min_confidence
        self.same_word_penalty = same_word_penalty
        self.max_closed_class_run = max_closed_class_run
        self.interpolated = interpolated
        self.kn_backoff = kn_backoff

        self.recall_scale = recall_scale
        self.pos_recall_scale = pos_recall_scale
        self.topic_recall_scale = topic_recall_scale
        self.state_scale = state_scale

        # Build type→words index from POS system
        self.type_words: Dict[int, List[int]] = {t: [] for t in range(N_POS)}
        TAG_PRIORITY = {
            POS2IDX["PUNCT"]: 0, POS2IDX["DET"]: 1, POS2IDX["PRON"]: 2,
            POS2IDX["AUX"]: 3, POS2IDX["CONJ"]: 4, POS2IDX["PART"]: 5,
            POS2IDX["PREP"]: 6, POS2IDX["NUM"]: 7, POS2IDX["ADV"]: 8,
            POS2IDX["ADJ"]: 9, POS2IDX["NOUN"]: 10, POS2IDX["VERB"]: 11,
            POS2IDX["X"]: 12,
        }
        for w, allowed in pos_system.allowed_types.items():
            if allowed:
                primary = min(allowed, key=lambda t: TAG_PRIORITY.get(t, 99))
                self.type_words[primary].append(w)

        # Pre-compute allowed POS transitions from grammar penalties
        self.allowed_transitions: Set[Tuple[int, int]] = set()
        for t1 in range(N_POS):
            for t2 in range(N_POS):
                penalty = pos_system.compute_grammar_penalty([t1], 0, t2)
                if penalty < 500:
                    self.allowed_transitions.add((t1, t2))

        # Compute unigram field h[] for top-k filtering
        # h[w] = log2(total_tokens / count(w)) — higher = rarer = higher energy
        self.h = np.zeros(self.vocab_size, dtype=np.int64)
        total = 0
        word_counts = np.zeros(self.vocab_size, dtype=np.int64)
        # Use document state's compatibility tables if available, else compute
        if document_state._built and document_state.topic_word_counts is not None:
            # Sum across all topic rows for unigram counts
            for row_idx in range(document_state.topic_word_counts.shape[0]):
                word_counts += document_state.topic_word_counts[row_idx]
            total = int(word_counts.sum())
        # Fallback: compute from type_words + simple heuristic
        if total == 0:
            self.h = np.ones(self.vocab_size, dtype=np.int64)

        for w in range(self.vocab_size):
            c = int(word_counts[w])
            if c > 0 and total > c:
                ratio = total // c
                if ratio >= 2:
                    self.h[w] = ratio.bit_length() - 1  # floor(log2(ratio))

        # Diagnostics
        self._stats = {
            'total_positions': 0,
            'recall_hit': 0,
            'copy_used': 0,
            'same_word_blocked': 0,
            'closed_loop_blocked': 0,
        }

    # ===================================================================
    # WORD TYPE HELPERS
    # ===================================================================

    def _get_word_type(self, word_idx: int) -> int:
        """Get primary POS type for a word."""
        TAG_PRIORITY = {
            POS2IDX["PUNCT"]: 0, POS2IDX["DET"]: 1, POS2IDX["PRON"]: 2,
            POS2IDX["AUX"]: 3, POS2IDX["CONJ"]: 4, POS2IDX["PART"]: 5,
            POS2IDX["PREP"]: 6, POS2IDX["NUM"]: 7, POS2IDX["ADV"]: 8,
            POS2IDX["ADJ"]: 9, POS2IDX["NOUN"]: 10, POS2IDX["VERB"]: 11,
            POS2IDX["X"]: 12,
        }
        if word_idx in self.pos_system.allowed_types and self.pos_system.allowed_types[word_idx]:
            return min(self.pos_system.allowed_types[word_idx],
                       key=lambda t: TAG_PRIORITY.get(t, 99))
        return POS2IDX["X"]

    def _get_valid_next_types(self, prev_type: int, types_history: List[int]) -> List[int]:
        """Get valid next POS types with hard constraints + anti-loop."""
        valid = [t for t in range(N_POS) if (prev_type, t) in self.allowed_transitions]
        if not valid:
            valid = list(range(N_POS))

        # Hard type constraints (e.g. PART -> VERB)
        if prev_type in self.HARD_TYPE_CONSTRAINTS:
            constrained = self.HARD_TYPE_CONSTRAINTS[prev_type]
            constrained_valid = [t for t in valid if t in constrained]
            if constrained_valid:
                valid = constrained_valid

        # Closed-class anti-loop
        closed_run = 0
        for t in reversed(types_history):
            if t in self.CLOSED_CLASS_IDS:
                closed_run += 1
            else:
                break
        if closed_run >= self.max_closed_class_run:
            open_types = [t for t in valid if t not in self.CLOSED_CLASS_IDS]
            if open_types:
                valid = open_types
                self._stats['closed_loop_blocked'] += 1

        return valid

    def _compute_type_energy(self, pos: int, type_idx: int,
                              types_history: List[int],
                              context_words: Optional[List[int]] = None) -> int:
        """
        Compute energy for a POS type at position pos. Pure integer.

        Energy components:
          - Grammar penalty (from POSTypeSystem)
          - Same-type penalty (avoid repeating the same type)
          - Recall type bias (if recall suggests a type, give energy bonus)
        """
        energy = 0

        # Grammar penalty
        types_for_check = list(types_history) + [type_idx]
        penalty = self.pos_system.compute_grammar_penalty(
            types_for_check, len(types_history), type_idx
        )
        energy += penalty

        # Same-type penalty (avoid VERB VERB VERB etc.)
        if len(types_history) > 0 and type_idx == types_history[-1]:
            if type_idx not in (POS2IDX['NOUN'], POS2IDX['X']):
                energy += 50

        # Recall type bias: if recall's top prediction has this type,
        # give it an energy bonus (lower energy = more likely)
        if context_words is not None and len(context_words) >= 2 and self.word_index is not None:
            recall_matches = self.word_index.lookup(context_words)
            if recall_matches:
                best_k = max(recall_matches.keys())
                best_conts = recall_matches[best_k]
                if best_k >= 2 and best_conts:
                    best_word, best_count, best_total = best_conts[0]
                    if best_count * 3 >= best_total:
                        recall_type = self._get_word_type(best_word)
                        if recall_type == type_idx:
                            energy -= 200  # Moderate recall bias

        return energy

    # ===================================================================
    # GENERATION
    # ===================================================================

    def generate(self, prompt: str = "the", length: int = 20) -> Dict:
        """
        Generate text autoregressively — v17 Multi-Scale Recall Architecture.

        At each position:
          1. Choose POS type: Boltzmann from type energy landscape
          2. Check copy mechanism (legitimate: it's a form of recall)
          3. Compute E(w|ctx) with multi-scale recall + document state + constraints
          4. Boltzmann sample: P(w) ~ exp(-beta * E(w))
          5. Update document state

        All energy computation and sampling is integer-only.
        """
        # Resolve prompt — tokenize ALL words
        prompt_words = prompt.strip().split()
        prompt_tokens = []
        for w in prompt_words:
            idx = self.vocab.word2idx.get(w)
            if idx is None:
                idx = self.vocab.word2idx.get(w.lower())
            if idx is not None and idx >= 4:  # Skip special tokens
                prompt_tokens.append(idx)
        # Fallback: if no words found, use "the"
        if not prompt_tokens:
            idx = self.vocab.word2idx.get("the", 4)
            prompt_tokens = [idx]

        words = list(prompt_tokens)
        types_list = [self._get_word_type(w) for w in words]
        consecutive_copies = 0
        diagnostics = []

        # Initialize document state from prompt
        self.document_state.reset()
        for w in words:
            word_str = self.vocab.idx2word.get(w, "")
            self.document_state.update(w, word_str=word_str)

        for pos in range(len(words), length):
            # === STEP 1: Choose POS type (BOLTZMANN) ===
            valid_types = self._get_valid_next_types(types_list[-1], types_list)

            # Check if recall suggests a type bias
            recall_type_bias = None
            if len(words) >= 2 and self.word_index is not None:
                recall_matches = self.word_index.lookup(words)
                if recall_matches:
                    best_k = max(recall_matches.keys())
                    best_conts = recall_matches[best_k]
                    if best_k >= 2 and best_conts:
                        best_word, best_count, best_total = best_conts[0]
                        if best_count * 3 >= best_total:
                            recall_type = self._get_word_type(best_word)
                            if recall_type in valid_types:
                                recall_type_bias = recall_type

            # Compute type energies
            type_energies = np.array([
                self._compute_type_energy(pos, t, types_list, words)
                for t in valid_types
            ], dtype=np.int64)

            # Add recall type bias as energy bonus (NOT an override)
            if recall_type_bias is not None:
                for i, t in enumerate(valid_types):
                    if t == recall_type_bias:
                        type_energies[i] -= 200  # Moderate recall bias

            chosen_type = valid_types[self.type_sampler.sample(type_energies)]

            # === STEP 2: Check copy mechanism ===
            copy_word = None
            if self.copy_enabled and len(words) >= self.copy_min_context and self.word_index is not None:
                copy_candidate = self.word_index.get_best_copy_candidate(
                    context_words=words,
                    min_context_length=self.copy_min_context,
                    min_confidence=self.copy_min_confidence,
                )
                if copy_candidate is not None:
                    copy_word_idx, _, _ = copy_candidate
                    # Check POS type compatibility
                    if copy_word_idx < self.vocab_size:
                        allowed = self.pos_system.allowed_types.get(copy_word_idx, set())
                        if chosen_type in allowed or not allowed:
                            # Don't copy same word twice
                            if len(words) >= 1 and copy_word_idx == words[-1]:
                                copy_word_idx = None
                            elif consecutive_copies >= 6:
                                copy_word_idx = None
                            else:
                                copy_word = copy_word_idx
                                consecutive_copies += 1
                                self._stats['copy_used'] += 1

            if copy_word is None:
                consecutive_copies = 0

            # === STEP 3: Get candidate words ===
            candidate_list = self.type_words.get(chosen_type, [])
            if not candidate_list:
                candidate_list = list(range(min(200, self.vocab_size)))
            candidate_words = np.array(candidate_list, dtype=np.int64)

            # Top-k filtering by field strength (unigram frequency)
            if len(candidate_words) > 300:
                field_vals = self.h[candidate_words]
                top_k = np.argsort(field_vals)[-300:]
                candidate_words = candidate_words[top_k]

            # === STEP 4: Compute energy ===
            # Determine closed_class_run for the energy computer
            closed_run = 0
            for t in reversed(types_list):
                if t in self.CLOSED_CLASS_IDS:
                    closed_run += 1
                else:
                    break

            prev_word = words[-1] if words else -1

            word_energies = self.energy_computer.compute_energy(
                context_words=words,
                candidate_words=candidate_words,
                current_type=chosen_type,
                prev_word=prev_word,
                closed_class_run=closed_run,
            )

            # Additional penalties that depend on context
            # Repetition penalty for recent words
            if len(words) > 0:
                recent = set(words[-5:])
                rep_penalty = max(200, self.recall_scale // 8)
                for i, w in enumerate(candidate_words):
                    if int(w) in recent:
                        word_energies[i] += rep_penalty

            # === STEP 5: Boltzmann sample ===
            chosen_energy = 0
            if copy_word is not None:
                chosen_word = copy_word
            else:
                word_idx = self.word_sampler.sample(word_energies)
                chosen_word = int(candidate_words[word_idx])
                chosen_energy = int(word_energies[word_idx])

            words.append(chosen_word)
            types_list.append(chosen_type)

            # === STEP 6: Update document state ===
            word_str = self.vocab.idx2word.get(chosen_word, "")
            self.document_state.update(chosen_word, word_str=word_str)

            # Track diagnostics
            self._stats['total_positions'] += 1
            recall_hit = False
            if self.word_index is not None:
                recall_matches = self.word_index.lookup(words[:-1])
                recall_hit = bool(recall_matches)
            if recall_hit:
                self._stats['recall_hit'] += 1

            diagnostics.append({
                'pos': pos,
                'type': IDX2POS.get(chosen_type, "UNK"),
                'word': self.vocab.idx2word.get(chosen_word, "<UNK>"),
                'copy': copy_word is not None,
                'recall_hit': recall_hit,
                'energy': chosen_energy,
            })

        text = self.vocab.decode(words)
        type_names = [IDX2POS.get(t, "UNK") for t in types_list]

        return {
            'text': text,
            'words': words,
            'types': types_list,
            'type_names': type_names,
            'diagnostics': diagnostics,
        }

    def generate_raw(self, length: int = 20) -> Tuple[List[int], List[int]]:
        """Generate with a random prompt."""
        start_idx = np.random.randint(4, min(54, self.vocab_size))
        prompt = self.vocab.idx2word.get(start_idx, "the")
        result = self.generate(prompt=prompt, length=length)
        return result['words'], result['types']

    # ===================================================================
    # PERPLEXITY COMPUTATION
    # ===================================================================

    def compute_perplexity(
        self,
        test_sequences: Optional[List[List[int]]] = None,
        n_samples: int = 100,
    ) -> float:
        """
        Compute perplexity on held-out test sequences.

        PPL = exp(-1/N * sum log P(w_t | ctx))

        where P(w_t | ctx) = exp(-beta * E(w_t)) / Z
        over all candidates of the same POS type.

        This uses the word_sampler's Boltzmann lookup table for efficient
        computation of the partition function.  All arithmetic is integer,
        with final PPL conversion to float for display only.

        v17 adaptation: Uses EnergyComputer for energy computation instead
        of the old _compute_word_energy method.  Document state is updated
        per-position to track discourse context.
        """
        if test_sequences is None:
            print("  Warning: No test sequences. Returning inf PPL.")
            return float('inf')

        sampler = self.word_sampler

        # Accumulate log2 probabilities as integers (x LOG2_SCALE)
        total_log2_prob = 0
        total_tokens = 0

        eval_seqs = test_sequences[:n_samples]

        for seq_idx, seq in enumerate(eval_seqs):
            if len(seq) < 3:
                continue

            # Reset document state for each new sequence
            self.document_state.reset()

            for pos in range(1, len(seq)):
                target_word = seq[pos]
                context_words = seq[:pos]
                context_types = [self._get_word_type(w) for w in context_words]

                # Determine the POS type for the target word
                word_type = self._get_word_type(target_word)

                # Get candidate words for this type
                candidate_list = self.type_words.get(word_type, [])
                if not candidate_list:
                    continue
                candidate_words = np.array(candidate_list, dtype=np.int64)

                # v17: recall-only mode is fast enough for all candidates,
                # but if the candidate set is very large, limit to top-500 + target
                if len(candidate_words) > 500:
                    field_vals = self.h[candidate_words]
                    top_k = np.argsort(field_vals)[-499:]
                    candidate_words = candidate_words[top_k]
                    # Always include target word
                    if int(target_word) not in set(candidate_words.tolist()):
                        candidate_words = np.append(candidate_words, target_word)

                # Check if target word is in candidates
                target_in_candidates = int(target_word) in set(candidate_words.tolist())
                if not target_in_candidates:
                    # Target not reachable; use smoothing
                    total_log2_prob += -15 * LOG2_SCALE
                    total_tokens += 1
                    # Still update document state
                    word_str = self.vocab.idx2word.get(target_word, "")
                    self.document_state.update(target_word, word_str=word_str)
                    continue

                # Compute closed_class_run for context
                closed_run = 0
                for t in reversed(context_types):
                    if t in self.CLOSED_CLASS_IDS:
                        closed_run += 1
                    else:
                        break

                prev_word = context_words[-1] if context_words else -1

                # Compute energies using EnergyComputer
                energies = self.energy_computer.compute_energy(
                    context_words=context_words,
                    candidate_words=candidate_words,
                    current_type=word_type,
                    prev_word=prev_word,
                    closed_class_run=closed_run,
                )

                # Additional repetition penalty (matches generation)
                if len(context_words) > 0:
                    recent = set(context_words[-5:])
                    rep_penalty = max(200, self.recall_scale // 8)
                    for i, w in enumerate(candidate_words):
                        if int(w) in recent:
                            energies[i] += rep_penalty

                # Compute log2 probabilities (integer, x LOG2_SCALE)
                log_probs = sampler.compute_log_probabilities(energies)

                # Find the target word's log2 probability
                target_idx = np.where(candidate_words == target_word)[0]
                if len(target_idx) > 0:
                    total_log2_prob += int(log_probs[target_idx[0]])
                else:
                    total_log2_prob += -15 * LOG2_SCALE

                total_tokens += 1

                # Update document state with actual next word
                word_str = self.vocab.idx2word.get(target_word, "")
                self.document_state.update(target_word, word_str=word_str)

        if total_tokens == 0:
            return float('inf')

        # PPL from integer log2 probabilities — INTEGER-ONLY until final display
        # PPL = 2^(-avg_log2_prob) = 2^(-total_log2_prob / (total_tokens * LOG2_SCALE))
        if total_log2_prob >= 0:
            perplexity = 1.0
        else:
            # Compute log2(PPL) in fixed-point with 16 fractional bits
            neg_avg = -total_log2_prob  # positive value
            log2_ppl_fp = (neg_avg << 16) // (total_tokens * LOG2_SCALE)
            int_part = log2_ppl_fp >> 16
            frac_part = log2_ppl_fp & 0xFFFF  # in [0, 65536)

            # 2^frac_part using Taylor expansion of exp(f*ln(2))
            # f = frac_part / 65536, f*ln(2) in [0, 0.693)
            FP = 48
            ONE_FP = 1 << FP
            f_fp = (frac_part * ONE_FP) >> 16
            x = (f_fp * LN2_NUM) // LN2_DEN
            x2 = (x * x) >> FP
            x3 = (x2 * x) >> FP
            x4 = (x3 * x) >> FP
            x5 = (x4 * x) >> FP
            x6 = (x5 * x) >> FP
            x7 = (x6 * x) >> FP
            x8 = (x7 * x) >> FP
            x9 = (x8 * x) >> FP
            x10 = (x9 * x) >> FP
            # exp(x) = 1 + x + x^2/2! + ... + x^10/10!
            exp_val = (ONE_FP + x + (x2 >> 1) + (x3 // 6) + (x4 // 24) +
                       (x5 // 120) + (x6 // 720) + (x7 // 5040) + (x8 // 40320) +
                       (x9 // 362880) + (x10 // 3628800))
            # PPL = 2^int_part * exp_val / 2^FP
            ppl_frac = exp_val / ONE_FP  # only float conversion for final display
            if int_part < 63:
                perplexity = float(1 << int_part) * ppl_frac
            else:
                perplexity = float('inf')

        print(f"  Perplexity: {perplexity:.2f} (evaluated on {total_tokens} tokens)")
        return perplexity

    # ===================================================================
    # DIAGNOSTICS
    # ===================================================================

    def get_stats(self) -> Dict:
        """Get generation statistics."""
        stats = self._stats.copy()
        total = max(1, stats['total_positions'])
        stats['recall_hit_rate'] = stats['recall_hit'] / total
        stats['copy_rate'] = stats['copy_used'] / total
        return stats
