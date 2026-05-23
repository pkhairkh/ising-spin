"""
Ising Spin Glass Language Model v17 — Multi-Scale Abstract Recall

Architecture:
  1. Word-level n-gram recall (5-gram, same as v1-v16)
  2. POS-level n-gram recall (15-gram, NEW — abstract generalization)
  3. Topic-level n-gram recall (10-gram, NEW — discourse coherence)
  4. Document state (7 evolving integer variables, NEW — full-document context)
  5. Hard constraints (POS type, same-word, closed-class)

Key insight: Each scale independently constrains predictions via Product of Experts.
No single scale dominates — they VETO each other's mistakes.

v17 replaces the 6500-line monolithic model.py with clean, focused DDD modules:
  - vocabulary/   : Vocabulary, POSTypeSystem, TopicAssigner
  - recall/       : WordNgramIndex, PosNgramIndex, TopicNgramIndex, MultiScaleRecall
  - state/        : DocumentState
  - energy/       : EnergyComputer
  - sampling/     : IntegerBoltzmannSampler
  - generator.py  : IsingLMGenerator
  - model_v17.py  : IsingLMModel (THIS FILE — training orchestrator)
"""

import math
import time
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .vocabulary import Vocabulary, POSTypeSystem, TopicAssigner
from .vocabulary.pos import COARSE_POS_TAGS, POS2IDX, IDX2POS, N_POS, CLOSED_CLASS
from .recall import WordNgramIndex, PosNgramIndex, TopicNgramIndex, MultiScaleRecall
from .state import DocumentState
from .energy import EnergyComputer
from .sampling import IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE


def _get_rss_mb() -> int:
    """Get current process RSS in MB."""
    try:
        import os
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except Exception:
        return 0


def _tokenize_texts(texts: List[str], vocab: Vocabulary) -> List[List[int]]:
    """Tokenize a list of texts using the vocabulary. Pure integer encoding."""
    sequences = []
    for text in texts:
        tokens = vocab.encode(text)
        if len(tokens) > 0:
            sequences.append(tokens)
    return sequences


def _truncate_sequences(sequences: List[List[int]], max_len: int = 30) -> List[List[int]]:
    """Truncate sequences to max_len and filter short ones."""
    return [seq[:max_len] for seq in sequences if len(seq) > 1]


def _load_fineweb_edu(
    n_samples: int = 50000,
    split: str = "train",
    subset: str = "sample-10BT",
    min_length: int = 20,
    max_length: int = 2000,
) -> List[str]:
    """Load text samples from the fineweb-edu dataset on HuggingFace."""
    from datasets import load_dataset

    print(f"Loading fineweb-edu ({subset}, split={split})...")

    dataset = None
    for name in ["HuggingFaceFW/fineweb-edu", "HuggingFW/fineweb-edu"]:
        try:
            dataset = load_dataset(name, name=subset, split=split, streaming=True)
            print(f"  Loaded from '{name}' with subset '{subset}'")
            break
        except Exception:
            continue

    if dataset is None:
        raise RuntimeError("Could not load fineweb-edu dataset. Install 'datasets' package.")

    texts = []
    for i, item in enumerate(dataset):
        if len(texts) >= n_samples:
            break
        text = item.get("text", "")
        if min_length <= len(text) <= max_length:
            texts.append(text)
        if (i + 1) % 10000 == 0:
            print(f"  Scanned {i + 1} items, collected {len(texts)} texts")

    return texts


class IsingLMModel:
    """
    v17: Multi-Scale Abstract Recall + Evolving Document State.

    Training pipeline:
      1. Load corpus / use provided texts
      2. Build vocabulary
      3. Tokenize texts → sequences
      4. Split train/test (90/10)
      5. Build POS type system
      6. Build topic assigner
      7. Build word n-gram index
      8. Build POS n-gram index
      9. Build topic n-gram index
      10. Build multi-scale recall
      11. Build document state
      12. Build energy computer
      13. Auto-calibrate beta
      14. Build generator
    """

    def __init__(
        self,
        # Vocabulary
        vocab_min_freq: int = 25,
        vocab_max_size: int = 4000,
        # N-gram
        ngram_max_n: int = 5,
        ngram_min_count: int = 2,
        ngram_max_sequences: int = 1000000,
        # POS
        pos_ngram_max_n: int = 10,
        pos_ngram_min_count: int = 2,
        # Topic
        n_topics: int = 16,
        topic_ngram_max_n: int = 10,
        topic_ngram_min_count: int = 3,
        # Energy scales
        recall_scale: int = 1600,
        pos_recall_scale: int = 400,
        topic_recall_scale: int = 200,
        state_scale: int = 200,
        # Hard constraints
        same_word_penalty: int = 200,
        max_closed_class_run: int = 2,
        # Beta
        beta_type: float = 0.01,
        beta_word: float = 0.1,
        auto_calibrate_beta: bool = True,
        # Interpolation
        interpolated: bool = True,
        kn_backoff: bool = True,
        # Copy mechanism
        copy_enabled: bool = True,
        copy_min_context: int = 3,
        copy_min_confidence: float = 0.4,
        # Misc
        max_seq_len: int = 30,
    ):
        # Store all params
        self.vocab_min_freq = vocab_min_freq
        self.vocab_max_size = vocab_max_size
        self.ngram_max_n = ngram_max_n
        self.ngram_min_count = ngram_min_count
        self.ngram_max_sequences = ngram_max_sequences
        self.pos_ngram_max_n = pos_ngram_max_n
        self.pos_ngram_min_count = pos_ngram_min_count
        self.n_topics = n_topics
        self.topic_ngram_max_n = topic_ngram_max_n
        self.topic_ngram_min_count = topic_ngram_min_count
        self.recall_scale = recall_scale
        self.pos_recall_scale = pos_recall_scale
        self.topic_recall_scale = topic_recall_scale
        self.state_scale = state_scale
        self.same_word_penalty = same_word_penalty
        self.max_closed_class_run = max_closed_class_run
        self.beta_type = beta_type
        self.beta_word = beta_word
        self.auto_calibrate_beta = auto_calibrate_beta
        self.interpolated = interpolated
        self.kn_backoff = kn_backoff
        self.copy_enabled = copy_enabled
        self.copy_min_context = copy_min_context
        self.copy_min_confidence = copy_min_confidence
        self.max_seq_len = max_seq_len

        # Built during training
        self.vocab: Optional[Vocabulary] = None
        self.pos_system: Optional[POSTypeSystem] = None
        self.topic_assigner: Optional[TopicAssigner] = None
        self.word_index: Optional[WordNgramIndex] = None
        self.pos_index: Optional[PosNgramIndex] = None
        self.topic_index: Optional[TopicNgramIndex] = None
        self.multiscale_recall: Optional[MultiScaleRecall] = None
        self.document_state: Optional[DocumentState] = None
        self.energy_computer: Optional[EnergyComputer] = None
        self.generator = None  # IsingLMGenerator

        self.sequences: Optional[List[List[int]]] = None
        self.test_sequences: Optional[List[List[int]]] = None
        self._word_freq: Optional[np.ndarray] = None

    # ===================================================================
    # TRAINING PIPELINE
    # ===================================================================

    def train(self, n_samples: int = 20000, texts=None) -> "IsingLMModel":
        """
        Full training pipeline for v17 Multi-Scale Abstract Recall.

        Much simpler than v1-v16: no PMI, no knowledge layer, no category
        layer, no logic layer, no Walsh, no graded couplings, no Grassmann,
        no context accumulator, no long-range coupling. Just recall at
        three scales + document state + hard constraints.
        """
        print("=" * 70)
        print("ISING SPIN GLASS LANGUAGE MODEL v17 — MULTI-SCALE ABSTRACT RECALL")
        print("=" * 70)
        print(f"\n  Architecture: 3-Scale Recall (word+POS+topic) + Document State")
        print(f"  Word n-gram:  max_n={self.ngram_max_n}, scale={self.recall_scale}")
        print(f"  POS n-gram:   max_n={self.pos_ngram_max_n}, scale={self.pos_recall_scale}")
        print(f"  Topic n-gram: max_n={self.topic_ngram_max_n}, scale={self.topic_recall_scale}")
        print(f"  Document state: scale={self.state_scale}")
        print(f"  Interpolated: {self.interpolated}, KN backoff: {self.kn_backoff}")
        print(f"  Auto-calibrate beta: {self.auto_calibrate_beta}")
        print(f"  Same-word penalty: {self.same_word_penalty}")
        print()

        t0 = time.time()

        # ------------------------------------------------------------------
        # Step 1: Load corpus
        # ------------------------------------------------------------------
        if texts is None:
            print("[1/14] Loading corpus...")
            texts = _load_fineweb_edu(n_samples=n_samples)
            print(f"  Loaded {len(texts)} texts ({time.time()-t0:.1f}s)")
        else:
            print(f"[1/14] Using provided texts ({len(texts)} texts)")

        # ------------------------------------------------------------------
        # Step 2: Build vocabulary
        # ------------------------------------------------------------------
        print("\n[2/14] Building vocabulary...")
        self.vocab = Vocabulary(
            min_freq=self.vocab_min_freq,
            max_size=self.vocab_max_size,
        )
        self.vocab.build(texts)
        print(f"  Vocabulary size: {len(self.vocab)} words")

        # ------------------------------------------------------------------
        # Step 3: Tokenize texts → sequences
        # ------------------------------------------------------------------
        print("\n[3/14] Tokenizing texts...")
        sequences = _tokenize_texts(texts, self.vocab)
        sequences = _truncate_sequences(sequences, max_len=self.max_seq_len)
        print(f"  Tokenized: {len(sequences):,} sequences")

        # ------------------------------------------------------------------
        # Step 4: Split train/test (90/10)
        # ------------------------------------------------------------------
        split_idx = int(len(sequences) * 0.9)
        self.sequences = sequences[:split_idx]
        self.test_sequences = sequences[split_idx:]
        print(f"  Train: {len(self.sequences):,}, Test: {len(self.test_sequences):,}")
        rss = _get_rss_mb()
        if rss > 0:
            print(f"  Memory (RSS): {rss:,} MB")

        # Compute word frequencies (used by multiple components)
        self._word_freq = np.zeros(len(self.vocab), dtype=np.int64)
        total_tokens = 0
        for seq in self.sequences:
            for w in seq:
                if w < len(self.vocab):
                    self._word_freq[w] += 1
                    total_tokens += 1
        print(f"  Total train tokens: {total_tokens:,}")

        # ------------------------------------------------------------------
        # Step 5: Build POS type system
        # ------------------------------------------------------------------
        print("\n[5/14] Building POS type system...")
        self.pos_system = POSTypeSystem(
            vocab_size=len(self.vocab),
            window=5,
        )
        self.pos_system.build_from_vocabulary(self.vocab.word2idx, self.vocab.idx2word)
        self.pos_system.build_grammar_penalties(penalty_strength=60)
        self.pos_system.compute_type_couplings(self.sequences, self.vocab.idx2word)
        n_typed = sum(1 for w in range(len(self.vocab)) if w in self.pos_system.allowed_types)
        print(f"  POS system: {N_POS} types, {n_typed} words typed")

        # ------------------------------------------------------------------
        # Step 6: Build topic assigner
        # ------------------------------------------------------------------
        print("\n[6/14] Building topic assigner...")
        self.topic_assigner = TopicAssigner(n_topics=self.n_topics)
        self.topic_assigner.build(texts, self.vocab)

        # ------------------------------------------------------------------
        # Step 7: Build word n-gram index
        # ------------------------------------------------------------------
        rss_pre = _get_rss_mb()
        print(f"\n[7/14] Building word n-gram index..."
              f" (RSS: {rss_pre:,} MB)" if rss_pre > 0 else "\n[7/14] Building word n-gram index...")

        # Cap n-gram sequences for OOM protection
        ngram_seqs = self.sequences
        if self.ngram_max_sequences > 0 and len(self.sequences) > self.ngram_max_sequences:
            import random as _rnd
            _rnd.seed(42)
            ngram_seqs = _rnd.sample(self.sequences, self.ngram_max_sequences)
            print(f"  Capped: {len(self.sequences):,} → {len(ngram_seqs):,} sequences")

        self.word_index = WordNgramIndex(
            max_n=self.ngram_max_n,
            min_count=self.ngram_min_count,
        )
        if len(ngram_seqs) > 500000:
            print(f"  Large corpus — using batched build")
            self.word_index.build_batched(ngram_seqs, batch_size=200000)
        else:
            self.word_index.build(ngram_seqs)

        rss_post = _get_rss_mb()
        if rss_post > 0:
            print(f"  Word index memory delta: +{rss_post - rss_pre:,} MB (RSS: {rss_post:,} MB)")

        # ------------------------------------------------------------------
        # Step 8: Build POS n-gram index
        # ------------------------------------------------------------------
        print("\n[8/14] Building POS n-gram index...")
        # Derive word→POS mapping from pos_system
        word_pos_tags = {}
        TAG_PRIORITY = {
            POS2IDX["PUNCT"]: 0, POS2IDX["DET"]: 1, POS2IDX["PRON"]: 2,
            POS2IDX["AUX"]: 3, POS2IDX["CONJ"]: 4, POS2IDX["PART"]: 5,
            POS2IDX["PREP"]: 6, POS2IDX["NUM"]: 7, POS2IDX["ADV"]: 8,
            POS2IDX["ADJ"]: 9, POS2IDX["NOUN"]: 10, POS2IDX["VERB"]: 11,
            POS2IDX["X"]: 12,
        }
        for w, allowed in self.pos_system.allowed_types.items():
            if allowed:
                word_pos_tags[w] = min(allowed, key=lambda t: TAG_PRIORITY.get(t, 99))

        self.pos_index = PosNgramIndex(
            max_n=self.pos_ngram_max_n,
            min_count=self.pos_ngram_min_count,
            pos_system=self.pos_system,
        )
        if len(ngram_seqs) > 500000:
            print(f"  Large corpus — using batched build")
            self.pos_index.build_batched(ngram_seqs, word_pos_tags=word_pos_tags, batch_size=200000)
        else:
            self.pos_index.build(ngram_seqs, word_pos_tags=word_pos_tags)

        # ------------------------------------------------------------------
        # Step 9: Build topic n-gram index
        # ------------------------------------------------------------------
        print("\n[9/14] Building topic n-gram index...")
        self.topic_index = TopicNgramIndex(
            max_n=self.topic_ngram_max_n,
            min_count=self.topic_ngram_min_count,
            n_topics=self.n_topics,
            word_topics=self.topic_assigner.word_topics,
        )
        if len(ngram_seqs) > 500000:
            print(f"  Large corpus — using batched build")
            self.topic_index.build_batched(ngram_seqs, batch_size=200000)
        else:
            self.topic_index.build(ngram_seqs)

        # ------------------------------------------------------------------
        # Step 10: Build multi-scale recall
        # ------------------------------------------------------------------
        print("\n[10/14] Building multi-scale recall...")
        self.multiscale_recall = MultiScaleRecall(
            word_index=self.word_index,
            pos_index=self.pos_index,
            topic_index=self.topic_index,
            word_scale=self.recall_scale,
            pos_scale=self.pos_recall_scale,
            topic_scale=self.topic_recall_scale,
        )
        print(f"  {self.multiscale_recall.summary()}")

        # ------------------------------------------------------------------
        # Step 11: Build document state
        # ------------------------------------------------------------------
        print("\n[11/14] Building document state...")
        self.document_state = DocumentState(
            vocab_size=len(self.vocab),
            n_topics=self.n_topics,
            pos_system=self.pos_system,
            word_topics=self.topic_assigner.word_topics,
        )
        self.document_state.build(self.sequences, idx2word=self.vocab.idx2word)

        # ------------------------------------------------------------------
        # Step 12: Build energy computer
        # ------------------------------------------------------------------
        print("\n[12/14] Building energy computer...")
        self.energy_computer = EnergyComputer(
            multiscale_recall=self.multiscale_recall,
            document_state=self.document_state,
            pos_system=self.pos_system,
            recall_scale=self.recall_scale,
            pos_recall_scale=self.pos_recall_scale,
            topic_recall_scale=self.topic_recall_scale,
            state_scale=self.state_scale,
            same_word_penalty=self.same_word_penalty,
            max_closed_class_run=self.max_closed_class_run,
        )

        # ------------------------------------------------------------------
        # Step 13: Auto-calibrate beta
        # ------------------------------------------------------------------
        if self.auto_calibrate_beta:
            print("\n[13/14] Auto-calibrating beta from recall energy distribution...")
            self._auto_calibrate_beta()
        else:
            print(f"\n[13/14] Using provided beta_word={self.beta_word:.6f}")

        # ------------------------------------------------------------------
        # Step 14: Build generator
        # ------------------------------------------------------------------
        print("\n[14/14] Building generator...")
        self._build_generator()

        t_total = time.time() - t0
        print(f"\nTraining complete: {t_total:.1f}s")
        print(f"  Integer-only: YES (v17 — ZERO float operations in hot path)")
        return self

    # ===================================================================
    # BETA CALIBRATION
    # ===================================================================

    def _auto_calibrate_beta(self) -> None:
        """
        Auto-calibrate beta from recall-only energy distribution.

        The theoretical optimal beta for the Boltzmann distribution over
        recall energies is:
            beta = 0.55 * ln(2) / recall_scale

        This gives P(w) ~ exp(-beta * E_recall(w)) ≈ P_ngram(w)^0.5,
        which is the correct temperature for sampling.

        We also compute an empirical beta from the p10 energy difference,
        which adapts to the actual energy distribution.
        """
        if self.multiscale_recall is None:
            return

        # Theoretical beta
        theoretical_beta = 0.55 * math.log(2) / self.recall_scale

        # Sample recall energies from training sequences
        energy_diffs = []
        sample_count = 0
        max_samples = 200

        for seq in self.sequences[:max_samples]:
            if len(seq) < 3:
                continue

            for pos in range(1, min(len(seq), 10)):
                context_words = seq[:pos]
                # Get candidates of the same type as the actual next word
                target_word = seq[pos]
                target_type = self._get_word_type(target_word)
                candidate_list = self._type_words.get(target_type, [])
                if len(candidate_list) < 5:
                    continue

                candidate_words = np.array(candidate_list[:200], dtype=np.int64)

                # Get recall energy only (no state, no hard constraints)
                recall_energies = self.multiscale_recall.compute_energy(
                    context_words, candidate_words,
                    longest_only=not self.interpolated,
                    interpolated=self.interpolated,
                    kn_backoff=self.kn_backoff,
                )

                # Compute energy differences from minimum
                e_min = recall_energies.min()
                diffs = recall_energies - e_min
                diffs = diffs[diffs > 0]

                if len(diffs) > 0:
                    median_diff = int(np.median(diffs))
                    if median_diff > 0:
                        energy_diffs.append(median_diff)

                sample_count += 1
                if sample_count >= 500:
                    break
            if sample_count >= 500:
                break

        if energy_diffs:
            median_delta_e = int(np.median(energy_diffs))
            p10_delta_e = int(np.percentile(energy_diffs, 10))
            p90_delta_e = int(np.percentile(energy_diffs, 90))

            print(f"    Theoretical beta = {theoretical_beta:.6f}")
            print(f"    Median dE (recall): {median_delta_e}")
            print(f"    dE spread: p10={p10_delta_e}, p90={p90_delta_e}")

            # Empirical beta from p10 delta E (decision boundary)
            empirical_beta = (3.5 * 1.5) / max(1, p10_delta_e)
            empirical_beta = max(0.00001, min(1.0, empirical_beta))

            # Use the larger of theoretical and empirical
            chosen_beta = max(theoretical_beta, empirical_beta)

            if 0.00001 <= chosen_beta <= 1.0:
                self.beta_word = chosen_beta
                print(f"    Empirical beta = {empirical_beta:.6f}")
                print(f"    Using beta_word = {self.beta_word:.6f}")
            else:
                print(f"    Kept beta_word = {self.beta_word:.6f} (calibrated out of range)")
        else:
            self.beta_word = max(0.00001, min(1.0, theoretical_beta))
            print(f"    No energy diffs found, using theoretical beta = {self.beta_word:.6f}")

    # ===================================================================
    # GENERATOR CONSTRUCTION
    # ===================================================================

    def _build_generator(self) -> None:
        """Build the v17 generator."""
        from .generator import IsingLMGenerator

        word_sampler = IntegerBoltzmannSampler(
            beta=self.beta_word, max_delta=50000
        )
        type_sampler = IntegerBoltzmannSampler(
            beta=self.beta_type, max_delta=50000
        )

        self.generator = IsingLMGenerator(
            vocab=self.vocab,
            pos_system=self.pos_system,
            multiscale_recall=self.multiscale_recall,
            document_state=self.document_state,
            energy_computer=self.energy_computer,
            word_sampler=word_sampler,
            type_sampler=type_sampler,
            word_index=self.word_index,  # for copy mechanism
            copy_enabled=self.copy_enabled,
            copy_min_context=self.copy_min_context,
            copy_min_confidence=self.copy_min_confidence,
            same_word_penalty=self.same_word_penalty,
            max_closed_class_run=self.max_closed_class_run,
            interpolated=self.interpolated,
            kn_backoff=self.kn_backoff,
            recall_scale=self.recall_scale,
            pos_recall_scale=self.pos_recall_scale,
            topic_recall_scale=self.topic_recall_scale,
            state_scale=self.state_scale,
        )

    # ===================================================================
    # HELPER: Get word's primary POS type
    # ===================================================================

    def _get_word_type(self, word_idx: int) -> int:
        """Get primary POS type for a word."""
        if word_idx in self.pos_system.allowed_types and self.pos_system.allowed_types[word_idx]:
            TAG_PRIORITY = {
                POS2IDX["PUNCT"]: 0, POS2IDX["DET"]: 1, POS2IDX["PRON"]: 2,
                POS2IDX["AUX"]: 3, POS2IDX["CONJ"]: 4, POS2IDX["PART"]: 5,
                POS2IDX["PREP"]: 6, POS2IDX["NUM"]: 7, POS2IDX["ADV"]: 8,
                POS2IDX["ADJ"]: 9, POS2IDX["NOUN"]: 10, POS2IDX["VERB"]: 11,
                POS2IDX["X"]: 12,
            }
            return min(self.pos_system.allowed_types[word_idx],
                       key=lambda t: TAG_PRIORITY.get(t, 99))
        return POS2IDX["X"]

    @property
    def _type_words(self) -> Dict[int, List[int]]:
        """Build type→words mapping (lazy)."""
        if not hasattr(self, '_type_words_cache') or self._type_words_cache is None:
            tw: Dict[int, List[int]] = {t: [] for t in range(N_POS)}
            for w, allowed in self.pos_system.allowed_types.items():
                if allowed:
                    primary = min(allowed, key=lambda t: {
                        POS2IDX["PUNCT"]: 0, POS2IDX["DET"]: 1, POS2IDX["PRON"]: 2,
                        POS2IDX["AUX"]: 3, POS2IDX["CONJ"]: 4, POS2IDX["PART"]: 5,
                        POS2IDX["PREP"]: 6, POS2IDX["NUM"]: 7, POS2IDX["ADV"]: 8,
                        POS2IDX["ADJ"]: 9, POS2IDX["NOUN"]: 10, POS2IDX["VERB"]: 11,
                        POS2IDX["X"]: 12,
                    }.get(t, 99))
                    tw[primary].append(w)
            self._type_words_cache = tw
        return self._type_words_cache

    # ===================================================================
    # CONVENIENCE WRAPPERS
    # ===================================================================

    def generate(self, prompt: str = "the", length: int = 20) -> Dict:
        """Generate text autoregressively."""
        if self.generator is None:
            self._build_generator()
        return self.generator.generate(prompt=prompt, length=length)

    def generate_with_trace(self, prompt: str = "the", length: int = 20) -> Dict:
        """Generate text with full diagnostics."""
        if self.generator is None:
            self._build_generator()
        result = self.generator.generate(prompt=prompt, length=length)
        result['stats'] = self.generator.get_stats()
        return result

    def compute_perplexity(
        self,
        test_sequences: Optional[List[List[int]]] = None,
        n_samples: int = 100,
    ) -> float:
        """
        Compute perplexity on held-out test sequences.

        PPL = exp(-1/N * sum log P(w_t | ctx))

        Uses the word_sampler's Boltzmann lookup table for efficient
        computation of the partition function.  All arithmetic is integer,
        with final PPL conversion to float for display only.
        """
        if self.generator is None:
            self._build_generator()

        if test_sequences is None:
            test_sequences = self.test_sequences

        if not test_sequences:
            print("  Warning: No test sequences available. Returning inf PPL.")
            return float('inf')

        return self.generator.compute_perplexity(
            test_sequences=test_sequences,
            n_samples=n_samples,
        )

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
