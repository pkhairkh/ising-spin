"""
Ising Spin Glass Language Model — unified architecture.

A single model class with config-driven feature flags instead of
version-suffixed file names (model_v17, model_v18, …).

Architecture (all energy terms are additive; disabled terms contribute 0):
  1. Word-level n-gram recall (5-gram)            — E_recall
  2. POS-level n-gram recall (10-gram)            — E_pos
  3. Topic-level n-gram recall (10-gram)          — E_topic
  4. Dense Associative Memory (random features)   — E_dense_am
  5. VSA qFHRR binding (compositional encoding)   — E_vsa
  6. Integer ESN Reservoir (~50 token lookback)   — E_reservoir
  7. Cross-Scale RFF (joint word+POS+topic)       — E_rff
  8. Factorial State Coupling (mean-field)         — E_coupling
  9. Document state (7 evolving integer variables) — E_state
 10. Hard constraints (POS type, same-word, etc.)  — E_hard

ALL energy computation and Boltzmann sampling is integer-only.
Float operations appear ONLY during module initialization.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .errors import BuildError, ConfigError, ValidationError
from .helpers import (
    TAG_PRIORITY,
    get_primary_pos,
    get_rss_mb,
    load_fineweb_edu,
    tokenize_texts,
    truncate_sequences,
)
from .vocabulary import Vocabulary, POSTypeSystem, TopicAssigner
from .vocabulary.pos import COARSE_POS_TAGS, POS2IDX, IDX2POS, N_POS, CLOSED_CLASS
from .recall import WordNgramIndex, PosNgramIndex, TopicNgramIndex, MultiScaleRecall
from .state import DocumentState
from .energy import EnergyComputer
from .sampling import IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
from .vsa import VSAEncoder
from .dense_am import RandomFeatureProjector, DenseAMEnergy
from .reservoir import IntegerESN
from .rff import CrossScaleRFF


# ===========================================================================
# CONFIGURATION
# ===========================================================================

@dataclass
class ModelConfig:
    """All model hyperparameters in one place — no version slugs needed."""

    # --- Vocabulary ---
    vocab_min_freq: int = 15
    vocab_max_size: int = 49000

    # --- Word n-gram ---
    ngram_max_n: int = 5
    ngram_min_count: int = 2
    ngram_max_sequences: int = 1_000_000

    # --- POS n-gram ---
    pos_ngram_max_n: int = 10
    pos_ngram_min_count: int = 2

    # --- Topic ---
    n_topics: int = 16
    topic_ngram_max_n: int = 10
    topic_ngram_min_count: int = 3

    # --- Energy scales ---
    recall_scale: int = 1600
    pos_recall_scale: int = 800
    topic_recall_scale: int = 400
    state_scale: int = 400
    vsa_scale: int = 800
    dense_am_scale: int = 1200
    reservoir_scale: int = 800
    coupling_scale: int = 200
    rff_scale: int = 600

    # --- VSA ---
    vsa_enabled: bool = True
    vsa_dimension: int = 512
    vsa_seed: int = 42

    # --- Dense AM ---
    dense_am_enabled: bool = True
    dense_am_dim: int = 256
    dense_am_degree: int = 2
    dense_am_seed: int = 42
    dense_am_hash_dim: int = 32

    # --- Reservoir ---
    reservoir_enabled: bool = True
    reservoir_dim: int = 512
    reservoir_alpha_q15: int = 31130  # ≈0.95 in Q15
    reservoir_seed: int = 42

    # --- Mean-field coupling ---
    mf_enabled: bool = True
    mf_iterations: int = 5
    mf_lambda_q15: int = 16384  # ≈0.5 in Q15

    # --- Cross-Scale RFF ---
    rff_enabled: bool = True
    rff_dim: int = 256
    rff_hash_dim: int = 32
    rff_seed: int = 42

    # --- Hard constraints ---
    same_word_penalty: int = 200
    max_closed_class_run: int = 2

    # --- Beta ---
    beta_type: float = 0.01
    beta_word: float = 0.1
    auto_calibrate_beta: bool = True

    # --- Interpolation ---
    interpolated: bool = True
    kn_backoff: bool = True

    # --- Copy mechanism ---
    copy_enabled: bool = True
    copy_min_context: int = 3
    copy_min_confidence: float = 0.4

    # --- Misc ---
    max_seq_len: int = 30

    def validate(self) -> None:
        """Raise ConfigError if any parameter is out of range."""
        if self.vocab_min_freq < 1:
            raise ConfigError(f"vocab_min_freq must be >= 1, got {self.vocab_min_freq}")
        if self.vocab_max_size < 100:
            raise ConfigError(f"vocab_max_size must be >= 100, got {self.vocab_max_size}")
        if self.ngram_max_n < 2:
            raise ConfigError(f"ngram_max_n must be >= 2, got {self.ngram_max_n}")
        if self.recall_scale <= 0:
            raise ConfigError(f"recall_scale must be > 0, got {self.recall_scale}")
        if self.dense_am_degree not in (1, 2):
            raise ConfigError(f"dense_am_degree must be 1 or 2, got {self.dense_am_degree}")
        if self.reservoir_dim < 16:
            raise ConfigError(f"reservoir_dim must be >= 16, got {self.reservoir_dim}")
        if not 0 < self.reservoir_alpha_q15 < 32768:
            raise ConfigError(
                f"reservoir_alpha_q15 must be in (0, 32768), got {self.reservoir_alpha_q15}"
            )


# ===========================================================================
# MODEL
# ===========================================================================

class IsingLMModel:
    """
    Ising Spin Glass Language Model — config-driven unified architecture.

    Instead of version-suffixed classes (IsingLMModelV17, IsingLMModelV18),
    this single class uses a ``ModelConfig`` dataclass to enable/disable
    features and set all hyperparameters.

    Training pipeline:
      1.  Load corpus
      2.  Build vocabulary
      3.  Tokenize → sequences
      4.  Split train/test (90/10)
      5.  Build POS type system
      6.  Build topic assigner
      7.  Build word n-gram index
      8.  Build POS n-gram index
      9.  Build topic n-gram index
      10. Build multi-scale recall
      11. Build document state
      12. Build Dense AM (if enabled)
      13. Build VSA encoder (if enabled)
      14. Build Cross-Scale RFF (if enabled)
      15. Build Integer ESN reservoir (if enabled)
      16. Build factorial state coupling (if enabled)
      17. Build energy computer
      18. Auto-calibrate beta
      19. Build generator
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.config.validate()

        # Built during training
        self.vocab: Optional[Vocabulary] = None
        self.pos_system: Optional[POSTypeSystem] = None
        self.topic_assigner: Optional[TopicAssigner] = None
        self.word_index: Optional[WordNgramIndex] = None
        self.pos_index: Optional[PosNgramIndex] = None
        self.topic_index: Optional[TopicNgramIndex] = None
        self.multiscale_recall: Optional[MultiScaleRecall] = None
        self.document_state: Optional[DocumentState] = None
        self.dense_am: Optional[DenseAMEnergy] = None
        self.vsa_encoder: Optional[VSAEncoder] = None
        self.reservoir: Optional[IntegerESN] = None
        self.rff: Optional[CrossScaleRFF] = None
        self.energy_computer: Optional[EnergyComputer] = None
        self.generator = None

        self.sequences: Optional[List[List[int]]] = None
        self.test_sequences: Optional[List[List[int]]] = None
        self._word_freq: Optional[np.ndarray] = None
        self._type_words_cache: Optional[Dict[int, List[int]]] = None

    # ===================================================================
    # TRAINING PIPELINE
    # ===================================================================

    def train(self, n_samples: int = 50000, texts=None) -> "IsingLMModel":
        """Full training pipeline."""
        cfg = self.config
        t0 = time.time()

        self._print_header(n_samples)

        # Step 1: Load corpus
        if texts is None:
            print("[1/19] Loading corpus...")
            texts = load_fineweb_edu(n_samples=n_samples)
            print(f"  Loaded {len(texts)} texts ({time.time()-t0:.1f}s)")
        else:
            print(f"[1/19] Using provided texts ({len(texts)} texts)")

        # Step 2: Build vocabulary
        print("\n[2/19] Building vocabulary...")
        self.vocab = Vocabulary(
            min_freq=cfg.vocab_min_freq,
            max_size=cfg.vocab_max_size,
        )
        self.vocab.build(texts)
        print(f"  Vocabulary size: {len(self.vocab)} words")

        # Step 3: Tokenize
        print("\n[3/19] Tokenizing texts...")
        sequences = tokenize_texts(texts, self.vocab)
        sequences = truncate_sequences(sequences, max_len=cfg.max_seq_len)
        print(f"  Tokenized: {len(sequences):,} sequences")

        # Step 4: Split train/test
        split_idx = int(len(sequences) * 0.9)
        self.sequences = sequences[:split_idx]
        self.test_sequences = sequences[split_idx:]
        print(f"  Train: {len(self.sequences):,}, Test: {len(self.test_sequences):,}")
        rss = get_rss_mb()
        if rss > 0:
            print(f"  Memory (RSS): {rss:,} MB")

        # Compute word frequencies
        self._word_freq = np.zeros(len(self.vocab), dtype=np.int64)
        total_tokens = 0
        for seq in self.sequences:
            for w in seq:
                if w < len(self.vocab):
                    self._word_freq[w] += 1
                    total_tokens += 1
        print(f"  Total train tokens: {total_tokens:,}")

        # Step 5: POS type system
        print("\n[5/19] Building POS type system...")
        self.pos_system = POSTypeSystem(vocab_size=len(self.vocab), window=5)
        self.pos_system.build_from_vocabulary(self.vocab.word2idx, self.vocab.idx2word)
        self.pos_system.build_grammar_penalties(penalty_strength=60)
        self.pos_system.compute_type_couplings(self.sequences, self.vocab.idx2word)
        n_typed = sum(1 for w in range(len(self.vocab)) if w in self.pos_system.allowed_types)
        print(f"  POS system: {N_POS} types, {n_typed} words typed")

        # Step 6: Topic assigner
        print("\n[6/19] Building topic assigner...")
        self.topic_assigner = TopicAssigner(n_topics=cfg.n_topics)
        self.topic_assigner.build(texts, self.vocab)

        # Step 7: Word n-gram index
        print(f"\n[7/19] Building word n-gram index...")
        ngram_seqs = self._cap_sequences(self.sequences, cfg.ngram_max_sequences)
        self.word_index = WordNgramIndex(
            max_n=cfg.ngram_max_n, min_count=cfg.ngram_min_count,
        )
        self._build_index(self.word_index, ngram_seqs, "word")

        # Step 8: POS n-gram index
        print("\n[8/19] Building POS n-gram index...")
        word_pos_tags = self._build_word_pos_tags()
        self.pos_index = PosNgramIndex(
            max_n=cfg.pos_ngram_max_n, min_count=cfg.pos_ngram_min_count,
            pos_system=self.pos_system,
        )
        self._build_index(self.pos_index, ngram_seqs, "POS",
                          build_kwargs={"word_pos_tags": word_pos_tags})

        # Step 9: Topic n-gram index
        print("\n[9/19] Building topic n-gram index...")
        self.topic_index = TopicNgramIndex(
            max_n=cfg.topic_ngram_max_n, min_count=cfg.topic_ngram_min_count,
            n_topics=cfg.n_topics, word_topics=self.topic_assigner.word_topics,
        )
        self._build_index(self.topic_index, ngram_seqs, "topic")

        # Step 10: Multi-scale recall
        print("\n[10/19] Building multi-scale recall...")
        self.multiscale_recall = MultiScaleRecall(
            word_index=self.word_index,
            pos_index=self.pos_index,
            topic_index=self.topic_index,
            word_scale=cfg.recall_scale,
            pos_scale=cfg.pos_recall_scale,
            topic_scale=cfg.topic_recall_scale,
        )
        print(f"  {self.multiscale_recall.summary()}")

        # Step 11: Document state
        print("\n[11/19] Building document state...")
        self.document_state = DocumentState(
            vocab_size=len(self.vocab),
            n_topics=cfg.n_topics,
            pos_system=self.pos_system,
            word_topics=self.topic_assigner.word_topics,
        )
        self.document_state.build(self.sequences, idx2word=self.vocab.idx2word)

        # Step 12: Dense AM
        self._build_dense_am(cfg)

        # Step 13: VSA encoder
        self._build_vsa(cfg)

        # Step 14: Cross-Scale RFF
        self._build_rff(cfg, word_pos_tags)

        # Step 15: ESN Reservoir
        self._build_reservoir(cfg)

        # Step 16: Factorial state coupling
        self._build_coupling(cfg)

        # Step 17: Energy computer
        print("\n[17/19] Building energy computer...")
        self.energy_computer = EnergyComputer(
            multiscale_recall=self.multiscale_recall,
            document_state=self.document_state,
            pos_system=self.pos_system,
            vsa_encoder=self.vsa_encoder,
            dense_am=self.dense_am,
            reservoir=self.reservoir,
            rff=self.rff,
            recall_scale=cfg.recall_scale,
            pos_recall_scale=cfg.pos_recall_scale,
            topic_recall_scale=cfg.topic_recall_scale,
            state_scale=cfg.state_scale,
            vsa_scale=cfg.vsa_scale,
            dense_am_scale=cfg.dense_am_scale,
            reservoir_scale=cfg.reservoir_scale,
            coupling_scale=cfg.coupling_scale,
            rff_scale=cfg.rff_scale,
            same_word_penalty=cfg.same_word_penalty,
            max_closed_class_run=cfg.max_closed_class_run,
            interpolated=cfg.interpolated,
            kn_backoff=cfg.kn_backoff,
            mf_enabled=cfg.mf_enabled,
        )

        # Step 18: Auto-calibrate beta
        if cfg.auto_calibrate_beta:
            print("\n[18/19] Auto-calibrating beta...")
            self._auto_calibrate_beta()
        else:
            print(f"\n[18/19] Using provided beta_word={cfg.beta_word:.6f}")

        # Step 19: Build generator
        print("\n[19/19] Building generator...")
        self._build_generator()

        t_total = time.time() - t0
        print(f"\nTraining complete: {t_total:.1f}s")
        print(f"  Integer-only: YES (ZERO float operations in hot path)")
        return self

    # ===================================================================
    # BUILD HELPERS (private)
    # ===================================================================

    def _print_header(self, n_samples: int) -> None:
        cfg = self.config
        print("=" * 70)
        print("ISING SPIN GLASS LANGUAGE MODEL — unified architecture")
        print("=" * 70)
        print(f"\n  Word n-gram:  max_n={cfg.ngram_max_n}, scale={cfg.recall_scale}")
        print(f"  POS n-gram:   max_n={cfg.pos_ngram_max_n}, scale={cfg.pos_recall_scale}")
        print(f"  Topic n-gram: max_n={cfg.topic_ngram_max_n}, scale={cfg.topic_recall_scale}")
        print(f"  Dense AM:     {'ON' if cfg.dense_am_enabled else 'OFF'} "
              f"(D={cfg.dense_am_dim}, degree={cfg.dense_am_degree}, scale={cfg.dense_am_scale})")
        print(f"  VSA:          {'ON' if cfg.vsa_enabled else 'OFF'} "
              f"(D={cfg.vsa_dimension}, scale={cfg.vsa_scale})")
        print(f"  Reservoir:    {'ON' if cfg.reservoir_enabled else 'OFF'} "
              f"(D={cfg.reservoir_dim}, alpha_q15={cfg.reservoir_alpha_q15}, scale={cfg.reservoir_scale})")
        print(f"  RFF:          {'ON' if cfg.rff_enabled else 'OFF'} "
              f"(D={cfg.rff_dim}, hash_dim={cfg.rff_hash_dim}, scale={cfg.rff_scale})")
        print(f"  Coupling:     {'ON' if cfg.mf_enabled else 'OFF'} "
              f"(scale={cfg.coupling_scale}, iters={cfg.mf_iterations})")
        print(f"  Document state: scale={cfg.state_scale}")
        print(f"  Interpolated: {cfg.interpolated}, KN backoff: {cfg.kn_backoff}")
        print(f"  Auto-calibrate beta: {cfg.auto_calibrate_beta}")
        print()

    def _cap_sequences(self, seqs, max_seqs):
        if max_seqs > 0 and len(seqs) > max_seqs:
            import random
            random.seed(42)
            return random.sample(seqs, max_seqs)
        return seqs

    def _build_index(self, index, seqs, label, build_kwargs=None):
        kwargs = build_kwargs or {}
        if len(seqs) > 500_000:
            print(f"  Large corpus — using batched build")
            index.build_batched(seqs, batch_size=200_000, **kwargs)
        else:
            index.build(seqs, **kwargs)
        rss = get_rss_mb()
        if rss > 0:
            print(f"  {label} index RSS: {rss:,} MB")

    def _build_word_pos_tags(self) -> Dict[int, int]:
        tags: Dict[int, int] = {}
        for w, allowed in self.pos_system.allowed_types.items():
            if allowed:
                tags[w] = get_primary_pos(allowed)
        return tags

    def _build_dense_am(self, cfg: ModelConfig) -> None:
        if not cfg.dense_am_enabled:
            print(f"\n[12/19] Dense AM DISABLED")
            self.dense_am = None
            return
        print(f"\n[12/19] Building Dense AM (D={cfg.dense_am_dim}, degree={cfg.dense_am_degree})...")
        projector = RandomFeatureProjector(
            vocab_size=len(self.vocab),
            D=cfg.dense_am_dim,
            context_hash_dim=cfg.dense_am_hash_dim,
            seed=cfg.dense_am_seed,
        )
        self.dense_am = DenseAMEnergy(
            projector=projector,
            vocab_size=len(self.vocab),
            degree=cfg.dense_am_degree,
            dense_am_scale=cfg.dense_am_scale,
        )
        max_seqs = min(len(self.sequences), 200_000)
        t1 = time.time()
        self.dense_am.preaggregate(self.sequences, max_sequences=max_seqs)
        print(f"    Pre-aggregation took {time.time()-t1:.1f}s")
        if self.dense_am.Phi is not None:
            mem_mb = self.dense_am.Phi.nbytes / (1024 * 1024)
            print(f"    Phi: shape={self.dense_am.Phi.shape}, memory={mem_mb:.1f} MB")

    def _build_vsa(self, cfg: ModelConfig) -> None:
        if not cfg.vsa_enabled:
            print(f"\n[13/19] VSA module DISABLED")
            self.vsa_encoder = None
            return
        print(f"\n[13/19] Building VSA encoder (D={cfg.vsa_dimension})...")
        self.vsa_encoder = VSAEncoder(
            vocab_size=len(self.vocab),
            n_pos=N_POS,
            n_topics=cfg.n_topics,
            dimension=cfg.vsa_dimension,
            seed=cfg.vsa_seed,
        )
        self.vsa_encoder.build(
            pos_system=self.pos_system,
            word_topics=self.topic_assigner.word_topics,
        )
        R = self.vsa_encoder.readout_matrix
        if R is not None:
            mem_mb = R.nbytes / (1024 * 1024)
            print(f"  VSA readout: shape={R.shape}, memory={mem_mb:.1f} MB")

    def _build_rff(self, cfg: ModelConfig, word_pos_tags: Dict) -> None:
        if not cfg.rff_enabled:
            print(f"\n[14/19] Cross-Scale RFF DISABLED")
            self.rff = None
            return
        print(f"\n[14/19] Building Cross-Scale RFF (D={cfg.rff_dim})...")
        self.rff = CrossScaleRFF(
            vocab_size=len(self.vocab),
            n_pos=N_POS,
            n_topics=cfg.n_topics,
            D=cfg.rff_dim,
            context_hash_dim=cfg.rff_hash_dim,
            seed=cfg.rff_seed,
            rff_scale=cfg.rff_scale,
        )
        max_seqs = min(len(self.sequences), 200_000)
        t1 = time.time()
        self.rff.build(
            self.sequences,
            word_pos_tags=word_pos_tags,
            word_topics=self.topic_assigner.word_topics,
            max_sequences=max_seqs,
        )
        print(f"    RFF build took {time.time()-t1:.1f}s")
        if self.rff.Theta is not None:
            mem_T = self.rff.Theta.nbytes / (1024 * 1024)
            print(f"    Theta: shape={self.rff.Theta.shape}, memory={mem_T:.1f} MB")

    def _build_reservoir(self, cfg: ModelConfig) -> None:
        if not cfg.reservoir_enabled:
            print(f"\n[15/19] ESN Reservoir DISABLED")
            self.reservoir = None
            return
        print(f"\n[15/19] Building Integer ESN Reservoir (D={cfg.reservoir_dim})...")
        self.reservoir = IntegerESN(
            vocab_size=len(self.vocab),
            reservoir_dim=cfg.reservoir_dim,
            alpha_q15=cfg.reservoir_alpha_q15,
            seed=cfg.reservoir_seed,
        )
        max_seqs = min(len(self.sequences), 200_000)
        t1 = time.time()
        self.reservoir.build(self.sequences, max_sequences=max_seqs)
        print(f"    ESN build took {time.time()-t1:.1f}s")
        if self.reservoir.R is not None:
            mem_R = self.reservoir.R.nbytes / (1024 * 1024)
            mem_W = self.reservoir.W_in.nbytes / (1024 * 1024)
            print(f"    W_in: {self.reservoir.W_in.shape}, memory={mem_W:.1f} MB")
            print(f"    R: {self.reservoir.R.shape}, memory={mem_R:.1f} MB")

    def _build_coupling(self, cfg: ModelConfig) -> None:
        if not cfg.mf_enabled:
            print(f"\n[16/19] Factorial Coupling DISABLED")
            return
        print(f"\n[16/19] Building Factorial State Coupling...")
        self.document_state.build_coupling(
            self.sequences,
            idx2word=self.vocab.idx2word,
            mf_iterations=cfg.mf_iterations,
            mf_lambda_q15=cfg.mf_lambda_q15,
        )

    # ===================================================================
    # BETA CALIBRATION
    # ===================================================================

    def _auto_calibrate_beta(self) -> None:
        if self.multiscale_recall is None:
            return
        cfg = self.config
        theoretical_beta = 0.55 * math.log(2) / cfg.recall_scale

        energy_diffs = []
        sample_count = 0

        for seq in self.sequences[:200]:
            if len(seq) < 3:
                continue
            for pos in range(1, min(len(seq), 10)):
                context_words = seq[:pos]
                target_word = seq[pos]
                target_type = get_primary_pos(
                    self.pos_system.allowed_types.get(target_word, {POS2IDX["X"]})
                )
                candidate_list = self._type_words.get(target_type, [])
                if len(candidate_list) < 5:
                    continue
                candidate_words = np.array(candidate_list[:200], dtype=np.int64)
                recall_energies = self.multiscale_recall.compute_energy(
                    context_words, candidate_words,
                    longest_only=not cfg.interpolated,
                    interpolated=cfg.interpolated,
                    kn_backoff=cfg.kn_backoff,
                )
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
            p10_delta_e = int(np.percentile(energy_diffs, 10))
            empirical_beta = (3.5 * 1.5) / max(1, p10_delta_e)
            empirical_beta = max(0.00001, min(1.0, empirical_beta))
            chosen_beta = max(theoretical_beta, empirical_beta)
            if 0.00001 <= chosen_beta <= 1.0:
                cfg.beta_word = chosen_beta
                print(f"    Theoretical beta = {theoretical_beta:.6f}")
                print(f"    Empirical beta   = {empirical_beta:.6f}")
                print(f"    Using beta_word  = {cfg.beta_word:.6f}")
            else:
                print(f"    Kept beta_word = {cfg.beta_word:.6f} (calibrated out of range)")
        else:
            cfg.beta_word = max(0.00001, min(1.0, theoretical_beta))
            print(f"    No energy diffs, using theoretical beta = {cfg.beta_word:.6f}")

    # ===================================================================
    # GENERATOR CONSTRUCTION
    # ===================================================================

    def _build_generator(self) -> None:
        from .generator import IsingLMGenerator
        cfg = self.config

        word_sampler = IntegerBoltzmannSampler(
            beta=cfg.beta_word, max_delta=50000
        )
        type_sampler = IntegerBoltzmannSampler(
            beta=cfg.beta_type, max_delta=50000
        )

        self.generator = IsingLMGenerator(
            vocab=self.vocab,
            pos_system=self.pos_system,
            multiscale_recall=self.multiscale_recall,
            document_state=self.document_state,
            energy_computer=self.energy_computer,
            word_sampler=word_sampler,
            type_sampler=type_sampler,
            word_index=self.word_index,
            reservoir=self.reservoir,
            copy_enabled=cfg.copy_enabled,
            copy_min_context=cfg.copy_min_context,
            copy_min_confidence=cfg.copy_min_confidence,
            same_word_penalty=cfg.same_word_penalty,
            max_closed_class_run=cfg.max_closed_class_run,
            interpolated=cfg.interpolated,
            kn_backoff=cfg.kn_backoff,
            recall_scale=cfg.recall_scale,
            pos_recall_scale=cfg.pos_recall_scale,
            topic_recall_scale=cfg.topic_recall_scale,
            state_scale=cfg.state_scale,
        )

    # ===================================================================
    # TYPE-WORDS MAPPING (lazy)
    # ===================================================================

    @property
    def _type_words(self) -> Dict[int, List[int]]:
        if self._type_words_cache is None:
            tw: Dict[int, List[int]] = {t: [] for t in range(N_POS)}
            for w, allowed in self.pos_system.allowed_types.items():
                if allowed:
                    for t in allowed:
                        tw[t].append(w)
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

    def compute_perplexity(
        self,
        test_sequences: Optional[List[List[int]]] = None,
        n_samples: int = 100,
    ) -> float:
        """Compute perplexity on held-out test sequences."""
        if self.generator is None:
            self._build_generator()
        if test_sequences is None:
            test_sequences = self.test_sequences
        if not test_sequences:
            print("  Warning: No test sequences available. Returning inf PPL.")
            return float('inf')
        return self.generator.compute_perplexity(
            test_sequences=test_sequences, n_samples=n_samples,
        )
