"""
Ising Spin Glass Language Model v18.3 — Cross-Scale RFF + Integer ESN Reservoir + Factorial State Coupling

Architecture (extends v18.2):
  1. Word-level n-gram recall (5-gram)
  2. POS-level n-gram recall (10-gram)
  3. Topic-level n-gram recall (10-gram)
  4. Dense AM (v18.1 — nonlinear pattern matching with random features)
  5. VSA qFHRR binding (v18.0 — compositional word+POS+topic encoding)
  6. Reservoir (v18.2 — Integer ESN for ~50 token lookback)
  7. Cross-Scale RFF (v18.3 NEW — joint word+POS+topic random Fourier features)
  8. Document state (7 evolving integer variables, REBALANCED scale=400)
  9. Factorial coupling (v18.2 — mean-field state inference + coupling energy)
 10. Hard constraints (POS type, same-word, closed-class)

Key insight (v18.3): Dense AM, VSA, and Reservoir each operate on a SINGLE
scale — word, word+POS+topic, or temporal. Cross-Scale RFF combines word+
POS+topic into a SINGLE feature vector via random projection with cosine
nonlinearity, capturing interactions that independent per-scale energy terms
miss (e.g., "NOUN in SPORTS context" vs "NOUN in POLITICS context").

Key insight (v18.2): Standard n-gram recall has a HARD window (5 tokens for
word, 10 for POS/topic). The Integer ESN provides a SOFT window via
exponential decay — tokens from 50 positions ago still have ~5% influence.
This is the "temporal dynamics" that transformers get from self-attention,
but implemented as a fixed random recurrent network with integer arithmetic.

Additionally, the 7 state variables are no longer independent. Pairwise
compatibility tables capture correlations (e.g., topic=SCIENCE co-occurs
with mode=DESCRIPTION). Mean-field inference iteratively refines state
values using coupling, and coupling energy penalizes unlikely combinations.

v18.3 changes from v18.2:
  - NEW: Cross-Scale RFF module (E_rff energy term)
  - NEW: --rff-dim, --rff-hash-dim, --rff-scale, --no-rff CLI flags
  - CHANGED: EnergyComputer now includes E_rff term

v18.2 changes from v18.1:
  - NEW: Integer ESN reservoir (E_reservoir energy term, ~50 token lookback)
  - NEW: Factorial state coupling with mean-field inference
  - NEW: E_coupling energy term (scalar offset for unlikely state combos)
  - NEW: --reservoir-dim, --reservoir-alpha, --reservoir-scale, --no-reservoir
  - NEW: --coupling-scale, --no-mf CLI flags
  - CHANGED: EnergyComputer now includes E_reservoir and E_coupling terms
  - CHANGED: DocumentState now has build_coupling() and run_mean_field()
  - CHANGED: Generator now tracks reservoir state per-token

v18.1 changes from v18.0:
  - NEW: Dense AM module with polynomial nonlinearity (E_dense_am energy term)
  - NEW: Random feature pre-aggregation (Phi matrix, V x D int16)
  - NEW: --dense-am-dim, --dense-am-degree, --no-dense-am CLI flags
  - CHANGED: EnergyComputer now includes E_dense_am term

v18.0 changes from v17.4:
  - NEW: VSA qFHRR binding module (E_vsa_bind energy term)
  - CHANGED: state_scale default 50 → 400 (state energy was <3% of total)
  - NEW: vsa_scale parameter (default 800)
  - NEW: --no-vsa ablation flag
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
from .vsa import VSAEncoder
from .dense_am import RandomFeatureProjector, DenseAMEnergy
from .reservoir import IntegerESN
from .rff import CrossScaleRFF


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


class IsingLMModelV18:
    """
    v18.3: Multi-Scale Abstract Recall + Dense AM + VSA Binding + Integer ESN
           Reservoir + Cross-Scale RFF + Factorial State Coupling.

    Training pipeline (extends v18.2 with step 14 for Cross-Scale RFF):
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
      12. Build Dense AM (random feature projector + pre-aggregation) (v18.1)
      13. Build VSA encoder and readout matrix (v18.0)
      14. Build Cross-Scale RFF (v18.3 NEW)
      15. Build Integer ESN reservoir (v18.2)
      16. Build factorial state coupling (v18.2)
      17. Build energy computer (with RFF + Reservoir + Coupling + Dense AM + VSA)
      18. Auto-calibrate beta
      19. Build generator
    """

    def __init__(
        self,
        # Vocabulary
        vocab_min_freq: int = 15,
        vocab_max_size: int = 49000,
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
        pos_recall_scale: int = 800,
        topic_recall_scale: int = 400,
        state_scale: int = 400,         # v18.0: increased from 50 for meaningful contribution
        vsa_scale: int = 800,           # v18.0: VSA binding energy scale
        dense_am_scale: int = 1200,     # v18.1: Dense AM energy scale
        reservoir_scale: int = 800,     # v18.2 NEW: ESN reservoir energy scale
        coupling_scale: int = 200,      # v18.2 NEW: Factorial state coupling scale
        # VSA
        vsa_enabled: bool = True,       # v18.0: enable/disable VSA module
        vsa_dimension: int = 512,       # v18.0: VSA vector dimension
        vsa_seed: int = 42,             # v18.0: VSA random seed
        # Dense AM
        dense_am_enabled: bool = True,  # v18.1: enable/disable Dense AM
        dense_am_dim: int = 256,        # v18.1: random feature dimension
        dense_am_degree: int = 2,       # v18.1: polynomial degree (1=linear, 2=Dense AM)
        dense_am_seed: int = 42,        # v18.1: random feature seed
        dense_am_hash_dim: int = 32,    # v18.1: context hash dimension
        # Reservoir (v18.2 NEW)
        reservoir_enabled: bool = True,   # v18.2: enable/disable ESN reservoir
        reservoir_dim: int = 512,         # v18.2: reservoir state dimension
        reservoir_alpha_q15: int = 31130, # v18.2: Q15 decay factor (~0.95)
        reservoir_seed: int = 42,         # v18.2: random seed for W_in
        # Coupling (v18.2 NEW)
        mf_enabled: bool = True,          # v18.2: enable/disable mean-field inference
        mf_iterations: int = 5,           # v18.2: number of mean-field iterations
        mf_lambda_q15: int = 16384,       # v18.2: coupling strength in Q15 (~0.5)
        # RFF (v18.3 NEW)
        rff_enabled: bool = True,           # v18.3: enable/disable Cross-Scale RFF
        rff_dim: int = 256,                 # v18.3: RFF feature dimension
        rff_hash_dim: int = 32,             # v18.3: RFF context hash dimension
        rff_seed: int = 42,                 # v18.3: RFF random seed
        rff_scale: int = 600,               # v18.3: RFF energy scale
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
        self.vsa_scale = vsa_scale
        self.dense_am_scale = dense_am_scale
        self.reservoir_scale = reservoir_scale
        self.coupling_scale = coupling_scale
        self.vsa_enabled = vsa_enabled
        self.vsa_dimension = vsa_dimension
        self.vsa_seed = vsa_seed
        self.dense_am_enabled = dense_am_enabled
        self.dense_am_dim = dense_am_dim
        self.dense_am_degree = dense_am_degree
        self.dense_am_seed = dense_am_seed
        self.dense_am_hash_dim = dense_am_hash_dim
        self.reservoir_enabled = reservoir_enabled
        self.reservoir_dim = reservoir_dim
        self.reservoir_alpha_q15 = reservoir_alpha_q15
        self.reservoir_seed = reservoir_seed
        self.mf_enabled = mf_enabled
        self.mf_iterations = mf_iterations
        self.mf_lambda_q15 = mf_lambda_q15
        self.rff_enabled = rff_enabled
        self.rff_dim = rff_dim
        self.rff_hash_dim = rff_hash_dim
        self.rff_seed = rff_seed
        self.rff_scale = rff_scale
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
        self.dense_am: Optional[DenseAMEnergy] = None       # v18.1
        self.vsa_encoder: Optional[VSAEncoder] = None       # v18.0
        self.reservoir: Optional[IntegerESN] = None         # v18.2 NEW
        self.rff: Optional[CrossScaleRFF] = None             # v18.3 NEW
        self.energy_computer: Optional[EnergyComputer] = None
        self.generator = None

        self.sequences: Optional[List[List[int]]] = None
        self.test_sequences: Optional[List[List[int]]] = None
        self._word_freq: Optional[np.ndarray] = None

    def train(self, n_samples: int = 50000, texts=None) -> "IsingLMModelV18":
        """
        Full training pipeline for v18.2 — Integer ESN Reservoir + Factorial State Coupling.
        """
        print("=" * 70)
        print("ISING SPIN GLASS LANGUAGE MODEL v18.3 — CROSS-SCALE RFF + ESN RESERVOIR + FACTORIAL COUPLING")
        print("=" * 70)
        print(f"\n  Architecture: 3-Scale Recall + Dense AM + VSA + Cross-Scale RFF + ESN Reservoir + Factorial Coupling")
        print(f"  v18.3 NEW: Cross-Scale RFF (D={self.rff_dim}, hash_dim={self.rff_hash_dim}, scale={self.rff_scale})")
        print(f"  v18.2: Integer ESN Reservoir (D={self.reservoir_dim}, alpha_q15={self.reservoir_alpha_q15}, ~50 token lookback)")
        print(f"  v18.2: Factorial State Coupling (5 pairs, mf_iterations={self.mf_iterations}, lambda_q15={self.mf_lambda_q15})")
        print(f"  v18.1: Dense AM (F(x)=x^{self.dense_am_degree}, D={self.dense_am_dim})")
        print(f"  v18.0: VSA qFHRR binding (E_vsa_bind energy term)")
        print(f"  Word n-gram:  max_n={self.ngram_max_n}, scale={self.recall_scale}")
        print(f"  POS n-gram:   max_n={self.pos_ngram_max_n}, scale={self.pos_recall_scale}")
        print(f"  Topic n-gram: max_n={self.topic_ngram_max_n}, scale={self.topic_recall_scale}")
        print(f"  Dense AM:     enabled={self.dense_am_enabled}, D={self.dense_am_dim}, "
              f"degree={self.dense_am_degree}, scale={self.dense_am_scale}")
        print(f"  VSA binding:  enabled={self.vsa_enabled}, D={self.vsa_dimension}, scale={self.vsa_scale}")
        print(f"  Reservoir:    enabled={self.reservoir_enabled}, D={self.reservoir_dim}, "
              f"alpha_q15={self.reservoir_alpha_q15}, scale={self.reservoir_scale}")
        print(f"  RFF:          enabled={self.rff_enabled}, D={self.rff_dim}, "
              f"hash_dim={self.rff_hash_dim}, scale={self.rff_scale}")
        print(f"  Coupling:     enabled={self.mf_enabled}, scale={self.coupling_scale}, "
              f"mf_iters={self.mf_iterations}, lambda_q15={self.mf_lambda_q15}")
        print(f"  Document state: scale={self.state_scale}")
        print(f"  Interpolated: {self.interpolated}, KN backoff: {self.kn_backoff}")
        print(f"  Auto-calibrate beta: {self.auto_calibrate_beta}")
        print()

        t0 = time.time()

        # ------------------------------------------------------------------
        # Step 1: Load corpus
        # ------------------------------------------------------------------
        if texts is None:
            print("[1/19] Loading corpus...")
            texts = _load_fineweb_edu(n_samples=n_samples)
            print(f"  Loaded {len(texts)} texts ({time.time()-t0:.1f}s)")
        else:
            print(f"[1/19] Using provided texts ({len(texts)} texts)")

        # ------------------------------------------------------------------
        # Step 2: Build vocabulary
        # ------------------------------------------------------------------
        print("\n[2/19] Building vocabulary...")
        self.vocab = Vocabulary(
            min_freq=self.vocab_min_freq,
            max_size=self.vocab_max_size,
        )
        self.vocab.build(texts)
        print(f"  Vocabulary size: {len(self.vocab)} words")

        # ------------------------------------------------------------------
        # Step 3: Tokenize texts → sequences
        # ------------------------------------------------------------------
        print("\n[3/19] Tokenizing texts...")
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

        # Compute word frequencies
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
        print("\n[5/19] Building POS type system...")
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
        print("\n[6/19] Building topic assigner...")
        self.topic_assigner = TopicAssigner(n_topics=self.n_topics)
        self.topic_assigner.build(texts, self.vocab)

        # ------------------------------------------------------------------
        # Step 7: Build word n-gram index
        # ------------------------------------------------------------------
        rss_pre = _get_rss_mb()
        print(f"\n[7/19] Building word n-gram index...")

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

        # ------------------------------------------------------------------
        # Step 8: Build POS n-gram index
        # ------------------------------------------------------------------
        print("\n[8/19] Building POS n-gram index...")
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
            self.pos_index.build_batched(ngram_seqs, word_pos_tags=word_pos_tags, batch_size=200000)
        else:
            self.pos_index.build(ngram_seqs, word_pos_tags=word_pos_tags)

        # ------------------------------------------------------------------
        # Step 9: Build topic n-gram index
        # ------------------------------------------------------------------
        print("\n[9/19] Building topic n-gram index...")
        self.topic_index = TopicNgramIndex(
            max_n=self.topic_ngram_max_n,
            min_count=self.topic_ngram_min_count,
            n_topics=self.n_topics,
            word_topics=self.topic_assigner.word_topics,
        )
        if len(ngram_seqs) > 500000:
            self.topic_index.build_batched(ngram_seqs, batch_size=200000)
        else:
            self.topic_index.build(ngram_seqs)

        # ------------------------------------------------------------------
        # Step 10: Build multi-scale recall
        # ------------------------------------------------------------------
        print("\n[10/19] Building multi-scale recall...")
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
        print("\n[11/19] Building document state...")
        self.document_state = DocumentState(
            vocab_size=len(self.vocab),
            n_topics=self.n_topics,
            pos_system=self.pos_system,
            word_topics=self.topic_assigner.word_topics,
        )
        self.document_state.build(self.sequences, idx2word=self.vocab.idx2word)

        # ------------------------------------------------------------------
        # Step 12: Build Dense AM (v18.1)
        # ------------------------------------------------------------------
        if self.dense_am_enabled:
            print(f"\n[12/19] Building Dense AM (D={self.dense_am_dim}, degree={self.dense_am_degree})...")
            projector = RandomFeatureProjector(
                vocab_size=len(self.vocab),
                D=self.dense_am_dim,
                context_hash_dim=self.dense_am_hash_dim,
                seed=self.dense_am_seed,
            )
            self.dense_am = DenseAMEnergy(
                projector=projector,
                vocab_size=len(self.vocab),
                degree=self.dense_am_degree,
                dense_am_scale=self.dense_am_scale,
            )

            # Pre-aggregate from training sequences
            # Use capped number of sequences for speed
            max_seqs = min(len(self.sequences), 200000)
            t_preagg = time.time()
            self.dense_am.preaggregate(self.sequences, max_sequences=max_seqs)
            print(f"    Pre-aggregation took {time.time()-t_preagg:.1f}s")

            if self.dense_am.Phi is not None:
                mem_mb = self.dense_am.Phi.nbytes / (1024 * 1024)
                print(f"    Dense AM Phi: shape={self.dense_am.Phi.shape}, memory={mem_mb:.1f} MB")
        else:
            print(f"\n[12/19] Dense AM DISABLED (--no-dense-am flag)")
            self.dense_am = None

        # ------------------------------------------------------------------
        # Step 13: Build VSA encoder and readout matrix (v18.0)
        # ------------------------------------------------------------------
        if self.vsa_enabled:
            print(f"\n[13/19] Building VSA encoder (D={self.vsa_dimension})...")
            self.vsa_encoder = VSAEncoder(
                vocab_size=len(self.vocab),
                n_pos=N_POS,
                n_topics=self.n_topics,
                dimension=self.vsa_dimension,
                seed=self.vsa_seed,
            )
            self.vsa_encoder.build(
                pos_system=self.pos_system,
                word_topics=self.topic_assigner.word_topics,
            )
            R = self.vsa_encoder.readout_matrix
            if R is not None:
                mem_mb = R.nbytes / (1024 * 1024)
                print(f"  VSA readout: shape={R.shape}, memory={mem_mb:.1f} MB")
        else:
            print(f"\n[13/19] VSA module DISABLED (--no-vsa flag)")
            self.vsa_encoder = None

        # ------------------------------------------------------------------
        # Step 14: Build Cross-Scale RFF (v18.3 NEW)
        # ------------------------------------------------------------------
        if self.rff_enabled:
            print(f"\n[14/19] Building Cross-Scale RFF (D={self.rff_dim}, "
                  f"hash_dim={self.rff_hash_dim})...")
            self.rff = CrossScaleRFF(
                vocab_size=len(self.vocab),
                n_pos=N_POS,
                n_topics=self.n_topics,
                D=self.rff_dim,
                context_hash_dim=self.rff_hash_dim,
                seed=self.rff_seed,
                rff_scale=self.rff_scale,
            )

            max_seqs = min(len(self.sequences), 200000)
            t_rff = time.time()
            self.rff.build(
                self.sequences,
                word_pos_tags=word_pos_tags,
                word_topics=self.topic_assigner.word_topics,
                max_sequences=max_seqs,
            )
            print(f"    RFF Theta build took {time.time()-t_rff:.1f}s")

            if self.rff.Theta is not None:
                mem_T = self.rff.Theta.nbytes / (1024 * 1024)
                print(f"    RFF Theta: shape={self.rff.Theta.shape}, memory={mem_T:.1f} MB")
        else:
            print(f"\n[14/19] Cross-Scale RFF DISABLED (--no-rff flag)")
            self.rff = None

        # ------------------------------------------------------------------
        # Step 15: Build Integer ESN reservoir (v18.2)
        # ------------------------------------------------------------------
        if self.reservoir_enabled:
            print(f"\n[15/19] Building Integer ESN Reservoir (D={self.reservoir_dim}, "
                  f"alpha_q15={self.reservoir_alpha_q15})...")
            self.reservoir = IntegerESN(
                vocab_size=len(self.vocab),
                reservoir_dim=self.reservoir_dim,
                alpha_q15=self.reservoir_alpha_q15,
                seed=self.reservoir_seed,
            )

            # Pre-aggregate readout matrix from training sequences
            max_seqs = min(len(self.sequences), 200000)
            t_esn = time.time()
            self.reservoir.build(self.sequences, max_sequences=max_seqs)
            print(f"    ESN readout build took {time.time()-t_esn:.1f}s")

            if self.reservoir.R is not None:
                mem_R = self.reservoir.R.nbytes / (1024 * 1024)
                mem_W = self.reservoir.W_in.nbytes / (1024 * 1024)
                print(f"    ESN W_in: {self.reservoir.W_in.shape}, memory={mem_W:.1f} MB")
                print(f"    ESN R: {self.reservoir.R.shape}, memory={mem_R:.1f} MB")
        else:
            print(f"\n[15/19] ESN Reservoir DISABLED (--no-reservoir flag)")
            self.reservoir = None

        # ------------------------------------------------------------------
        # Step 16: Build factorial state coupling (v18.2)
        # ------------------------------------------------------------------
        if self.mf_enabled:
            print(f"\n[16/19] Building Factorial State Coupling "
                  f"(5 pairs, mf_iters={self.mf_iterations})...")
            self.document_state.build_coupling(
                self.sequences,
                idx2word=self.vocab.idx2word,
                mf_iterations=self.mf_iterations,
                mf_lambda_q15=self.mf_lambda_q15,
            )
        else:
            print(f"\n[16/19] Factorial Coupling DISABLED (--no-mf flag)")

        # ------------------------------------------------------------------
        # Step 17: Build energy computer (with RFF + Reservoir + Coupling + Dense AM + VSA)
        # ------------------------------------------------------------------
        print("\n[17/19] Building energy computer...")
        self.energy_computer = EnergyComputer(
            multiscale_recall=self.multiscale_recall,
            document_state=self.document_state,
            pos_system=self.pos_system,
            vsa_encoder=self.vsa_encoder,
            dense_am=self.dense_am,            # v18.1
            reservoir=self.reservoir,          # v18.2
            rff=self.rff,                      # v18.3 NEW
            recall_scale=self.recall_scale,
            pos_recall_scale=self.pos_recall_scale,
            topic_recall_scale=self.topic_recall_scale,
            state_scale=self.state_scale,
            vsa_scale=self.vsa_scale,
            dense_am_scale=self.dense_am_scale,
            reservoir_scale=self.reservoir_scale,   # v18.2
            coupling_scale=self.coupling_scale,      # v18.2
            rff_scale=self.rff_scale,               # v18.3 NEW
            same_word_penalty=self.same_word_penalty,
            max_closed_class_run=self.max_closed_class_run,
            interpolated=self.interpolated,
            kn_backoff=self.kn_backoff,
            mf_enabled=self.mf_enabled,             # v18.2
        )

        # ------------------------------------------------------------------
        # Step 18: Auto-calibrate beta
        # ------------------------------------------------------------------
        if self.auto_calibrate_beta:
            print("\n[18/19] Auto-calibrating beta from recall energy distribution...")
            self._auto_calibrate_beta()
        else:
            print(f"\n[18/19] Using provided beta_word={self.beta_word:.6f}")

        # ------------------------------------------------------------------
        # Step 19: Build generator
        # ------------------------------------------------------------------
        print("\n[19/19] Building generator...")
        self._build_generator()

        t_total = time.time() - t0
        print(f"\nTraining complete: {t_total:.1f}s")
        print(f"  Integer-only: YES (v18.3 — ZERO float operations in hot path)")
        print(f"  Dense AM: {'ENABLED' if self.dense_am else 'DISABLED'} "
              f"(degree={self.dense_am_degree})" if self.dense_am else "")
        print(f"  VSA binding: {'ENABLED' if self.vsa_encoder else 'DISABLED'}")
        print(f"  Reservoir: {'ENABLED' if self.reservoir else 'DISABLED'} "
              f"(D={self.reservoir_dim})" if self.reservoir else "")
        print(f"  RFF: {'ENABLED' if self.rff else 'DISABLED'} "
              f"(D={self.rff_dim})" if self.rff else "")
        print(f"  Coupling: {'ENABLED' if self.mf_enabled else 'DISABLED'} "
              f"(scale={self.coupling_scale})")
        print(f"  State scale: {self.state_scale} (was 50 in v17)")
        return self

    # ===================================================================
    # BETA CALIBRATION
    # ===================================================================

    def _auto_calibrate_beta(self) -> None:
        """Auto-calibrate beta from recall energy distribution."""
        if self.multiscale_recall is None:
            return

        theoretical_beta = 0.55 * math.log(2) / self.recall_scale

        energy_diffs = []
        sample_count = 0
        max_samples = 200

        for seq in self.sequences[:max_samples]:
            if len(seq) < 3:
                continue

            for pos in range(1, min(len(seq), 10)):
                context_words = seq[:pos]
                target_word = seq[pos]
                target_type = self._get_word_type(target_word)
                candidate_list = self._type_words.get(target_type, [])
                if len(candidate_list) < 5:
                    continue

                candidate_words = np.array(candidate_list[:200], dtype=np.int64)

                recall_energies = self.multiscale_recall.compute_energy(
                    context_words, candidate_words,
                    longest_only=not self.interpolated,
                    interpolated=self.interpolated,
                    kn_backoff=self.kn_backoff,
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
            median_delta_e = int(np.median(energy_diffs))
            p10_delta_e = int(np.percentile(energy_diffs, 10))

            print(f"    Theoretical beta = {theoretical_beta:.6f}")
            print(f"    Median dE (recall): {median_delta_e}")
            print(f"    dE p10={p10_delta_e}")

            empirical_beta = (3.5 * 1.5) / max(1, p10_delta_e)
            empirical_beta = max(0.00001, min(1.0, empirical_beta))

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
        """Build the v18 generator."""
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
            word_index=self.word_index,
            reservoir=self.reservoir,  # v18.2: ESN reservoir
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
    # HELPERS
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
        """Build type→words mapping (lazy). Multi-type — words in ALL allowed buckets."""
        if not hasattr(self, '_type_words_cache') or self._type_words_cache is None:
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
            test_sequences=test_sequences,
            n_samples=n_samples,
        )
