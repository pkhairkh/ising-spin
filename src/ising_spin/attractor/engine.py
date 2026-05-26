"""
Attractor Language Machine — THE ENGINE.

The attractor dynamics of a Dense Associative Memory ARE a language model.
Not an approximation. Not a component. They ARE one.

This engine implements:
  - DAM attractor dynamics with NONLINEAR F-lookup energy (exponential capacity)
  - Hierarchical DAM states (L0-L3) with Wilsonian RG flow on COUPLINGS
  - ONLY L0 trained (Hebbian); higher levels get J from RG decimation
  - Content-addressable episodic memory (sparse pattern storage)
  - Sparse Distributed Representations (kWTA, ~2% active bits)
  - UV completeness checks (Ward identities + cutoff independence)

DEEP FIXES (v28 — from knowledge base analysis):
  1. F_EXP_APPROX: piecewise integer exponential — TRUE exponential capacity
  2. RG-derived J_eff REPLACES J at higher levels (not just diagnostic)
  3. Ward identity UV checks (not just spectral gap / cutoff sensitivity)
  4. Pure Hebbian ONLY (PCD removed — unnecessary at right sparsity)
  5. Anomalous dimensions from operator spectrum of J (not running correlations)
  6. DAM energy + bigram J2 + skip J2 drives word selection (v54: POS-driven + bigram-dominant generation)
  7. D decreasing: 512->256->128->64 (RG reduces DOF at coarser scales)

WHAT'S KEPT:
  - Vocabulary building and tokenization
  - POS type system (grammar constraints are still needed)
  - IntegerBoltzmannSampler (for final word selection from DAM energies)
  - Data loading infrastructure

HOW PREDICTION WORKS:
  1. Encode context words as SDRs -> context_field for L0
  2. Run hierarchical attractor dynamics (L0->L3 bottom-up, L3->L0 top-down)
  3. Compute DAM energy for each candidate word using F-LOOKUP (exp_approx)
  4. Add episodic memory energy
  5. Add POS grammar constraints
  6. Boltzmann sample from energy distribution
  7. Update all layer states and episodic memory

TRAINING (SIMPLIFIED — v28):
  Phase 1: Build SDR encoder (deterministic, no learning)
  Phase 2: Train L0 via batch Hebbian (RG fixed point)
  Phase 3: Compute coupling flow, apply J_eff to all higher levels
  Phase 4: Check UV completeness (Ward identities + cutoff independence)
  Phase 5: Build episodic memory
  Phase 6: Calibrate beta

ALL INTEGER ARITHMETIC. Runs on Raspberry Pi 5.
"""

import math
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

from .sdr import SDREncoder
from .dam import DAMLayer
from .hierarchy import HierarchicalDAM
from .episodic import EpisodicMemory
from .binding import BindingContext


class AttractorLanguageModel:
    """
    Attractor Language Machine — Dense Associative Memory as the ENGINE.

    The full language model: SDR encoder + hierarchical DAM + episodic
    memory + POS constraints. Training via Hebbian (L0 only) + RG flow.
    """

    def __init__(
        self,
        # Vocabulary
        vocab_min_freq: int = 5,
        vocab_max_size: int = 2000,
        # SDR
        sdr_dim: int = 512,
        sdr_sparsity: float = 0.02,
        # Hierarchy
        layers_config: Optional[List[Tuple[int, int, int]]] = None,
        # Coupling
        j_clip: int = 500,
        # UV-complete
        uv_regularize: bool = True,
        uv_lambda: int = 5,
        topdown_scale: int = 200,
        # DAM
        dam_scale: int = 1600,
        f_type: int = 2,  # F_EXP_APPROX
        exp_temperature: int = 100,
        # Episodic
        max_episodes: int = 10000,
        episodic_scale: int = 100,
        # Energy scales
        grammar_penalty_scale: int = 60,
        same_word_penalty: int = 800,
        # Generation
        beta: float = 0.01,
        max_seq_len: int = 30,
        # VSA Binding (v39)
        bind_window: int = 8,
        bind_weight: int = 30,
        n_unbind_words: int = 3,
        bind_density: int = 0,  # 0 = auto (2*k=20)
        # Bigram DAM (v49)
        bigram_weight: int = 0,  # 0 = disabled; v49 default in train.py is 8
        # Skip bigram (v51)
        skip_weight: int = 0,  # 0 = disabled; weight for J2[words[-2], c]
        # POS skeleton (v51)
        pos_weight: int = 0,  # 0 = disabled for PPL; weight for POS trigram transitions
        # Frequency penalty (v53)
        freq_penalty_weight: int = 0,  # 0 = disabled; weight for log2(freq+1) during generation
        # POS generation bonus (v53)
        pos_gen_weight: int = 10,  # POS skeleton energy bonus during generation (always active)
        # POS type pre-selection (v53)
        pos_type_top_k: int = 3,  # Number of top POS types to consider during generation
        # Bigram generation weight (v54)
        bigram_gen_weight: int = 0,  # 0 = same as bigram_weight; generation-only bigram boost
        # Skip bigram generation weight (v54)
        skip_gen_weight: int = 0,  # 0 = same as skip_weight; generation-only skip boost
        # Memory
        memory_budget_mb: int = 0,
        # Seeds
        seed: int = 42,
    ):
        self.vocab_min_freq = vocab_min_freq
        self.vocab_max_size = vocab_max_size
        self.sdr_dim = sdr_dim
        self.sdr_sparsity = sdr_sparsity
        self.dam_scale = dam_scale
        self.grammar_penalty_scale = grammar_penalty_scale
        self.same_word_penalty = same_word_penalty
        self.beta = beta
        self.max_seq_len = max_seq_len
        self.memory_budget_mb = memory_budget_mb
        self.seed = seed
        self._j_clip = j_clip
        self._uv_regularize = uv_regularize
        self._uv_lambda = uv_lambda
        self._topdown_scale = topdown_scale
        self._max_episodes = max_episodes
        self._episodic_scale = episodic_scale
        self._f_type = f_type
        self._exp_temperature = exp_temperature
        self._bind_window = bind_window
        self._bind_weight = bind_weight
        self._n_unbind_words = n_unbind_words
        self._bind_density = bind_density
        self._bigram_weight = bigram_weight
        self._skip_weight = skip_weight
        self._pos_weight = pos_weight
        self._freq_penalty_weight = freq_penalty_weight
        self._pos_gen_weight = pos_gen_weight
        self._pos_type_top_k = pos_type_top_k
        self._bigram_gen_weight = bigram_gen_weight
        self._skip_gen_weight = skip_gen_weight

        # Built during training
        self.vocab = None
        self.J2 = None  # v49: Bigram coupling matrix (V, V) int32, log-normalized
        self.J_pos_bi = None  # v51: POS bigram transition matrix (N_POS, N_POS) int32, log-normalized
        self.J_pos_tri = None  # v51: POS trigram transition matrix (N_POS, N_POS, N_POS) int32, log-normalized
        self.pos_system = None
        self.sdr_encoder: Optional[SDREncoder] = None
        self.hierarchy: Optional[HierarchicalDAM] = None
        self.episodic: Optional[EpisodicMemory] = None
        self.binding: Optional[BindingContext] = None

        self.sequences = None
        self.test_sequences = None
        self._word_freq = None

        self.type_words: Dict[int, List[int]] = {}
        self._sampler = None

        self._stats = {
            'total_steps': 0,
            'dam_hits': 0,
            'episodic_hits': 0,
        }

    # ===================================================================
    # TRAINING PIPELINE
    # ===================================================================

    def train(self, n_samples: int = 500000, texts=None) -> "AttractorLanguageModel":
        """
        Full training pipeline for the Attractor Language Machine.

        Phases:
          1. Load corpus, build vocabulary, tokenize
          2. Build POS type system
          3. Build SDR encoder
          4. Build hierarchical DAM (F_EXP_APPROX by default)
          5. Batch Hebbian training (L0 only — RG fixed point)
          6. Compute coupling flow and apply J_eff to all levels
          7. Check UV completeness (Ward identities + cutoff independence)
          8. Build episodic memory
          9. Calibrate beta
        """
        from ..vocabulary import Vocabulary, POSTypeSystem
        from ..vocabulary.pos import POS2IDX, IDX2POS, N_POS, CLOSED_CLASS
        from ..utils import (
            primary_pos_tag, tokenize_texts, truncate_sequences,
            DATASET_LOADERS, DEFAULT_DATASET, get_rss_mb,
        )

        f_type_name = {0: 'quadratic', 1: 'cubic', 2: 'exp_approx'}.get(
            self._f_type, 'unknown'
        )

        print("=" * 70, flush=True)
        print("ATTRACTOR LANGUAGE MACHINE v54 — POS-DRIVEN GENERATION", flush=True)
        print(f"  F function: {f_type_name}, T={self._exp_temperature/100:.2f}", flush=True)
        print("  RG flow: J_eff[l] decimated (not layers[l].J), Kadanoff rescaling", flush=True)
        print("  Energy: NORMALIZED log2-F (LOG2_NORM=512, NO k division, NO h)", flush=True)
        print("  Binding: VSA permutation bind(a,hash(b)), kWTA sparsification", flush=True)
        print(f"  Bind window={self._bind_window}, weight={self._bind_weight}, n_unbind={self._n_unbind_words}, density={self._bind_density if self._bind_density > 0 else 'auto'}", flush=True)
        print("  M_bind: attractor dynamics ONLY (not DAM energy) — v45 reverted", flush=True)
        print(f"  Bigram DAM: J2 weight={self._bigram_weight}{' (disabled)' if self._bigram_weight == 0 else ''} (LOG-normalized)", flush=True)
        print(f"  Skip bigram: J2[words[-2]] weight={self._skip_weight}{' (disabled)' if self._skip_weight == 0 else ''} (v51)", flush=True)
        print(f"  POS skeleton: PPL weight={self._pos_weight}{' (disabled for PPL)' if self._pos_weight == 0 else ''}, gen bonus weight={self._pos_gen_weight}, type top-k={self._pos_type_top_k} (v53)", flush=True)
        print(f"  Frequency penalty: weight={self._freq_penalty_weight}{' (disabled)' if self._freq_penalty_weight == 0 else ''} (generation only, v53)", flush=True)
        print(f"  Bigram gen weight: {self._bigram_gen_weight}{' (=bigram_weight)' if self._bigram_gen_weight == 0 else ''} (v54: generation-only bigram boost)", flush=True)
        print(f"  Skip gen weight: {self._skip_gen_weight}{' (=skip_weight)' if self._skip_gen_weight == 0 else ''} (v54: generation-only skip boost)", flush=True)
        print("  Training: Positional VSA DAM + bigram J2 + skip J2 + POS-driven generation (v54)", flush=True)
        print("  UV checks: Ward identities + cutoff independence", flush=True)
        print("  Learning: Hebbian L0 only, PCD REMOVED", flush=True)
        print("=" * 70, flush=True)

        t0 = time.time()

        # Step 1: Load corpus
        if texts is None:
            print(f"\n[1/9] Loading corpus...")
            loader = DATASET_LOADERS[DEFAULT_DATASET]
            texts = loader(n_samples=n_samples)
        else:
            print(f"\n[1/9] Using provided texts ({len(texts):,})")

        # Step 2: Build vocabulary
        print("\n[2/9] Building vocabulary...")
        self.vocab = Vocabulary(
            min_freq=self.vocab_min_freq,
            max_size=self.vocab_max_size,
        )
        self.vocab.build(texts)
        V = len(self.vocab)
        print(f"  Vocabulary: {V} words")

        # Step 3: Tokenize
        print("\n[3/9] Tokenizing...")
        sequences = tokenize_texts(texts, self.vocab)
        sequences = truncate_sequences(sequences, max_len=self.max_seq_len)
        split_idx = int(len(sequences) * 0.9)
        self.sequences = sequences[:split_idx]
        self.test_sequences = sequences[split_idx:]
        print(f"  Train: {len(self.sequences):,}, Test: {len(self.test_sequences):,}")

        self._word_freq = np.zeros(V, dtype=np.int64)
        for seq in self.sequences:
            for w in seq:
                if w < V:
                    self._word_freq[w] += 1

        # Step 4: Build POS type system
        print("\n[4/9] Building POS type system...")
        self.pos_system = POSTypeSystem(vocab_size=V, window=5)
        self.pos_system.build_from_vocabulary(self.vocab.word2idx, self.vocab.idx2word)
        self.pos_system.build_grammar_penalties(penalty_strength=self.grammar_penalty_scale)
        self.pos_system.compute_type_couplings(self.sequences, self.vocab.idx2word)

        # v40: Filter special tokens (idx < 4: PAD, UNK, BOS, EOS) from candidate lists.
        # These should never be generated — they caused <UNK> spam in v39.
        self.type_words = {t: [] for t in range(N_POS)}
        for w, allowed in self.pos_system.allowed_types.items():
            if allowed and w >= 4:  # v40: skip special tokens
                primary = primary_pos_tag(allowed)
                self.type_words[primary].append(w)

        n_typed = sum(1 for w in range(V) if w in self.pos_system.allowed_types)
        print(f"  POS system: {N_POS} types, {n_typed} words typed")

        # Step 5: Build SDR encoder
        print(f"\n[5/9] Building SDR encoder (D={self.sdr_dim}, sparsity={self.sdr_sparsity})...")
        self.sdr_encoder = SDREncoder(
            vocab_size=V,
            D=self.sdr_dim,
            sparsity=self.sdr_sparsity,
            seed=self.seed,
        )
        self.sdr_encoder.build(word_freq=self._word_freq)

        # Step 6: Build hierarchical DAM
        print(f"\n[6/9] Building hierarchical DAM...")
        D0 = self.sdr_dim
        k0 = self.sdr_encoder.k
        sdr_sp = self.sdr_sparsity
        layers_config = []
        D = D0
        while D >= 32:
            k = max(2, int(D * sdr_sp))
            scale = max(200, int(1600 * D / D0))
            layers_config.append((D, k, scale))
            D = D // 2

        print(f"  Hierarchy layers: {[(D, k, s) for D, k, s in layers_config]}")
        print(f"  F function: {f_type_name}, T={self._exp_temperature/100:.2f}")

        self.hierarchy = HierarchicalDAM(
            layers_config=layers_config,
            j_clip=self._j_clip,
            uv_regularize=self._uv_regularize,
            uv_lambda=self._uv_lambda,
            topdown_scale=self._topdown_scale,
            f_type=self._f_type,
            exp_temperature=self._exp_temperature,
            seed=self.seed,
        )
        self.hierarchy.build(self.sdr_encoder)

        # Step 7: Batch Hebbian training (L0 only, then RG flow)
        print(f"\n[7/9] Batch Hebbian training (L0 only, then RG flow)...", flush=True)
        self._batch_hebbian_train()

        # Step 8: Check UV completeness
        print(f"\n[8/9] UV completeness check (Ward identities + cutoff independence)...")
        uv_results = self.hierarchy.check_uv_completeness()
        print(f"  UV completeness score: {uv_results['overall_uv_score']:.2f}")
        print(f"  RG applied: {uv_results['rg_applied']}")
        print(f"  Flow consistent: {uv_results['flow_consistent']}")
        for l in range(self.hierarchy.n_layers):
            l_uv = uv_results[f'L{l}']
            print(f"  L{l}: cutoff_sensitivity={l_uv['cutoff_sensitivity']:.3f}, "
                  f"ward_violation={l_uv['ward_violation']:.4f}, "
                  f"relevant={l_uv['n_relevant']}, irrelevant={l_uv['n_irrelevant']}")

        # Step 9: Build episodic memory + binding context + calibrate beta
        print(f"\n[9/9] Building episodic memory, binding context, and calibrating beta...")
        self.episodic = EpisodicMemory(
            D=self.sdr_dim,
            k=self.sdr_encoder.k,
            max_episodes=self._max_episodes,
            field_scale=self._episodic_scale,
            seed=self.seed,
        )
        self._populate_episodic_memory()

        # v39: Build VSA binding context with multi-step unbinding
        # v47: Binding with reverted params
        bind_density_arg = self._bind_density if self._bind_density > 0 else 0  # 0 = auto
        self.binding = BindingContext(
            D=self.sdr_dim,
            k=self.sdr_encoder.k,
            window=self._bind_window,
            bind_weight=self._bind_weight,
            n_unbind_words=self._n_unbind_words,
            target_density=bind_density_arg,
        )
        actual_density = self.binding.target_density
        print(f"    Binding context: D={self.sdr_dim}, k={self.sdr_encoder.k}, "
              f"window={self._bind_window}, weight={self._bind_weight}, "
              f"n_unbind={self._n_unbind_words}, density={actual_density}")

        # v48: Build bigram coupling matrix J2
        if self._bigram_weight > 0:
            self._build_bigram_j2()

        # v53: Always build POS skeleton (needed for constrained decoding during generation)
        self._build_pos_skeleton()

        self._calibrate_beta()

        # v42: Reset binding context after calibration — avoids showing
        # calibration residue in diagnostics (deque had only 1-2 entries)
        if self.binding:
            self.binding.reset()

        from ..sampling import IntegerBoltzmannSampler
        self._sampler = IntegerBoltzmannSampler(
            beta=self.beta, max_delta=50000  # v49: increased from 5000 for bigger dE range
        )

        t_total = time.time() - t0
        rss = get_rss_mb()
        print(f"\nTraining complete: {t_total:.1f}s")
        print(f"  Vocab: {V} words")
        if rss > 0:
            print(f"  Memory (RSS): {rss:,} MB")
        print(f"  Integer-only: YES — ZERO float operations in hot path")
        print(f"  Architecture: Dense Associative Memory (DAM) Engine v54")
        print(f"  F function: {f_type_name}, T={self._exp_temperature/100:.2f}")
        print(f"  Learning: Hebbian (L0 only, RG flow to higher levels)")
        print(f"  Energy: NORMALIZED log2-F ({f_type_name}, LOG2_NORM=512, NO k div, NO h)")
        print(f"  Binding: VSA permutation (window={self._bind_window}, weight={self._bind_weight}, n_unbind={self._n_unbind_words}, density={self._bind_density if self._bind_density > 0 else 'auto'})")
        print(f"  Repetition: penalty={self.same_word_penalty}, window=15, distance-decay")
        print(f"  Generation: top-k=10 + Boltzmann + POS-driven + bigram-dominant (v54)")
        print(f"  Bigram DAM: J2 weight={self._bigram_weight}{' (disabled)' if self._bigram_weight == 0 else ''} (LOG-normalized)")
        print(f"  Skip bigram: weight={self._skip_weight}{' (disabled)' if self._skip_weight == 0 else ''} (v51)")
        print(f"  POS skeleton: PPL weight={self._pos_weight}{' (disabled for PPL)' if self._pos_weight == 0 else ''}, gen bonus weight={self._pos_gen_weight} (v53)")
        print(f"  Frequency penalty: weight={self._freq_penalty_weight}{' (disabled)' if self._freq_penalty_weight == 0 else ''} (generation only, v53)")
        print(f"  Bigram gen weight: {self._bigram_gen_weight}{' (=bigram_weight)' if self._bigram_gen_weight == 0 else ''} (v54)")
        print(f"  Skip gen weight: {self._skip_gen_weight}{' (=skip_weight)' if self._skip_gen_weight == 0 else ''} (v54)")

        self._print_diagnostics()

        return self

    def _batch_hebbian_train(self) -> None:
        """
        Phase 1 training: batch Hebbian storage (L0 ONLY).

        DEEP FIX: Only L0 is trained. Higher levels get J from RG flow.
        This ensures RG consistency: J at every level is derivable from L0.

        v46: REVERTED v45 order-sensitive training. The v45 approach trained
        the DAM on binding-augmented context SDRs, which caused PPL regression
        from 221 to 1909. The binding bits contaminated the J matrix because:
        1) Same BOW context produces different SDRs depending on word order
        2) With k=10 active bits, the DAM can't represent all variations
        3) Inconsistent training signals pollute the energy landscape
        The DAM captures co-occurrence (BOW), binding captures order (runtime).

        OPTIMIZED: Uses vectorized batch SDR encoding instead of per-pair
        Python loops. ~50-100x faster on the encoding step.
        """
        V = len(self.vocab)
        context_window = 10
        total_pairs = 0
        n_seqs = len(self.sequences)

        # Adaptive batch size: D=4096 with 50K pairs = 4096*50K*4bytes = 800MB per array
        # Reduce batch size for large D to keep memory manageable
        if self.sdr_dim >= 4096:
            hebbian_batch = 2000  # ~32MB per array at D=4096
        elif self.sdr_dim >= 2048:
            hebbian_batch = 5000
        else:
            hebbian_batch = 50000

        print(f"    Vectorized Hebbian training over {n_seqs:,} sequences...", flush=True)
        print(f"    Encoding batch size: {hebbian_batch} (adaptive for D={self.sdr_dim})",
              flush=True)
        print(f"    v52: Using POSITIONAL VSA context encoding (replaces BOW)", flush=True)

        def progress_callback(seq_idx, total):
            print(f"      Hebbian encoding: {seq_idx:,} seqs, {total:,} pairs encoded",
                  flush=True)

        # v52: POSITIONAL VSA context encoding.
        # Each word's SDR is rotated by its relative position hash before
        # superposition, preserving word ORDER in the context SDR.
        # The DAM learns position-dependent patterns, not just co-occurrence.
        for ctx_arr, tgt_arr in self.sdr_encoder.encode_contexts_batch_positional(
            self.sequences,
            context_window=context_window,
            batch_size=hebbian_batch,
            callback=progress_callback,
        ):
            batch_n = ctx_arr.shape[0]
            total_pairs += batch_n
            t_batch = time.time()
            # defer_rg=True: skip RG flow per batch — compute once at the end
            self.hierarchy.train_l0_hebbian(ctx_arr, tgt_arr, eta=1, defer_rg=True)
            t_batch = time.time() - t_batch
            print(f"      Hebbian batch: {batch_n:,} pairs stored "
                  f"(total: {total_pairs:,}) [{t_batch:.1f}s]", flush=True)

        # Compute RG flow ONCE after all Hebbian batches
        print(f"    Computing RG flow from L0 to all higher levels...", flush=True)
        t_rg = time.time()
        self.hierarchy.finalize_rg_flow()
        print(f"    RG flow computed in {time.time()-t_rg:.1f}s", flush=True)

        print(f"    Hebbian training complete: {total_pairs:,} pairs stored")
        print(f"    RG flow applied to all higher levels: {self.hierarchy._rg_applied}")

        # v34: Enhanced RG flow diagnostics
        for l, layer in enumerate(self.hierarchy.layers):
            diag = layer.get_diagnostics()
            source = "Hebbian (L0)" if l == 0 else "RG flow"
            j_eff_info = ""
            if l > 0 and self.hierarchy.J_eff[l] is not None:
                je = self.hierarchy.J_eff[l]
                j_eff_info = f", J_eff_max={int(np.max(np.abs(je)))}, J_eff_nnz={int(np.sum(je != 0))}"
            print(f"    L{l} [{source}]: J_max={diag['J_max']}, J_nnz={diag['J_nnz']}, "
                  f"h_max={diag['h_max']}, h_nnz={diag['h_nnz']}{j_eff_info}")

        # Print RG beta functions (should be ~1.0 if rescaling works)
        for l in range(self.hierarchy.n_layers - 1):
            if self.hierarchy.rg_beta[l] is not None:
                print(f"    RG beta L{l}->L{l+1}: {self.hierarchy.rg_beta[l]:.4f}")

    def _build_bigram_j2(self) -> None:
        """v49: Build log-normalized bigram coupling matrix J2.

        J2[prev_word, candidate_word] = log2(count+1) of times candidate follows prev.
        This compresses the dynamic range from [0, ~87570] to [0, ~16],
        making bigram energy comparable to DAM dE instead of dominating it.

        v48 BUG: Raw counts gave max_count=87570 × weight=5 = 437850 energy,
        completely dominating dE (51768 vs DAM's 122). This caused generation
        loops: "there was a time" repeated forever because the bigram chain
        was unbreakable. Repetition penalty of 800 was invisible on dE=51768.

        v49 FIX: log2(count+1) compression:
          - count=1 → log2(2)=1, count=10→3.5, count=100→6.7,
            count=1000→10, count=87570→16.4
          - Range [0,16] × weight=8 = max bigram energy ~128
          - Comparable to DAM dE=122 → bigram is significant but not dominant
          - Repetition penalty 800 is now 4x dE → effective anti-loop

        Memory: V*V * 4 bytes = 2005*2005*4 ≈ 16 MB (same as v48).
        """
        V = len(self.vocab)
        J2_raw = np.zeros((V, V), dtype=np.int64)

        n_bigrams = 0
        for seq in self.sequences:
            for i in range(len(seq) - 1):
                prev_w = seq[i]
                next_w = seq[i + 1]
                if 0 <= prev_w < V and 0 <= next_w < V:
                    J2_raw[prev_w, next_w] += 1
                    n_bigrams += 1

        # v49: Log-normalize counts → compress dynamic range
        # log2(count+1): count=0→0, count=1→1, count=87570→16.4
        J2_log = np.log2(J2_raw.astype(np.float64) + 1.0).astype(np.int32)

        self.J2 = J2_log
        self._j2_raw_max = int(np.max(J2_raw))  # Store for diagnostics
        j2_nnz = int(np.sum(J2_log > 0))
        j2_log_max = int(np.max(J2_log))
        j2_mem_mb = J2_log.nbytes / (1024 * 1024)
        print(f"    Bigram J2 (LOG-normalized): {n_bigrams:,} bigrams, "
              f"{j2_nnz:,} non-zero entries, raw_max={self._j2_raw_max}, "
              f"log_max={j2_log_max}, weight={self._bigram_weight}, "
              f"skip_weight={self._skip_weight}, memory={j2_mem_mb:.1f} MB")

    def _compute_bigram_energy(
        self,
        prev_word: int,
        candidate_words: np.ndarray,
        weight: int = 0,  # v54: 0 = use self._bigram_weight
    ) -> np.ndarray:
        """v49: Compute bigram energy bonus for each candidate word.

        E_bigram(c) = -J2[prev_word, c] * weight

        v49: J2 now stores log2(count+1), not raw count.
        Range [0, ~16] × weight → max bigram energy ~128 (comparable to DAM dE=122).

        v54: Added optional weight parameter for generation-time boost.
        bigram_gen_weight=64 → max energy = 16*64 = 1024 >> DAM dE~407,
        making bigram the DOMINANT signal during generation.

        Higher log-count → more negative energy → more likely.
        Zero count → zero bonus (no penalty, just no help).

        This is O(n_candidates) — just a lookup per candidate.
        Much cheaper than VSA unbinding which requires rotations + overlaps.
        """
        if self.J2 is None or prev_word < 0 or prev_word >= self.J2.shape[0]:
            return np.zeros(len(candidate_words), dtype=np.int64)

        w = weight if weight > 0 else self._bigram_weight

        # Vectorized lookup: J2[prev_word, candidates]
        valid_mask = (candidate_words >= 0) & (candidate_words < self.J2.shape[1])
        energies = np.zeros(len(candidate_words), dtype=np.int64)

        if np.any(valid_mask):
            bigram_log_counts = self.J2[prev_word, candidate_words[valid_mask]]
            energies[valid_mask] = -(bigram_log_counts.astype(np.int64) * w)

        return energies

    def _build_pos_skeleton(self) -> None:
        """v51: Build POS bigram/trigram transition matrices for syntactic backbone.

        The POS skeleton captures the SYNTACTIC STRUCTURE of language:
        - DET → ADJ/NOUN is highly likely (sentence-initial pattern)
        - NOUN → VERB is highly likely (subject-verb)
        - VERB → DET/PREP/ADV is highly likely (post-verbal)
        - etc.

        This is the "missing layer" between the grammar system (which only
        penalizes INVALID transitions) and the word-level J2 (which has no
        notion of syntactic category). J_pos provides a BONUS for likely
        POS transitions, guiding the model toward syntactically coherent
        type selection.

        Matrices:
          J_pos_bi[prev_type, next_type]: 13×13 int32, log2(count+1)
          J_pos_tri[prev2_type, prev1_type, next_type]: 13×13×13 int32, log2(count+1)

        Memory: 169 + 2197 = 2366 entries ≈ 9.4 KB (trivial).

        The trigram backs off to bigram when count is 0.
        """
        from ..vocabulary.pos import POS2IDX, N_POS
        from ..utils import primary_pos_tag

        # Count POS transitions from training sequences
        pos_bi_counts = np.zeros((N_POS, N_POS), dtype=np.int64)
        pos_tri_counts = np.zeros((N_POS, N_POS, N_POS), dtype=np.int64)

        for seq in self.sequences:
            # Get POS type sequence
            types_seq = []
            for w in seq:
                allowed = self.pos_system.allowed_types.get(w, set())
                t = primary_pos_tag(allowed) if allowed else POS2IDX["X"]
                types_seq.append(t)

            # Count bigram transitions
            for i in range(len(types_seq) - 1):
                t1 = types_seq[i]
                t2 = types_seq[i + 1]
                pos_bi_counts[t1, t2] += 1

            # Count trigram transitions
            for i in range(len(types_seq) - 2):
                t1 = types_seq[i]
                t2 = types_seq[i + 1]
                t3 = types_seq[i + 2]
                pos_tri_counts[t1, t2, t3] += 1

        # Log-normalize (same as J2): log2(count+1)
        self.J_pos_bi = np.log2(pos_bi_counts.astype(np.float64) + 1.0).astype(np.int32)
        self.J_pos_tri = np.log2(pos_tri_counts.astype(np.float64) + 1.0).astype(np.int32)

        bi_log_max = int(np.max(self.J_pos_bi))
        tri_log_max = int(np.max(self.J_pos_tri))
        bi_nnz = int(np.sum(self.J_pos_bi > 0))
        tri_nnz = int(np.sum(self.J_pos_tri > 0))

        print(f"    POS skeleton (v51): bigram {bi_nnz} nnz (log_max={bi_log_max}), "
              f"trigram {tri_nnz} nnz (log_max={tri_log_max}), "
              f"weight={self._pos_weight}, "
              f"max_energy={bi_log_max * self._pos_weight}")

    def _compute_pos_energy(
        self,
        types_history: List[int],
        candidate_types: np.ndarray,
    ) -> np.ndarray:
        """v51: Compute POS transition energy bonus for each candidate word.

        Uses POS trigram (if available) backed off to bigram.
        E_pos(c) = -J_pos[prev_types, type_of(c)] * pos_weight

        This gives ALL words of a syntactically-favored POS type a bonus.
        For example, after DET ADJ, NOUN gets a big bonus; after VERB, PREP/DET get bonus.

        Vectorized: uses fancy indexing instead of Python loops.
        """
        if self.J_pos_bi is None:
            return np.zeros(len(candidate_types), dtype=np.int64)

        N_POS = self.J_pos_bi.shape[0]
        n = len(candidate_types)
        energies = np.zeros(n, dtype=np.int64)

        # Clip candidate types to valid range
        valid = (candidate_types >= 0) & (candidate_types < N_POS)

        # Get recent POS types
        if len(types_history) >= 2 and self.J_pos_tri is not None:
            t_prev2 = types_history[-2]
            t_prev1 = types_history[-1]
            if 0 <= t_prev2 < N_POS and 0 <= t_prev1 < N_POS:
                # Vectorized trigram lookup
                tri_vals = np.zeros(n, dtype=np.int32)
                bi_vals = np.zeros(n, dtype=np.int32)
                tri_vals[valid] = self.J_pos_tri[t_prev2, t_prev1, candidate_types[valid]]
                bi_vals[valid] = self.J_pos_bi[t_prev1, candidate_types[valid]]
                # Use trigram where available, bigram backoff elsewhere
                use_tri = tri_vals > 0
                use_bi = (~use_tri) & (bi_vals > 0)
                energies[use_tri] = -(tri_vals[use_tri].astype(np.int64) * self._pos_weight)
                energies[use_bi] = -(bi_vals[use_bi].astype(np.int64) * self._pos_weight * 3 // 4)  # 75% backoff
        elif len(types_history) >= 1:
            t_prev1 = types_history[-1]
            if 0 <= t_prev1 < N_POS:
                # Vectorized bigram lookup
                bi_vals = np.zeros(n, dtype=np.int32)
                bi_vals[valid] = self.J_pos_bi[t_prev1, candidate_types[valid]]
                nonzero = bi_vals > 0
                energies[nonzero] = -(bi_vals[nonzero].astype(np.int64) * self._pos_weight)
        # else: no history, no POS bonus

        return energies

    def _compute_pos_energy_raw(
        self,
        types_history: List[int],
        candidate_types: np.ndarray,
    ) -> np.ndarray:
        """v53: Compute raw POS transition energy (weight=1) for generation bonus.

        Unlike _compute_pos_energy which uses self._pos_weight (0 for PPL),
        this returns the raw log2(count+1) values scaled by weight=1.
        The caller then scales by _pos_gen_weight for generation-only bonus.
        """
        if self.J_pos_bi is None:
            return np.zeros(len(candidate_types), dtype=np.int64)

        N_POS = self.J_pos_bi.shape[0]
        n = len(candidate_types)
        energies = np.zeros(n, dtype=np.int64)

        # Clip candidate types to valid range
        valid = (candidate_types >= 0) & (candidate_types < N_POS)

        if len(types_history) >= 2 and self.J_pos_tri is not None:
            t_prev2 = types_history[-2]
            t_prev1 = types_history[-1]
            if 0 <= t_prev2 < N_POS and 0 <= t_prev1 < N_POS:
                tri_vals = np.zeros(n, dtype=np.int32)
                bi_vals = np.zeros(n, dtype=np.int32)
                tri_vals[valid] = self.J_pos_tri[t_prev2, t_prev1, candidate_types[valid]]
                bi_vals[valid] = self.J_pos_bi[t_prev1, candidate_types[valid]]
                use_tri = tri_vals > 0
                use_bi = (~use_tri) & (bi_vals > 0)
                energies[use_tri] = -(tri_vals[use_tri].astype(np.int64))
                energies[use_bi] = -(bi_vals[use_bi].astype(np.int64) * 3 // 4)
        elif len(types_history) >= 1:
            t_prev1 = types_history[-1]
            if 0 <= t_prev1 < N_POS:
                bi_vals = np.zeros(n, dtype=np.int32)
                bi_vals[valid] = self.J_pos_bi[t_prev1, candidate_types[valid]]
                nonzero = bi_vals > 0
                energies[nonzero] = -(bi_vals[nonzero].astype(np.int64))

        return energies

    def _compute_skip_energy(
        self,
        skip_word: int,
        candidate_words: np.ndarray,
        weight: int = 0,  # v54: 0 = use self._skip_weight
    ) -> np.ndarray:
        """v51: Compute skip bigram energy — J2[words[-2], c] at reduced weight.

        The skip bigram captures patterns like:
        - "the ___ girl" → what follows after a 1-word gap?
        - "a little ___" → what follows "a" with "little" in between?

        This provides a 2-word context window using only the existing J2 matrix.
        Weight is typically bigram_weight // 3 or // 4.

        v54: Added optional weight parameter for generation-time boost.
        """
        w = weight if weight > 0 else self._skip_weight
        if self.J2 is None or w <= 0:
            return np.zeros(len(candidate_words), dtype=np.int64)

        if skip_word < 0 or skip_word >= self.J2.shape[0]:
            return np.zeros(len(candidate_words), dtype=np.int64)

        valid_mask = (candidate_words >= 0) & (candidate_words < self.J2.shape[1])
        energies = np.zeros(len(candidate_words), dtype=np.int64)

        if np.any(valid_mask):
            skip_log_counts = self.J2[skip_word, candidate_words[valid_mask]]
            energies[valid_mask] = -(skip_log_counts.astype(np.int64) * w)

        return energies

    def _populate_episodic_memory(self) -> None:
        """Pre-populate episodic memory from training sequences."""
        V = len(self.vocab)
        context_window = 10

        n_stored = 0
        for seq_idx, seq in enumerate(self.sequences[:50000]):
            if len(seq) < 5:
                continue

            for pos in range(5, len(seq), 3):
                context_words = seq[max(0, pos - context_window):pos]
                context_sdr = self.sdr_encoder.encode_context_positional(context_words, context_window)
                if np.sum(context_sdr) > 0:
                    self.episodic.store(context_sdr)
                    n_stored += 1

            if (seq_idx + 1) % 10000 == 0:
                print(f"      Episodic memory: {seq_idx+1} seqs, {n_stored} episodes stored")

        print(f"    Episodic memory: {n_stored} episodes stored")

    def _calibrate_beta(self) -> None:
        """Calibrate Boltzmann beta from the normalized DAM energy distribution.

        v39: LOG2_NORM=512 gives dE ~ O(200-300), beta ~ 0.01.
        Includes VSA binding energy in the total energy distribution
        so that beta accounts for the binding bonus.

        v37 FIX: Three changes from v36:
        1. Removed h from word energy — h is frequency bias, not context signal.
        2. Removed k from energy divisor — k=10 always, dividing by k just
           loses 10x precision to integer truncation.
        3. Episodic scale reduced from 500→100.

        Target: beta * p10_dE ≈ 3.0 (stronger selectivity than v36's 2.0).
        The p10 candidate gets exp(-3) ≈ 5% of the best's probability.

        Also stores _median_de for auto-scaling penalties.
        """
        from ..utils import primary_pos_tag

        energy_diffs = []
        n_samples = 0

        # v38: Reset binding context for calibration
        if self.binding:
            self.binding.reset()

        for seq in self.test_sequences[:100]:
            if len(seq) < 3:
                continue

            # v38: Reset binding context for each sequence
            if self.binding:
                self.binding.reset()

            for pos in range(1, min(len(seq), 5)):
                target_word = seq[pos]
                context_words = seq[:pos]

                # v39: Update binding context with the previous word
                if self.binding and pos > 0:
                    prev_word = seq[pos - 1]
                    if 0 <= prev_word < self.sdr_encoder.vocab_size:
                        self.binding.add_word(self.sdr_encoder.word_active_bits[prev_word])

                word_type = primary_pos_tag(
                    self.pos_system.allowed_types.get(target_word, set())
                )
                candidates = self.type_words.get(word_type, [])
                if len(candidates) < 5:
                    continue

                candidate_arr = np.array(candidates[:200], dtype=np.int64)
                context_sdr = self.sdr_encoder.encode_context_positional(context_words)

                # v39: Do NOT OR M_bind into context_sdr for DAM energy.
                # The DAM was trained on standard context SDRs — binding bits
                # add noise to coupling energy. M_bind is only used for
                # the binding energy bonus (separate signal).

                # Compute energies using normalized log2-F (v39: LOG2_NORM=512)
                energies = self.hierarchy.compute_word_energies(
                    context_sdr, candidate_arr, self.sdr_encoder, self.dam_scale
                )

                # Add episodic energy (normalized scale)
                if self.episodic and len(self.episodic.episodes) > 0:
                    ep_energy = self.episodic.compute_word_episodic_energy(
                        candidate_arr, self.sdr_encoder, self.episodic.field_scale
                    )
                    energies += ep_energy

                # v39: Add binding energy bonus (multi-step unbinding)
                if self.binding and len(self.binding._recent_words) > 0:
                    bind_energy = self.binding.compute_binding_energy(
                        candidate_arr, self.sdr_encoder
                    )
                    energies += bind_energy

                # v48: Add bigram energy from J2
                if self.J2 is not None and pos > 0:
                    prev_word = seq[pos - 1]
                    bigram_energy = self._compute_bigram_energy(prev_word, candidate_arr)
                    energies += bigram_energy

                # v51: Add skip bigram energy
                if self.J2 is not None and self._skip_weight > 0 and pos >= 2:
                    skip_word = seq[pos - 2]
                    skip_energy = self._compute_skip_energy(skip_word, candidate_arr)
                    energies += skip_energy

                # v51: Add POS skeleton energy
                if self.J_pos_bi is not None and self._pos_weight > 0:
                    type_hist = [self._get_word_type(w) for w in seq[:pos]]
                    candidate_types = np.array([
                        self._get_word_type(int(w)) for w in candidate_arr
                    ], dtype=np.int64)
                    pos_energy = self._compute_pos_energy(type_hist, candidate_types)
                    energies += pos_energy

                e_min = energies.min()
                diffs = energies - e_min
                diffs = diffs[diffs > 0]

                if len(diffs) > 0:
                    median_diff = int(np.median(diffs))
                    if median_diff > 0:
                        energy_diffs.append(median_diff)

                n_samples += 1
                if n_samples >= 300:
                    break
            if n_samples >= 300:
                break

        if energy_diffs:
            median_de = int(np.median(energy_diffs))
            p10_de = int(np.percentile(energy_diffs, 10))

            # v37: Beta calibration for NORMALIZED energy scale.
            # With h removed and k removed from divisor, DAM dE ~ O(20-40).
            # Target: beta * p10_dE ≈ 3.0 for strong selectivity + diversity.
            # v36 used 2.0 but with dE dominated by episodic frequency bias.
            # With DAM-dominated dE, 3.0 gives better context discrimination.
            empirical_beta = 3.0 / max(1, p10_de)

            # Clamp to reasonable range: [0.01, 5.0]
            # v36 clamped to [0.05, 5.0] which forced beta too high when
            # dE was large. With dE now dominated by DAM coupling signal,
            # beta should be free to go lower.
            self.beta = max(0.01, min(5.0, empirical_beta))

            # v36: Store energy scale for auto-scaling penalties
            self._median_de = median_de
            self._p10_de = p10_de

            print(f"    Median dE: {median_de}, p10 dE: {p10_de}")
            print(f"    Empirical beta: {empirical_beta:.4f}")
            print(f"    Using beta: {self.beta:.4f}")
            print(f"    Expected selectivity: exp(-beta*p10_dE) = {math.exp(-self.beta * p10_de):.4f}")
        else:
            self.beta = 1.0
            self._median_de = 10
            self._p10_de = 5
            print(f"    No energy diffs — using default beta: {self.beta:.4f}")

    # ===================================================================
    # GENERATION
    # ===================================================================

    def generate(self, prompt: str = "the", length: int = 20) -> Dict:
        """
        Generate text autoregressively using DAM attractor dynamics.

        v39: Uses VSA binding context for order-sensitive composition.
        The binding context encodes bigram order and provides an
        expectation signal that goes beyond the DAM's co-occurrence
        statistics. M_bind is used for attractor dynamics but NOT
        for DAM energy computation (the DAM was trained without it).

        Uses F-lookup nonlinearity (exp_approx) for energy computation.
        """
        from ..vocabulary.pos import POS2IDX, IDX2POS, N_POS, CLOSED_CLASS
        from ..utils import primary_pos_tag

        if self._sampler is None:
            from ..sampling import IntegerBoltzmannSampler
            self._sampler = IntegerBoltzmannSampler(beta=self.beta, max_delta=50000)

        prompt_words = prompt.strip().split()
        prompt_tokens = []
        for w in prompt_words:
            idx = self.vocab.word2idx.get(w)
            if idx is None:
                idx = self.vocab.word2idx.get(w.lower())
            if idx is not None and idx >= 4:
                prompt_tokens.append(idx)
        if not prompt_tokens:
            idx = self.vocab.word2idx.get("the", 4)
            prompt_tokens = [idx]

        words = list(prompt_tokens)
        types_list = [self._get_word_type(w) for w in words]
        diagnostics = []

        self.hierarchy.reset()
        self.episodic.reset()
        if self.binding:
            self.binding.reset()

        # Initialize layer states and binding context from prompt
        for i, w in enumerate(words):
            context_sdr = self.sdr_encoder.encode_context_positional(words[:i+1], 10)

            # v46: M_bind for attractor dynamics ONLY (not DAM energy)
            # The DAM was trained on BOW-only contexts — injecting binding
            # bits into DAM energy adds noise (v45 proved this: PPL 1909).
            # M_bind shapes attractor basins through the context field,
            # and binding bonus is a separate energy term.
            if self.binding and np.sum(self.binding.M_bind) > 0:
                context_sdr_for_dynamics = self.binding.get_context_or(context_sdr)
            else:
                context_sdr_for_dynamics = context_sdr

            context_field = np.zeros(self.sdr_dim, dtype=np.int32)
            active = np.where(context_sdr_for_dynamics > 0)[0]
            context_field[active] = self.dam_scale

            self.hierarchy.step_all(context_field, n_sweeps=1)
            self.episodic.store(context_sdr)

            # v39: Update binding context with this word
            if self.binding and 0 <= w < self.sdr_encoder.vocab_size:
                self.binding.add_word(self.sdr_encoder.word_active_bits[w])

        for pos in range(len(words), length):
            # ═══════════════════════════════════════════════════════════
            # v54: HARD POS TYPE SELECTION — POS trigram picks ONE type
            # ═══════════════════════════════════════════════════════════
            # v53 BUG: The multi-type energy comparison allowed the DAM to
            # override syntactically correct type sequences. For example,
            # after "there was", DET should win (for "a") but VERB's hub
            # word "keep" had lower energy, so VERB was selected instead.
            #
            # v54 FIX: The POS trigram model selects EXACTLY ONE type.
            # The DAM only chooses WHICH word of that type. This ensures
            # syntactically coherent POS sequences (DET→ADJ→NOUN, etc.)
            # and eliminates the type-selection error cascade.
            ranked_types = self._select_pos_types(types_list)
            chosen_type = None
            for t in ranked_types:
                if self.type_words.get(t, []):
                    chosen_type = t
                    break
            if chosen_type is None:
                # Emergency fallback: pick any type with candidates
                for t in range(N_POS):
                    if self.type_words.get(t, []):
                        chosen_type = t
                        break
            if chosen_type is None:
                break  # No candidates at all — stop generation

            # Get candidates of the chosen type
            candidate_list = self.type_words[chosen_type]
            candidate_arr = np.array(candidate_list[:300], dtype=np.int64)

            # ═══════════════════════════════════════════════════════════
            # v54: BIGRAM PRE-FILTERING — rank by bigram, take top-K
            # ═══════════════════════════════════════════════════════════
            # Before computing expensive DAM energies, pre-filter candidates
            # by bigram score. This ensures the model follows bigram chains
            # faithfully and prevents the DAM from selecting implausible words
            # that have low DAM energy but terrible bigram scores.
            #
            # With bigram_gen_weight=64, the bigram signal dominates DAM
            # (max bigram energy = 16*64 = 1024 >> DAM dE~407), so
            # pre-filtering by bigram is a good approximation to the final
            # ranking and prevents wasting computation on low-bigram words.
            gen_bigram_weight = self._bigram_gen_weight if self._bigram_gen_weight > 0 else self._bigram_weight
            gen_skip_weight = self._skip_gen_weight if self._skip_gen_weight > 0 else self._skip_weight

            if self.J2 is not None and len(words) > 0 and gen_bigram_weight > 0:
                prev_word = words[-1]
                valid_cand_mask = (candidate_arr >= 0) & (candidate_arr < self.J2.shape[1])
                bigram_scores = np.full(len(candidate_arr), -1, dtype=np.int32)
                bigram_scores[valid_cand_mask] = self.J2[prev_word, candidate_arr[valid_cand_mask]]
                # Take top-50 by bigram score (highest J2 = most likely bigram)
                n_pre_filter = min(50, len(candidate_arr))
                if n_pre_filter < len(candidate_arr):
                    top_indices = np.argpartition(bigram_scores, -n_pre_filter)[-n_pre_filter:]
                    candidate_arr = candidate_arr[top_indices]

            # ═══════════════════════════════════════════════════════════
            # CONTEXT ENCODING + ATTRACTOR DYNAMICS
            # ═══════════════════════════════════════════════════════════
            context_sdr = self.sdr_encoder.encode_context_positional(words, 10)

            # M_bind for attractor dynamics ONLY, not DAM energy.
            if self.binding and np.sum(self.binding.M_bind) > 0:
                context_sdr_for_dynamics = self.binding.get_context_or(context_sdr)
            else:
                context_sdr_for_dynamics = context_sdr

            context_field = np.zeros(self.sdr_dim, dtype=np.int32)
            active = np.where(context_sdr_for_dynamics > 0)[0]
            context_field[active] = self.dam_scale

            # Run attractor dynamics
            self.hierarchy.step_all(context_field, n_sweeps=2)

            # ═══════════════════════════════════════════════════════════
            # COMPUTE ENERGIES (single type — no multi-type loop)
            # ═══════════════════════════════════════════════════════════

            # DAM energy with NORMALIZED log2-F
            dam_energies = self.hierarchy.compute_word_energies(
                context_sdr, candidate_arr, self.sdr_encoder, self.dam_scale
            )

            # Episodic energy
            ep_energies = self.episodic.compute_word_episodic_energy(
                candidate_arr, self.sdr_encoder, self.episodic.field_scale
            )

            # Grammar penalty
            test_types = list(types_list[-5:]) + [chosen_type]
            test_pos = len(test_types) - 1
            try:
                grammar_penalty = self.pos_system.compute_grammar_penalty(
                    test_types, test_pos, chosen_type
                )
            except (IndexError, ValueError):
                grammar_penalty = 0
            median_de = max(1, getattr(self, '_median_de', 10))
            grammar_scaled = grammar_penalty * (median_de // 2) // 60
            grammar_energies = np.full(len(candidate_arr), grammar_scaled, dtype=np.int64)

            total_energies = dam_energies + ep_energies + grammar_energies

            # Binding energy bonus
            if self.binding and len(self.binding._recent_words) > 0:
                bind_energy = self.binding.compute_binding_energy(
                    candidate_arr, self.sdr_encoder
                )
                total_energies += bind_energy

            # v54: Bigram energy with GENERATION weight (DOMINANT signal).
            # bigram_gen_weight=64 → max bigram energy = 16*64 = 1024 >> DAM dE~407.
            # The bigram model only depends on the previous word, which is
            # always correct — making it far more reliable than the DAM during
            # generation when context is self-generated and may contain errors.
            if self.J2 is not None and len(words) > 0:
                prev_word = words[-1]
                bigram_energy = self._compute_bigram_energy(
                    prev_word, candidate_arr, weight=gen_bigram_weight
                )
                total_energies += bigram_energy

            # v54: Skip bigram with GENERATION weight.
            # skip_gen_weight=24 → max skip energy = 16*24 = 384 ≈ DAM dE.
            # Provides 2-word context (prev2, prev1) for better selectivity.
            if self.J2 is not None and gen_skip_weight > 0 and len(words) >= 2:
                skip_word = words[-2]
                skip_energy = self._compute_skip_energy(
                    skip_word, candidate_arr, weight=gen_skip_weight
                )
                total_energies += skip_energy

            # Repetition penalty (v40: distance-based decay)
            rep_window = 15
            median_de = max(1, getattr(self, '_median_de', 10))
            rep_base = max(self.same_word_penalty, median_de * 4)
            recent_words = words[-rep_window:]
            for i, w in enumerate(candidate_arr):
                w_int = int(w)
                for d, rw in enumerate(reversed(recent_words)):
                    if w_int == rw:
                        decay = max(1, rep_window - d)
                        total_energies[i] += (rep_base * decay) // rep_window
                        break

            # Top-k filtering + Boltzmann sampling
            top_k = 10
            if len(total_energies) > top_k:
                kth = min(top_k, len(total_energies))
                top_indices = np.argpartition(total_energies, kth)[:kth]
                top_indices = top_indices[np.argsort(total_energies[top_indices])]
                candidate_arr = candidate_arr[top_indices]
                total_energies = total_energies[top_indices]

            chosen_idx = self._sampler.sample(total_energies)
            chosen_word = int(candidate_arr[chosen_idx])
            chosen_energy = int(total_energies[chosen_idx])

            words.append(chosen_word)
            types_list.append(chosen_type)

            context_sdr = self.sdr_encoder.encode_context_positional(words, 10)
            self.episodic.store(context_sdr)

            # v53: Sentence boundary reset
            chosen_word_str = self.vocab.idx2word.get(chosen_word, "")
            if chosen_word_str in (".", "!", "?"):
                self.hierarchy.reset()

            # Update binding context with chosen word
            if self.binding and 0 <= chosen_word < self.sdr_encoder.vocab_size:
                self.binding.add_word(self.sdr_encoder.word_active_bits[chosen_word])

            self._stats['total_steps'] += 1

            diagnostics.append({
                'pos': pos,
                'type': IDX2POS.get(chosen_type, "UNK"),
                'word': self.vocab.idx2word.get(chosen_word, "<UNK>"),
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

    # ===================================================================
    # PERPLEXITY
    # ===================================================================

    def compute_perplexity(self, n_samples: int = 100) -> float:
        """Compute perplexity on test sequences using F-lookup energies."""
        from ..vocabulary.pos import POS2IDX, CLOSED_CLASS
        from ..utils import primary_pos_tag
        from ..sampling import IntegerBoltzmannSampler, LOG2_SCALE

        if self._sampler is None:
            self._sampler = IntegerBoltzmannSampler(beta=self.beta, max_delta=50000)

        total_log2_prob = 0
        total_tokens = 0

        eval_seqs = self.test_sequences[:n_samples]

        for seq_idx, seq in enumerate(eval_seqs):
            if len(seq) < 3:
                continue

            self.hierarchy.reset()
            self.episodic.reset()
            if self.binding:
                self.binding.reset()

            for pos in range(1, len(seq)):
                target_word = seq[pos]
                context_words = seq[:pos]

                if target_word < 0 or target_word >= len(self.vocab):
                    total_log2_prob += -15 * LOG2_SCALE
                    total_tokens += 1
                    continue

                word_type = self._get_word_type(target_word)
                candidate_list = self.type_words.get(word_type, [])
                if not candidate_list:
                    total_log2_prob += -15 * LOG2_SCALE
                    total_tokens += 1
                    continue

                candidate_arr = np.array(candidate_list[:500], dtype=np.int64)

                if target_word not in candidate_arr:
                    candidate_arr = np.append(candidate_arr, target_word)

                context_sdr = self.sdr_encoder.encode_context_positional(context_words, 10)

                # v46: M_bind for attractor dynamics ONLY, not DAM energy.
                # DAM was trained on BOW-only contexts (v45 order-sensitive reverted).
                if self.binding and np.sum(self.binding.M_bind) > 0:
                    context_sdr_for_dynamics = self.binding.get_context_or(context_sdr)
                else:
                    context_sdr_for_dynamics = context_sdr

                context_field = np.zeros(self.sdr_dim, dtype=np.int32)
                active = np.where(context_sdr_for_dynamics > 0)[0]
                context_field[active] = self.dam_scale
                self.hierarchy.step_all(context_field, n_sweeps=1)

                # NORMALIZED log2-F DAM energies
                # v46: Use BOW-only context_sdr for DAM energy (NOT M_bind)
                dam_energies = self.hierarchy.compute_word_energies(
                    context_sdr, candidate_arr, self.sdr_encoder, self.dam_scale
                )

                # Episodic energy (v39: reduced scale)
                ep_energies = self.episodic.compute_word_episodic_energy(
                    candidate_arr, self.sdr_encoder, self.episodic.field_scale
                )

                total_energies = dam_energies + ep_energies

                # v39: Add binding energy bonus (multi-step unbinding)
                if self.binding and len(self.binding._recent_words) > 0:
                    bind_energy = self.binding.compute_binding_energy(
                        candidate_arr, self.sdr_encoder
                    )
                    total_energies += bind_energy

                # v48: Add bigram energy from J2
                if self.J2 is not None and pos > 0:
                    prev_word = seq[pos - 1]
                    bigram_energy = self._compute_bigram_energy(prev_word, candidate_arr)
                    total_energies += bigram_energy

                # v51: Add skip bigram energy
                if self.J2 is not None and self._skip_weight > 0 and pos >= 2:
                    skip_word = seq[pos - 2]
                    skip_energy = self._compute_skip_energy(skip_word, candidate_arr)
                    total_energies += skip_energy

                # v51: Add POS skeleton energy
                if self.J_pos_bi is not None and self._pos_weight > 0:
                    # Build type history from the sequence
                    type_hist = [self._get_word_type(w) for w in seq[:pos]]
                    candidate_types = np.array([
                        self._get_word_type(int(w)) for w in candidate_arr
                    ], dtype=np.int64)
                    pos_energy = self._compute_pos_energy(type_hist, candidate_types)
                    total_energies += pos_energy

                # v41: Repetition penalty REMOVED from PPL evaluation.
                # The repetition penalty is a generation-time anti-loop mechanism.
                # Applying it during PPL evaluation artificially inflates PPL by
                # penalizing correct target words that naturally repeat in text
                # (e.g., "the little girl saw a little cat" — predicting 2nd "little"
                # gets +800 penalty on dE scale of 122 = 6.5x overkill).
                # v40 PPL regression: 248 → 450 was entirely caused by this.

                log_probs = self._sampler.compute_log_probabilities(total_energies)

                target_idx = np.where(candidate_arr == target_word)[0]
                if len(target_idx) > 0:
                    total_log2_prob += int(log_probs[target_idx[0]])
                else:
                    total_log2_prob += -15 * LOG2_SCALE

                total_tokens += 1
                self.episodic.store(context_sdr)

                # v39: Update binding context with the target word
                if self.binding and 0 <= target_word < self.sdr_encoder.vocab_size:
                    self.binding.add_word(self.sdr_encoder.word_active_bits[target_word])

        if total_tokens == 0:
            return float('inf')

        if total_log2_prob >= 0:
            return 1.0

        from ..sampling import LN2_NUM, LN2_DEN
        neg_avg = -total_log2_prob
        log2_ppl_fp = (neg_avg << 16) // (total_tokens * LOG2_SCALE)
        int_part = log2_ppl_fp >> 16
        frac_part = log2_ppl_fp & 0xFFFF

        FP = 48
        ONE_FP = 1 << FP
        f_fp = (frac_part * ONE_FP) >> 16
        x = (f_fp * LN2_NUM) // LN2_DEN
        x2 = (x * x) >> FP
        x3 = (x2 * x) >> FP
        x4 = (x3 * x) >> FP
        x5 = (x4 * x) >> FP
        exp_val = ONE_FP + x + (x2 >> 1) + (x3 // 6) + (x4 // 24) + (x5 // 120)
        ppl_frac = exp_val / ONE_FP

        if int_part < 63:
            perplexity = float(1 << int_part) * ppl_frac
        else:
            perplexity = float('inf')

        print(f"  Perplexity: {perplexity:.2f} (evaluated on {total_tokens} tokens)")
        return perplexity

    # ===================================================================
    # HELPERS
    # ===================================================================

    def _get_word_type(self, word_idx: int) -> int:
        """Get primary POS type for a word."""
        from ..vocabulary.pos import POS2IDX
        from ..utils import primary_pos_tag
        allowed = self.pos_system.allowed_types.get(word_idx, set())
        return primary_pos_tag(allowed)

    def _get_valid_next_types(self, prev_type: int, types_history: List[int]) -> List[int]:
        """Get valid next POS types based on grammar constraints."""
        from ..vocabulary.pos import POS2IDX, N_POS, CLOSED_CLASS

        valid = list(range(N_POS))

        if self.pos_system is not None:
            allowed = set()
            for t in range(N_POS):
                test_types = list(types_history[-5:]) + [t]
                test_pos = len(test_types) - 1
                try:
                    penalty = self.pos_system.compute_grammar_penalty(
                        test_types, test_pos, t,
                    )
                    if penalty < 500:
                        allowed.add(t)
                except (IndexError, ValueError):
                    allowed.add(t)
            if allowed:
                valid = list(allowed)

        CLOSED_CLASS_IDS = frozenset({
            POS2IDX["DET"], POS2IDX["PREP"], POS2IDX["PART"],
            POS2IDX["PRON"], POS2IDX["AUX"], POS2IDX["CONJ"],
            POS2IDX["PUNCT"],  # v40: PUNCT is functionally closed-class; suppress after 2 consecutive
        })
        closed_run = 0
        for t in reversed(types_history):
            if t in CLOSED_CLASS_IDS:
                closed_run += 1
            else:
                break
        if closed_run >= 2:
            open_types = [t for t in valid if t not in CLOSED_CLASS_IDS]
            if open_types:
                valid = open_types

        return valid if valid else list(range(N_POS))

    def _select_pos_types(self, types_list: List[int]) -> List[int]:
        """v54: Rank valid POS types by trigram/bigram transition score.

        Returns types sorted from most to least likely, filtered to those
        that have candidate words available. The POS trigram model is the
        SOLE determinant of type ordering — the DAM does NOT influence
        type selection. This prevents the DAM from overriding syntactically
        correct type sequences with low-energy hub words of the wrong type.

        The POS trigram captures reliable syntactic patterns:
        - DET → ADJ/NOUN (determiner phrase)
        - NOUN → VERB (subject-verb)
        - VERB → DET/PREP/ADV (post-verbal)
        - ADJ → NOUN (modifier-head)
        These patterns are far more reliable than the DAM's noisy type
        preferences, especially during generation when context is self-
        generated and may contain errors.
        """
        from ..vocabulary.pos import POS2IDX, N_POS

        prev_type = types_list[-1] if types_list else POS2IDX["X"]
        valid_types = self._get_valid_next_types(prev_type, types_list)

        if not valid_types:
            return list(range(N_POS))

        if len(valid_types) == 1:
            return valid_types

        # Score each type using POS trigram (or bigram backoff)
        type_scores = {}
        for t in valid_types:
            # Skip types with no candidate words
            candidates = self.type_words.get(t, [])
            if not candidates:
                continue

            if len(types_list) >= 2 and self.J_pos_tri is not None:
                t_prev2 = types_list[-2]
                t_prev1 = types_list[-1]
                if (0 <= t_prev2 < self.J_pos_tri.shape[0] and
                    0 <= t_prev1 < self.J_pos_tri.shape[1] and
                    0 <= t < self.J_pos_tri.shape[2]):
                    tri_val = int(self.J_pos_tri[t_prev2, t_prev1, t])
                    bi_val = int(self.J_pos_bi[t_prev1, t])
                    # Use trigram where available, bigram backoff with 75% weight
                    score = tri_val if tri_val > 0 else (bi_val * 3 // 4)
                else:
                    score = 0
            elif len(types_list) >= 1:
                t_prev1 = types_list[-1]
                if 0 <= t_prev1 < self.J_pos_bi.shape[0] and 0 <= t < self.J_pos_bi.shape[1]:
                    score = int(self.J_pos_bi[t_prev1, t])
                else:
                    score = 0
            else:
                score = 0
            type_scores[t] = score

        if not type_scores:
            # Fallback: return all valid types that have candidates
            return [t for t in valid_types if self.type_words.get(t, [])]

        # Return types sorted by POS transition score (highest first)
        return sorted(type_scores, key=type_scores.get, reverse=True)

    def _print_diagnostics(self) -> None:
        """Print full model diagnostics."""
        f_type_name = {0: 'quadratic', 1: 'cubic', 2: 'exp_approx'}.get(
            self._f_type, 'unknown'
        )

        print("\n" + "=" * 70)
        print("ATTRACTOR LANGUAGE MACHINE v54 — DIAGNOSTICS")
        print("=" * 70)

        if self.sdr_encoder:
            print(f"  SDR: D={self.sdr_encoder.D}, k={self.sdr_encoder.k} "
                  f"({self.sdr_encoder.sparsity*100:.1f}% sparse)")

        if self.hierarchy:
            hdiag = self.hierarchy.get_diagnostics()
            for l in range(hdiag['n_layers']):
                ld = hdiag[f'L{l}']
                source = "Hebbian (L0)" if l == 0 else "RG flow"
                print(f"  L{l} [{source}]: D={ld['D']}, k={ld['k']}, "
                      f"J_max={ld['J_max']}, J_nnz={ld['J_nnz']}, "
                      f"h_max={ld['h_max']}, f_type={ld['f_type']}")
            print(f"  RG applied: {hdiag['rg_applied']}")
            print(f"  Total hierarchy memory: {hdiag['total_memory_kb']:.1f} KB")

            if 'uv_completeness' in hdiag:
                uv = hdiag['uv_completeness']
                print(f"  UV completeness score: {uv['overall_uv_score']:.2f}")
                print(f"  Ward identity violations:")
                for l in range(hdiag['n_layers']):
                    l_uv = uv[f'L{l}']
                    print(f"    L{l}: ward_violation={l_uv['ward_violation']:.4f}")

        if self.episodic:
            ediag = self.episodic.get_diagnostics()
            print(f"  Episodic: {ediag['n_episodes']} episodes, "
                  f"memory={ediag['memory_kb']:.1f} KB")

        if self.binding:
            bdiag = self.binding.get_diagnostics()
            print(f"  Binding: VSA permutation, window={bdiag['window']}, "
                  f"weight={bdiag['bind_weight']}, "
                  f"n_unbind={bdiag['n_unbind_words']}, "
                  f"M_bind density={bdiag['target_density']}/{self.sdr_dim}")
            print(f"  Binding: uniform kWTA (v43 revert), "
                  f"fill={bdiag['window_fill']}/{bdiag['window']}, "
                  f"actual_density={bdiag['m_bind_density']}")

        print(f"  Beta: {self.beta:.6f}")
        print(f"  Vocab: {len(self.vocab)} words")
        print(f"  F function: {f_type_name}, T={self._exp_temperature/100:.2f}")
        print(f"  Learning: Hebbian (L0 only, RG flow to higher levels)")
        print(f"  Energy: NORMALIZED log2-F ({f_type_name}, LOG2_NORM=512, NO k div, NO h)")
        print(f"  Binding: VSA permutation (window={self._bind_window}, weight={self._bind_weight}, n_unbind={self._n_unbind_words})")

        # v49: Bigram J2 diagnostics (log-normalized)
        if self.J2 is not None:
            j2_nnz = int(np.sum(self.J2 > 0))
            j2_log_max = int(np.max(self.J2))
            j2_mem_mb = self.J2.nbytes / (1024 * 1024)
            raw_max = getattr(self, '_j2_raw_max', '?')
            print(f"  Bigram J2 (LOG): {j2_nnz:,} non-zero, log_max={j2_log_max}, "
                  f"raw_max={raw_max}, weight={self._bigram_weight}, "
                  f"max_energy={j2_log_max * self._bigram_weight}, memory={j2_mem_mb:.1f} MB")
            if self._skip_weight > 0:
                print(f"  Skip bigram: weight={self._skip_weight}, "
                      f"max_energy={j2_log_max * self._skip_weight} (reuses J2)")
        else:
            print(f"  Bigram J2: disabled (weight=0)")

        # v53: POS skeleton diagnostics (always built for constrained decoding)
        if self.J_pos_bi is not None:
            bi_log_max = int(np.max(self.J_pos_bi))
            tri_log_max = int(np.max(self.J_pos_tri)) if self.J_pos_tri is not None else 0
            print(f"  POS skeleton: bi_log_max={bi_log_max}, tri_log_max={tri_log_max}, "
                  f"PPL weight={self._pos_weight}, gen bonus weight={self._pos_gen_weight}, "
                  f"type top-k={self._pos_type_top_k}")
            print(f"  Frequency penalty: weight={self._freq_penalty_weight}")
            print(f"  Bigram gen weight: {self._bigram_gen_weight}{' (=bigram_weight)' if self._bigram_gen_weight == 0 else ''} (v54)")
            print(f"  Skip gen weight: {self._skip_gen_weight}{' (=skip_weight)' if self._skip_gen_weight == 0 else ''} (v54)")
        else:
            print(f"  POS skeleton: not built")

        print("=" * 70)

    def reset_stats(self) -> None:
        """Reset generation statistics."""
        self._stats = {
            'total_steps': 0,
            'dam_hits': 0,
            'episodic_hits': 0,
        }

    def get_stats(self) -> Dict:
        """Get generation statistics."""
        return self._stats.copy()
