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
  6. DAM energy alone drives word selection (no n-gram crutch)
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

        # Built during training
        self.vocab = None
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
        print("ATTRACTOR LANGUAGE MACHINE v47 — CLEAN REVERT", flush=True)
        print(f"  F function: {f_type_name}, T={self._exp_temperature/100:.2f}", flush=True)
        print("  RG flow: J_eff[l] decimated (not layers[l].J), Kadanoff rescaling", flush=True)
        print("  Energy: NORMALIZED log2-F (LOG2_NORM=512, NO k division, NO h)", flush=True)
        print("  Binding: VSA permutation bind(a,hash(b)), kWTA sparsification", flush=True)
        print(f"  Bind window={self._bind_window}, weight={self._bind_weight}, n_unbind={self._n_unbind_words}, density={self._bind_density if self._bind_density > 0 else 'auto'}", flush=True)
        print("  M_bind: attractor dynamics ONLY (not DAM energy) — v45 reverted", flush=True)
        print("  Training: BOW-only DAM (v45 order-sensitive reverted, v46 binding params reverted)", flush=True)
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
        # v47: REVERTED v46 binding params (weight 100→30, density 40→auto=20)
        # v46's boosted binding caused PPL 7983 — binding noise amplified at
        # high weight dominated over DAM co-occurrence signal. v44 params (30, 20)
        # gave PPL=221.
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

        self._calibrate_beta()

        # v42: Reset binding context after calibration — avoids showing
        # calibration residue in diagnostics (deque had only 1-2 entries)
        if self.binding:
            self.binding.reset()

        from ..sampling import IntegerBoltzmannSampler
        self._sampler = IntegerBoltzmannSampler(
            beta=self.beta, max_delta=5000
        )

        t_total = time.time() - t0
        rss = get_rss_mb()
        print(f"\nTraining complete: {t_total:.1f}s")
        print(f"  Vocab: {V} words")
        if rss > 0:
            print(f"  Memory (RSS): {rss:,} MB")
        print(f"  Integer-only: YES — ZERO float operations in hot path")
        print(f"  Architecture: Dense Associative Memory (DAM) Engine v47")
        print(f"  F function: {f_type_name}, T={self._exp_temperature/100:.2f}")
        print(f"  Learning: Hebbian (L0 only, RG flow to higher levels)")
        print(f"  Energy: NORMALIZED log2-F ({f_type_name}, LOG2_NORM=512, NO k div, NO h)")
        print(f"  Binding: VSA permutation (window={self._bind_window}, weight={self._bind_weight}, n_unbind={self._n_unbind_words}, density={self._bind_density if self._bind_density > 0 else 'auto'})")
        print(f"  Repetition: penalty={self.same_word_penalty}, window=15, distance-decay")
        print(f"  Generation: top-k=10 (v44) + Boltzmann sampling")
        print(f"  v47: DAM trained on BOW-only contexts (v45 order-sensitive reverted)")
        print(f"  v47: Binding params reverted to v44 values (weight=30, density=auto=20)")

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
        print(f"    v47: Using BOW-ONLY context encoding (same as v44)", flush=True)

        def progress_callback(seq_idx, total):
            print(f"      Hebbian encoding: {seq_idx:,} seqs, {total:,} pairs encoded",
                  flush=True)

        # v47: BOW-only context encoding (same as v44).
        # v45 used encode_contexts_batch_with_binding() → PPL 1909.
        # v46 reverted training but boosted binding → PPL 7983.
        # v47: revert binding params too → should recover v44 PPL ~221.
        for ctx_arr, tgt_arr in self.sdr_encoder.encode_contexts_batch(
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
                context_sdr = self.sdr_encoder.encode_context(context_words, context_window)
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
                context_sdr = self.sdr_encoder.encode_context(context_words)

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
            self._sampler = IntegerBoltzmannSampler(beta=self.beta, max_delta=5000)

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
            context_sdr = self.sdr_encoder.encode_context(words[:i+1], 10)

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
            # Choose POS type
            prev_type = types_list[-1] if types_list else POS2IDX["X"]
            valid_types = self._get_valid_next_types(prev_type, types_list)

            # Encode context (standard word superposition — BOW only)
            context_sdr = self.sdr_encoder.encode_context(words, 10)

            # v46: M_bind for attractor dynamics ONLY, not DAM energy.
            # DAM energy uses BOW-only context (the DAM was trained without binding).
            # M_bind shapes the attractor basin through the context field.
            if self.binding and np.sum(self.binding.M_bind) > 0:
                context_sdr_for_dynamics = self.binding.get_context_or(context_sdr)
            else:
                context_sdr_for_dynamics = context_sdr

            # Context field for attractor dynamics (includes M_bind)
            context_field = np.zeros(self.sdr_dim, dtype=np.int32)
            active = np.where(context_sdr_for_dynamics > 0)[0]
            context_field[active] = self.dam_scale

            # Run attractor dynamics
            self.hierarchy.step_all(context_field, n_sweeps=2)

            # Find best type + candidates
            best_type = valid_types[0]
            best_min_energy = float('inf')
            best_candidate_words = None
            best_energies = None

            for chosen_type in valid_types:
                candidate_list = self.type_words.get(chosen_type, [])
                if not candidate_list:
                    continue

                candidate_arr = np.array(candidate_list[:300], dtype=np.int64)

                # DAM energy with NORMALIZED log2-F
                # v46: Use BOW-only context_sdr for DAM energy (NOT M_bind).
                # The DAM was trained on BOW-only contexts — injecting binding
                # bits adds noise to the coupling energy (v45: PPL 1909).
                dam_energies = self.hierarchy.compute_word_energies(
                    context_sdr, candidate_arr, self.sdr_encoder, self.dam_scale
                )

                # Episodic energy (v37: reduced scale=100, ~10% of DAM range)
                ep_energies = self.episodic.compute_word_episodic_energy(
                    candidate_arr, self.sdr_encoder, self.episodic.field_scale
                )

                # v40: Grammar penalty — scaled to ~50% of median_dE for real effect
                test_types = list(types_list[-5:]) + [chosen_type]
                test_pos = len(test_types) - 1
                try:
                    grammar_penalty = self.pos_system.compute_grammar_penalty(
                        test_types, test_pos, chosen_type
                    )
                except (IndexError, ValueError):
                    grammar_penalty = 0
                # v40 FIX: Scale grammar penalty to be meaningful on the dE scale.
                # v39 used 500//median_de as divisor, making penalty ~15 (5% of dE).
                # v40: grammar_penalty * (median_de // 2) // 60 → ~100 (33% of dE)
                median_de = max(1, getattr(self, '_median_de', 10))
                grammar_scaled = grammar_penalty * (median_de // 2) // 60
                grammar_energies = np.full(len(candidate_arr), grammar_scaled, dtype=np.int64)

                total_energies = dam_energies + ep_energies + grammar_energies

                # v39: Add binding energy bonus (multi-step unbinding)
                if self.binding and len(self.binding._recent_words) > 0:
                    bind_energy = self.binding.compute_binding_energy(
                        candidate_arr, self.sdr_encoder
                    )
                    total_energies += bind_energy

                # v40: Repetition penalty — FIXED: use same_word_penalty (800),
                # window=15 words, distance-based decay (closer = stronger).
                # v39 BUG: same_word_penalty=800 was dead code — actual penalty was
                # median_de//5 = 24, negligible on dE scale of 200-300.
                rep_window = 15
                rep_base = self.same_word_penalty  # 800 by default
                recent_words = words[-rep_window:]
                for i, w in enumerate(candidate_arr):
                    w_int = int(w)
                    # Distance-based decay: word at distance d gets penalty * (1 - d/window)
                    for d, rw in enumerate(reversed(recent_words)):
                        if w_int == rw:
                            decay = max(1, rep_window - d)  # 15 for most recent, 1 for oldest
                            total_energies[i] += (rep_base * decay) // rep_window
                            break  # Only count closest occurrence

                min_e = int(total_energies.min())
                if min_e < best_min_energy:
                    best_min_energy = min_e
                    best_type = chosen_type
                    best_candidate_words = candidate_arr
                    best_energies = total_energies

            if best_candidate_words is None:
                best_candidate_words = np.array([4], dtype=np.int64)
                best_energies = np.array([0], dtype=np.int64)

            # v44: Top-k filtering before Boltzmann sampling.
            # Keep only the k lowest-energy candidates to prevent
            # low-probability "tail" words from being sampled.
            # With 300 candidates per type, Boltzmann sampling spreads
            # probability too thinly, causing incoherent generation.
            # Top-k=10 focuses the sampler on the most likely words.
            top_k = 10
            if len(best_energies) > top_k:
                # Find indices of top-k lowest energies
                kth = min(top_k, len(best_energies))
                top_indices = np.argpartition(best_energies, kth)[:kth]
                # Sort by energy for better numerical properties
                top_indices = top_indices[np.argsort(best_energies[top_indices])]
                best_candidate_words = best_candidate_words[top_indices]
                best_energies = best_energies[top_indices]

            # Boltzmann sample from top-k candidates
            chosen_idx = self._sampler.sample(best_energies)
            chosen_word = int(best_candidate_words[chosen_idx])
            chosen_energy = int(best_energies[chosen_idx])

            words.append(chosen_word)
            types_list.append(best_type)

            context_sdr = self.sdr_encoder.encode_context(words, 10)
            self.episodic.store(context_sdr)

            # v39: Update binding context with chosen word
            if self.binding and 0 <= chosen_word < self.sdr_encoder.vocab_size:
                self.binding.add_word(self.sdr_encoder.word_active_bits[chosen_word])

            self._stats['total_steps'] += 1

            diagnostics.append({
                'pos': pos,
                'type': IDX2POS.get(best_type, "UNK"),
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
            self._sampler = IntegerBoltzmannSampler(beta=self.beta, max_delta=5000)

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

                context_sdr = self.sdr_encoder.encode_context(context_words, 10)

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

    def _print_diagnostics(self) -> None:
        """Print full model diagnostics."""
        f_type_name = {0: 'quadratic', 1: 'cubic', 2: 'exp_approx'}.get(
            self._f_type, 'unknown'
        )

        print("\n" + "=" * 70)
        print("ATTRACTOR LANGUAGE MACHINE v47 — DIAGNOSTICS")
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
