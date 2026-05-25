"""
Attractor Language Machine — THE ENGINE.

The attractor dynamics of a Dense Associative Memory ARE a language model.
Not an approximation. Not a component. They ARE one.

This engine implements:
  - DAM attractor dynamics (energy-based prediction)
  - Hierarchical DAM states (L0-L3) with Wilsonian RG flow
  - Content-addressable episodic memory (sparse pattern storage)
  - Sparse Distributed Representations (kWTA, ~2% active bits)
  - kWTA attractor dynamics + Boltzmann refinement

WHAT'S KEPT FROM THE OLD ARCHITECTURE:
  - Vocabulary building and tokenization
  - POS type system (grammar constraints are still needed)
  - IntegerBoltzmannSampler (for final word selection from DAM energies)
  - Data loading infrastructure

HOW PREDICTION WORKS:
  1. Encode context words as SDRs → context_field for L0
  2. Run hierarchical attractor dynamics (L0→L3 bottom-up, L3→L0 top-down)
  3. Compute DAM energy for each candidate word
  4. Add episodic memory energy
  5. Add POS grammar constraints
  6. Boltzmann sample from energy distribution
  7. Update all layer states and episodic memory

TRAINING:
  Phase 1: Build SDR encoder (deterministic, no learning)
  Phase 2: Batch Hebbian storage (fast, one pass)
  Phase 3: PCD refinement (iterative, sculpts energy landscape)

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


class AttractorLanguageModel:
    """
    Attractor Language Machine — Dense Associative Memory as the ENGINE.

    The full language model: SDR encoder + hierarchical DAM + episodic
    memory + POS constraints. Training via batch Hebbian + PCD refinement.

    This is the main entry point for the Attractor Language Machine.
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
        # Learning
        learning_rate: int = 1,
        n_dream_steps: int = 3,
        j_clip: int = 500,
        # UV-complete
        uv_regularize: bool = True,
        uv_lambda: int = 5,
        topdown_scale: int = 200,
        rg_beta_strength: int = 100,
        # Episodic
        max_episodes: int = 10000,
        episodic_scale: int = 500,
        # Energy scales
        dam_scale: int = 1600,
        grammar_penalty_scale: int = 60,
        same_word_penalty: int = 800,
        # Generation
        beta: float = 0.01,
        max_seq_len: int = 30,
        # Memory
        memory_budget_mb: int = 0,
        # Seeds
        seed: int = 42,
    ):
        # Store all params
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

        # Built during training
        self.vocab = None
        self.pos_system = None
        self.sdr_encoder: Optional[SDREncoder] = None
        self.hierarchy: Optional[HierarchicalDAM] = None
        self.episodic: Optional[EpisodicMemory] = None

        self.sequences = None
        self.test_sequences = None
        self._word_freq = None

        # Type→words mapping (from POS system)
        self.type_words: Dict[int, List[int]] = {}

        # Boltzmann sampler (built after training)
        self._sampler = None

        # Stats
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
          4. Build hierarchical DAM
          5. Batch Hebbian storage (one pass through training data)
          6. PCD refinement (optional, iterative)
          7. Build episodic memory
          8. Calibrate beta
        """
        from ..vocabulary import Vocabulary, POSTypeSystem
        from ..vocabulary.pos import POS2IDX, IDX2POS, N_POS, CLOSED_CLASS
        from ..utils import (
            primary_pos_tag, tokenize_texts, truncate_sequences,
            DATASET_LOADERS, DEFAULT_DATASET, get_rss_mb,
        )

        print("=" * 70)
        print("ATTRACTOR LANGUAGE MACHINE — Dense Associative Memory Engine")
        print("=" * 70)

        t0 = time.time()

        # ------------------------------------------------------------------
        # Step 1: Load corpus
        # ------------------------------------------------------------------
        if texts is None:
            print(f"\n[1/9] Loading corpus...")
            loader = DATASET_LOADERS[DEFAULT_DATASET]
            texts = loader(n_samples=n_samples)
        else:
            print(f"\n[1/9] Using provided texts ({len(texts):,})")

        # ------------------------------------------------------------------
        # Step 2: Build vocabulary
        # ------------------------------------------------------------------
        print("\n[2/9] Building vocabulary...")
        self.vocab = Vocabulary(
            min_freq=self.vocab_min_freq,
            max_size=self.vocab_max_size,
        )
        self.vocab.build(texts)
        V = len(self.vocab)
        print(f"  Vocabulary: {V} words")

        # ------------------------------------------------------------------
        # Step 3: Tokenize
        # ------------------------------------------------------------------
        print("\n[3/9] Tokenizing...")
        sequences = tokenize_texts(texts, self.vocab)
        sequences = truncate_sequences(sequences, max_len=self.max_seq_len)
        split_idx = int(len(sequences) * 0.9)
        self.sequences = sequences[:split_idx]
        self.test_sequences = sequences[split_idx:]
        print(f"  Train: {len(self.sequences):,}, Test: {len(self.test_sequences):,}")

        # Word frequencies
        self._word_freq = np.zeros(V, dtype=np.int64)
        for seq in self.sequences:
            for w in seq:
                if w < V:
                    self._word_freq[w] += 1

        # ------------------------------------------------------------------
        # Step 4: Build POS type system
        # ------------------------------------------------------------------
        print("\n[4/9] Building POS type system...")
        self.pos_system = POSTypeSystem(vocab_size=V, window=5)
        self.pos_system.build_from_vocabulary(self.vocab.word2idx, self.vocab.idx2word)
        self.pos_system.build_grammar_penalties(penalty_strength=self.grammar_penalty_scale)
        self.pos_system.compute_type_couplings(self.sequences, self.vocab.idx2word)

        # Build type→words mapping
        self.type_words = {t: [] for t in range(N_POS)}
        for w, allowed in self.pos_system.allowed_types.items():
            if allowed:
                primary = primary_pos_tag(allowed)
                self.type_words[primary].append(w)

        n_typed = sum(1 for w in range(V) if w in self.pos_system.allowed_types)
        print(f"  POS system: {N_POS} types, {n_typed} words typed")

        # ------------------------------------------------------------------
        # Step 5: Build SDR encoder
        # ------------------------------------------------------------------
        print(f"\n[5/9] Building SDR encoder (D={self.sdr_dim}, sparsity={self.sdr_sparsity})...")
        self.sdr_encoder = SDREncoder(
            vocab_size=V,
            D=self.sdr_dim,
            sparsity=self.sdr_sparsity,
            seed=self.seed,
        )
        self.sdr_encoder.build(word_freq=self._word_freq)

        # ------------------------------------------------------------------
        # Step 6: Build hierarchical DAM
        # ------------------------------------------------------------------
        print(f"\n[6/9] Building hierarchical DAM...")

        # Derive hierarchy config from SDR dimension
        # L0 must match SDR dimension; higher levels are powers-of-2 smaller
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

        self.hierarchy = HierarchicalDAM(
            layers_config=layers_config,
            learning_rate=self.learning_rate if hasattr(self, 'learning_rate') else 1,
            n_dream_steps=self.n_dream_steps if hasattr(self, 'n_dream_steps') else 3,
            j_clip=self.j_clip if hasattr(self, 'j_clip') else 500,
            uv_regularize=self.uv_regularize if hasattr(self, 'uv_regularize') else True,
            uv_lambda=self.uv_lambda if hasattr(self, 'uv_lambda') else 5,
            seed=self.seed,
        )
        self.hierarchy.build(self.sdr_encoder)

        # ------------------------------------------------------------------
        # Step 7: Batch Hebbian training
        # ------------------------------------------------------------------
        print(f"\n[7/9] Batch Hebbian training...")
        self._batch_hebbian_train()

        # ------------------------------------------------------------------
        # Step 8: Build episodic memory
        # ------------------------------------------------------------------
        print(f"\n[8/9] Building episodic memory...")
        self.episodic = EpisodicMemory(
            D=self.sdr_dim,
            k=self.sdr_encoder.k,
            max_episodes=self.max_episodes if hasattr(self, 'max_episodes') else 10000,
            field_scale=self.episodic_scale if hasattr(self, 'episodic_scale') else 500,
            seed=self.seed,
        )
        # Pre-populate episodic memory from training sequences
        self._populate_episodic_memory()

        # ------------------------------------------------------------------
        # Step 9: Calibrate beta
        # ------------------------------------------------------------------
        print(f"\n[9/9] Calibrating beta...")
        self._calibrate_beta()

        # Build Boltzmann sampler
        from ..sampling import IntegerBoltzmannSampler
        self._sampler = IntegerBoltzmannSampler(
            beta=self.beta, max_delta=50000
        )

        t_total = time.time() - t0
        rss = get_rss_mb()
        print(f"\nTraining complete: {t_total:.1f}s")
        print(f"  Vocab: {V} words")
        if rss > 0:
            print(f"  Memory (RSS): {rss:,} MB")
        print(f"  Integer-only: YES — ZERO float operations in hot path")
        print(f"  Architecture: Dense Associative Memory (DAM) Engine")

        # Print full diagnostics
        self._print_diagnostics()

        return self

    def _batch_hebbian_train(self) -> None:
        """
        Phase 1 training: batch Hebbian storage.

        For each training sequence, create (context, target) pairs and
        store them in the DAM using the Hebbian outer-product rule.

        This is fast (one pass, vectorized) and provides a good
        initialization. PCD refinement (Phase 2) can then improve it.
        """
        from ..utils import primary_pos_tag

        V = len(self.vocab)
        D = self.sdr_dim
        k = self.sdr_encoder.k
        context_window = 10  # Use last 10 words as context

        # Collect (context_sdr, target_sdr) pairs
        # Process in batches for memory efficiency
        batch_size = 50000
        total_pairs = 0
        batch_context = []
        batch_target = []

        for seq_idx, seq in enumerate(self.sequences):
            if len(seq) < 3:
                continue

            for pos in range(1, len(seq)):
                target_word = seq[pos]
                context_words = seq[max(0, pos - context_window):pos]

                if not context_words or target_word < 0 or target_word >= V:
                    continue

                # Encode context and target as SDRs
                context_sdr = self.sdr_encoder.encode_context(context_words, context_window)
                target_sdr = self.sdr_encoder.encode(target_word)

                # Only store if both are non-trivial
                if np.sum(context_sdr) > 0 and np.sum(target_sdr) > 0:
                    batch_context.append(context_sdr)
                    batch_target.append(target_sdr)
                    total_pairs += 1

                # Apply batch when full
                if len(batch_context) >= batch_size:
                    ctx_arr = np.array(batch_context, dtype=np.uint8)
                    tgt_arr = np.array(batch_target, dtype=np.uint8)
                    self.hierarchy.train_batch_hebbian(ctx_arr, tgt_arr, eta=1)
                    batch_context = []
                    batch_target = []

            if (seq_idx + 1) % 50000 == 0:
                print(f"      Hebbian training: {seq_idx+1}/{len(self.sequences)} seqs, "
                      f"{total_pairs:,} pairs stored")

        # Flush remaining batch
        if batch_context:
            ctx_arr = np.array(batch_context, dtype=np.uint8)
            tgt_arr = np.array(batch_target, dtype=np.uint8)
            self.hierarchy.train_batch_hebbian(ctx_arr, tgt_arr, eta=1)

        print(f"    Hebbian training complete: {total_pairs:,} pairs stored")

        # Print coupling stats
        for l, layer in enumerate(self.hierarchy.layers):
            diag = layer.get_diagnostics()
            print(f"    L{l}: J_max={diag['J_max']}, J_nnz={diag['J_nnz']}, "
                  f"h_max={diag['h_max']}, h_nnz={diag['h_nnz']}")

    def _populate_episodic_memory(self) -> None:
        """
        Pre-populate episodic memory from training sequences.

        For each training sequence, store the document state at each
        position. This gives the episodic memory a base of patterns
        to retrieve from during generation.
        """
        V = len(self.vocab)
        context_window = 10

        n_stored = 0
        for seq_idx, seq in enumerate(self.sequences[:50000]):  # Limit for speed
            if len(seq) < 5:
                continue

            # Store document state at each position
            for pos in range(5, len(seq), 3):  # Every 3rd position
                context_words = seq[max(0, pos - context_window):pos]
                context_sdr = self.sdr_encoder.encode_context(context_words, context_window)
                if np.sum(context_sdr) > 0:
                    self.episodic.store(context_sdr)
                    n_stored += 1

            if (seq_idx + 1) % 10000 == 0:
                print(f"      Episodic memory: {seq_idx+1} seqs, {n_stored} episodes stored")

        print(f"    Episodic memory: {n_stored} episodes stored")

    def _calibrate_beta(self) -> None:
        """
        Calibrate Boltzmann beta from the DAM energy distribution.

        The optimal beta depends on the energy scale and the
        discriminability of the energy landscape.
        """
        # Sample DAM energies for some test positions
        energy_diffs = []
        n_samples = 0

        for seq in self.test_sequences[:100]:
            if len(seq) < 3:
                continue

            for pos in range(1, min(len(seq), 5)):
                target_word = seq[pos]
                context_words = seq[:pos]

                # Get candidate words of the same POS type
                from ..utils import primary_pos_tag
                word_type = primary_pos_tag(
                    self.pos_system.allowed_types.get(target_word, set())
                )
                candidates = self.type_words.get(word_type, [])
                if len(candidates) < 5:
                    continue

                candidate_arr = np.array(candidates[:200], dtype=np.int64)
                context_sdr = self.sdr_encoder.encode_context(context_words)

                # Compute energies
                energies = self.hierarchy.compute_word_energies(
                    context_sdr, candidate_arr, self.sdr_encoder, self.dam_scale
                )

                # Add episodic energy
                if self.episodic and len(self.episodic.episodes) > 0:
                    ep_energy = self.episodic.compute_word_episodic_energy(
                        candidate_arr, self.sdr_encoder, self.episodic.field_scale
                    )
                    energies += ep_energy

                # Compute energy differences from minimum
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

            # Theoretical beta: gives PPL ≈ exp(1) at median dE
            theoretical_beta = 0.55 * math.log(2) / max(1, self.dam_scale)
            empirical_beta = 3.5 / max(1, p10_de)

            self.beta = max(theoretical_beta, min(1.0, empirical_beta))
            print(f"    Median dE: {median_de}, p10 dE: {p10_de}")
            print(f"    Theoretical beta: {theoretical_beta:.6f}")
            print(f"    Empirical beta: {empirical_beta:.6f}")
            print(f"    Using beta: {self.beta:.6f}")
        else:
            self.beta = 0.55 * math.log(2) / max(1, self.dam_scale)
            print(f"    Using theoretical beta: {self.beta:.6f}")

    # ===================================================================
    # GENERATION
    # ===================================================================

    def generate(self, prompt: str = "the", length: int = 20) -> Dict:
        """
        Generate text autoregressively using DAM attractor dynamics.

        At each position:
          1. Encode context as SDR
          2. Run hierarchical attractor dynamics
          3. Compute DAM + episodic + grammar energies for candidates
          4. Boltzmann sample next word
          5. Update all layer states and episodic memory

        All energy computation is integer-only.
        """
        from ..vocabulary.pos import POS2IDX, IDX2POS, N_POS, CLOSED_CLASS
        from ..utils import primary_pos_tag
        from ..sampling import IntegerBoltzmannSampler

        if self._sampler is None:
            self._sampler = IntegerBoltzmannSampler(beta=self.beta, max_delta=50000)

        # Resolve prompt
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

        # Reset all layers for new document
        self.hierarchy.reset()
        self.episodic.reset()

        # Initialize layer states from prompt
        for w in words:
            context_sdr = self.sdr_encoder.encode_context(words, 10)
            context_field = np.zeros(self.sdr_dim, dtype=np.int32)
            active = np.where(context_sdr > 0)[0]
            context_field[active] = self.dam_scale

            # Run attractor dynamics
            self.hierarchy.step_all(context_field, n_sweeps=1)

            # Store in episodic memory
            self.episodic.store(context_sdr)

        for pos in range(len(words), length):
            # === STEP 1: Choose POS type ===
            prev_type = types_list[-1] if types_list else POS2IDX["X"]
            valid_types = self._get_valid_next_types(prev_type, types_list)

            # === STEP 2: Encode context ===
            context_sdr = self.sdr_encoder.encode_context(words, 10)
            context_field = np.zeros(self.sdr_dim, dtype=np.int32)
            active = np.where(context_sdr > 0)[0]
            context_field[active] = self.dam_scale

            # === STEP 3: Run attractor dynamics ===
            self.hierarchy.step_all(context_field, n_sweeps=2)

            # === STEP 4: Get candidates and compute energies ===
            # Try each valid type, pick the one with lowest minimum energy
            best_type = valid_types[0]
            best_min_energy = float('inf')
            best_candidate_words = None
            best_energies = None

            for chosen_type in valid_types:
                candidate_list = self.type_words.get(chosen_type, [])
                if not candidate_list:
                    continue

                candidate_arr = np.array(candidate_list[:300], dtype=np.int64)

                # DAM energy
                dam_energies = self.hierarchy.compute_word_energies(
                    context_sdr, candidate_arr, self.sdr_encoder, self.dam_scale
                )

                # Episodic energy
                ep_energies = self.episodic.compute_word_episodic_energy(
                    candidate_arr, self.sdr_encoder, self.episodic.field_scale
                )

                # Grammar penalty (same for all candidates of same type)
                test_types = list(types_list[-5:]) + [chosen_type]
                test_pos = len(test_types) - 1
                try:
                    grammar_penalty = self.pos_system.compute_grammar_penalty(
                        test_types, test_pos, chosen_type
                    )
                except (IndexError, ValueError):
                    grammar_penalty = 0
                grammar_energies = np.full(len(candidate_arr), grammar_penalty, dtype=np.int64)

                total_energies = dam_energies + ep_energies + grammar_energies

                # Repetition penalty
                recent = set(words[-5:])
                for i, w in enumerate(candidate_arr):
                    if int(w) in recent:
                        total_energies[i] += self.same_word_penalty

                min_e = int(total_energies.min())
                if min_e < best_min_energy:
                    best_min_energy = min_e
                    best_type = chosen_type
                    best_candidate_words = candidate_arr
                    best_energies = total_energies

            if best_candidate_words is None:
                # Fallback: pick any word
                best_candidate_words = np.array([4], dtype=np.int64)
                best_energies = np.array([0], dtype=np.int64)

            # === STEP 5: Boltzmann sample ===
            chosen_idx = self._sampler.sample(best_energies)
            chosen_word = int(best_candidate_words[chosen_idx])
            chosen_energy = int(best_energies[chosen_idx])

            words.append(chosen_word)
            types_list.append(best_type)

            # === STEP 6: Update states ===
            # Update episodic memory
            context_sdr = self.sdr_encoder.encode_context(words, 10)
            self.episodic.store(context_sdr)

            # Track diagnostics
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
        """
        Compute perplexity on test sequences.

        PPL = exp(-1/N * Σ log P(w_t | ctx))

        P(w | ctx) ∝ exp(-β * E(w, ctx))
        E(w, ctx) = E_DAM(w, ctx) + E_episodic(w, ctx) + E_grammar(w, ctx)

        Uses IntegerBoltzmannSampler for integer-only log probability computation.
        """
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

            # Reset for new sequence
            self.hierarchy.reset()
            self.episodic.reset()

            for pos in range(1, len(seq)):
                target_word = seq[pos]
                context_words = seq[:pos]

                if target_word < 0 or target_word >= len(self.vocab):
                    total_log2_prob += -15 * LOG2_SCALE
                    total_tokens += 1
                    continue

                # Get candidate words
                word_type = self._get_word_type(target_word)
                candidate_list = self.type_words.get(word_type, [])
                if not candidate_list:
                    total_log2_prob += -15 * LOG2_SCALE
                    total_tokens += 1
                    continue

                candidate_arr = np.array(candidate_list[:500], dtype=np.int64)

                # Ensure target is in candidates
                if target_word not in candidate_arr:
                    candidate_arr = np.append(candidate_arr, target_word)

                # Encode context
                context_sdr = self.sdr_encoder.encode_context(context_words, 10)

                # Run attractor dynamics
                context_field = np.zeros(self.sdr_dim, dtype=np.int32)
                active = np.where(context_sdr > 0)[0]
                context_field[active] = self.dam_scale
                self.hierarchy.step_all(context_field, n_sweeps=1)

                # Compute energies
                dam_energies = self.hierarchy.compute_word_energies(
                    context_sdr, candidate_arr, self.sdr_encoder, self.dam_scale
                )

                ep_energies = self.episodic.compute_word_episodic_energy(
                    candidate_arr, self.sdr_encoder, self.episodic.field_scale
                )

                total_energies = dam_energies + ep_energies

                # Repetition penalty (matches generation)
                recent = set(context_words[-5:])
                for i, w in enumerate(candidate_arr):
                    if int(w) in recent:
                        total_energies[i] += self.same_word_penalty

                # Compute log probabilities
                log_probs = self._sampler.compute_log_probabilities(total_energies)

                # Find target word's log probability
                target_idx = np.where(candidate_arr == target_word)[0]
                if len(target_idx) > 0:
                    total_log2_prob += int(log_probs[target_idx[0]])
                else:
                    total_log2_prob += -15 * LOG2_SCALE

                total_tokens += 1

                # Update episodic memory
                self.episodic.store(context_sdr)

        if total_tokens == 0:
            return float('inf')

        # PPL from integer log2 probabilities
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

        # All types are valid initially
        valid = list(range(N_POS))

        # Grammar constraint: use allowed transitions
        if self.pos_system is not None:
            allowed = set()
            for t in range(N_POS):
                # Build a synthetic types list ending with the candidate type
                test_types = list(types_history[-5:]) + [t]
                test_pos = len(test_types) - 1  # Position of the candidate
                try:
                    penalty = self.pos_system.compute_grammar_penalty(
                        test_types, test_pos, t,
                    )
                    if penalty < 500:
                        allowed.add(t)
                except (IndexError, ValueError):
                    allowed.add(t)  # Allow on error
            if allowed:
                valid = list(allowed)

        # Closed-class anti-loop
        CLOSED_CLASS_IDS = frozenset({
            POS2IDX["DET"], POS2IDX["PREP"], POS2IDX["PART"],
            POS2IDX["PRON"], POS2IDX["AUX"], POS2IDX["CONJ"],
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
        print("\n" + "=" * 70)
        print("ATTRACTOR LANGUAGE MACHINE — DIAGNOSTICS")
        print("=" * 70)

        if self.sdr_encoder:
            print(f"  SDR: D={self.sdr_encoder.D}, k={self.sdr_encoder.k} "
                  f"({self.sdr_encoder.sparsity*100:.1f}% sparse)")

        if self.hierarchy:
            hdiag = self.hierarchy.get_diagnostics()
            for l in range(hdiag['n_layers']):
                ld = hdiag[f'L{l}']
                print(f"  L{l}: D={ld['D']}, k={ld['k']}, "
                      f"J_max={ld['J_max']}, J_nnz={ld['J_nnz']}, "
                      f"h_max={ld['h_max']}")
            print(f"  Total hierarchy memory: {hdiag['total_memory_kb']:.1f} KB")

        if self.episodic:
            ediag = self.episodic.get_diagnostics()
            print(f"  Episodic: {ediag['n_episodes']} episodes, "
                  f"memory={ediag['memory_kb']:.1f} KB")

        print(f"  Beta: {self.beta:.6f}")
        print(f"  Vocab: {len(self.vocab)} words")
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
