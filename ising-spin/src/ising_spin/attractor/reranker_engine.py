"""
Integer EBM Re-ranker Engine v76h.

ARCHITECTURAL FIXES from v76g:

  FLAW 1 FIXED: GPT-2 BPE-to-word misalignment.
  v76g passed word-level IDs directly to GPT-2, which interprets them
  as BPE tokens — producing semantic garbage. v76h uses the new
  get_top_k_words() method that converts context to text → BPE tokens,
  runs GPT-2 properly, and returns word-level candidates. This:
    - Eliminates <unk> tokens in generation
    - Gives GPT-2 proper semantic context for candidate generation
    - Makes base model log-probs actually meaningful

  FLAW 2 FIXED: Bigram energy scale mismatch.
  v76g scaled bigram energy by log_prob_scale * bigram_weight = 100*10 = 1000x,
  giving bigram values ~7700 that DOMINATED the z-score normalized DAM energy
  (~84 std). v76h:
    - Separates bigram from DAM-side energy (it's log-prob based, same scale
      as base_energies, not DAM energy)
    - Z-score normalizes bigram energy independently
    - Uses gamma parameter to mix bigram with base model
    - combined = base_energies + alpha * dam_side + gamma * bigram_scaled

  FLAW 3 FIXED: No repetition penalty.
  v76g had no mechanism to prevent repetition loops ("they in they in...").
  v76h adds a repetition penalty that increases the energy of candidates
  that appear in the recent N generated words.

Pipeline:
  1. Base model generates top-K WORD-LEVEL candidates + log-probabilities
  2. Spin state tracker updates from context (Z/X/Y bands)
  3. DAM discriminator scores each candidate: E = E_DAM + E_spin + E_episodic + E_bind
  4. Z-score normalize DAM-side energy, rescale to base model range
  5. Z-score normalize bigram energy independently
  6. Add repetition penalty for recent words
  7. Combined = base_energies + alpha * dam_side + gamma * bigram_scaled + rep_penalty
  8. Integer Boltzmann sampling from combined distribution

Training:
  NCE (Noise Contrastive Estimation) with 3 corruption types:
    - RANDOM_SUB: lexical coherence
    - POS_VIOLATE: grammatical structure
    - TOPIC_VIOLATE: semantic coherence
  Word swap is REMOVED (the DAM cannot learn it; bigram table handles order).
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple

from .dam import DAMLayer
from .sdr import SDREncoder
from .three_band import ThreeBandState
from .episodic import EpisodicMemory
from .binding import BindingContext
from .nce import NCETrainer
from .corruptions import Corruptor, CORRUPTION_NAMES
from .base_model import BaseLMInterface, DummyBaseLM
from ..sampling.boltzmann import IntegerBoltzmannSampler


# Scale factor: convert base model log-probs (nats) to integer energy scale
LOG_PROB_SCALE_DEFAULT = 100


class ReRankerEngine:
    """
    Integer EBM Re-ranker v76h.

    The DAM discriminates between correct and incorrect next-word
    candidates produced by a pretrained base model. DAM energies
    are z-score normalized before combining with base model log-probs.
    Bigram energies are z-score normalized independently.
    """

    def __init__(
        self,
        base_model,
        sdr_encoder: SDREncoder,
        vocab_words: List[str],
        word2idx: dict,
        idx2word: dict,
        pos_types: Optional[np.ndarray] = None,
        word_freq: Optional[np.ndarray] = None,
        # DAM parameters
        dam_scale: int = 1600,
        j_clip: int = 64000,  # v76h: increased from 32000
        f_type: int = DAMLayer.F_EXP_APPROX,
        exp_temperature: int = 100,
        # NCE parameters
        nce_eta: int = 10,
        nce_negatives: int = 3,  # v76g: 3 types only (no word_swap)
        nce_epochs: int = 1,
        context_window: int = 10,
        # Re-ranking parameters
        top_k: int = 50,
        log_prob_scale: int = LOG_PROB_SCALE_DEFAULT,
        dam_alpha: float = 1.0,  # v76g: mixing weight for normalized DAM energy
        gamma: float = 0.5,  # v76h: mixing weight for bigram energy
        spin_weight: int = 5,
        episodic_weight: int = 2,
        bind_weight: int = 5,
        # Repetition penalty
        rep_penalty: float = 50.0,  # v76h: energy penalty for repeated words
        rep_window: int = 5,  # v76h: how many recent words to penalize
        # UV regularization
        uv_regularize: bool = True,
        uv_lambda: int = 10,
        # Episodic memory
        max_episodes: int = 10000,
        # Seed
        seed: int = 42,
    ):
        self.base_model = base_model
        self.sdr_encoder = sdr_encoder
        self.vocab_words = vocab_words
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.V = len(vocab_words)

        # DAM discriminator (single layer, not hierarchical)
        D = sdr_encoder.D
        k = sdr_encoder.k
        self.dam = DAMLayer(
            D=D, k=k, scale=dam_scale, j_clip=j_clip,
            f_type=f_type, exp_temperature=exp_temperature,
            uv_regularize=uv_regularize, uv_lambda=uv_lambda,
            seed=seed,
        )

        # Spin state
        self.three_band = ThreeBandState(D=D)

        # Episodic memory
        self.episodic = EpisodicMemory(D=D, k=k, max_episodes=max_episodes)

        # Binding context
        self.binding = BindingContext(D=D, k=k)

        # Corruptor for NCE (v76g: no word_swap in training)
        self.corruptor = Corruptor(
            vocab_words=vocab_words,
            word2idx=word2idx,
            idx2word=idx2word,
            pos_types=pos_types,
            word_freq=word_freq,
            seed=seed,
        )

        # NCE trainer (v76g: 3 corruption types only, no word_swap)
        self.nce_trainer = NCETrainer(
            dam=self.dam,
            sdr_encoder=sdr_encoder,
            corruptor=self.corruptor,
            context_window=context_window,
            eta=nce_eta,
            j_clip=j_clip,
            uv_regularize=uv_regularize,
            uv_lambda=uv_lambda,
        )

        # Re-ranking parameters
        self.top_k = top_k
        self.log_prob_scale = log_prob_scale
        self.dam_alpha = dam_alpha
        self.gamma = gamma  # v76h: bigram mixing weight
        self.spin_weight = spin_weight
        self.episodic_weight = episodic_weight
        self.bind_weight = bind_weight
        self.dam_scale = dam_scale
        self.nce_epochs = nce_epochs
        self.nce_negatives = nce_negatives
        self.context_window = context_window
        self.seed = seed

        # v76h: Repetition penalty parameters
        self.rep_penalty = rep_penalty
        self.rep_window = rep_window

        # v76g: Bigram log-prob table for word order
        self._bigram_logprob: Optional[np.ndarray] = None  # (V, V) float64

        # v76g: Energy normalization statistics (computed after training)
        self._dam_energy_mean: float = 0.0
        self._dam_energy_std: float = 1.0
        self._base_energy_mean: float = 0.0
        self._base_energy_std: float = 1.0

        # v76h: Bigram energy normalization statistics
        self._bigram_energy_mean: float = 0.0
        self._bigram_energy_std: float = 1.0

        # Boltzmann sampler (will be calibrated after training)
        self.sampler = IntegerBoltzmannSampler(beta=0.01, max_delta=50000)
        self.rerank_beta = 0.01

        # v76h: Use word-level candidates (fixes <unk> problem)
        self._use_word_candidates = not isinstance(base_model, DummyBaseLM)

        # Diagnostics
        self._training_log = []

    def train(self, sequences: List[List[int]], callback=None) -> Dict:
        """
        Full NCE training pipeline.

        Args:
            sequences: List of tokenized sequences (word IDs).
            callback: Optional callback(epoch, stats).

        Returns:
            Dict with training statistics.
        """
        print(f"  NCE training: {len(sequences)} sequences, "
              f"{self.nce_epochs} epochs, {self.nce_negatives} negatives (no word_swap)")

        all_stats = []
        for epoch in range(self.nce_epochs):
            stats = self.nce_trainer.train_epoch(
                sequences=sequences,
                epoch=epoch,
                n_negatives=self.nce_negatives,
            )
            all_stats.append(stats)

            print(f"    Epoch {epoch+1}/{self.nce_epochs}: "
                  f"disc_acc={stats['disc_accuracy']:.3f}, "
                  f"energy_gap={stats['energy_gap']:.1f}, "
                  f"J_nnz={stats['J_nnz']}, "
                  f"J_max={stats['J_max']}, "
                  f"time={stats['time_s']:.1f}s")

            if callback:
                callback(epoch, stats)

        # Prune J matrix for sharper attractor basins
        self._prune_j()

        # Build bigram table from training data
        self._build_bigram_table(sequences)

        # Populate episodic memory
        self._populate_episodic_memory(sequences)

        # Calibrate energy normalization and re-ranking
        self._calibrate_reranking(sequences)

        return {
            'epochs': all_stats,
            'final_J_nnz': int(np.count_nonzero(self.dam.J)),
            'final_J_max': int(np.max(np.abs(self.dam.J))),
        }

    def _prune_j(self, target_density: float = 0.05):
        """Prune J matrix — keep strongest entries by absolute value."""
        J = self.dam.J
        D = self.dam.D
        total_entries = D * D
        current_nnz = int(np.count_nonzero(J))
        current_density = current_nnz / total_entries

        if current_density <= target_density:
            print(f"    J pruning: density {current_density:.3f} already ≤ "
                  f"target {target_density:.3f} — skipping")
            return

        target_nnz = max(1, int(target_density * total_entries))

        if current_nnz <= target_nnz:
            print(f"    J pruning: nnz {current_nnz} already ≤ "
                  f"target {target_nnz} — skipping")
            return

        abs_J = np.abs(J)
        flat_abs = abs_J.ravel()
        if target_nnz < len(flat_abs):
            kth = len(flat_abs) - target_nnz
            threshold = int(np.partition(flat_abs, kth)[kth])
        else:
            threshold = 0

        if threshold <= 0:
            print(f"    J pruning: all entries near zero, keeping as-is "
                  f"(J_nnz={current_nnz})")
            return

        prune_mask = (abs_J < threshold) & (abs_J > 0)
        J[prune_mask] = 0

        final_nnz = int(np.count_nonzero(J))
        final_density = final_nnz / total_entries
        print(f"    J pruning: density {current_density:.3f} → {final_density:.3f} "
              f"(threshold={threshold}, J_nnz={current_nnz}→{final_nnz})")

    def _build_bigram_table(self, sequences: List[List[int]]):
        """
        v76g: Build bigram log-probability table from training data.

        This is the explicit word-order model that replaces the DAM's
        failed attempt to learn word swaps. Bigrams directly capture
        "word A should come before word B" statistics.

        The table stores log P(next_word | prev_word) with Laplace
        smoothing. During re-ranking, the bigram energy for a candidate
        word is the negative log-probability given the previous word.
        """
        V = self.V
        print(f"    Building bigram table (V={V})...")

        # Count bigrams
        bigram_counts = np.zeros((V, V), dtype=np.int32)
        for seq in sequences:
            for i in range(1, len(seq)):
                prev = seq[i - 1]
                curr = seq[i]
                if 0 <= prev < V and 0 <= curr < V:
                    bigram_counts[prev, curr] += 1

        # Laplace smoothing + log-probability
        alpha_smooth = 1  # Laplace alpha
        row_sums = bigram_counts.sum(axis=1, keepdims=True) + alpha_smooth * V
        self._bigram_logprob = np.log(
            (bigram_counts.astype(np.float64) + alpha_smooth) / row_sums.astype(np.float64)
        )

        n_nonzero = int(np.sum(bigram_counts > 0))
        print(f"    Bigram table: {n_nonzero} non-zero pairs, "
              f"mean_logprob={np.mean(self._bigram_logprob[self._bigram_logprob > -20]):.3f}")

    def _populate_episodic_memory(self, sequences: List[List[int]]):
        """Store SDR patterns in episodic memory."""
        count = 0
        for seq in sequences:
            for word_id in seq:
                if word_id < 0 or word_id >= self.V:
                    continue
                sdr = self.sdr_encoder.encode(word_id)
                self.episodic.store(sdr)
                count += 1
                if count >= self.episodic.max_episodes:
                    break
            if count >= self.episodic.max_episodes:
                break
        print(f"    Episodic memory: {count} episodes stored")

    def _calibrate_reranking(self, sequences: List[List[int]]):
        """
        v76h: Calibrate re-ranking with z-score normalized energies.

        Three-step calibration:
        1. Compute energy statistics for z-score normalization
           (DAM energy, base energy, AND bigram energy)
        2. Grid search over alpha (DAM mixing weight), gamma (bigram weight),
           and beta (Boltzmann temp)

        With z-score normalization, alpha and gamma are on meaningful scales:
          alpha=0 → no DAM influence
          alpha=1 → DAM influence equal to base model std
          gamma=0 → no bigram influence
          gamma=1 → bigram influence equal to base model std
        """
        print("    Calibrating re-ranking (z-score normalization)...")

        # Step 1: Collect energy samples for normalization statistics
        dam_energy_samples = []
        base_energy_samples = []
        bigram_energy_samples = []

        for seq in sequences[:300]:
            if len(seq) < 3:
                continue
            for pos in range(1, min(len(seq), 5)):
                ctx = seq[:pos]
                target = seq[pos]

                # Get candidates — use word-level if available
                if self._use_word_candidates and hasattr(self.base_model, 'get_top_k_words'):
                    candidates, log_probs = self.base_model.get_top_k_words(
                        ctx, k=min(self.top_k, self.V)
                    )
                else:
                    candidates, log_probs = self.base_model.get_top_k(
                        ctx, k=min(self.top_k, self.V)
                    )

                # Compute DAM energy for target
                ctx_sdr = self.sdr_encoder.encode_context_positional(
                    ctx[-self.context_window:]
                )
                context_field = self.dam.compute_field(ctx_sdr)
                if 0 <= target < self.V:
                    tgt_sdr = self.sdr_encoder.encode(target)
                    active = np.where(tgt_sdr > 0)[0]
                    dam_e = -int(np.sum(context_field[active]))
                    dam_energy_samples.append(dam_e)

                # Compute base energy for all candidates
                base_e = -(log_probs * self.log_prob_scale).astype(np.int64)
                base_energy_samples.extend(base_e.tolist())

                # Compute bigram energy samples
                if len(ctx) > 0:
                    prev_word = ctx[-1]
                    if 0 <= prev_word < self.V and self._bigram_logprob is not None:
                        for cand_id in candidates[:10]:  # Sample 10 candidates
                            if 0 <= cand_id < self.V:
                                lp = self._bigram_logprob[prev_word, cand_id]
                                bigram_e = -lp  # v76h: raw negative log-prob (no scale)
                                bigram_energy_samples.append(bigram_e)

        if not dam_energy_samples:
            self._dam_energy_mean = 0.0
            self._dam_energy_std = 1.0
            self._base_energy_mean = 0.0
            self._base_energy_std = 1.0
            self._bigram_energy_mean = 0.0
            self._bigram_energy_std = 1.0
            self.rerank_beta = 0.01
            print(f"    Using defaults: alpha={self.dam_alpha}, "
                  f"gamma={self.gamma}, beta={self.rerank_beta}")
            return

        # Compute normalization statistics
        self._dam_energy_mean = float(np.mean(dam_energy_samples))
        self._dam_energy_std = max(1.0, float(np.std(dam_energy_samples)))
        self._base_energy_mean = float(np.mean(base_energy_samples))
        self._base_energy_std = max(1.0, float(np.std(base_energy_samples)))

        # v76h: Bigram normalization stats
        if bigram_energy_samples:
            self._bigram_energy_mean = float(np.mean(bigram_energy_samples))
            self._bigram_energy_std = max(0.1, float(np.std(bigram_energy_samples)))
        else:
            self._bigram_energy_mean = 7.7  # Typical -log P for bigrams
            self._bigram_energy_std = 2.0

        print(f"    Energy stats: DAM mean={self._dam_energy_mean:.1f} "
              f"std={self._dam_energy_std:.1f}, "
              f"Base mean={self._base_energy_mean:.1f} "
              f"std={self._base_energy_std:.1f}")
        print(f"    Bigram stats: mean={self._bigram_energy_mean:.3f} "
              f"std={self._bigram_energy_std:.3f}")

        # Step 2: Grid search over alpha, gamma, and beta
        test_pairs = []
        for seq in sequences[:200]:
            if len(seq) < 3:
                continue
            for pos in range(1, min(len(seq), 5)):
                test_pairs.append((seq[:pos], seq[pos]))

        if not test_pairs:
            self.rerank_beta = 0.01
            print(f"    Using default beta={self.rerank_beta}")
            return

        # Precompute re-ranking data
        rerank_data = []
        for ctx, target in test_pairs[:300]:
            if self._use_word_candidates and hasattr(self.base_model, 'get_top_k_words'):
                candidates, log_probs = self.base_model.get_top_k_words(
                    ctx, k=min(self.top_k, self.V)
                )
            else:
                candidates, log_probs = self.base_model.get_top_k(
                    ctx, k=min(self.top_k, self.V)
                )

            if target not in candidates:
                continue
            target_idx = int(np.where(candidates == target)[0][0])
            rerank_data.append((ctx, candidates, log_probs, target_idx))

        if not rerank_data:
            self.rerank_beta = 0.01
            print(f"    Using default beta={self.rerank_beta}")
            return

        best_alpha = self.dam_alpha
        best_gamma = self.gamma
        best_beta = 0.01
        best_acc = 0.0

        for alpha in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
            for gamma in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
                for beta in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
                    correct = 0
                    total = 0
                    for ctx, candidates, log_probs, target_idx in rerank_data:
                        # Temporarily override alpha and gamma
                        orig_alpha = self.dam_alpha
                        orig_gamma = self.gamma
                        self.dam_alpha = alpha
                        self.gamma = gamma
                        combined_energies, _ = self.rerank(ctx, candidates, log_probs)
                        self.dam_alpha = orig_alpha
                        self.gamma = orig_gamma

                        if len(combined_energies) == 0:
                            continue

                        # Boltzmann-weighted selection
                        E_min = np.min(combined_energies)
                        deltas = (combined_energies - E_min).astype(np.float64)
                        deltas = np.clip(deltas, 0, 700)
                        weights = np.exp(-deltas * beta)
                        total_weight = np.sum(weights)
                        if total_weight <= 0:
                            continue

                        chosen_idx = int(np.argmax(weights))
                        if chosen_idx == target_idx:
                            correct += 1
                        total += 1

                    acc = correct / max(1, total)
                    if acc > best_acc:
                        best_acc = acc
                        best_alpha = alpha
                        best_gamma = gamma
                        best_beta = beta

        self.dam_alpha = best_alpha
        self.gamma = best_gamma
        self.rerank_beta = best_beta
        self.sampler = IntegerBoltzmannSampler(beta=best_beta, max_delta=50000)
        print(f"    Calibrated: alpha={best_alpha:.1f}, gamma={best_gamma:.1f}, "
              f"beta={best_beta}, rerank_acc={best_acc:.3f} ({len(rerank_data)} pairs)")

    def _compute_bigram_energy(
        self,
        context_word_ids: List[int],
        candidates: np.ndarray,
    ) -> np.ndarray:
        """
        v76h: Compute bigram log-probability energy for each candidate.

        The bigram energy is the negative log P(candidate | prev_word).
        Lower energy = more likely bigram = better word order.

        v76h CHANGE: Energy is in natural log units (NOT scaled by
        log_prob_scale). It gets z-score normalized independently
        before combining with base_energies.
        """
        K = len(candidates)
        bigram_energies = np.zeros(K, dtype=np.float64)

        if self._bigram_logprob is None or len(context_word_ids) == 0:
            return bigram_energies

        prev_word = context_word_ids[-1]
        if prev_word < 0 or prev_word >= self.V:
            return bigram_energies

        for i, cand_id in enumerate(candidates):
            if cand_id < 0 or cand_id >= self.V:
                bigram_energies[i] = self._bigram_energy_mean
                continue
            # v76h: Raw negative log-prob — NOT scaled by log_prob_scale
            lp = self._bigram_logprob[prev_word, cand_id]
            bigram_energies[i] = -lp

        return bigram_energies

    def _compute_rep_penalty(
        self,
        candidates: np.ndarray,
        recent_words: List[int],
    ) -> np.ndarray:
        """
        v76h: Compute repetition penalty for each candidate.

        Candidates that appear in the recent N generated words get
        an energy penalty proportional to how recently they appeared.
        This prevents loops like "they in they in they in...".
        """
        K = len(candidates)
        penalties = np.zeros(K, dtype=np.int64)

        if self.rep_penalty <= 0 or len(recent_words) == 0:
            return penalties

        # Only penalize the last rep_window words
        recent = recent_words[-self.rep_window:]
        for i, cand_id in enumerate(candidates):
            for j, prev_word in enumerate(recent):
                if cand_id == prev_word:
                    # More recent repetitions get higher penalty
                    recency = len(recent) - j  # 1 = most recent
                    penalties[i] += int(self.rep_penalty * recency / len(recent))

        return penalties

    def rerank(
        self,
        context_word_ids: List[int],
        candidates: np.ndarray,
        base_log_probs: np.ndarray,
        recent_words: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Re-rank candidates using DAM discriminator + bigram model.

        v76h KEY CHANGES:
        1. Bigram energy is z-score normalized INDEPENDENTLY from DAM
           (it's log-prob based, same units as base_energies, not DAM energy)
        2. Repetition penalty added for recent words
        3. Combined = base + alpha*dam_side + gamma*bigram_scaled + rep_penalty

        Args:
            context_word_ids: Context word IDs.
            candidates: Candidate word IDs, shape (K,).
            base_log_probs: Base model log-probabilities, shape (K,).
            recent_words: v76h: Recently generated words for rep penalty.

        Returns:
            (energies, dam_energies) — combined energies and DAM-side energies.
        """
        K = len(candidates)
        if K == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        # 1. Encode context (positional VSA)
        ctx_window = context_word_ids[-self.context_window:]
        ctx_sdr = self.sdr_encoder.encode_context_positional(
            ctx_window, context_window=self.context_window
        )

        # 2. Compute context field for DAM
        context_field = self.dam.compute_field(ctx_sdr)

        # 3. Update spin state
        self.three_band.update(self.dam.state)

        # 4. Compute DAM energy for each candidate (LINEAR field sum)
        dam_energies = np.zeros(K, dtype=np.int64)
        for i, cand_id in enumerate(candidates):
            if cand_id < 0 or cand_id >= self.V:
                dam_energies[i] = 0
                continue
            cand_sdr = self.sdr_encoder.encode(cand_id)
            active = np.where(cand_sdr > 0)[0]
            dam_energies[i] = -int(np.sum(context_field[active]))

        # 5. Z-score normalize DAM energies, rescale to base energy std
        dam_float = dam_energies.astype(np.float64)
        if self._dam_energy_std > 0:
            dam_norm = (dam_float - self._dam_energy_mean) / self._dam_energy_std
        else:
            dam_norm = np.zeros_like(dam_float)
        dam_scaled = (dam_norm * self._base_energy_std).astype(np.int64)

        # 6. Compute spin energy for each candidate
        spin_energies = np.zeros(K, dtype=np.int64)
        if self.spin_weight > 0:
            for i, cand_id in enumerate(candidates):
                if cand_id < 0 or cand_id >= self.V:
                    continue
                cand_sdr = self.sdr_encoder.encode(cand_id)
                spin_e = self._compute_spin_energy(cand_sdr)
                spin_energies[i] = spin_e

        # 7. Compute episodic energy
        ep_energies = np.zeros(K, dtype=np.int64)
        if self.episodic_weight > 0 and len(self.episodic.episodes) > 0:
            for i, cand_id in enumerate(candidates):
                if cand_id < 0 or cand_id >= self.V:
                    continue
                cand_sdr = self.sdr_encoder.encode(cand_id)
                active = np.where(cand_sdr > 0)[0]
                ep_overlap = int(np.sum(self.episodic.episode_counts[active]))
                ep_energies[i] = -ep_overlap

        # 8. Compute binding energy
        bind_energies = np.zeros(K, dtype=np.int64)
        if self.bind_weight > 0 and len(self.binding._recent_words) > 0:
            bind_energies = self.binding.compute_binding_energy(
                candidates.astype(np.int64), self.sdr_encoder
            )

        # 9. DAM-side energy: DAM + spin + episodic + binding (NO bigram)
        dam_side = (dam_scaled
                    + spin_energies * self.spin_weight
                    + ep_energies * self.episodic_weight
                    + bind_energies * self.bind_weight)

        # 10. v76h: Bigram energy — z-score normalized INDEPENDENTLY
        # Bigram is log-prob based (like base_energies), not DAM energy.
        # We normalize it to base_energy_std scale so gamma is meaningful.
        bigram_raw = self._compute_bigram_energy(context_word_ids, candidates)
        if self._bigram_energy_std > 0:
            bigram_norm = (bigram_raw - self._bigram_energy_mean) / self._bigram_energy_std
        else:
            bigram_norm = np.zeros_like(bigram_raw)
        bigram_scaled = (bigram_norm * self._base_energy_std).astype(np.int64)

        # 11. v76h: Repetition penalty
        rep_penalties = np.zeros(K, dtype=np.int64)
        if recent_words is not None and len(recent_words) > 0:
            rep_penalties = self._compute_rep_penalty(candidates, recent_words)

        # 12. Convert base model log-probs to integer energy scale
        base_energies = -(base_log_probs * self.log_prob_scale).astype(np.int64)

        # 13. v76h: Combined energy
        # base + alpha * DAM-side + gamma * bigram + rep_penalty
        combined = (base_energies
                    + (dam_side * self.dam_alpha).astype(np.int64)
                    + (bigram_scaled * self.gamma).astype(np.int64)
                    + rep_penalties)

        return combined, dam_side.astype(np.float64)

    def _rerank_index(
        self,
        context_word_ids: List[int],
        candidates: np.ndarray,
        base_log_probs: np.ndarray,
        beta: float,
        recent_words: Optional[List[int]] = None,
    ) -> int:
        """Get the index of the best candidate after re-ranking."""
        combined, _ = self.rerank(context_word_ids, candidates, base_log_probs,
                                  recent_words=recent_words)
        if len(combined) == 0:
            return 0
        return int(np.argmin(combined))

    def _compute_spin_energy(self, candidate_sdr: np.ndarray) -> int:
        """
        Compute spin-field energy correction for a candidate word.
        """
        D = self.dam.D
        active = np.where(candidate_sdr > 0)[0]
        if len(active) == 0:
            return 0

        z_vals = self.three_band.m_z[:D]
        x_vals = self.three_band.m_x[:D]
        y_vals = self.three_band.m_y[:D]

        z_overlap = int(np.sum(z_vals[active]))
        x_overlap = int(np.sum(x_vals[active]))
        y_overlap = int(np.sum(y_vals[active]))

        spin_score = (z_overlap * 3 + x_overlap * 1 + y_overlap * 2) // 10
        spin_score = max(-10000, min(10000, spin_score))
        return -spin_score

    def generate(
        self,
        prompt_ids: List[int],
        length: int = 200,
        temperature: float = 1.0,
    ) -> List[int]:
        """
        Generate text using base model + DAM re-ranking.

        v76h KEY CHANGES:
        1. Uses get_top_k_words() for GPT-2 (word-level candidates)
        2. All candidates are valid word IDs (no <unk>)
        3. Repetition penalty prevents loops
        4. Properly filtered candidates (only word IDs in [4, V))

        Pipeline per step:
          1. Base model generates top-K WORD-LEVEL candidates + log-probs
          2. DAM discriminator scores each candidate
          3. Bigram model scores word-order likelihood
          4. Repetition penalty added for recent words
          5. Combined energy determines ranking
          6. Boltzmann sample from combined distribution
        """
        generated = list(prompt_ids)

        # Reset spin state
        self.three_band.full_reset()
        self.episodic.reset()
        self.binding.reset()

        # Initialize spin state from prompt
        for i, word_id in enumerate(generated):
            if word_id < 0 or word_id >= self.V:
                continue
            sdr = self.sdr_encoder.encode(word_id)
            self.three_band.update(sdr)
            self.episodic.store(sdr)
            self.binding.add_word(self.sdr_encoder.word_active_bits[word_id])

        for step in range(length):
            # Get top-K candidates — use word-level if available
            if self._use_word_candidates and hasattr(self.base_model, 'get_top_k_words'):
                candidates, log_probs = self.base_model.get_top_k_words(
                    generated, k=min(self.top_k, self.V)
                )
            else:
                candidates, log_probs = self.base_model.get_top_k(
                    generated, k=min(self.top_k, self.V)
                )

            if len(candidates) == 0:
                break

            # Filter to valid word IDs only (no special tokens, no OOV)
            valid_mask = (candidates >= 4) & (candidates < self.V)
            if not np.any(valid_mask):
                break
            candidates = candidates[valid_mask]
            log_probs = log_probs[valid_mask]

            # Re-rank with DAM + bigram + repetition penalty
            combined_energies, dam_energies = self.rerank(
                generated, candidates, log_probs,
                recent_words=generated[-self.rep_window:]
            )

            # Boltzmann sample from combined energies
            if len(combined_energies) > 1:
                idx = self.sampler.sample(combined_energies)
            else:
                idx = 0

            chosen_id = int(candidates[idx])
            generated.append(chosen_id)

            # Update spin state, episodic, binding
            if 0 <= chosen_id < self.V:
                sdr = self.sdr_encoder.encode(chosen_id)
                self.three_band.update(sdr)
                self.episodic.store(sdr)
                self.binding.add_word(
                    self.sdr_encoder.word_active_bits[chosen_id]
                )

        return generated

    def compute_perplexity(
        self,
        sequences: List[List[int]],
        n_samples: int = 100,
    ) -> Dict:
        """
        Compute both base model PPL and re-ranked PPL.
        """
        base_log_probs = []
        reranked_log_probs = []
        n_tokens = 0

        for seq in sequences[:n_samples]:
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                ctx = seq[:pos]
                target = seq[pos]

                # Use word-level candidates if available
                if self._use_word_candidates and hasattr(self.base_model, 'get_top_k_words'):
                    candidates, log_probs = self.base_model.get_top_k_words(
                        ctx, k=min(self.top_k, self.V)
                    )
                else:
                    candidates, log_probs = self.base_model.get_top_k(
                        ctx, k=min(self.top_k, self.V)
                    )

                target_mask = candidates == target
                if not np.any(target_mask):
                    # Target not in candidates — use base model log-prob
                    if hasattr(self.base_model, 'compute_word_sequence_log_prob'):
                        base_lp = (self.base_model.compute_word_sequence_log_prob(ctx + [target])
                                   - self.base_model.compute_word_sequence_log_prob(ctx))
                    else:
                        base_lp = self.base_model.compute_sequence_log_prob(
                            ctx + [target]
                        ) - self.base_model.compute_sequence_log_prob(ctx)
                    base_log_probs.append(base_lp)
                    reranked_log_probs.append(base_lp)
                else:
                    target_idx = np.where(target_mask)[0][0]
                    base_lp = log_probs[target_idx]
                    base_log_probs.append(base_lp)

                    combined_energies, _ = self.rerank(ctx, candidates, log_probs)

                    E_min = np.min(combined_energies)
                    deltas = (combined_energies - E_min).astype(np.float64)
                    deltas = np.clip(deltas, 0, 700)
                    weights = np.exp(-deltas * self.rerank_beta)
                    total_weight = np.sum(weights)
                    if total_weight > 0:
                        target_prob = weights[target_idx] / total_weight
                        if target_prob > 1e-300:
                            reranked_log_probs.append(np.log(target_prob))
                        else:
                            reranked_log_probs.append(-690)
                    else:
                        reranked_log_probs.append(base_lp)

                n_tokens += 1

        base_ppl = np.exp(-np.mean(base_log_probs)) if base_log_probs else float('inf')
        reranked_ppl = np.exp(-np.mean(reranked_log_probs)) if reranked_log_probs else float('inf')

        return {
            'base_ppl': base_ppl,
            'reranked_ppl': reranked_ppl,
            'n_tokens': n_tokens,
            'n_sequences': min(n_samples, len(sequences)),
        }

    def compute_discriminative_accuracy(
        self,
        sequences: List[List[int]],
        n_samples: int = 1000,
    ) -> Dict:
        """
        How often does the DAM assign LOWER energy to correct vs corrupted?
        """
        correct = 0
        total = 0
        type_correct = {}
        type_total = {}

        n_pairs = 0
        for seq in sequences[:n_samples]:
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                ctx = seq[:pos]
                target = seq[pos]

                # Positive energy
                ctx_sdr = self.sdr_encoder.encode_context_positional(
                    ctx[-self.context_window:]
                )
                tgt_sdr = self.sdr_encoder.encode(target)
                field = self.dam.compute_field(ctx_sdr)
                active = np.where(tgt_sdr > 0)[0]
                pos_energy = -int(np.sum(field[active]))

                # Generate corruptions
                negatives = self.corruptor.generate_negatives(
                    ctx, target, n_negatives=4
                )

                for neg_ctx, neg_cand, ctype in negatives:
                    neg_ctx_sdr = self.sdr_encoder.encode_context_positional(
                        neg_ctx[-self.context_window:]
                    )
                    neg_tgt_sdr = self.sdr_encoder.encode(neg_cand)
                    neg_field = self.dam.compute_field(neg_ctx_sdr)
                    neg_active = np.where(neg_tgt_sdr > 0)[0]
                    neg_energy = -int(np.sum(neg_field[neg_active]))

                    if pos_energy < neg_energy:
                        correct += 1
                        type_correct[ctype] = type_correct.get(ctype, 0) + 1
                    total += 1
                    type_total[ctype] = type_total.get(ctype, 0) + 1

                n_pairs += 1
                if n_pairs >= n_samples:
                    break
            if n_pairs >= n_samples:
                break

        result = {
            'overall_accuracy': correct / max(1, total),
            'total_comparisons': total,
        }

        for ctype in sorted(type_total.keys()):
            name = CORRUPTION_NAMES.get(ctype, f"type_{ctype}")
            acc = type_correct.get(ctype, 0) / max(1, type_total[ctype])
            result[f"{name}_accuracy"] = acc

        return result
