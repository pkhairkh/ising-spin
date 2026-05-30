"""
Integer EBM Re-ranker Engine v76f.

ARCHITECTURAL PIVOT from v75:
  The DAM is no longer a standalone language model. It is a DISCRIMINATOR
  that re-ranks candidates from a frozen neural LM (GPT-2) or DummyBaseLM.

  "Base model proposes, discriminator disposes."

Pipeline:
  1. Base model generates top-K candidates + log-probabilities
  2. Spin state tracker updates from context (Z/X/Y bands)
  3. DAM discriminator scores each candidate: E = E_DAM + E_spin + E_episodic + E_bind
  4. Combined energy = base_log_prob * LOG_PROB_SCALE + DAM_side_energy
  5. Integer Boltzmann sampling from combined distribution

Training:
  NCE (Noise Contrastive Estimation) with 4 corruption types.
  No PCD. No n-gram matrices. No POS cascade.

v76f CRITICAL FIXES:
  Bug 14: "a a a a" degenerate repetition caused by double-counting word
  frequency. The DAM's bias h encodes word frequency (frequent words get
  large positive h at their active SDR bits → lower energy). The base model
  ALSO favors frequent words via log-probs. Result: both systems push
  toward the same frequent words, creating runaway repetition.
  FIX: Use context-only DAM energy E = -s^T(J@ctx) for re-ranking,
  NOT the full energy E = -s^T(J@ctx + h). The h term is the marginal
  (frequency prior); the J@ctx term is the conditional (context signal).
  The base model already provides the frequency prior, so the DAM should
  only add context-dependent information. This eliminates double-counting.

  Bug 15: Calibration uses np.argmax(Boltzmann_weights) which is equivalent
  to np.argmin(energies) regardless of beta — beta never affects the result.
  FIX: Calibrate using log-likelihood of the target word under the
  Boltzmann distribution. This properly optimizes beta because higher beta
  sharpens the distribution (which can help or hurt the target's probability).

  Bug 16: No repetition penalty during generation. Even with context-only
  energy, short loops like "a a a" can form because the base model's top-K
  is dominated by frequent words and the DAM's context signal is weak.
  FIX: Add configurable repetition penalty that increases energy for tokens
  that appeared in the last N positions.

What was REMOVED from v75:
  - All n-gram matrices (J2, J3, skip bigram)
  - POS trigram cascade and grammar penalties
  - Frequency penalty, repetition window, unseen n-gram penalty
  - PCD training
  - Hierarchical DAM (single DAMLayer as discriminator)
  - Weight calibration with Adam (only 2-3 params to calibrate)

What was KEPT from v75:
  - SDREncoder (word-level sparse binary codes)
  - DAMLayer (single layer, NCE-trained)
  - IntegerBoltzmannSampler (integer-only sampling)
  - ThreeBandState (Z/X/Y spin, now as context representation)
  - EpisodicMemory (long-range context retrieval)
  - BindingContext (VSA permutation for word order)
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
# GPT-2 log-probs are typically in [-15, 0]. Multiply by 100 to get [-1500, 0]
# which is comparable to DAM energies ([-500, 0] range).
LOG_PROB_SCALE_DEFAULT = 100


class ReRankerEngine:
    """
    Integer EBM Re-ranker v76f.

    The DAM discriminates between correct and incorrect next-word
    candidates produced by a pretrained base model.

    v76f: Context-only DAM energy + log-likelihood calibration +
    repetition penalty.
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
        j_clip: int = 32000,
        f_type: int = DAMLayer.F_EXP_APPROX,
        exp_temperature: int = 100,
        # NCE parameters
        nce_eta: int = 10,
        nce_negatives: int = 4,
        nce_epochs: int = 1,
        context_window: int = 10,
        # Re-ranking parameters
        top_k: int = 50,
        log_prob_scale: int = LOG_PROB_SCALE_DEFAULT,
        dam_weight_shift: int = 8,  # bit-shift for DAM energy scaling (>>8 = /256)
        spin_weight: int = 5,
        episodic_weight: int = 2,
        bind_weight: int = 5,
        # UV regularization
        uv_regularize: bool = True,
        uv_lambda: int = 10,
        # Word swap (v76e)
        n_word_swap: int = 3,
        word_swap_weight: int = 2,
        # Repetition penalty (v76f)
        repetition_penalty: int = 500,
        repetition_window: int = 3,
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

        # Corruptor for NCE
        self.corruptor = Corruptor(
            vocab_words=vocab_words,
            word2idx=word2idx,
            idx2word=idx2word,
            pos_types=pos_types,
            word_freq=word_freq,
            seed=seed,
        )

        # NCE trainer (v76e: extra word_swap params)
        self.nce_trainer = NCETrainer(
            dam=self.dam,
            sdr_encoder=sdr_encoder,
            corruptor=self.corruptor,
            context_window=context_window,
            eta=nce_eta,
            j_clip=j_clip,
            uv_regularize=uv_regularize,
            uv_lambda=uv_lambda,
            n_word_swap=n_word_swap,
            word_swap_weight=word_swap_weight,
        )

        # Re-ranking parameters
        self.top_k = top_k
        self.log_prob_scale = log_prob_scale
        self.dam_weight_shift = dam_weight_shift
        self.spin_weight = spin_weight
        self.episodic_weight = episodic_weight
        self.bind_weight = bind_weight
        self.dam_scale = dam_scale
        self.nce_epochs = nce_epochs
        self.nce_negatives = nce_negatives
        self.context_window = context_window
        self.seed = seed

        # v76f: Repetition penalty
        self.repetition_penalty = repetition_penalty
        self.repetition_window = repetition_window

        # Boltzmann sampler (will be calibrated after training)
        self.sampler = IntegerBoltzmannSampler(beta=0.01, max_delta=50000)
        self.rerank_beta = 0.01

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
              f"{self.nce_epochs} epochs, {self.nce_negatives} negatives")

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

        # Populate episodic memory
        self._populate_episodic_memory(sequences)

        # Calibrate re-ranking
        self._calibrate_reranking(sequences)

        return {
            'epochs': all_stats,
            'final_J_nnz': int(np.count_nonzero(self.dam.J)),
            'final_J_max': int(np.max(np.abs(self.dam.J))),
        }

    def _prune_j(self, target_density: float = 0.05):
        """Prune J matrix — keep strongest entries by absolute value.

        v76 FIX: The old median-based pruning zeroed EVERYTHING when
        UV regularization caused values to cluster. The median of
        clustered values ≥ max value → single iteration kills all entries.

        New approach: keep top-N entries by |J|, where N = target_density * D^2.
        This is deterministic, never over-prunes, and respects the actual
        distribution of coupling strengths.
        """
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

        # Find the threshold: keep the top-target_nnz entries by |J|
        abs_J = np.abs(J)
        # Get the value at the target_nnz-th largest position
        # partition is O(N) and doesn't sort the whole array
        flat_abs = abs_J.ravel()
        if target_nnz < len(flat_abs):
            # np.partition puts the target_nnz-th smallest at [target_nnz]
            # We want the target_nnz-th largest = (N - target_nnz)-th smallest
            kth = len(flat_abs) - target_nnz
            threshold = int(np.partition(flat_abs, kth)[kth])
        else:
            threshold = 0

        if threshold <= 0:
            # All entries are 0 or near-0; nothing worth pruning
            print(f"    J pruning: all entries near zero, keeping as-is "
                  f"(J_nnz={current_nnz})")
            return

        # Zero entries below threshold (but keep entries AT threshold
        # only if we need them to fill up to target_nnz)
        prune_mask = (abs_J < threshold) & (abs_J > 0)
        J[prune_mask] = 0

        final_nnz = int(np.count_nonzero(J))
        final_density = final_nnz / total_entries
        print(f"    J pruning: density {current_density:.3f} → {final_density:.3f} "
              f"(threshold={threshold}, J_nnz={current_nnz}→{final_nnz})")

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
        """Calibrate re-ranking beta and dam_weight_shift.

        v76f: Three major fixes over v76e:
        1. Use log-likelihood of the target under the Boltzmann distribution
           as the optimization target, NOT argmax (which ignores beta).
           argmax(weights) = argmin(energies) for ANY beta > 0, so beta
           never affected the result. Log-likelihood properly optimizes beta
           because sharper distributions (higher beta) can help or hurt.
        2. Search smaller shifts [0, 2, 4, 6, 8] since context-only
           DAM energy is much smaller than full energy (no h bias).
        3. Also track rerank accuracy (argmin) for reporting.
        """
        print("    Calibrating re-ranking...")

        # Sample test pairs
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

        # Precompute re-ranking data for all test pairs
        rerank_data = []
        for ctx, target in test_pairs[:300]:
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

        # Precompute combined energies for each shift value
        # (energies don't depend on beta, only on shift)
        shift_energies = {}
        for shift in [0, 2, 4, 6, 8, 10]:
            orig_shift = self.dam_weight_shift
            self.dam_weight_shift = shift
            energies_list = []
            for ctx, candidates, log_probs, target_idx in rerank_data:
                combined_energies, _ = self.rerank(ctx, candidates, log_probs)
                energies_list.append((combined_energies, target_idx))
            shift_energies[shift] = energies_list
            self.dam_weight_shift = orig_shift

        # Grid search over (shift, beta) using log-likelihood
        best_beta = 0.01
        best_shift = self.dam_weight_shift
        best_ll = -float('inf')
        best_acc = 0.0

        for shift in [0, 2, 4, 6, 8, 10]:
            for beta in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
                total_ll = 0.0
                correct = 0
                total = 0
                for combined_energies, target_idx in shift_energies[shift]:
                    if len(combined_energies) == 0:
                        continue

                    # Compute Boltzmann weights
                    E_min = np.min(combined_energies)
                    deltas = (combined_energies - E_min).astype(np.float64)
                    deltas = np.clip(deltas, 0, 700)
                    weights = np.exp(-deltas * beta)
                    total_weight = np.sum(weights)
                    if total_weight <= 0:
                        continue

                    # Log-likelihood of the target
                    target_prob = weights[target_idx] / total_weight
                    if target_prob > 1e-300:
                        total_ll += np.log(target_prob)
                    else:
                        total_ll += -690  # log(1e-300)

                    # Also track argmin accuracy for reporting
                    if int(np.argmin(combined_energies)) == target_idx:
                        correct += 1
                    total += 1

                acc = correct / max(1, total)
                # Select by log-likelihood (proper Bayesian optimization)
                if total_ll > best_ll:
                    best_ll = total_ll
                    best_beta = beta
                    best_shift = shift
                    best_acc = acc

        self.dam_weight_shift = best_shift
        self.rerank_beta = best_beta
        self.sampler = IntegerBoltzmannSampler(beta=best_beta, max_delta=50000)
        print(f"    Calibrated: beta={best_beta}, "
              f"dam_weight_shift={best_shift} (>>{best_shift} = /{2**best_shift}), "
              f"rerank_acc={best_acc:.3f} ({len(rerank_data)} pairs), "
              f"log_likelihood={best_ll:.1f}")

    def rerank(
        self,
        context_word_ids: List[int],
        candidates: np.ndarray,
        base_log_probs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Re-rank candidates using DAM discriminator.

        Args:
            context_word_ids: Context word IDs.
            candidates: Candidate word IDs, shape (K,).
            base_log_probs: Base model log-probabilities, shape (K,).

        Returns:
            (energies, re_ranked_probs) — combined energies and
            Boltzmann probabilities for each candidate.
        """
        K = len(candidates)
        if K == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        # 1. Encode context (positional VSA, same as v75)
        ctx_window = context_word_ids[-self.context_window:]
        ctx_sdr = self.sdr_encoder.encode_context_positional(
            ctx_window, context_window=self.context_window
        )

        # 2. Compute context field for DAM
        context_field = self.dam.compute_field(ctx_sdr)  # J@ctx + h

        # v76f FIX: Use context-only DAM energy (remove frequency bias from h).
        # The base model already provides the frequency prior via log-probs.
        # The DAM's h encodes the same frequency prior, causing double-counting
        # that makes frequent words like "a" get even lower energy, leading to
        # degenerate repetition ("a a a a"). By using only the context-dependent
        # part (J@ctx), the DAM provides ONLY information the base model lacks:
        # which words fit the current CONTEXT, not which words are common overall.
        #
        # Math: E_full = -s^T(J@ctx + h)  vs  E_context = -s^T(J@ctx)
        # E_context = E_full + s^T h  (removing the bias contribution)
        # The h term is constant across contexts but varies across candidates
        # (different active bits → different s^T h). Removing it eliminates
        # the frequency bias while keeping all context-dependent signal.
        dam_bias = self.dam.h.astype(np.int32)  # h alone
        context_only_field = context_field - dam_bias  # = J@ctx only

        # 3. Update spin state
        self.three_band.update(self.dam.state)

        # 4. Compute DAM energy for each candidate (LINEAR energy, no F)
        # v76b FIX: The F(piecewise_exp) function is designed for attractor
        # dynamics, NOT for ranking. F(5000) = 2^50 makes every candidate
        # get infinite energy — zero discrimination. For re-ranking, we use
        # the LINEAR field sum: E = -sum(field[active]). This is consistent
        # with the discriminative accuracy metric used during training.
        # v76f: Using context_only_field (J@ctx) instead of full field (J@ctx + h).
        dam_energies = np.zeros(K, dtype=np.int64)
        for i, cand_id in enumerate(candidates):
            if cand_id < 0 or cand_id >= self.V:
                dam_energies[i] = 0
                continue
            cand_sdr = self.sdr_encoder.encode(cand_id)
            active = np.where(cand_sdr > 0)[0]
            # Context-only energy: E = -sum(J@ctx)[active]
            dam_energies[i] = -int(np.sum(context_only_field[active]))

        # 5. Compute spin energy for each candidate
        spin_energies = np.zeros(K, dtype=np.int64)
        if self.spin_weight > 0:
            for i, cand_id in enumerate(candidates):
                if cand_id < 0 or cand_id >= self.V:
                    continue
                cand_sdr = self.sdr_encoder.encode(cand_id)
                spin_e = self._compute_spin_energy(cand_sdr)
                spin_energies[i] = spin_e

        # 6. Compute episodic energy
        # Use episode_counts as a fast overlap proxy: overlap = sum(counts[active_bits])
        ep_energies = np.zeros(K, dtype=np.int64)
        if self.episodic_weight > 0 and len(self.episodic.episodes) > 0:
            for i, cand_id in enumerate(candidates):
                if cand_id < 0 or cand_id >= self.V:
                    continue
                cand_sdr = self.sdr_encoder.encode(cand_id)
                active = np.where(cand_sdr > 0)[0]
                ep_overlap = int(np.sum(self.episodic.episode_counts[active]))
                ep_energies[i] = -ep_overlap  # Higher overlap = lower energy

        # 7. Compute binding energy (uses existing binding interface)
        bind_energies = np.zeros(K, dtype=np.int64)
        if self.bind_weight > 0 and len(self.binding._recent_words) > 0:
            bind_energies = self.binding.compute_binding_energy(
                candidates.astype(np.int64), self.sdr_encoder
            )

        # 8. Total DAM-side energy (with scaling for balance)
        # v76f: Context-only dam_energies are SMALLER than full energies
        # (no h bias). Range is roughly [-100000, +50000] instead of
        # [-320000, 0]. The sign can be positive (some candidates have
        # POSITIVE context-only field → they DON'T fit the context).
        # Scale DAM down by >>dam_weight_shift to balance with base model.
        # With context-only energy, smaller shifts (0-6) are typically best.
        dam_scaled = dam_energies >> self.dam_weight_shift
        total_dam = (dam_scaled
                    + spin_energies * self.spin_weight
                    + ep_energies * self.episodic_weight
                    + bind_energies * self.bind_weight)

        # 9. Convert base model log-probs to integer energy scale
        base_energies = -(base_log_probs * self.log_prob_scale).astype(np.int64)

        # 10. Combined energy: base model + DAM discriminator
        combined = base_energies + total_dam

        return combined, total_dam

    def _rerank_index(
        self,
        context_word_ids: List[int],
        candidates: np.ndarray,
        base_log_probs: np.ndarray,
        beta: float,
    ) -> int:
        """Get the index of the best candidate after re-ranking."""
        combined, _ = self.rerank(context_word_ids, candidates, base_log_probs)
        if len(combined) == 0:
            return 0
        # Greedy: pick lowest energy (most probable)
        return int(np.argmin(combined))

    def _compute_spin_energy(self, candidate_sdr: np.ndarray) -> int:
        """
        Compute spin-field energy correction for a candidate word.

        Uses the three-band spin state as a structured context representation.
        Lower energy = better fit with current spin state.
        """
        D = self.dam.D
        active = np.where(candidate_sdr > 0)[0]
        if len(active) == 0:
            return 0

        # Compute overlap with each spin band's nonzero entries
        z_vals = self.three_band.m_z[:D]
        x_vals = self.three_band.m_x[:D]
        y_vals = self.three_band.m_y[:D]

        # Sum spin values at candidate's active positions
        z_overlap = int(np.sum(z_vals[active]))
        x_overlap = int(np.sum(x_vals[active]))
        y_overlap = int(np.sum(y_vals[active]))

        # Weighted spin energy: negative = better alignment
        # Use int32-safe arithmetic (spin values are typically <100)
        spin_score = (z_overlap * 3 + x_overlap * 1 + y_overlap * 2) // 10
        # Clamp to prevent overflow
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

        Pipeline per step:
          1. Base model generates top-K candidates + log-probs
          2. DAM discriminator scores each candidate
          3. Combined energy determines the final ranking
          4. Boltzmann sample from combined distribution

        Args:
            prompt_ids: Initial context word IDs.
            length: Number of tokens to generate.
            temperature: Base model temperature (not used for DAM).

        Returns:
            List of generated word IDs (including prompt).
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
            # Get top-K candidates from base model
            candidates, log_probs = self.base_model.get_top_k(
                generated, k=min(self.top_k, self.V)
            )

            if len(candidates) == 0:
                break

            # Re-rank with DAM
            combined_energies, dam_energies = self.rerank(
                generated, candidates, log_probs
            )

            # v76f: Repetition penalty — penalize words that appeared
            # in recent context. Even with context-only DAM energy,
            # the system can fall into repetition loops because frequent
            # words are always top-ranked by the base model. Adding a
            # direct energy penalty for recently-used tokens breaks
            # these degenerate attractors.
            if self.repetition_penalty > 0 and len(generated) > 0:
                recent_ids = generated[-self.repetition_window:]
                for i, cand_id in enumerate(candidates):
                    count = sum(1 for rid in recent_ids if int(rid) == int(cand_id))
                    if count > 0:
                        combined_energies[i] += count * self.repetition_penalty

            # Boltzmann sample from combined energies
            if len(combined_energies) > 1:
                # Convert energies to weights: w = exp(-beta * E)
                E_min = np.min(combined_energies)
                deltas = (combined_energies - E_min).astype(np.int64)
                # Use IntegerBoltzmannSampler
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

        Args:
            sequences: Test sequences (word IDs).
            n_samples: Number of sequences to evaluate.

        Returns:
            Dict with base_ppl, reranked_ppl, n_tokens.
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

                # Base model log-prob for target
                candidates, log_probs = self.base_model.get_top_k(
                    ctx, k=min(self.top_k, self.V)
                )

                # Find target in candidates
                target_mask = candidates == target
                if not np.any(target_mask):
                    # Target not in top-K: use base model's log-prob directly
                    base_lp = self.base_model.compute_sequence_log_prob(
                        ctx + [target]
                    ) - self.base_model.compute_sequence_log_prob(ctx)
                    base_log_probs.append(base_lp)
                    # Can't re-rank what we can't see — use same value
                    reranked_log_probs.append(base_lp)
                else:
                    target_idx = np.where(target_mask)[0][0]
                    base_lp = log_probs[target_idx]
                    base_log_probs.append(base_lp)

                    # Re-rank and get probability of target
                    combined_energies, _ = self.rerank(
                        ctx, candidates, log_probs
                    )

                    # Convert energies to probabilities via softmax
                    # Use double-precision for numerical stability
                    E_min = np.min(combined_energies)
                    deltas = (combined_energies - E_min).astype(np.float64)
                    # Clip deltas to prevent overflow in exp
                    deltas = np.clip(deltas, 0, 700)  # exp(700) ≈ 1e304
                    weights = np.exp(-deltas * self.rerank_beta)
                    total_weight = np.sum(weights)
                    if total_weight > 0:
                        target_prob = weights[target_idx] / total_weight
                        if target_prob > 1e-300:
                            reranked_log_probs.append(np.log(target_prob))
                        else:
                            reranked_log_probs.append(-690)  # log(1e-300)
                    else:
                        reranked_log_probs.append(base_lp)

                n_tokens += 1

        # Compute PPL: PPL = exp(-avg_log_prob)
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

        This is the DIRECT measure of DAM quality.
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

                    if pos_energy < neg_energy:  # Lower = more likely
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

        # Per-type accuracy
        for ctype in sorted(type_total.keys()):
            name = CORRUPTION_NAMES.get(ctype, f"type_{ctype}")
            acc = type_correct.get(ctype, 0) / max(1, type_total[ctype])
            result[f"{name}_accuracy"] = acc

        return result
