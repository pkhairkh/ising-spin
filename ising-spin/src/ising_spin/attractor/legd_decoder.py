"""
Local Energy-Guided Decoding (LEGD) — Phase 2 of the Architectural Rethink.

v77: Kill the reranker. Make the energy a per-step guide.

The v76h reranker scores COMPLETE sequences, which is too late and too noisy.
LEGD computes LOCAL energy changes at each generation step:

  delta_E(s_t) = hash_energy(context[-1], s_t)
               + trigram_weight * hash_energy(context[-2], context[-1], s_t)

This is O(n_hashes) per candidate — typically 3 integer lookups.
Compare: v76h DAM needed O(D^2) matrix lookup + O(D) field computation.

KEY ADVANTAGE:
  Local delta_E has ~100x better SNR than global E because:
  1. We only look up the relevant pair/trigram entries
  2. No variance explosion from summing thousands of couplings
  3. The hash table was trained on EXACTLY these local decisions

DECODING FORMULA:
  P(s_t) proportional to P_base(s_t) * exp(-alpha * delta_E(s_t) / T)

Where:
  - P_base(s_t) = base model probability (GPT-2 or DummyBaseLM)
  - alpha = mixing weight (calibrated via grid search)
  - T = temperature (controls sharpness of energy correction)
  - delta_E(s_t) = local hash energy change (integer)

ADAPTIVE METROPOLIS GATE (hard rejection):
  If delta_E > rejection_threshold, HARD-REJECT the candidate.
  The hash energy acts as a discrete filter / bouncer, not a soft score.
  This is the Metropolis-Hastings rejection step, but without the accept
  side — we just pre-filter candidates that are clearly wrong.

REPETITION PENALTY:
  Carried over from v76h — still essential to prevent loops.

NO SDR, NO DAM, NO J-MATRIX, NO SPIN STATE, NO BINDING CONTEXT.
Just hash tables + base model + repetition penalty.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

from .hash_energy import HashEnergyTable
from ..sampling.boltzmann import IntegerBoltzmannSampler


class LEGDDecoder:
    """
    Local Energy-Guided Decoder.

    Replaces the entire ReRankerEngine with a simpler, faster pipeline:
      1. Base model generates top-K candidates
      2. Hash energy computes local delta_E for each candidate
      3. Metropolis gate hard-rejects candidates above threshold
      4. Combined score adjusts base model probabilities
      5. Sample from adjusted distribution

    No SDR encoding, no DAM field computation, no spin state updates.
    """

    def __init__(
        self,
        base_model,
        hash_energy: HashEnergyTable,
        vocab_words: List[str],
        word2idx: dict,
        idx2word: dict,
        # Decoding parameters
        top_k: int = 50,
        alpha: float = 1.0,
        temperature: float = 1.0,
        metropolis_threshold: int = 0,
        # Repetition penalty
        rep_penalty: float = 50.0,
        rep_window: int = 5,
        # Scale for converting log-probs to integer energy
        log_prob_scale: int = 100,
        # Boltzmann sampling
        beta: float = 0.01,
        # Seed
        seed: int = 42,
    ):
        """
        Args:
            base_model: Base language model (GPT-2 or DummyBaseLM).
            hash_energy: Trained HashEnergyTable from Phase 1.
            vocab_words: List of vocabulary words.
            word2idx: Dict mapping word -> index.
            idx2word: Dict mapping index -> word.
            top_k: Number of candidates from base model.
            alpha: Mixing weight for hash energy correction.
                   alpha=0 → pure base model.
                   alpha>0 → energy lowers probability of bad candidates.
            temperature: Temperature for energy correction.
                         Lower = sharper (more deterministic).
                         Higher = softer (more random).
            metropolis_threshold: Hard rejection threshold for delta_E.
                                  If delta_E > threshold, candidate is rejected.
                                  0 = disabled (no hard rejection).
                                  Positive = reject candidates with delta_E > threshold.
            rep_penalty: Energy penalty for repeated words (0=disabled).
            rep_window: How many recent words to check for repetition.
            log_prob_scale: Scale factor for base model log-probs to integer.
            beta: Boltzmann temperature for final sampling.
            seed: Random seed.
        """
        self.base_model = base_model
        self.hash_energy = hash_energy
        self.vocab_words = vocab_words
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.V = len(vocab_words)

        # Decoding parameters
        self.top_k = top_k
        self.alpha = alpha
        self.temperature = temperature
        self.metropolis_threshold = metropolis_threshold
        self.log_prob_scale = log_prob_scale
        self.seed = seed

        # Repetition penalty
        self.rep_penalty = rep_penalty
        self.rep_window = rep_window

        # Boltzmann sampler
        self.beta = beta
        self.sampler = IntegerBoltzmannSampler(beta=beta, max_delta=50000)

        # Use word-level candidates if available
        self._use_word_candidates = not isinstance(base_model, type)

        # Energy normalization statistics (calibrated after training)
        self._energy_mean: float = 0.0
        self._energy_std: float = 1.0
        self._base_energy_mean: float = 0.0
        self._base_energy_std: float = 1.0

        # Calibrated parameters
        self._calibrated = False

    def train_hash_energy(
        self,
        sequences: List[List[int]],
        n_epochs: int = 3,
        n_negatives: int = 3,
        corruptor=None,
    ) -> Dict:
        """
        Train the hash energy table via NCE.

        This is Phase 1 — just wraps HashEnergyTable.train_nce().
        """
        return self.hash_energy.train_nce(
            sequences=sequences,
            n_epochs=n_epochs,
            n_negatives=n_negatives,
            corruptor=corruptor,
        )

    def calibrate(
        self,
        sequences: List[List[int]],
        corruptor=None,
    ) -> Dict:
        """
        Calibrate alpha, temperature, beta, and metropolis_threshold.

        Steps:
        1. Collect energy statistics for z-score normalization
        2. Grid search over alpha, temperature, beta
        3. Optional: search for metropolis_threshold

        This replaces the v76h _calibrate_reranking method, but is
        MUCH faster because hash energy is O(1) per candidate instead
        of O(D^2).
        """
        print("    Calibrating LEGD decoder...")

        # Step 1: Collect energy samples
        energy_samples = []
        base_energy_samples = []

        for seq in sequences[:300]:
            if len(seq) < 3:
                continue
            for pos in range(1, min(len(seq), 5)):
                ctx = seq[:pos]
                target = seq[pos]

                # Get candidates
                if self._use_word_candidates and hasattr(self.base_model, 'get_top_k_words'):
                    candidates, log_probs = self.base_model.get_top_k_words(
                        ctx, k=min(self.top_k, self.V)
                    )
                else:
                    candidates, log_probs = self.base_model.get_top_k(
                        ctx, k=min(self.top_k, self.V)
                    )

                # Compute hash energy for target
                if 0 <= target < self.V:
                    e = self.hash_energy.compute_local_energy(ctx, target)
                    energy_samples.append(e)

                # Collect base energies
                base_e = -(log_probs * self.log_prob_scale).astype(np.int64)
                base_energy_samples.extend(base_e.tolist())

        if not energy_samples:
            print("    No energy samples — using defaults")
            self._calibrated = True
            return {'alpha': self.alpha, 'beta': self.beta}

        # Compute normalization stats
        self._energy_mean = float(np.mean(energy_samples))
        self._energy_std = max(1.0, float(np.std(energy_samples)))
        self._base_energy_mean = float(np.mean(base_energy_samples))
        self._base_energy_std = max(1.0, float(np.std(base_energy_samples)))

        print(f"    Energy stats: mean={self._energy_mean:.1f} "
              f"std={self._energy_std:.1f}")
        print(f"    Base stats: mean={self._base_energy_mean:.1f} "
              f"std={self._base_energy_std:.1f}")

        # Step 2: Grid search over alpha, beta
        test_pairs = []
        for seq in sequences[:200]:
            if len(seq) < 3:
                continue
            for pos in range(1, min(len(seq), 5)):
                test_pairs.append((seq[:pos], seq[pos]))

        if not test_pairs:
            self._calibrated = True
            return {'alpha': self.alpha, 'beta': self.beta}

        # Precompute re-ranking data
        rerank_data = []
        for ctx, target in test_pairs[:500]:
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
            print("    No test pairs — using defaults")
            self._calibrated = True
            return {'alpha': self.alpha, 'beta': self.beta}

        print(f"    Grid search on {len(rerank_data)} pairs...")

        best_alpha = self.alpha
        best_beta = self.beta
        best_acc = 0.0
        best_threshold = 0

        for alpha in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            for beta in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
                correct = 0
                total = 0
                for ctx, candidates, log_probs, target_idx in rerank_data:
                    combined = self._compute_combined_energy(
                        ctx, candidates, log_probs,
                        alpha=alpha,
                        recent_words=None,
                    )

                    if len(combined) == 0:
                        continue

                    # Boltzmann-weighted selection
                    E_min = np.min(combined)
                    deltas = (combined - E_min).astype(np.float64)
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
                    best_beta = beta

        # Step 3: Search for metropolis threshold
        if best_alpha > 0:
            # Find a threshold that rejects the worst 10-30% of candidates
            all_energies = []
            for ctx, candidates, log_probs, target_idx in rerank_data[:200]:
                hash_energies = self.hash_energy.compute_local_energy_batch(
                    ctx, candidates
                )
                all_energies.extend(hash_energies.tolist())

            if all_energies:
                # Threshold at 70th percentile (reject top 30%)
                best_threshold = int(np.percentile(all_energies, 70))

        self.alpha = best_alpha
        self.beta = best_beta
        self.metropolis_threshold = best_threshold
        self.sampler = IntegerBoltzmannSampler(beta=best_beta, max_delta=50000)
        self._calibrated = True

        print(f"    Calibrated: alpha={best_alpha:.1f}, "
              f"beta={best_beta}, "
              f"metropolis_threshold={best_threshold}, "
              f"rerank_acc={best_acc:.3f}")

        return {
            'alpha': best_alpha,
            'beta': best_beta,
            'metropolis_threshold': best_threshold,
            'rerank_acc': best_acc,
        }

    def _compute_combined_energy(
        self,
        context_word_ids: List[int],
        candidates: np.ndarray,
        base_log_probs: np.ndarray,
        alpha: Optional[float] = None,
        recent_words: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Compute combined energy for candidates.

        combined = base_energy + alpha * hash_energy_normalized + rep_penalty

        Hash energy is z-score normalized and rescaled to base energy std,
        same approach as v76h but with much lower variance.

        Args:
            context_word_ids: Context word IDs.
            candidates: Candidate word IDs, shape (K,).
            base_log_probs: Base model log-probs, shape (K,).
            alpha: Override alpha for grid search.
            recent_words: Recent words for rep penalty.

        Returns:
            Combined energies, shape (K,).
        """
        if alpha is None:
            alpha = self.alpha

        K = len(candidates)
        if K == 0:
            return np.array([], dtype=np.int64)

        # 1. Base model energies (negative log-prob scaled to integer)
        base_energies = -(base_log_probs * self.log_prob_scale).astype(np.int64)

        # 2. Hash energies (z-score normalized, rescaled to base std)
        hash_energies = self.hash_energy.compute_local_energy_batch(
            context_word_ids, candidates
        )

        if self._energy_std > 0:
            hash_norm = (
                (hash_energies.astype(np.float64) - self._energy_mean)
                / self._energy_std
            )
            hash_scaled = (hash_norm * self._base_energy_std).astype(np.int64)
        else:
            hash_scaled = np.zeros(K, dtype=np.int64)

        # 3. Metropolis gate — hard reject candidates above threshold
        if self.metropolis_threshold > 0 and alpha > 0:
            reject_mask = hash_energies > self.metropolis_threshold
            hash_scaled[reject_mask] = 100000  # Very high energy = rejected

        # 4. Repetition penalty
        rep_penalties = np.zeros(K, dtype=np.int64)
        if recent_words is not None and len(recent_words) > 0 and self.rep_penalty > 0:
            recent = recent_words[-self.rep_window:]
            for i, cand_id in enumerate(candidates):
                for j, prev_word in enumerate(recent):
                    if cand_id == prev_word:
                        recency = len(recent) - j
                        rep_penalties[i] += int(
                            self.rep_penalty * recency / len(recent)
                        )

        # 5. Combined energy
        combined = (base_energies
                    + (hash_scaled * alpha).astype(np.int64)
                    + rep_penalties)

        return combined

    def generate(
        self,
        prompt_ids: List[int],
        length: int = 200,
    ) -> List[int]:
        """
        Generate text using Local Energy-Guided Decoding.

        Per step:
          1. Base model produces top-K candidates
          2. Hash energy computes local delta_E for each candidate
          3. Metropolis gate rejects candidates above threshold
          4. Combined score adjusts base model probabilities
          5. Boltzmann sample from adjusted distribution

        No spin state, no SDR encoding, no DAM field. Just lookups.
        """
        generated = list(prompt_ids)

        for step in range(length):
            # Get top-K candidates
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

            # Filter to valid word IDs
            valid_mask = (candidates >= 4) & (candidates < self.V)
            if not np.any(valid_mask):
                break
            candidates = candidates[valid_mask]
            log_probs = log_probs[valid_mask]

            # Compute combined energy
            combined = self._compute_combined_energy(
                generated, candidates, log_probs,
                recent_words=generated[-self.rep_window:],
            )

            # Boltzmann sample
            if len(combined) > 1:
                idx = self.sampler.sample(combined)
            else:
                idx = 0

            chosen_id = int(candidates[idx])
            generated.append(chosen_id)

        return generated

    def compute_perplexity(
        self,
        sequences: List[List[int]],
        n_samples: int = 100,
    ) -> Dict:
        """
        Compute both base model PPL and LEGD-adjusted PPL.
        """
        base_log_probs = []
        legd_log_probs = []
        n_tokens = 0

        for seq in sequences[:n_samples]:
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                ctx = seq[:pos]
                target = seq[pos]

                # Get candidates
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
                    # Target not in candidates
                    if hasattr(self.base_model, 'compute_word_sequence_log_prob'):
                        base_lp = (
                            self.base_model.compute_word_sequence_log_prob(ctx + [target])
                            - self.base_model.compute_word_sequence_log_prob(ctx)
                        )
                    else:
                        base_lp = self.base_model.compute_sequence_log_prob(
                            ctx + [target]
                        ) - self.base_model.compute_sequence_log_prob(ctx)
                    base_log_probs.append(base_lp)
                    legd_log_probs.append(base_lp)
                else:
                    target_idx = np.where(target_mask)[0][0]
                    base_lp = log_probs[target_idx]
                    base_log_probs.append(base_lp)

                    # LEGD-adjusted probability
                    combined = self._compute_combined_energy(
                        ctx, candidates, log_probs
                    )

                    E_min = np.min(combined)
                    deltas = (combined - E_min).astype(np.float64)
                    deltas = np.clip(deltas, 0, 700)
                    weights = np.exp(-deltas * self.beta)
                    total_weight = np.sum(weights)
                    if total_weight > 0:
                        target_prob = weights[target_idx] / total_weight
                        if target_prob > 1e-300:
                            legd_log_probs.append(np.log(target_prob))
                        else:
                            legd_log_probs.append(-690)
                    else:
                        legd_log_probs.append(base_lp)

                n_tokens += 1

        base_ppl = np.exp(-np.mean(base_log_probs)) if base_log_probs else float('inf')
        legd_ppl = np.exp(-np.mean(legd_log_probs)) if legd_log_probs else float('inf')

        return {
            'base_ppl': base_ppl,
            'legd_ppl': legd_ppl,
            'n_tokens': n_tokens,
            'n_sequences': min(n_samples, len(sequences)),
        }

    def compute_discriminative_accuracy(
        self,
        sequences: List[List[int]],
        corruptor=None,
        n_samples: int = 1000,
    ) -> Dict:
        """
        How often does the hash energy assign LOWER energy to correct
        vs corrupted next words?

        This directly tests: "does the energy function know what word
        should come next?"
        """
        correct = 0
        total = 0
        type_correct = {}
        type_total = {}

        rng = np.random.RandomState(self.seed)

        n_pairs = 0
        for seq in sequences[:n_samples]:
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                ctx = seq[:pos]
                target = seq[pos]

                # Positive energy
                pos_energy = self.hash_energy.compute_local_energy(ctx, target)

                # Generate corruptions
                if corruptor is not None:
                    negatives = corruptor.generate_negatives(
                        ctx, target, n_negatives=4
                    )
                else:
                    # Simple random corruption
                    negatives = []
                    for _ in range(3):
                        neg_cand = rng.randint(4, self.V)
                        while neg_cand == target:
                            neg_cand = rng.randint(4, self.V)
                        negatives.append((ctx, neg_cand, 0))

                for neg_ctx, neg_cand, ctype in negatives:
                    neg_energy = self.hash_energy.compute_local_energy(
                        neg_ctx, neg_cand
                    )

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

        from .corruptions import CORRUPTION_NAMES
        for ctype in sorted(type_total.keys()):
            name = CORRUPTION_NAMES.get(ctype, f"type_{ctype}")
            acc = type_correct.get(ctype, 0) / max(1, type_total[ctype])
            result[f"{name}_accuracy"] = acc

        return result

    def diagnostics(self) -> Dict:
        """Return diagnostic information about the decoder state."""
        energy_stats = self.hash_energy.energy_statistics()
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'metropolis_threshold': self.metropolis_threshold,
            'rep_penalty': self.rep_penalty,
            'rep_window': self.rep_window,
            'energy_mean': self._energy_mean,
            'energy_std': self._energy_std,
            'base_energy_mean': self._base_energy_mean,
            'base_energy_std': self._base_energy_std,
            'calibrated': self._calibrated,
            **{f'hash_{k}': v for k, v in energy_stats.items()},
        }
