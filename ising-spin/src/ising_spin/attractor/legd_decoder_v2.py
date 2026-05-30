"""
LEGD Decoder v2 — Local Energy-Guided Decoding with Feature-Hashed Energy.

v78 extension: The LEGD decoder now works with FeatureHashEnergyTable
which provides both POS-level (generalizable) and lexical-level (memorized)
energy signals.

The decoding formula is unchanged:
  P(s_t) proportional to P_base(s_t) * exp(-alpha * delta_E(s_t) / T)

But delta_E now has two components:
  delta_E = pos_weight * E_pos(POS(prev), POS(candidate))
          + lex_weight * E_lex(prev_id, candidate_id)

The POS component provides CATEGORY-LEVEL generalization — learning
"the cat" improves "a dog" because both are DET→NOUN transitions.
The lexical component provides TOKEN-SPECIFIC knowledge.

Calibration now also searches over pos_weight and lex_weight to find
the optimal balance between generalizable rules and specific facts.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

from .hash_energy import HashEnergyTable
from ..sampling.boltzmann import IntegerBoltzmannSampler


class LEGDDecoderV2:
    """
    Local Energy-Guided Decoder V2 — with Feature-Hashed Energy.

    Works with either HashEnergyTable (v77) or FeatureHashEnergyTable (v78).
    The energy table must implement:
      - compute_local_energy_batch(context, candidates) -> np.ndarray
      - train_nce(sequences, n_epochs, n_negatives, corruptor) -> Dict
      - energy_statistics() -> Dict
      - memory_mb() -> float
    """

    def __init__(
        self,
        base_model,
        hash_energy,  # HashEnergyTable or FeatureHashEnergyTable
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

        # Energy normalization statistics
        self._energy_mean: float = 0.0
        self._energy_std: float = 1.0
        self._base_energy_mean: float = 0.0
        self._base_energy_std: float = 1.0

        # POS-specific energy normalization (for FeatureHashEnergyTable)
        self._pos_energy_mean: float = 0.0
        self._pos_energy_std: float = 1.0
        self._lex_energy_mean: float = 0.0
        self._lex_energy_std: float = 1.0

        # Calibrated parameters
        self._calibrated = False

        # Feature hash support flag
        self._has_feature_hash = hasattr(hash_energy, 'word_pos')

    def train_hash_energy(
        self,
        sequences: List[List[int]],
        n_epochs: int = 3,
        n_negatives: int = 3,
        corruptor=None,
    ) -> Dict:
        """Train the hash energy table via NCE."""
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
        Calibrate alpha, beta, metropolis_threshold, and feature weights.

        For FeatureHashEnergyTable, also searches over pos_weight and
        lex_weight to find the optimal balance.
        """
        print("    Calibrating LEGD v2 decoder...")

        # Step 1: Collect energy samples for normalization
        energy_samples = []
        pos_energy_samples = []
        lex_energy_samples = []
        base_energy_samples = []

        for seq in sequences[:300]:
            if len(seq) < 3:
                continue
            for pos in range(1, min(len(seq), 5)):
                ctx = seq[:pos]
                target = seq[pos]

                # Get candidates
                candidates, log_probs = self._get_candidates(ctx)

                # Compute hash energy for target
                if 0 <= target < self.V:
                    e = self.hash_energy.compute_local_energy(ctx, target)
                    energy_samples.append(e)

                    # If feature hash, also collect POS and lexical separately
                    if self._has_feature_hash:
                        prev_pos = int(self.hash_energy.word_pos[ctx[-1]])
                        target_pos = int(self.hash_energy.word_pos[target])

                        # POS energy
                        pe = 0
                        from .feature_hash_energy import _double_hash
                        for h_idx in range(self.hash_energy.n_pos_hashes):
                            slot = _double_hash(
                                prev_pos, target_pos,
                                h_idx, self.hash_energy.pos_table_size
                            )
                            pe += int(self.hash_energy._pos_bigram_tables[h_idx][slot])
                        pos_energy_samples.append(pe)

                        # Lexical energy
                        le = 0
                        for h_idx in range(self.hash_energy.n_lex_hashes):
                            slot = _double_hash(
                                ctx[-1], target,
                                h_idx, self.hash_energy.lex_table_size
                            )
                            le += int(self.hash_energy._lex_bigram_tables[h_idx][slot])
                        lex_energy_samples.append(le)

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

        if pos_energy_samples:
            self._pos_energy_mean = float(np.mean(pos_energy_samples))
            self._pos_energy_std = max(1.0, float(np.std(pos_energy_samples)))
        if lex_energy_samples:
            self._lex_energy_mean = float(np.mean(lex_energy_samples))
            self._lex_energy_std = max(1.0, float(np.std(lex_energy_samples)))

        print(f"    Combined energy: mean={self._energy_mean:.1f} "
              f"std={self._energy_std:.1f}")
        if self._has_feature_hash:
            print(f"    POS energy: mean={self._pos_energy_mean:.1f} "
                  f"std={self._pos_energy_std:.1f}")
            print(f"    Lex energy: mean={self._lex_energy_mean:.1f} "
                  f"std={self._lex_energy_std:.1f}")
        print(f"    Base stats: mean={self._base_energy_mean:.1f} "
              f"std={self._base_energy_std:.1f}")

        # Step 2: Grid search
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
            candidates, log_probs = self._get_candidates(ctx)
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
        best_pos_weight = getattr(self.hash_energy, 'pos_weight', 1.0)
        best_lex_weight = getattr(self.hash_energy, 'lex_weight', 1.0)

        # Search over alpha, beta
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

        # Step 3: If feature hash, also search over pos_weight/lex_weight
        if self._has_feature_hash:
            print(f"    Searching feature weights (pos_weight, lex_weight)...")
            # Use best alpha/beta from above
            for pw in [0.5, 1.0, 2.0, 3.0, 5.0]:
                for lw in [0.5, 1.0, 2.0, 3.0]:
                    self.hash_energy.pos_weight = pw
                    self.hash_energy.lex_weight = lw

                    correct = 0
                    total = 0
                    for ctx, candidates, log_probs, target_idx in rerank_data:
                        combined = self._compute_combined_energy(
                            ctx, candidates, log_probs,
                            alpha=best_alpha,
                            recent_words=None,
                        )

                        if len(combined) == 0:
                            continue

                        E_min = np.min(combined)
                        deltas = (combined - E_min).astype(np.float64)
                        deltas = np.clip(deltas, 0, 700)
                        weights = np.exp(-deltas * best_beta)
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
                        best_pos_weight = pw
                        best_lex_weight = lw

            # Apply best feature weights
            self.hash_energy.pos_weight = best_pos_weight
            self.hash_energy.lex_weight = best_lex_weight

        # Step 4: Search for metropolis threshold
        if best_alpha > 0:
            all_energies = []
            for ctx, candidates, log_probs, target_idx in rerank_data[:200]:
                hash_energies = self.hash_energy.compute_local_energy_batch(
                    ctx, candidates
                )
                all_energies.extend(hash_energies.tolist())

            if all_energies:
                best_threshold = int(np.percentile(all_energies, 70))

        self.alpha = best_alpha
        self.beta = best_beta
        self.metropolis_threshold = best_threshold
        self.sampler = IntegerBoltzmannSampler(beta=best_beta, max_delta=50000)
        self._calibrated = True

        result = {
            'alpha': best_alpha,
            'beta': best_beta,
            'metropolis_threshold': best_threshold,
            'rerank_acc': best_acc,
        }

        if self._has_feature_hash:
            result['pos_weight'] = best_pos_weight
            result['lex_weight'] = best_lex_weight

        print(f"    Calibrated: alpha={best_alpha:.1f}, "
              f"beta={best_beta}, "
              f"metropolis_threshold={best_threshold}, "
              f"rerank_acc={best_acc:.3f}")
        if self._has_feature_hash:
            print(f"    Feature weights: pos_weight={best_pos_weight}, "
                  f"lex_weight={best_lex_weight}")

        return result

    def _get_candidates(self, ctx):
        """Get top-K candidates from base model."""
        if self._use_word_candidates and hasattr(self.base_model, 'get_top_k_words'):
            return self.base_model.get_top_k_words(
                ctx, k=min(self.top_k, self.V)
            )
        else:
            return self.base_model.get_top_k(
                ctx, k=min(self.top_k, self.V)
            )

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
        """
        if alpha is None:
            alpha = self.alpha

        K = len(candidates)
        if K == 0:
            return np.array([], dtype=np.int64)

        # 1. Base model energies
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

        # 3. Metropolis gate
        if self.metropolis_threshold > 0 and alpha > 0:
            reject_mask = hash_energies > self.metropolis_threshold
            hash_scaled[reject_mask] = 100000

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
        """Generate text using Local Energy-Guided Decoding."""
        generated = list(prompt_ids)

        for step in range(length):
            candidates, log_probs = self._get_candidates(generated)

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
        """Compute both base model PPL and LEGD-adjusted PPL."""
        base_log_probs = []
        legd_log_probs = []
        n_tokens = 0

        for seq in sequences[:n_samples]:
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                ctx = seq[:pos]
                target = seq[pos]

                candidates, log_probs = self._get_candidates(ctx)

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
        """How often does hash energy prefer correct vs corrupted next words?"""
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
        diag = {
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
            'has_feature_hash': self._has_feature_hash,
        }

        if self._has_feature_hash:
            diag['pos_weight'] = self.hash_energy.pos_weight
            diag['lex_weight'] = self.hash_energy.lex_weight
            diag['pos_energy_mean'] = self._pos_energy_mean
            diag['pos_energy_std'] = self._pos_energy_std
            diag['lex_energy_mean'] = self._lex_energy_mean
            diag['lex_energy_std'] = self._lex_energy_std

        diag.update({f'hash_{k}': v for k, v in energy_stats.items()})
        return diag
