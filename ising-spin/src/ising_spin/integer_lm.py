"""
Integer Language Model — the main class.

Pure integer language model. No neural nets. No torch dependency.
Runs on a Pi 5. Produces grammatically coherent text for simple domains.

Architecture (v82 — Multi-Class):
  1. BigramModel: P(word | prev) from integer counts (the base)
  2. Dynamic FeatureHashEnergy: MULTI-CLASS features
     - Multiple word class systems running simultaneously
     - Frequency buckets ("freq") + distributional clusters ("dist")
     - Each feature declares which class system it uses via class_key
     - add_feature() / remove_feature() for full flexibility
  3. LEGD: P(c) proportional to P_base(c) * exp(-alpha * E_norm(c))
  4. Metropolis gate: hard-reject high-energy candidates
  5. Repetition penalty: unigram + bigram n-gram blocking

KEY CHANGES from v81:
  - MULTI-CLASS: features use BOTH freq buckets AND dist clusters
  - Distributional clusters capture syntactic behavior (not just frequency)
  - N-gram blocking: prevent repeated bigrams, not just unigrams
  - Adaptive clipping: prevent energy saturation

Generation loop (per step):
  1. Bigram model proposes top-K candidates with probabilities
  2. Each feature computes its energy contribution (using its class array)
  3. LEGD combines: P(c) proportional to P_base(c) * exp(-alpha * E_norm(c))
  4. Metropolis gate zeros out candidates above threshold
  5. Repetition penalty reduces probability of recent words AND bigrams
  6. Sample from resulting distribution

Training:
  1. Count bigrams (integer)
  2. NCE train all features (integer, with class-balanced negatives)
  3. Calibrate alpha (grid search on LEGD probs)

Memory footprint: ~25 MB for V=2000 with 8 features. Runs anywhere.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

from .bigram_model import BigramModel
from .feature_hash_energy import (
    FeatureHashEnergyTable, FeatureSpec, default_features,
)
from .vocabulary import Vocabulary, IDX2POS


class IntegerLM:
    """
    Pure Integer Language Model with Multi-Class Word System.

    No neural nets. No torch. No float32 matrix multiplies in the hot path.
    Just integer counting, hash table lookups, and LEGD probability adjustment.

    The LEGD formula:
        P(c | ctx) proportional to P_base(c | prev) * exp(-alpha * E_norm(c))

    Lower energy = more likely = higher probability.
    alpha controls how much the energy correction overrides the base model.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        # Feature configuration — either pass a list of features OR use defaults
        features: Optional[List[FeatureSpec]] = None,
        # Generation
        top_k: int = 50,
        alpha: float = 0.1,
        temperature: float = 1.0,
        metropolis_threshold: int = 0,
        rep_penalty: float = 3.0,
        rep_window: int = 5,
        bigram_block_window: int = 3,
        # Bigram model
        smoothing_alpha: float = 1.0,
        log_prob_scale: int = 100,
        # Seed
        seed: int = 42,
    ):
        self.vocab = vocab
        self.V = vocab.V
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Bigram base model
        self.bigram = BigramModel(
            vocab_size=self.V,
            smoothing_alpha=smoothing_alpha,
            log_prob_scale=log_prob_scale,
            seed=seed,
        )

        # Feature hash energy — MULTI-CLASS feature registry
        # v82: pass ALL available class systems
        word_classes = vocab.get_word_classes()
        self.energy = FeatureHashEnergyTable(
            vocab_size=self.V,
            word_classes=word_classes,
            seed=seed,
        )

        # Register features
        if features is not None:
            for feat in features:
                self.energy.add_feature(feat)
        else:
            include_dist = "dist" in word_classes
            for feat in default_features(include_dist=include_dist):
                self.energy.add_feature(feat)

        # Generation parameters
        self.top_k = top_k
        self.alpha = alpha
        self.temperature = temperature
        self.log_prob_scale = log_prob_scale
        self.metropolis_threshold = metropolis_threshold
        self.rep_penalty = rep_penalty
        self.rep_window = rep_window
        self.bigram_block_window = bigram_block_window  # v82: bigram n-gram blocking

        # Normalization stats (calibrated after training)
        self._e_mean = 0.0
        self._e_std = 1.0
        self._calibrated = False

    # -------------------------------------------------------------------
    # Feature management — add/remove features at any time
    # -------------------------------------------------------------------

    def add_feature(self, feature: FeatureSpec):
        """Add a feature to the energy table. Must be done before training."""
        self.energy.add_feature(feature)

    def remove_feature(self, name: str):
        """Remove a feature by name."""
        self.energy.remove_feature(name)

    def list_features(self) -> List[str]:
        """List registered feature names."""
        return list(self.energy.features.keys())

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------

    def train(self, sequences: List[List[int]], n_epochs: int = 3, n_negatives: int = 3) -> Dict:
        """
        Train the model: build bigram counts + NCE train all features.
        """
        print("  Training bigram model...", flush=True)
        self.bigram.build(sequences)
        print(f"    {self.bigram.statistics()['total_bigrams']:,} bigrams, "
              f"{self.bigram.statistics()['nonzero_bigrams']:,} nonzero", flush=True)

        print("  Training feature hash energy...", flush=True)
        energy_stats = self.energy.train_nce(
            sequences=sequences,
            n_epochs=n_epochs,
            n_negatives=n_negatives,
        )
        return energy_stats

    def calibrate(self, sequences: List[List[int]]) -> Dict:
        """
        Calibrate alpha, metropolis_threshold, and feature weights.

        Searches alpha in [0.001, 0.5] and feature weights independently.
        v82: Also tries pruning features with weight=0.
        """
        print("  Calibrating...", flush=True)

        # Collect energy samples for normalization
        e_samples = []
        for seq in sequences[:300]:
            if len(seq) < 3:
                continue
            for pos in range(1, min(len(seq), 5)):
                ctx = seq[:pos]
                target = seq[pos]
                if 0 <= target < self.V:
                    e_samples.append(self.energy.compute_local_energy(ctx, target))

        if e_samples:
            self._e_mean = float(np.mean(e_samples))
            self._e_std = max(1.0, float(np.std(e_samples)))

        print(f"    Energy: mean={self._e_mean:.1f}, std={self._e_std:.1f}", flush=True)

        # Build test data for re-ranking accuracy
        rerank_data = []
        for seq in sequences[:200]:
            if len(seq) < 3:
                continue
            for pos in range(1, min(len(seq), 5)):
                ctx, target = seq[:pos], seq[pos]
                cands, lps = self.bigram.get_top_k(ctx, k=self.top_k)
                if target not in cands:
                    continue
                tidx = int(np.where(cands == target)[0][0])
                rerank_data.append((ctx, cands, lps, tidx))

        if not rerank_data:
            self._calibrated = True
            return {'alpha': self.alpha}

        print(f"    Grid search on {len(rerank_data)} pairs...", flush=True)

        # Phase 1: Search alpha
        best_alpha, best_acc = self.alpha, 0.0
        for alpha in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            acc = self._eval_rerank(rerank_data, alpha)
            if acc > best_acc:
                best_acc, best_alpha = acc, alpha

        print(f"    Best alpha: {best_alpha} (acc={best_acc:.3f})", flush=True)

        # Phase 2: Search feature weights (only if there are features with alpha > 0)
        best_weights = {feat.name: feat.weight for feat in self.energy.features.values()}
        if len(self.energy.features) > 0 and best_alpha > 0:
            print(f"    Searching feature weights...", flush=True)

            for feat in self.energy.features.values():
                original_weight = feat.weight
                best_w = original_weight
                best_feat_acc = best_acc

                for w in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
                    feat.weight = w
                    acc = self._eval_rerank(rerank_data, best_alpha)
                    if acc > best_feat_acc:
                        best_feat_acc = acc
                        best_w = w

                feat.weight = best_w
                if best_feat_acc > best_acc:
                    best_acc = best_feat_acc
                    best_weights[feat.name] = best_w

            # Restore best weights
            for feat in self.energy.features.values():
                feat.weight = best_weights.get(feat.name, feat.weight)

        self.alpha = best_alpha

        # Metropolis threshold
        best_threshold = 0
        if best_alpha > 0:
            all_e = []
            for ctx, cands, lps, _ in rerank_data[:200]:
                he = self.energy.compute_local_energy_batch(ctx, cands)
                all_e.extend(he.tolist())
            if all_e:
                best_threshold = int(np.percentile(all_e, 70))

        self.metropolis_threshold = best_threshold
        self._calibrated = True

        result = {
            'alpha': best_alpha,
            'metropolis_threshold': best_threshold,
            'rerank_acc': best_acc,
            'feature_weights': best_weights,
        }
        w_str = ", ".join(f"{k}={v:.1f}" for k, v in best_weights.items())
        print(f"    Result: alpha={best_alpha}, threshold={best_threshold}, "
              f"acc={best_acc:.3f}", flush=True)
        print(f"    Weights: {w_str}", flush=True)
        return result

    def _eval_rerank(self, data, alpha):
        """Evaluate re-ranking accuracy with given alpha."""
        correct = total = 0
        for ctx, cands, lps, tidx in data:
            probs = self._compute_legd_probs(ctx, cands, lps, alpha)
            if len(probs) == 0:
                continue
            if np.argmax(probs) == tidx:
                correct += 1
            total += 1
        return correct / max(1, total)

    # -------------------------------------------------------------------
    # LEGD probability computation — the core formula
    # -------------------------------------------------------------------

    def _compute_legd_probs(
        self,
        context: List[int],
        candidates: np.ndarray,
        log_probs: np.ndarray,
        alpha: float,
        recent_words: Optional[List[int]] = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Compute LEGD-adjusted probabilities.

        P(c) proportional to P_base(c)^(1/T) * exp(-alpha * E_norm(c) / T)

        The energy is z-score normalized: E_norm = (E - mean) / std
        """
        K = len(candidates)
        if K == 0:
            return np.array([])

        # Base probabilities (numerically stable)
        lps_scaled = log_probs / temperature
        base_probs = np.exp(lps_scaled - np.max(lps_scaled))

        # Hash energy (z-score normalized)
        hash_e = self.energy.compute_local_energy_batch(context, candidates)
        h_norm = (hash_e.astype(np.float64) - self._e_mean) / max(1.0, self._e_std)

        # LEGD: P(c) proportional to P_base(c) * exp(-alpha * h_norm(c) / T)
        legd_weights = base_probs * np.exp(-alpha * h_norm / temperature)

        # Metropolis gate: zero out candidates above threshold
        if self.metropolis_threshold > 0 and alpha > 0:
            gate = hash_e > self.metropolis_threshold
            legd_weights[gate] = 0.0

        # Repetition penalty — unigram + bigram blocking
        if recent_words and self.rep_penalty > 0:
            recent = recent_words[-self.rep_window:]

            # Unigram penalty (original)
            for i, cid in enumerate(candidates):
                for j, pw in enumerate(recent):
                    if cid == pw:
                        recency = (len(recent) - j) / len(recent)
                        legd_weights[i] *= np.exp(-self.rep_penalty * recency / temperature)

            # v82: Bigram blocking — prevent repeating recent bigrams
            # If context[-1] → candidate appeared recently, penalize heavily
            if len(recent_words) >= 2 and self.bigram_block_window > 0:
                prev_word = recent_words[-1] if recent_words else None
                if prev_word is not None:
                    # Check recent bigrams
                    for j in range(max(0, len(recent_words) - self.bigram_block_window),
                                   len(recent_words) - 1):
                        if recent_words[j] == prev_word:
                            # This bigram (prev_word → next) appeared before
                            next_word = recent_words[j + 1]
                            for i, cid in enumerate(candidates):
                                if cid == next_word:
                                    # Heavy penalty for repeating a bigram
                                    legd_weights[i] *= np.exp(
                                        -self.rep_penalty * 2.0 / temperature
                                    )

        # Normalize
        total = legd_weights.sum()
        if total > 0:
            return legd_weights / total
        else:
            return np.ones(K) / K

    # -------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------

    def generate(self, prompt_ids: List[int], length: int = 100) -> List[int]:
        """
        Generate text using LEGD-adjusted probability sampling.
        v82: Includes bigram n-gram blocking.
        """
        generated = list(prompt_ids)
        for _ in range(length):
            candidates, log_probs = self.bigram.get_top_k(generated, k=self.top_k)
            if len(candidates) == 0:
                break

            # Filter special tokens
            valid = (candidates >= 4) & (candidates < self.V)
            if not np.any(valid):
                break
            candidates = candidates[valid]
            log_probs = log_probs[valid]

            probs = self._compute_legd_probs(
                generated, candidates, log_probs, self.alpha,
                recent_words=generated[-self.rep_window:],
                temperature=self.temperature,
            )

            idx = self.rng.choice(len(candidates), p=probs)
            generated.append(int(candidates[idx]))

        return generated

    def generate_text(self, prompt: str, length: int = 100) -> str:
        """Generate text from a string prompt."""
        words = prompt.lower().split()
        ids = [self.vocab.word2idx.get(w, 1) for w in words]
        ids = [i for i in ids if i >= 4]
        if not ids:
            return ""
        generated = self.generate(ids, length=length)
        return self.vocab.decode(generated)

    # -------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------

    def perplexity(self, sequences: List[List[int]], n_samples: int = 100) -> Dict:
        """
        Compute base PPL and LEGD-adjusted PPL.
        """
        base_lps, legd_lps, n_tok = [], [], 0

        for seq in sequences[:n_samples]:
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                ctx, target = seq[:pos], seq[pos]
                base_lp = self.bigram.compute_log_prob(ctx, target)
                base_lps.append(base_lp)

                # LEGD-adjusted probability
                cands, lps = self.bigram.get_top_k(ctx, k=self.top_k)
                if target in cands:
                    tidx = int(np.where(cands == target)[0][0])
                    probs = self._compute_legd_probs(ctx, cands, lps, self.alpha)
                    tp = probs[tidx]
                    if tp > 0:
                        legd_lps.append(float(np.log(tp)))
                    else:
                        legd_lps.append(base_lp)
                else:
                    legd_lps.append(base_lp)
                n_tok += 1

        base_ppl = np.exp(-np.mean(base_lps)) if base_lps else float('inf')
        legd_ppl = np.exp(-np.mean(legd_lps)) if legd_lps else float('inf')
        return {'base_ppl': base_ppl, 'legd_ppl': legd_ppl, 'n_tokens': n_tok}

    def discriminative_accuracy(self, sequences: List[List[int]], n_samples: int = 500) -> Dict:
        """How often does energy prefer correct vs random next words?"""
        correct = total = 0
        rng = np.random.RandomState(self.seed)

        n = 0
        for seq in sequences[:n_samples]:
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                ctx, target = seq[:pos], seq[pos]
                pos_e = self.energy.compute_local_energy(ctx, target)

                for _ in range(3):
                    neg = rng.randint(4, self.V)
                    while neg == target:
                        neg = rng.randint(4, self.V)
                    neg_e = self.energy.compute_local_energy(ctx, neg)
                    if pos_e < neg_e:
                        correct += 1
                    total += 1

                n += 1
                if n >= n_samples:
                    break
            if n >= n_samples:
                break

        return {'accuracy': correct / max(1, total), 'comparisons': total}

    def class_transition_matrix(self, class_key: Optional[str] = None) -> np.ndarray:
        """Return K×K class transition energy matrix for visualization."""
        return self.energy.get_class_matrix(class_key=class_key)

    def diagnostics(self) -> Dict:
        """Full diagnostic report."""
        e_stats = self.energy.statistics()
        b_stats = self.bigram.statistics()
        feat_weights = {feat.name: feat.weight for feat in self.energy.features.values()}
        feat_class_keys = {feat.name: feat.class_key for feat in self.energy.features.values()}
        return {
            'alpha': self.alpha,
            'temperature': self.temperature,
            'metropolis_threshold': self.metropolis_threshold,
            'rep_penalty': self.rep_penalty,
            'rep_window': self.rep_window,
            'bigram_block_window': self.bigram_block_window,
            'n_features': e_stats['n_features'],
            'feature_names': e_stats['feature_names'],
            'feature_weights': feat_weights,
            'feature_class_keys': feat_class_keys,
            'class_systems': e_stats.get('class_systems', []),
            'n_classes_map': e_stats.get('n_classes_map', {}),
            'energy_mean': self._e_mean,
            'energy_std': self._e_std,
            'calibrated': self._calibrated,
            'energy_memory_mb': e_stats['memory_mb'],
            'bigram_memory_mb': b_stats.get('memory_mb', 0),
            'bigram_nnz': b_stats.get('nonzero_bigrams', 0),
        }
