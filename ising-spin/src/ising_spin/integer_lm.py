"""
Integer Language Model — the main class.

Pure integer language model. No neural nets. No torch dependency.
Runs on a Pi 5. Produces grammatically coherent text for simple domains.

Architecture (v90 — Architectural Fix):
  1. BigramModel: P(word | prev) with Jelinek-Mercer interpolation + alpha=0.01
     (v89 used Laplace alpha=1.0 which destroyed bigram signal — 95%+ smoothing mass)
  2. Dynamic FeatureHashEnergy: MULTI-CLASS features
     - Per-feature z-score normalization BEFORE weighting
     - Multiple word class systems: POS + freq + dist
     - Independent hash functions (double-hashing scheme)
     - Per-feature balanced negatives (aligned to each feature's class_key)
  3. LEGD: P(c) proportional to P_base(c) * exp(-alpha * E(c))
  4. Repetition penalty: simple bigram blocking (no Metropolis gate)
  5. top_k=200 for PPL evaluation (was 50 — 15-25% of tokens were invisible)

v90 FUNDAMENTAL FIXES over v89 (PPL=13.40, only 3/9 features active):

  Bigram base model (THE BIGGEST IMPACT):
  - Laplace alpha=1.0→0.01: With V=2000, alpha*V=2000 pseudo-observations
    dominated the probability for most context words. The base model was
    nearly a unigram model (PPL=27.74). With alpha=0.01, alpha*V=20,
    so the bigram signal dominates for contexts with 20+ observations.
  - Jelinek-Mercer interpolation: P(w|prev) = λ*P_bigram + (1-λ)*P_unigram
    where λ = total/(total + alpha*V). Automatic backoff for rare contexts.
    Expected base PPL improvement: 27.74 → ~15-18.

  LEGD inference:
  - Metropolis gate REMOVED: It killed 25% of candidates including correct
    ones. The soft exp(-α*E) reweighting already handles high-energy words.
  - top_k=200 for PPL evaluation: The old top_k=50 was a hard expressiveness
    ceiling. 15-25% of correct tokens fell outside top-50 and got base PPL
    as a "free ride", masking the LEGD model's true performance.
  - Alpha search expanded to [0, 5.0]: With per-feature normalization,
    higher alpha values are safe. v89 capped at 1.0 which was too low.
  - Repetition penalty reduced: 3.0→1.0 (was killing repeated function words)
  - Calibration data expanded: More sequences, all positions, held-out data.

  Feature hash energy:
  - Hash independence fixed (double-hashing, collision correlation 65%→3%)
  - LexBigramFeature removed (redundant with P_base)
  - Table sizes increased (65537→262147 for lexical, load factor 3.05→0.76)
  - Class clip reduced (50→20, prevents binary saturation)
  - Per-feature balanced negatives (each feature uses its own class_key)
  - POS class system added (syntactically meaningful class→word features)

Memory footprint: ~30 MB for V=2000 with 8 features. Runs anywhere.
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
        top_k: int = 200,
        alpha: float = 0.5,
        temperature: float = 1.0,
        metropolis_threshold: int = 0,  # v90: DEPRECATED — always 0
        rep_penalty: float = 1.0,
        rep_window: int = 5,
        bigram_block_window: int = 3,
        # Bigram model
        smoothing_alpha: float = 0.01,
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
            include_pos = "pos" in word_classes
            for feat in default_features(include_dist=include_dist, include_pos=include_pos):
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
        # Store training stats for disc-aware calibration
        self.energy._last_train_stats = energy_stats
        return energy_stats

    def calibrate(self, sequences: List[List[int]]) -> Dict:
        """
        Calibrate alpha, metropolis_threshold, and feature weights.

        v89 FUNDAMENTAL CHANGES over v88:
        1. Per-feature z-score normalization is now used (computed in
           _compute_feature_stats, applied in compute_local_energy_batch).
           This means the energy is already properly scaled — no global
           z-score needed.
        2. Alpha search uses PPL MINIMIZATION instead of argmax accuracy.
           Argmax-based search selects alpha=2.0 which makes energy
           dominate P_base (good for argmax, catastrophic for PPL).
           PPL-based search finds the sweet spot where energy gently
           corrects the base model.

        v83 IMPROVEMENTS (preserved):
        - Disc-aware weight pruning: features with disc < 0.60 get weight=0
        - Disc-proportional weight initialization before grid search
        """
        print("  Calibrating...", flush=True)

        # v89: Compute per-feature stats — NOW USED for normalization
        self._compute_feature_stats(sequences)

        # Print per-feature stats (now actually used for normalization)
        if self.energy.feature_stats:
            print(f"    Per-feature stats (USED for z-score normalization):", flush=True)
            for fname, stats in self.energy.feature_stats.items():
                print(f"      {fname}: mean={stats['mean']:.1f}, std={stats['std']:.1f}",
                      flush=True)

        # Compute overall energy stats for diagnostics
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

        print(f"    Energy (normalized): mean={self._e_mean:.2f}, std={self._e_std:.2f}",
              flush=True)

        # Build test data for PPL-based calibration
        # v90: Use more sequences (500 vs 200), all positions (not just first 4)
        rerank_data = []
        for seq in sequences[:500]:
            if len(seq) < 3:
                continue
            for pos in range(1, len(seq)):
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

        # Phase 0: Disc-aware weight pruning
        last_train_stats = getattr(self.energy, '_last_train_stats', None)
        if last_train_stats and 'epochs' in last_train_stats and last_train_stats['epochs']:
            last_epoch = last_train_stats['epochs'][-1]
            feat_disc = last_epoch.get('feature_disc', {})
            n_pruned = 0
            for feat in self.energy.features.values():
                disc = feat_disc.get(feat.name, 0.5)
                if disc < 0.60:
                    print(f"    PRUNE: {feat.name} disc={disc:.3f} < 0.60 → weight=0",
                          flush=True)
                    feat.weight = 0.0
                    n_pruned += 1
                elif disc > 0.85:
                    feat.weight = max(feat.weight, 1.0)
                    print(f"    BOOST: {feat.name} disc={disc:.3f} → weight={feat.weight:.1f}",
                          flush=True)
            if n_pruned > 0:
                print(f"    Pruned {n_pruned} weak features (disc < 0.60)", flush=True)

        # Phase 1: Search alpha — v90: EXPANDED range up to 5.0
        # With per-feature normalization and the improved base model (alpha=0.01),
        # higher alpha values are safe and likely optimal.
        best_alpha, best_ppl = self.alpha, float('inf')
        for alpha in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
            ppl = self._eval_ppl(rerank_data, alpha)
            if ppl < best_ppl:
                best_ppl, best_alpha = ppl, alpha

        # Also report argmax accuracy for comparison
        best_acc = self._eval_rerank(rerank_data, best_alpha)
        print(f"    Best alpha: {best_alpha} (PPL={best_ppl:.2f}, acc={best_acc:.3f})",
              flush=True)

        # Phase 2: Search feature weights (PPL-based, only active features)
        best_weights = {feat.name: feat.weight for feat in self.energy.features.values()}
        active_features = [f for f in self.energy.features.values() if f.weight > 0]

        if len(active_features) > 0 and best_alpha > 0:
            print(f"    Searching feature weights ({len(active_features)} active, PPL-based)...",
                  flush=True)

            # v90: Wider weight grid to match expanded alpha range
            weight_grid = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]

            for feat in active_features:
                original_weight = feat.weight
                best_w = original_weight
                best_feat_ppl = best_ppl

                for w in weight_grid:
                    feat.weight = w
                    ppl = self._eval_ppl(rerank_data, best_alpha)
                    if ppl < best_feat_ppl:
                        best_feat_ppl = ppl
                        best_w = w

                feat.weight = best_w
                if best_feat_ppl < best_ppl:
                    best_ppl = best_feat_ppl
                    best_weights[feat.name] = best_w

            # Restore best weights
            for feat in self.energy.features.values():
                feat.weight = best_weights.get(feat.name, feat.weight)

        self.alpha = best_alpha

        # v90: Metropolis gate REMOVED — set threshold to 0 always
        # The soft exp(-alpha * E) reweighting already handles high-energy words.
        # The old Metropolis gate at 75th percentile killed 25% of candidates
        # including correct ones, and the PPL evaluation masked this damage.
        best_threshold = 0
        self.metropolis_threshold = best_threshold
        self._calibrated = True

        result = {
            'alpha': best_alpha,
            'metropolis_threshold': best_threshold,
            'cal_ppl': best_ppl,
            'rerank_acc': best_acc,
            'feature_weights': best_weights,
        }
        w_str = ", ".join(f"{k}={v:.1f}" for k, v in best_weights.items())
        print(f"    Result: alpha={best_alpha}, threshold={best_threshold}, "
              f"PPL={best_ppl:.2f}, acc={best_acc:.3f}", flush=True)
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

    def _eval_ppl(self, data, alpha):
        """v89: Evaluate PPL with given alpha. Lower is better."""
        log_probs = []
        for ctx, cands, lps, tidx in data:
            probs = self._compute_legd_probs(ctx, cands, lps, alpha)
            if len(probs) == 0:
                continue
            tp = probs[tidx]
            if tp > 0:
                log_probs.append(float(np.log(tp)))
            else:
                log_probs.append(-20.0)  # Very bad but not -inf
        if not log_probs:
            return float('inf')
        return float(np.exp(-np.mean(log_probs)))

    def _compute_feature_stats(self, sequences: List[List[int]]):
        """
        v89: Compute per-feature mean and std — NOW USED for normalization.

        These stats are stored in self.energy.feature_stats and used by
        compute_local_energy_batch() for per-feature z-score normalization.
        This is the key fix: without per-feature normalization, lex_bi
        (mean=-454) drowns out cls_tri_freq (mean=-103).
        """
        feature_energies = {feat.name: [] for feat in self.energy.features.values()}

        n_samples = 0
        for seq in sequences[:500]:
            if len(seq) < 3:
                continue
            for pos in range(1, min(len(seq), 10)):  # v89: more positions (was 8)
                ctx = seq[:pos]
                target = seq[pos]
                if not (0 <= target < self.V):
                    continue
                candidates = np.array([target], dtype=np.int64)
                for feat in self.energy.features.values():
                    wc = self.energy._get_class_array(feat)
                    e = feat.energy_batch(ctx, candidates, wc)
                    feature_energies[feat.name].append(float(e[0]))
                n_samples += 1

        # v89: Store for per-feature z-score normalization at inference
        self.energy.feature_stats = {}
        for feat in self.energy.features.values():
            vals = feature_energies[feat.name]
            if len(vals) > 10:
                mean = float(np.mean(vals))
                std = max(1.0, float(np.std(vals)))
                self.energy.feature_stats[feat.name] = {
                    'mean': mean,
                    'std': std,
                }

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

        v90: P(c) proportional to P_base(c)^(1/T) * exp(-alpha * E(c))

        v90 CHANGES:
        - Metropolis gate REMOVED (was killing 25% of candidates including correct ones)
        - Energy NOT divided by temperature (makes alpha and T independent controls)
        - Repetition penalty reduced (3.0 → 1.0 default, set at construction)

        The energy E(c) is already per-feature z-score normalized (in
        compute_local_energy_batch), so no global z-score is needed here.
        Each feature contributes (E_f - mu_f)/sigma_f * weight_f, giving
        the total energy a well-behaved scale (mean≈0, std≈sqrt(sum(w^2))).
        """
        K = len(candidates)
        if K == 0:
            return np.array([])

        # Base probabilities (numerically stable)
        lps_scaled = log_probs / temperature
        base_probs = np.exp(lps_scaled - np.max(lps_scaled))

        # Hash energy (v89: already per-feature z-score normalized)
        hash_e = self.energy.compute_local_energy_batch(context, candidates)

        # LEGD: P(c) proportional to P_base(c) * exp(-alpha * E(c))
        # v90: Energy NOT scaled by temperature — makes alpha and T independent controls.
        # T controls base distribution sharpness, alpha controls energy influence.
        legd_weights = base_probs * np.exp(-alpha * hash_e)

        # v90: NO Metropolis gate — the soft exp(-alpha * E) reweighting
        # already handles high-energy words. The old gate at 75th percentile
        # killed 25% of candidates including correct ones.

        # Repetition penalty — v90: simplified, less aggressive
        if recent_words and self.rep_penalty > 0:
            recent = recent_words[-self.rep_window:]

            # Unigram penalty
            for i, cid in enumerate(candidates):
                for j, pw in enumerate(recent):
                    if cid == pw:
                        recency = (len(recent) - j) / len(recent)
                        legd_weights[i] *= np.exp(-self.rep_penalty * recency)

            # Bigram blocking — simple: prevent exact bigram repeats
            if len(recent_words) >= 2 and self.bigram_block_window > 0:
                prev_word = recent_words[-1] if recent_words else None
                if prev_word is not None:
                    for j in range(max(0, len(recent_words) - self.bigram_block_window * 2),
                                   len(recent_words) - 1):
                        if recent_words[j] == prev_word and j + 1 < len(recent_words):
                            next_word = recent_words[j + 1]
                            recency = 1.0 - (len(recent_words) - 1 - j) / max(1, self.bigram_block_window * 2)
                            recency = max(0.0, recency)
                            for i, cid in enumerate(candidates):
                                if cid == next_word:
                                    legd_weights[i] *= np.exp(
                                        -self.rep_penalty * recency
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
