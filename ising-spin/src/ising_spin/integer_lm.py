"""
Integer Language Model — the main class.

Pure integer language model. No neural nets. No torch dependency.
Runs on a Pi 5. Produces grammatically coherent text for simple domains.

Architecture:
  1. BigramModel: P(word | prev) from integer counts (the base)
  2. FeatureHashEnergy: POS rules + lexical facts + skip-gram patterns
  3. LEGD: P(c) ∝ P_base(c) × exp(-α × E_norm(c))
  4. Metropolis gate: hard-reject grammatically invalid tokens
  5. Repetition penalty: prevent loops

Generation loop (per step):
  1. Bigram model proposes top-K candidates with probabilities
  2. Feature hash energy computes E for each candidate
  3. LEGD combines: P(c) ∝ P_base(c) × exp(-α × E_norm(c))
  4. Metropolis gate zeros out candidates above threshold
  5. Repetition penalty reduces probability of recent words
  6. Sample from resulting distribution

Training:
  1. Count bigrams (integer)
  2. NCE train energy tables (integer, with balanced POS negatives)
  3. Calibrate alpha + feature weights (grid search on LEGD probs)

Memory footprint: ~20 MB for V=2000. Runs anywhere.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

from .bigram_model import BigramModel
from .feature_hash_energy import FeatureHashEnergyTable
from .vocabulary import Vocabulary, IDX2POS


class IntegerLM:
    """
    Pure Integer Language Model.

    No neural nets. No torch. No float32 matrix multiplies in the hot path.
    Just integer counting, hash table lookups, and LEGD probability adjustment.

    The LEGD formula:
        P(c | ctx) ∝ P_base(c | prev) × exp(-α × E_norm(c, ctx))

    Lower energy = more likely = higher probability.
    α controls how much the energy correction overrides the base model.
    α=0: pure bigram model. α→∞: pure energy model.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        # Energy table parameters
        n_pos_hashes: int = 2,
        pos_table_size: int = 1009,
        pos_eta: int = 3,
        pos_clip: int = 500,
        n_lex_hashes: int = 3,
        lex_table_size: int = 65537,
        lex_eta: int = 1,
        lex_clip: int = 1000,
        use_skip: bool = True,
        n_skip_hashes: int = 2,
        skip_table_size: int = 65537,
        skip_eta: int = 1,
        skip_clip: int = 800,
        use_trigram: bool = True,
        trigram_weight: int = 1,
        # Energy combination weights
        pos_weight: float = 1.0,
        lex_weight: float = 1.0,
        skip_weight: float = 0.5,
        # Bigram model
        smoothing_alpha: float = 1.0,
        log_prob_scale: int = 100,
        # Generation
        top_k: int = 50,
        alpha: float = 1.0,
        temperature: float = 1.0,
        metropolis_threshold: int = 0,
        rep_penalty: float = 3.0,
        rep_window: int = 5,
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

        # Feature hash energy
        self.energy = FeatureHashEnergyTable(
            vocab_size=self.V,
            word_pos=vocab.word_pos,
            n_pos_types=13,
            n_pos_hashes=n_pos_hashes,
            pos_table_size=pos_table_size,
            pos_eta=pos_eta,
            pos_clip=pos_clip,
            n_lex_hashes=n_lex_hashes,
            lex_table_size=lex_table_size,
            lex_eta=lex_eta,
            lex_clip=lex_clip,
            use_skip=use_skip,
            n_skip_hashes=n_skip_hashes,
            skip_table_size=skip_table_size,
            skip_eta=skip_eta,
            skip_clip=skip_clip,
            use_trigram=use_trigram,
            trigram_weight=trigram_weight,
            pos_weight=pos_weight,
            lex_weight=lex_weight,
            skip_weight=skip_weight,
            seed=seed,
        )

        # Generation parameters
        self.top_k = top_k
        self.alpha = alpha
        self.temperature = temperature
        self.log_prob_scale = log_prob_scale
        self.metropolis_threshold = metropolis_threshold
        self.rep_penalty = rep_penalty
        self.rep_window = rep_window

        # Normalization stats (calibrated after training)
        self._e_mean = 0.0
        self._e_std = 1.0
        self._calibrated = False

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------

    def train(self, sequences: List[List[int]], n_epochs: int = 3, n_negatives: int = 3) -> Dict:
        """
        Train the model: build bigram counts + NCE train energy tables.
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

        # Build test data
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

        best_alpha, best_acc = self.alpha, 0.0
        best_pw = self.energy.pos_weight
        best_lw = self.energy.lex_weight
        best_sw = self.energy.skip_weight

        # Phase 1: search alpha
        for alpha in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            acc = self._eval_rerank(rerank_data, alpha)
            if acc > best_acc:
                best_acc, best_alpha = acc, alpha

        # Phase 2: search feature weights
        print(f"    Best alpha: {best_alpha} (acc={best_acc:.3f})", flush=True)
        print(f"    Searching feature weights...", flush=True)
        for pw in [0.5, 1.0, 2.0, 3.0, 5.0]:
            for lw in [0.5, 1.0, 2.0, 3.0]:
                for sw in [0.0, 0.3, 0.5, 1.0]:
                    self.energy.pos_weight = pw
                    self.energy.lex_weight = lw
                    self.energy.skip_weight = sw
                    acc = self._eval_rerank(rerank_data, best_alpha)
                    if acc > best_acc:
                        best_acc, best_pw, best_lw, best_sw = acc, pw, lw, sw

        self.energy.pos_weight = best_pw
        self.energy.lex_weight = best_lw
        self.energy.skip_weight = best_sw
        self.alpha = best_alpha

        # Metropolis threshold: reject candidates in the top 30% of energy
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
            'pos_weight': best_pw, 'lex_weight': best_lw, 'skip_weight': best_sw,
        }
        print(f"    Result: alpha={best_alpha}, "
              f"pos_w={best_pw}, lex_w={best_lw}, skip_w={best_sw}, "
              f"threshold={best_threshold}, acc={best_acc:.3f}", flush=True)
        return result

    def _eval_rerank(self, data, alpha):
        """Evaluate re-ranking accuracy with given alpha. Uses argmax of LEGD probs."""
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

        P(c) ∝ P_base(c)^(1/T) × exp(-α × E_norm(c) / T)

        This is the proper LEGD (Local Energy-Guided Decoding) formula:
          - Base probability is adjusted by energy in probability space
          - α controls energy influence (0 = pure bigram, ∞ = pure energy)
          - T controls sampling diversity (1.0 = normal, >1 = more random)

        The key insight: we work in PROBABILITY space, not energy space.
        The old formula combined base_e + α×h_scaled and applied Boltzmann,
        which produced P_base^{β×scale} = P_base^{10} — absurdly peaked.
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

        # LEGD: P(c) ∝ P_base(c) × exp(-alpha × h_norm(c) / T)
        legd_weights = base_probs * np.exp(-alpha * h_norm / temperature)

        # Metropolis gate: zero out candidates above threshold
        if self.metropolis_threshold > 0 and alpha > 0:
            gate = hash_e > self.metropolis_threshold
            legd_weights[gate] = 0.0

        # Repetition penalty (in probability space)
        # rep_penalty is in natural-log units: weight *= exp(-penalty × recency / T)
        # Default 3.0 means the most recent word gets exp(-3) ≈ 0.05 weight
        if recent_words and self.rep_penalty > 0:
            recent = recent_words[-self.rep_window:]
            for i, cid in enumerate(candidates):
                for j, pw in enumerate(recent):
                    if cid == pw:
                        recency = (len(recent) - j) / len(recent)
                        legd_weights[i] *= np.exp(-self.rep_penalty * recency / temperature)

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

        Per step:
          1. Bigram model proposes top-K candidates
          2. LEGD computes adjusted probabilities
          3. Sample from distribution
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
        """Generate text from a string prompt. Convenience method."""
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

        LEGD PPL uses the proper formula:
          P_legd(target) = P_base(target) × exp(-α × E_norm(target)) / Z
        where Z = Σ_c P_base(c) × exp(-α × E_norm(c))
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
                    # Target not in top-K: use base prob (energy can't help)
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

                # Compare against 3 random corruptions
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

    def pos_transition_matrix(self) -> np.ndarray:
        """Return 13x13 POS transition energy matrix."""
        return self.energy.get_pos_matrix()

    def diagnostics(self) -> Dict:
        """Full diagnostic report."""
        e_stats = self.energy.statistics()
        b_stats = self.bigram.statistics()
        return {
            'alpha': self.alpha,
            'temperature': self.temperature,
            'metropolis_threshold': self.metropolis_threshold,
            'rep_penalty': self.rep_penalty,
            'rep_window': self.rep_window,
            'pos_weight': e_stats['pos_weight'],
            'lex_weight': e_stats['lex_weight'],
            'skip_weight': e_stats['skip_weight'],
            'energy_mean': self._e_mean,
            'energy_std': self._e_std,
            'calibrated': self._calibrated,
            'energy_memory_mb': e_stats['memory_mb'],
            'bigram_memory_mb': b_stats.get('memory_mb', 0),
            'bigram_nnz': b_stats.get('nonzero_bigrams', 0),
            'pos_nnz': e_stats['pos_nnz'],
            'lex_nnz': e_stats['lex_nnz'],
        }
