"""
Integer Language Model — the main class.

Pure integer language model. No neural nets. No torch dependency.
Runs on a Pi 5. Produces grammatically coherent text for simple domains.

Architecture:
  1. BigramModel: P(word | prev) from integer counts (the base)
  2. FeatureHashEnergy: POS rules + lexical facts + skip-gram patterns
  3. Metropolis gate: hard-reject grammatically invalid tokens
  4. Repetition penalty: prevent loops
  5. Boltzmann sampler: stochastic generation from energy distribution

Generation loop (per step):
  1. Bigram model proposes top-K candidates with log-probabilities
  2. Feature hash energy computes delta_E for each candidate
  3. Metropolis gate hard-rejects candidates above energy threshold
  4. Repetition penalty added for recently-used words
  5. Combined energy = base_energy + alpha * hash_energy + rep_penalty
  6. Boltzmann sample from adjusted distribution

Training:
  1. Count bigrams (integer)
  2. NCE train energy tables (integer)
  3. Calibrate alpha/beta/weights (grid search)

Memory footprint: ~20 MB for V=2000. Runs anywhere.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

from .bigram_model import BigramModel
from .feature_hash_energy import FeatureHashEnergyTable
from .boltzmann import IntegerBoltzmannSampler
from .vocabulary import Vocabulary, IDX2POS


class IntegerLM:
    """
    Pure Integer Language Model.

    No neural nets. No torch. No float32 matrix multiplies in the hot path.
    Just integer counting, hash table lookups, and Boltzmann sampling.

    Combines:
      - BigramModel: base probability P(word | prev) from counts
      - FeatureHashEnergy: POS + lexical + skip-gram energy correction
      - Metropolis gate: hard rejection for bad grammar
      - Repetition penalty: prevent loops
      - Boltzmann sampler: stochastic selection
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
        beta: float = 0.01,
        metropolis_threshold: int = 0,
        rep_penalty: float = 50.0,
        rep_window: int = 5,
        # Seed
        seed: int = 42,
    ):
        self.vocab = vocab
        self.V = vocab.V
        self.seed = seed

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
        self.log_prob_scale = log_prob_scale
        self.metropolis_threshold = metropolis_threshold
        self.rep_penalty = rep_penalty
        self.rep_window = rep_window

        # Boltzmann sampler
        self.beta = beta
        self.sampler = IntegerBoltzmannSampler(beta=beta, max_delta=50000)

        # Normalization stats (calibrated after training)
        self._e_mean = 0.0
        self._e_std = 1.0
        self._b_mean = 0.0
        self._b_std = 1.0
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
        Calibrate alpha, beta, metropolis_threshold, and feature weights.
        """
        print("  Calibrating...", flush=True)

        # Collect energy samples
        e_samples, b_samples = [], []
        for seq in sequences[:300]:
            if len(seq) < 3:
                continue
            for pos in range(1, min(len(seq), 5)):
                ctx = seq[:pos]
                target = seq[pos]
                if 0 <= target < self.V:
                    e_samples.append(self.energy.compute_local_energy(ctx, target))
                candidates, log_probs = self.bigram.get_top_k(ctx, k=self.top_k)
                b_samples.extend((-(log_probs * self.log_prob_scale)).astype(np.int64).tolist())

        if e_samples:
            self._e_mean = float(np.mean(e_samples))
            self._e_std = max(1.0, float(np.std(e_samples)))
            self._b_mean = float(np.mean(b_samples))
            self._b_std = max(1.0, float(np.std(b_samples)))

        print(f"    Energy: mean={self._e_mean:.1f}, std={self._e_std:.1f}", flush=True)
        print(f"    Base:   mean={self._b_mean:.1f}, std={self._b_std:.1f}", flush=True)

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
            return {'alpha': self.alpha, 'beta': self.beta}

        print(f"    Grid search on {len(rerank_data)} pairs...", flush=True)

        best_alpha, best_beta, best_acc = self.alpha, self.beta, 0.0
        best_pw, best_lw, best_sw = self.energy.pos_weight, self.energy.lex_weight, self.energy.skip_weight

        # Phase 1: search alpha/beta
        for alpha in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
            for beta in [0.001, 0.005, 0.01, 0.05, 0.1]:
                acc = self._eval_rerank(rerank_data, alpha, beta)
                if acc > best_acc:
                    best_acc, best_alpha, best_beta = acc, alpha, beta

        # Phase 2: search feature weights
        print(f"    Best alpha/beta: {best_alpha}, {best_beta} (acc={best_acc:.3f})", flush=True)
        print(f"    Searching feature weights...", flush=True)
        for pw in [0.5, 1.0, 2.0, 3.0, 5.0]:
            for lw in [0.5, 1.0, 2.0, 3.0]:
                for sw in [0.0, 0.3, 0.5, 1.0]:
                    self.energy.pos_weight = pw
                    self.energy.lex_weight = lw
                    self.energy.skip_weight = sw
                    acc = self._eval_rerank(rerank_data, best_alpha, best_beta)
                    if acc > best_acc:
                        best_acc, best_pw, best_lw, best_sw = acc, pw, lw, sw

        self.energy.pos_weight = best_pw
        self.energy.lex_weight = best_lw
        self.energy.skip_weight = best_sw
        self.alpha = best_alpha
        self.beta = best_beta

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
        self.sampler = IntegerBoltzmannSampler(beta=best_beta, max_delta=50000)
        self._calibrated = True

        result = {
            'alpha': best_alpha, 'beta': best_beta,
            'metropolis_threshold': best_threshold,
            'rerank_acc': best_acc,
            'pos_weight': best_pw, 'lex_weight': best_lw, 'skip_weight': best_sw,
        }
        print(f"    Result: alpha={best_alpha}, beta={best_beta}, "
              f"pos_w={best_pw}, lex_w={best_lw}, skip_w={best_sw}, "
              f"threshold={best_threshold}, acc={best_acc:.3f}", flush=True)
        return result

    def _eval_rerank(self, data, alpha, beta):
        """Evaluate re-ranking accuracy with given alpha/beta."""
        correct = total = 0
        sampler = IntegerBoltzmannSampler(beta=beta, max_delta=50000)
        for ctx, cands, lps, tidx in data:
            combined = self._combined_energy(ctx, cands, lps, alpha)
            if len(combined) == 0:
                continue
            idx = sampler.sample(combined)
            if idx == tidx:
                correct += 1
            total += 1
        return correct / max(1, total)

    # -------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------

    def generate(self, prompt_ids: List[int], length: int = 100) -> List[int]:
        """
        Generate text using integer-only energy-guided decoding.

        Per step:
          1. Bigram model proposes top-K candidates
          2. Feature hash energy computes delta_E
          3. Metropolis gate rejects bad candidates
          4. Repetition penalty added
          5. Boltzmann sample from combined distribution
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

            combined = self._combined_energy(
                generated, candidates, log_probs, self.alpha,
                recent_words=generated[-self.rep_window:]
            )

            idx = self.sampler.sample(combined) if len(combined) > 1 else 0
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

    def _combined_energy(
        self,
        context: List[int],
        candidates: np.ndarray,
        log_probs: np.ndarray,
        alpha: float,
        recent_words: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Compute combined energy: base + alpha*hash + rep_penalty."""
        K = len(candidates)
        if K == 0:
            return np.array([], dtype=np.int64)

        # Base energy (negative log-prob scaled to integer)
        base_e = -(log_probs * self.log_prob_scale).astype(np.int64)

        # Hash energy (z-score normalized, rescaled to base std)
        hash_e = self.energy.compute_local_energy_batch(context, candidates)
        if self._e_std > 0:
            h_norm = (hash_e.astype(np.float64) - self._e_mean) / self._e_std
            h_scaled = (h_norm * self._b_std).astype(np.int64)
        else:
            h_scaled = np.zeros(K, dtype=np.int64)

        # Metropolis gate
        if self.metropolis_threshold > 0 and alpha > 0:
            h_scaled[hash_e > self.metropolis_threshold] = 100000

        # Repetition penalty
        rep_pen = np.zeros(K, dtype=np.int64)
        if recent_words and self.rep_penalty > 0:
            recent = recent_words[-self.rep_window:]
            for i, cid in enumerate(candidates):
                for j, pw in enumerate(recent):
                    if cid == pw:
                        rep_pen[i] += int(self.rep_penalty * (len(recent) - j) / len(recent))

        return base_e + (h_scaled * alpha).astype(np.int64) + rep_pen

    # -------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------

    def perplexity(self, sequences: List[List[int]], n_samples: int = 100) -> Dict:
        """Compute base PPL and energy-guided PPL."""
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
                    combined = self._combined_energy(ctx, cands, lps)
                    e_min = np.min(combined)
                    deltas = np.clip((combined - e_min).astype(np.float64), 0, 700)
                    weights = np.exp(-deltas * self.beta)
                    total_w = np.sum(weights)
                    if total_w > 0:
                        tp = weights[tidx] / total_w
                        legd_lps.append(float(np.log(max(tp, 1e-300))))
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
            'beta': self.beta,
            'metropolis_threshold': self.metropolis_threshold,
            'rep_penalty': self.rep_penalty,
            'rep_window': self.rep_window,
            'pos_weight': e_stats['pos_weight'],
            'lex_weight': e_stats['lex_weight'],
            'skip_weight': e_stats['skip_weight'],
            'energy_mean': self._e_mean,
            'energy_std': self._e_std,
            'base_mean': self._b_mean,
            'base_std': self._b_std,
            'calibrated': self._calibrated,
            'energy_memory_mb': e_stats['memory_mb'],
            'bigram_memory_mb': b_stats.get('memory_mb', 0),
            'bigram_nnz': b_stats.get('nonzero_bigrams', 0),
            'pos_nnz': e_stats['pos_nnz'],
            'lex_nnz': e_stats['lex_nnz'],
        }
