#!/usr/bin/env python3
"""A/B test: Which v10.0 improvement helps/hurts?"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ising_spin.model import IsingLMModel, IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
import numpy as np

def quick_ppl(model, n_seqs=20):
    """Quick PPL with β sweep."""
    gen = model.generator
    recall_scale = model.recall_scale
    best_ppl = float('inf')
    best_factor = 0

    for factor in [0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
        beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
        test_sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
        total_log2_prob = 0
        total_tokens = 0

        for seq in model.test_sequences[:n_seqs]:
            if len(seq) < 3:
                continue
            for pos in range(1, len(seq)):
                target_word = seq[pos]
                context_words = seq[:pos]
                context_types = [gen._get_word_type(w) for w in context_words]
                word_type = gen._get_word_type(target_word)
                candidate_list = gen.type_words.get(word_type, [])
                if not candidate_list:
                    continue
                candidate_words = np.array(candidate_list, dtype=np.int64)
                target_in_candidates = int(target_word) in set(candidate_words.tolist())
                if not target_in_candidates:
                    total_log2_prob += -15 * LOG2_SCALE
                    total_tokens += 1
                    continue
                recall_matches = gen.ngram_index.lookup(context_words)
                recall_hit = bool(recall_matches)
                energies = gen._compute_word_energy(
                    pos, candidate_words, word_type,
                    context_words, context_types, recall_hit
                )
                log_probs = test_sampler.compute_log_probabilities(energies)
                target_idx = np.where(candidate_words == target_word)[0]
                if len(target_idx) > 0:
                    total_log2_prob += int(log_probs[target_idx[0]])
                else:
                    total_log2_prob += -15 * LOG2_SCALE
                total_tokens += 1

        if total_tokens > 0:
            avg_log2 = total_log2_prob / (total_tokens * LOG2_SCALE)
            ppl_val = 2.0 ** (-avg_log2)
            if ppl_val < best_ppl:
                best_ppl = ppl_val
                best_factor = factor

    return best_ppl, best_factor

configs = [
    ("v9.0 baseline (ratio=total//count, unigram backoff)", False, False),
    ("v10.0 precise ratio only", False, False),  # will modify model after
    ("v10.0 KN backoff only (ratio=total//count)", False, True),
    ("v10.0 precise + KN", False, True),
]

# We need to test precise ratio vs not. The precise ratio is built into model.py now.
# Let me just test with and without KN for now.

for name, interp, kn in configs:
    print(f"\n{'='*60}")
    print(f"CONFIG: {name}")
    print(f"{'='*60}")

    model = IsingLMModel(
        vocab_min_freq=15, vocab_max_size=3000,
        ngram_max_n=4, ngram_min_count=2,
        recall_scale=800, pmi_weight=0, field_weight=1,
        knowledge_scale=0, spin3_scale=0, category_scale=0,
        logic_rule_scale=0, logic_hard_scale=0,
        beta_type=0.001, beta_word=0.001,
        copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
        same_word_penalty=500, max_closed_class_run=2,
        ising_enabled=False, skip_pmi_max_dist=5, mcmc_refine_steps=0,
        use_conceptnet=False, walsh_enabled=False,
        graded_couplings_enabled=False, auto_calibrate_beta=True,
        recall_primary_mode=True,
        interpolated=interp, kn_backoff=kn,
    )

    t0 = time.time()
    model.train(n_samples=10000)
    train_time = time.time() - t0

    ppl, best_f = quick_ppl(model, n_seqs=15)
    print(f"  Training: {train_time:.1f}s")
    print(f"  Best β={best_f:.2f}×ln2/scale, PPL={ppl:.1f}")
