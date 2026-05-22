#!/usr/bin/env python3
"""
v11.1: Careful PPL optimization — build on proven v9/v10 baseline.

Strategy:
  Phase 1: v9.0 baseline (proven PPL~73 at 20K, ~112 at 50K)
    - Fine-grained integer log2
    - Longest-only n-gram matching
    - recall_scale=800, β=f*ln(2)/scale
  Phase 2: Add Topic Spin with tiny penalty (20 = 2.5% of recall_scale)
  Phase 3: Try longer n-grams (6-gram, 7-gram)
  Phase 4: β sweep at each step
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ising_spin.model import (
    IsingLMModel, IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
)

CACHE = "/home/z/my-project/ising-spin/cached_fineweb_50k.json"
t0 = time.time()

# Load cached data
print("Loading cached data...")
with open(CACHE) as f:
    texts = json.load(f)
print(f"Loaded {len(texts)} texts")


def beta_sweep(model, n_seqs=15, factors=None):
    """Quick β sweep, returns (best_ppl, best_factor, all_results)."""
    if factors is None:
        factors = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1]

    gen = model.generator
    recall_scale = model.recall_scale
    best_ppl = float('inf')
    best_factor = 0.55
    results = {}

    for factor in factors:
        beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
        sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
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
                if int(target_word) not in set(candidate_words.tolist()):
                    total_log2_prob += -15 * LOG2_SCALE
                    total_tokens += 1
                    continue
                recall_matches = gen.ngram_index.lookup(context_words)
                recall_hit = bool(recall_matches)
                energies = gen._compute_word_energy(
                    pos, candidate_words, word_type,
                    context_words, context_types, recall_hit
                )
                log_probs = sampler.compute_log_probabilities(energies)
                target_idx = np.where(candidate_words == target_word)[0]
                if len(target_idx) > 0:
                    total_log2_prob += int(log_probs[target_idx[0]])
                else:
                    total_log2_prob += -15 * LOG2_SCALE
                total_tokens += 1

        if total_tokens > 0:
            avg_log2 = total_log2_prob / (total_tokens * LOG2_SCALE)
            ppl_val = 2.0 ** (-avg_log2)
            results[factor] = ppl_val
            print(f"  f={factor:.2f}: PPL={ppl_val:.1f}")
            if ppl_val < best_ppl:
                best_ppl = ppl_val
                best_factor = factor

    return best_ppl, best_factor, results


def apply_best_beta(model, factor):
    """Apply the best β factor to the model."""
    recall_scale = model.recall_scale
    beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
    model.generator.word_sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
    model.generator.beta_word = beta_val
    model.beta_word = beta_val


# ====================================================================
# PHASE 1: v9.0 Baseline (5-gram, no Topic Spin)
# ====================================================================
print("\n" + "=" * 70)
print("PHASE 1: v9.0 Baseline (5-gram, no Topic Spin)")
print("=" * 70)

model1 = IsingLMModel(
    vocab_min_freq=15, vocab_max_size=5000,
    ngram_max_n=5, ngram_min_count=2,
    recall_scale=800, pmi_weight=0, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=500, max_closed_class_run=2,
    ising_enabled=False, skip_pmi_max_dist=5, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=False,
    kn_backoff=False,
    topic_spin_enabled=False,
)
model1.train(n_samples=50000, texts=texts)
print(f"Phase 1 training: {time.time()-t0:.1f}s")

print("\nPhase 1 β sweep:")
ppl1, f1, results1 = beta_sweep(model1, n_seqs=15)
print(f"Phase 1 BEST: f={f1:.2f}, PPL={ppl1:.1f}")
apply_best_beta(model1, f1)

# Full PPL
ppl1_full = model1.compute_perplexity(n_samples=50)
print(f"Phase 1 PPL (full): {ppl1_full:.2f}")


# ====================================================================
# PHASE 2: Add Topic Spin (small penalty = 2.5% of recall_scale)
# ====================================================================
print("\n" + "=" * 70)
print("PHASE 2: + Topic Spin (penalty=20, 2.5% of recall_scale)")
print("=" * 70)

model2 = IsingLMModel(
    vocab_min_freq=15, vocab_max_size=5000,
    ngram_max_n=5, ngram_min_count=2,
    recall_scale=800, pmi_weight=0, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=500, max_closed_class_run=2,
    ising_enabled=False, skip_pmi_max_dist=5, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=False,
    kn_backoff=False,
    topic_spin_enabled=True, topic_n_topics=16,
    topic_coherence_penalty=20,
    topic_spin_flip_interval=20,
    topic_context_window=30,
    topic_coupling_scale=50,
)
model2.train(n_samples=50000, texts=texts)
print(f"Phase 2 training: {time.time()-t0:.1f}s")

print("\nPhase 2 β sweep:")
ppl2, f2, results2 = beta_sweep(model2, n_seqs=15)
print(f"Phase 2 BEST: f={f2:.2f}, PPL={ppl2:.1f}")
apply_best_beta(model2, f2)

ppl2_full = model2.compute_perplexity(n_samples=50)
print(f"Phase 2 PPL (full): {ppl2_full:.2f}")


# ====================================================================
# PHASE 3: Longer n-grams (7-gram) + Topic Spin
# ====================================================================
print("\n" + "=" * 70)
print("PHASE 3: 7-gram + Topic Spin (penalty=20)")
print("=" * 70)

model3 = IsingLMModel(
    vocab_min_freq=15, vocab_max_size=5000,
    ngram_max_n=7, ngram_min_count=2,
    recall_scale=800, pmi_weight=0, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=500, max_closed_class_run=2,
    ising_enabled=False, skip_pmi_max_dist=5, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=False,
    kn_backoff=False,
    topic_spin_enabled=True, topic_n_topics=16,
    topic_coherence_penalty=20,
    topic_spin_flip_interval=20,
    topic_context_window=30,
    topic_coupling_scale=50,
)
model3.train(n_samples=50000, texts=texts)
print(f"Phase 3 training: {time.time()-t0:.1f}s")

print("\nPhase 3 β sweep:")
ppl3, f3, results3 = beta_sweep(model3, n_seqs=15)
print(f"Phase 3 BEST: f={f3:.2f}, PPL={ppl3:.1f}")
apply_best_beta(model3, f3)

ppl3_full = model3.compute_perplexity(n_samples=50)
print(f"Phase 3 PPL (full): {ppl3_full:.2f}")


# ====================================================================
# Pick the best model and generate 400 tokens
# ====================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Phase 1 (5-gram, no Topic): PPL={ppl1_full:.2f} (f={f1:.2f})")
print(f"Phase 2 (5-gram + Topic):   PPL={ppl2_full:.2f} (f={f2:.2f})")
print(f"Phase 3 (7-gram + Topic):   PPL={ppl3_full:.2f} (f={f3:.2f})")

best_ppl = min(ppl1_full, ppl2_full, ppl3_full)
if best_ppl == ppl1_full:
    best_model = model1
    best_name = "5-gram baseline"
    best_f = f1
elif best_ppl == ppl2_full:
    best_model = model2
    best_name = "5-gram + Topic Spin"
    best_f = f2
else:
    best_model = model3
    best_name = "7-gram + Topic Spin"
    best_f = f3

print(f"\nBest: {best_name}, PPL={best_ppl:.2f}")

# Generate 400 tokens with best model
print("\n" + "=" * 70)
print(f"GENERATING 400 TOKENS ({best_name}, PPL={best_ppl:.2f})")
print("=" * 70)

result = best_model.generator.generate(prompt="the history of", length=400)
text = result['text']
words = text.split()
n_recalls = sum(1 for d in result['diagnostics'] if d['recall_hit'])
n_copies = sum(1 for d in result['diagnostics'] if d['copy'])
print(f"Words: {len(words)}, recalls={n_recalls}, copies={n_copies}")
print(text)

# Save
output_path = "/home/z/my-project/download/ising_v11_400tokens.txt"
with open(output_path, 'w') as f:
    f.write(f"Ising Spin Glass Language Model v11.0 — 400-Token Generation\n")
    f.write(f"Best config: {best_name}\n")
    f.write(f"PPL: {best_ppl:.2f}\n")
    f.write(f"Prompt: the history of\n")
    f.write(f"Training: 50K FineWeb-Edu, 5K vocab\n")
    f.write(f"recall_scale=800, beta_factor={best_f:.2f}\n")
    f.write(f"Integer-only: YES (ZERO float operations)\n")
    f.write("=" * 70 + "\n\n")
    f.write(text)
    f.write(f"\n\n--- Phase Results ---\n")
    f.write(f"Phase 1 (5-gram, no Topic): PPL={ppl1_full:.2f} (f={f1:.2f})\n")
    f.write(f"Phase 2 (5-gram + Topic):   PPL={ppl2_full:.2f} (f={f2:.2f})\n")
    f.write(f"Phase 3 (7-gram + Topic):   PPL={ppl3_full:.2f} (f={f3:.2f})\n")
    f.write(f"\n--- β Sweep Details ---\n")
    f.write(f"Phase 1:\n")
    for factor in sorted(results1.keys()):
        f.write(f"  f={factor:.2f}: PPL={results1[factor]:.1f}\n")
    f.write(f"Phase 2:\n")
    for factor in sorted(results2.keys()):
        f.write(f"  f={factor:.2f}: PPL={results2[factor]:.1f}\n")
    f.write(f"Phase 3:\n")
    for factor in sorted(results3.keys()):
        f.write(f"  f={factor:.2f}: PPL={results3[factor]:.1f}\n")
    f.write(f"\n--- Diagnostics ---\n")
    f.write(f"Words: {len(words)}\n")
    f.write(f"Recall hits: {n_recalls}\n")
    f.write(f"Copy hits: {n_copies}\n")
print(f"\nSaved to: {output_path}")

total_time = time.time() - t0
print(f"\nTotal: {total_time:.1f}s ({total_time/60:.1f}min)")
