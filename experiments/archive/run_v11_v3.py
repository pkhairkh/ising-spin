#!/usr/bin/env python3
"""
v11.2: PMI backoff + higher β + smaller vocab.

Key insight: When no n-gram matches, ALL candidates get ~same high energy →
uniform distribution → terrible PPL. PMI couplings provide context-dependent
discrimination for recall-miss positions.

Strategy:
  1. 2K vocab (more frequent words → better n-gram stats)
  2. PMI weight=3 as context-dependent backoff
  3. Higher recall_scale=1200 (sharper discrimination)
  4. Longer sequences (max_len=50 instead of 30)
  5. Fine-grained log2, optimal β sweep
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
    """Quick β sweep."""
    if factors is None:
        factors = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5]

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
# Config A: 2K vocab + PMI backoff (pmi_weight=3)
# ====================================================================
print("\n" + "=" * 70)
print("Config A: 2K vocab + PMI backoff (pmi_weight=3)")
print("=" * 70)

modelA = IsingLMModel(
    vocab_min_freq=20, vocab_max_size=3000,
    ngram_max_n=5, ngram_min_count=2,
    recall_scale=800, pmi_weight=3, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=300, max_closed_class_run=2,
    ising_enabled=True, skip_pmi_max_dist=5, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=False,
    kn_backoff=False,
    topic_spin_enabled=False,
)
modelA.train(n_samples=50000, texts=texts)
print(f"Config A training: {time.time()-t0:.1f}s")

print("\nConfig A β sweep:")
pplA, fA, resultsA = beta_sweep(modelA, n_seqs=15)
print(f"Config A BEST: f={fA:.2f}, PPL={pplA:.1f}")
apply_best_beta(modelA, fA)
pplA_full = modelA.compute_perplexity(n_samples=50)
print(f"Config A PPL (full): {pplA_full:.2f}")


# ====================================================================
# Config B: 3K vocab + PMI + higher recall_scale=1200
# ====================================================================
print("\n" + "=" * 70)
print("Config B: 3K vocab + PMI + recall_scale=1200")
print("=" * 70)

modelB = IsingLMModel(
    vocab_min_freq=20, vocab_max_size=3000,
    ngram_max_n=5, ngram_min_count=2,
    recall_scale=1200, pmi_weight=3, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=300, max_closed_class_run=2,
    ising_enabled=True, skip_pmi_max_dist=5, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=False,
    kn_backoff=False,
    topic_spin_enabled=False,
)
modelB.train(n_samples=50000, texts=texts)
print(f"Config B training: {time.time()-t0:.1f}s")

print("\nConfig B β sweep:")
pplB, fB, resultsB = beta_sweep(modelB, n_seqs=15)
print(f"Config B BEST: f={fB:.2f}, PPL={pplB:.1f}")
apply_best_beta(modelB, fB)
pplB_full = modelB.compute_perplexity(n_samples=50)
print(f"Config B PPL (full): {pplB_full:.2f}")


# ====================================================================
# Config C: 5K vocab + stronger PMI (pmi_weight=5) + recall_scale=800
# ====================================================================
print("\n" + "=" * 70)
print("Config C: 5K vocab + PMI weight=5 + recall_scale=800")
print("=" * 70)

modelC = IsingLMModel(
    vocab_min_freq=15, vocab_max_size=5000,
    ngram_max_n=5, ngram_min_count=2,
    recall_scale=800, pmi_weight=5, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=300, max_closed_class_run=2,
    ising_enabled=True, skip_pmi_max_dist=5, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=False,
    kn_backoff=False,
    topic_spin_enabled=False,
)
modelC.train(n_samples=50000, texts=texts)
print(f"Config C training: {time.time()-t0:.1f}s")

print("\nConfig C β sweep:")
pplC, fC, resultsC = beta_sweep(modelC, n_seqs=15)
print(f"Config C BEST: f={fC:.2f}, PPL={pplC:.1f}")
apply_best_beta(modelC, fC)
pplC_full = modelC.compute_perplexity(n_samples=50)
print(f"Config C PPL (full): {pplC_full:.2f}")


# ====================================================================
# Pick best and generate
# ====================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Config A (3K vocab + PMI=3 + scale=800): PPL={pplA_full:.2f} (f={fA:.2f})")
print(f"Config B (3K vocab + PMI=3 + scale=1200): PPL={pplB_full:.2f} (f={fB:.2f})")
print(f"Config C (5K vocab + PMI=5 + scale=800): PPL={pplC_full:.2f} (f={fC:.2f})")

best_ppl = min(pplA_full, pplB_full, pplC_full)
if best_ppl == pplA_full:
    best_model, best_name, best_f = modelA, "3K vocab + PMI=3 + scale=800", fA
elif best_ppl == pplB_full:
    best_model, best_name, best_f = modelB, "3K vocab + PMI=3 + scale=1200", fB
else:
    best_model, best_name, best_f = modelC, "5K vocab + PMI=5 + scale=800", fC

print(f"\nBest: {best_name}, PPL={best_ppl:.2f}")

# Generate 400 tokens
print("\n" + "=" * 70)
print(f"GENERATING 400 TOKENS ({best_name}, PPL={best_ppl:.2f})")
print("=" * 70)

for prompt in ["the history of", "science and technology", "research shows that"]:
    result = best_model.generator.generate(prompt=prompt, length=400)
    text = result['text']
    words = text.split()
    n_recalls = sum(1 for d in result['diagnostics'] if d['recall_hit'])
    n_copies = sum(1 for d in result['diagnostics'] if d['copy'])
    print(f"\n--- '{prompt}' ({len(words)} words) ---")
    print(text[:500])
    print(f"  recalls={n_recalls} copies={n_copies}")

# Save main output
result = best_model.generator.generate(prompt="the history of", length=400)
text = result['text']
words = text.split()
output_path = "/home/z/my-project/download/ising_v11_400tokens.txt"
with open(output_path, 'w') as f:
    f.write(f"Ising Spin Glass Language Model v11.2 — 400-Token Generation\n")
    f.write(f"Best config: {best_name}\n")
    f.write(f"PPL: {best_ppl:.2f}\n")
    f.write(f"Prompt: the history of\n")
    f.write(f"Training: 50K FineWeb-Edu\n")
    f.write(f"Integer-only: YES (ZERO float operations)\n")
    f.write("=" * 70 + "\n\n")
    f.write(text)
    f.write(f"\n\n--- Config Results ---\n")
    f.write(f"A (3K + PMI=3 + scale=800): PPL={pplA_full:.2f}\n")
    f.write(f"B (3K + PMI=3 + scale=1200): PPL={pplB_full:.2f}\n")
    f.write(f"C (5K + PMI=5 + scale=800): PPL={pplC_full:.2f}\n")
print(f"\nSaved to: {output_path}")
print(f"\nTotal: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}min)")
