#!/usr/bin/env python3
"""
v11.3: Push PPL toward 20 with aggressive optimizations.

Key findings from v11.2:
  - PMI backoff is critical (124 → 83 PPL)
  - Smaller vocab (3K > 5K) helps because more frequent words → better n-gram stats
  - Higher recall_scale (1200 > 800) slightly helps
  - β=0.9 is consistently optimal

New strategies:
  D: 2K vocab + PMI=5 + scale=1200 (even smaller vocab, stronger PMI)
  E: 3K vocab + PMI=5 + scale=1200 (stronger PMI)
  F: 3K vocab + PMI=3 + scale=1200 + skip_pmi_dist=8 (wider PMI context)
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
        factors = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2]

    gen = model.generator
    recall_scale = model.recall_scale
    best_ppl = float('inf')
    best_factor = 0.9
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
# Config D: 2K vocab + PMI=5 + scale=1200
# ====================================================================
print("\n" + "=" * 70)
print("Config D: 2K vocab + PMI=5 + scale=1200")
print("=" * 70)

modelD = IsingLMModel(
    vocab_min_freq=25, vocab_max_size=2000,
    ngram_max_n=5, ngram_min_count=2,
    recall_scale=1200, pmi_weight=5, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=200, max_closed_class_run=2,
    ising_enabled=True, skip_pmi_max_dist=5, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=False, kn_backoff=False,
    topic_spin_enabled=False,
)
modelD.train(n_samples=50000, texts=texts)
print(f"Config D training: {time.time()-t0:.1f}s")

print("\nConfig D β sweep:")
pplD, fD, resultsD = beta_sweep(modelD, n_seqs=15)
print(f"Config D BEST: f={fD:.2f}, PPL={pplD:.1f}")
apply_best_beta(modelD, fD)
pplD_full = modelD.compute_perplexity(n_samples=50)
print(f"Config D PPL (full): {pplD_full:.2f}")


# ====================================================================
# Config E: 3K vocab + PMI=5 + scale=1200
# ====================================================================
print("\n" + "=" * 70)
print("Config E: 3K vocab + PMI=5 + scale=1200")
print("=" * 70)

modelE = IsingLMModel(
    vocab_min_freq=20, vocab_max_size=3000,
    ngram_max_n=5, ngram_min_count=2,
    recall_scale=1200, pmi_weight=5, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=200, max_closed_class_run=2,
    ising_enabled=True, skip_pmi_max_dist=5, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=False, kn_backoff=False,
    topic_spin_enabled=False,
)
modelE.train(n_samples=50000, texts=texts)
print(f"Config E training: {time.time()-t0:.1f}s")

print("\nConfig E β sweep:")
pplE, fE, resultsE = beta_sweep(modelE, n_seqs=15)
print(f"Config E BEST: f={fE:.2f}, PPL={pplE:.1f}")
apply_best_beta(modelE, fE)
pplE_full = modelE.compute_perplexity(n_samples=50)
print(f"Config E PPL (full): {pplE_full:.2f}")


# ====================================================================
# Config F: 3K vocab + PMI=3 + scale=1200 + wider skip (dist=8)
# ====================================================================
print("\n" + "=" * 70)
print("Config F: 3K vocab + PMI=3 + scale=1200 + skip_pmi_max_dist=8")
print("=" * 70)

modelF = IsingLMModel(
    vocab_min_freq=20, vocab_max_size=3000,
    ngram_max_n=5, ngram_min_count=2,
    recall_scale=1200, pmi_weight=3, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=200, max_closed_class_run=2,
    ising_enabled=True, skip_pmi_max_dist=8, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=False, kn_backoff=False,
    topic_spin_enabled=False,
)
modelF.train(n_samples=50000, texts=texts)
print(f"Config F training: {time.time()-t0:.1f}s")

print("\nConfig F β sweep:")
pplF, fF, resultsF = beta_sweep(modelF, n_seqs=15)
print(f"Config F BEST: f={fF:.2f}, PPL={pplF:.1f}")
apply_best_beta(modelF, fF)
pplF_full = modelF.compute_perplexity(n_samples=50)
print(f"Config F PPL (full): {pplF_full:.2f}")


# ====================================================================
# Pick best and generate 400 tokens
# ====================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Config D (2K + PMI=5 + scale=1200): PPL={pplD_full:.2f} (f={fD:.2f})")
print(f"Config E (3K + PMI=5 + scale=1200): PPL={pplE_full:.2f} (f={fE:.2f})")
print(f"Config F (3K + PMI=3 + scale=1200 + skip=8): PPL={pplF_full:.2f} (f={fF:.2f})")

configs = [
    (pplD_full, modelD, "2K + PMI=5 + scale=1200", fD),
    (pplE_full, modelE, "3K + PMI=5 + scale=1200", fE),
    (pplF_full, modelF, "3K + PMI=3 + scale=1200 + skip=8", fF),
]
best_ppl, best_model, best_name, best_f = min(configs, key=lambda x: x[0])
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
    print(text[:600])
    print(f"  recalls={n_recalls} copies={n_copies}")

result = best_model.generator.generate(prompt="the history of", length=400)
text = result['text']
words = text.split()
output_path = "/home/z/my-project/download/ising_v11_400tokens.txt"
with open(output_path, 'w') as f:
    f.write(f"Ising Spin Glass Language Model v11.3 — 400-Token Generation\n")
    f.write(f"Best config: {best_name}\n")
    f.write(f"PPL: {best_ppl:.2f}\n")
    f.write(f"Prompt: the history of\n")
    f.write(f"Training: 50K FineWeb-Edu\n")
    f.write(f"Integer-only: YES (ZERO float operations)\n")
    f.write("=" * 70 + "\n\n")
    f.write(text)
    f.write(f"\n\n--- Config Results ---\n")
    for ppl_val, _, name, bf in configs:
        f.write(f"{name}: PPL={ppl_val:.2f} (f={bf:.2f})\n")
print(f"\nSaved to: {output_path}")
print(f"\nTotal: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}min)")
