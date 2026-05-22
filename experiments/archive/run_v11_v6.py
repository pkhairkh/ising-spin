#!/usr/bin/env python3
"""
v11.5: Longer sequences (max_len=80) + 2K vocab + PMI backoff.

Previous best: PPL=52.04 (2K vocab, PMI=5, scale=1600, max_len=30)
New: max_len=80 → 2.7× longer contexts → more n-gram matches
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ising_spin.model import (
    IsingLMModel, IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
)

CACHE = "/home/z/my-project/ising-spin/cached_fineweb_50k.json"
t0 = time.time()

with open(CACHE) as f:
    texts = json.load(f)
print(f"Loaded {len(texts)} texts")


def beta_sweep(model, n_seqs=15, factors=None):
    if factors is None:
        factors = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5, 2.0]
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
            if len(seq) < 3: continue
            for pos in range(1, len(seq)):
                target_word = seq[pos]
                context_words = seq[:pos]
                context_types = [gen._get_word_type(w) for w in context_words]
                word_type = gen._get_word_type(target_word)
                candidate_list = gen.type_words.get(word_type, [])
                if not candidate_list: continue
                candidate_words = np.array(candidate_list, dtype=np.int64)
                if int(target_word) not in set(candidate_words.tolist()):
                    total_log2_prob += -15 * LOG2_SCALE; total_tokens += 1; continue
                recall_matches = gen.ngram_index.lookup(context_words)
                recall_hit = bool(recall_matches)
                energies = gen._compute_word_energy(pos, candidate_words, word_type, context_words, context_types, recall_hit)
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
                best_ppl = ppl_val; best_factor = factor
    return best_ppl, best_factor, results


def apply_best_beta(model, factor):
    recall_scale = model.recall_scale
    beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
    model.generator.word_sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
    model.generator.beta_word = beta_val
    model.beta_word = beta_val


# ====================================================================
# Config J: 2K vocab + PMI=5 + scale=1600 + max_len=80
# ====================================================================
print("\n" + "=" * 70)
print("Config J: 2K vocab + PMI=5 + scale=1600 + max_len=80")
print("=" * 70)

modelJ = IsingLMModel(
    vocab_min_freq=25, vocab_max_size=2000,
    ngram_max_n=5, ngram_min_count=2,
    recall_scale=1600, pmi_weight=5, field_weight=1,
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
modelJ.train(n_samples=50000, texts=texts)
print(f"Config J training: {time.time()-t0:.1f}s")

print("\nConfig J β sweep:")
pplJ, fJ, resultsJ = beta_sweep(modelJ, n_seqs=15)
print(f"Config J BEST: f={fJ:.2f}, PPL={pplJ:.1f}")
apply_best_beta(modelJ, fJ)
pplJ_full = modelJ.compute_perplexity(n_samples=50)
print(f"Config J PPL (full): {pplJ_full:.2f}")


# ====================================================================
# Config K: 2K vocab + PMI=5 + scale=1600 + max_len=80 + min_count=1
# ====================================================================
print("\n" + "=" * 70)
print("Config K: 2K vocab + PMI=5 + scale=1600 + max_len=80 + min_count=1")
print("=" * 70)

modelK = IsingLMModel(
    vocab_min_freq=25, vocab_max_size=2000,
    ngram_max_n=5, ngram_min_count=1,
    recall_scale=1600, pmi_weight=5, field_weight=1,
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
modelK.train(n_samples=50000, texts=texts)
print(f"Config K training: {time.time()-t0:.1f}s")

print("\nConfig K β sweep:")
pplK, fK, resultsK = beta_sweep(modelK, n_seqs=15)
print(f"Config K BEST: f={fK:.2f}, PPL={pplK:.1f}")
apply_best_beta(modelK, fK)
pplK_full = modelK.compute_perplexity(n_samples=50)
print(f"Config K PPL (full): {pplK_full:.2f}")


# Summary and generate
print(f"\n{'='*70}\nRESULTS SUMMARY\n{'='*70}")
print(f"Config J (2K + PMI=5 + scale=1600 + max_len=80): PPL={pplJ_full:.2f} (f={fJ:.2f})")
print(f"Config K (2K + PMI=5 + scale=1600 + max_len=80 + min_count=1): PPL={pplK_full:.2f} (f={fK:.2f})")
print(f"Previous best (max_len=30): PPL=52.04")

if pplJ_full <= pplK_full:
    best_model, best_ppl, best_f = modelJ, pplJ_full, fJ
    best_name = "2K + PMI=5 + scale=1600 + max_len=80"
else:
    best_model, best_ppl, best_f = modelK, pplK_full, fK
    best_name = "2K + PMI=5 + scale=1600 + max_len=80 + min_count=1"

print(f"\nBest: {best_name}, PPL={best_ppl:.2f}")

# Generate 400 tokens
print(f"\n{'='*70}\nGENERATING 400 TOKENS (PPL={best_ppl:.2f})\n{'='*70}")
for prompt in ["the history of", "science and technology", "research shows that"]:
    result = best_model.generator.generate(prompt=prompt, length=400)
    text = result['text']
    words = text.split()
    n_recalls = sum(1 for d in result['diagnostics'] if d['recall_hit'])
    n_copies = sum(1 for d in result['diagnostics'] if d['copy'])
    print(f"\n--- '{prompt}' ({len(words)} words) ---")
    print(text[:800])
    print(f"  recalls={n_recalls} copies={n_copies}")

result = best_model.generator.generate(prompt="the history of", length=400)
text = result['text']
output_path = "/home/z/my-project/download/ising_v11_400tokens.txt"
with open(output_path, 'w') as f:
    f.write(f"Ising Spin Glass Language Model v11.5 — 400-Token Generation\n")
    f.write(f"Config: {best_name}\n")
    f.write(f"PPL: {best_ppl:.2f}\n")
    f.write(f"Prompt: the history of\n")
    f.write(f"Integer-only: YES (ZERO float operations)\n")
    f.write("=" * 70 + "\n\n")
    f.write(text)
    f.write(f"\n\n--- Results ---\n")
    f.write(f"Config J (max_len=80, min_count=2): PPL={pplJ_full:.2f}\n")
    f.write(f"Config K (max_len=80, min_count=1): PPL={pplK_full:.2f}\n")
    f.write(f"Previous (max_len=30): PPL=52.04\n")
print(f"\nSaved to: {output_path}")
print(f"\nTotal: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}min)")
