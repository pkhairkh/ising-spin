#!/usr/bin/env python3
"""
Precise PPL comparison: v8.0 model vs pure recall diagnostic.
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.model import IsingLMModel, IntegerBoltzmannSampler

def precise_ppl(gen, test_seqs, n_samples=50):
    """Compute PPL matching the diagnostic exactly."""
    sampler = gen.word_sampler
    total_log_prob = 0.0
    total_tokens = 0
    n_target_missing = 0
    
    eval_seqs = test_seqs[:n_samples]
    
    for seq in eval_seqs:
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
            
            # NO top-k filtering — use ALL candidates
            
            target_in_candidates = int(target_word) in set(candidate_words.tolist())
            if not target_in_candidates:
                total_log_prob += -15.0
                total_tokens += 1
                n_target_missing += 1
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
                total_log_prob += float(log_probs[target_idx[0]])
            else:
                total_log_prob += -15.0
            
            total_tokens += 1
    
    if total_tokens == 0:
        return float('inf'), 0
    
    avg_log_prob = total_log_prob / total_tokens
    perplexity = math.exp(-avg_log_prob)
    return perplexity, total_tokens


def main():
    print("=" * 70)
    print("PRECISE PPL COMPARISON: v8.0 vs Pure Recall")
    print("=" * 70)
    
    # Train the model
    model = IsingLMModel(
        vocab_min_freq=3,
        vocab_max_size=8000,
        ngram_max_n=5,
        ngram_min_count=1,
        pmi_window=5,
        pmi_min_count=2,
        pmi_cap=10,
        
        recall_scale=800,
        pmi_weight=0,
        field_weight=1,
        knowledge_scale=0,
        spin3_scale=0,
        category_scale=0,
        logic_rule_scale=0,
        logic_hard_scale=0,
        
        beta_type=0.001,
        beta_word=0.001,
        copy_enabled=False,  # Disable copy for clean comparison
        same_word_penalty=0,  # Disable penalty for clean comparison
        ising_enabled=False,
        mcmc_refine_steps=0,
        use_conceptnet=False,
        walsh_enabled=False,
        graded_couplings_enabled=False,
        recall_primary_mode=True,
        auto_calibrate_beta=False,  # We'll set β manually
    )
    
    model.train(n_samples=20000)
    gen = model.generator
    
    # ====================================================================
    # Test 1: Pure recall with β = 0.5*ln2/scale (theoretical optimal)
    # ====================================================================
    beta_theory = 0.5 * math.log(2) / 800
    gen.word_sampler = IntegerBoltzmannSampler(beta=beta_theory, max_delta=5000)
    gen.beta_word = beta_theory
    
    # Remove all layers for pure recall
    gen.graded_couplings = None
    gen.knowledge_layer = None
    gen.category_layer = None
    gen.markov_logic_layer = None
    gen.ising_enabled = False
    
    ppl, n_tok = precise_ppl(gen, model.test_sequences, n_samples=50)
    print(f"\nPure recall (β=0.5*ln2/scale={beta_theory:.6f}): PPL = {ppl:.2f} ({n_tok} tokens)")
    
    # ====================================================================
    # Test 2: Sweep β to find optimal
    # ====================================================================
    print("\nβ sweep for pure recall:")
    for beta_mult in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
        beta = beta_mult * math.log(2) / 800
        gen.word_sampler = IntegerBoltzmannSampler(beta=beta, max_delta=5000)
        gen.beta_word = beta
        ppl, _ = precise_ppl(gen, model.test_sequences, n_samples=50)
        print(f"  β={beta:.6f} ({beta_mult:.1f}× ln2/scale): PPL = {ppl:.2f}")
    
    # ====================================================================
    # Test 3: Recall + field
    # ====================================================================
    print("\nRecall + field:")
    beta_theory = 0.5 * math.log(2) / 800
    gen.word_sampler = IntegerBoltzmannSampler(beta=beta_theory, max_delta=5000)
    gen.beta_word = beta_theory
    ppl, _ = precise_ppl(gen, model.test_sequences, n_samples=50)
    print(f"  PPL = {ppl:.2f}")
    
    # ====================================================================
    # Test 4: Now add back same_word_penalty
    # ====================================================================
    print("\nRecall + field + same_word_penalty:")
    gen.same_word_penalty = 200  # Small penalty
    ppl, _ = precise_ppl(gen, model.test_sequences, n_samples=50)
    print(f"  same_word_penalty=200: PPL = {ppl:.2f}")
    
    gen.same_word_penalty = 50000  # Large penalty
    ppl, _ = precise_ppl(gen, model.test_sequences, n_samples=50)
    print(f"  same_word_penalty=50000: PPL = {ppl:.2f}")
    
    gen.same_word_penalty = 0  # Reset
    ppl, _ = precise_ppl(gen, model.test_sequences, n_samples=50)
    print(f"  same_word_penalty=0: PPL = {ppl:.2f}")


if __name__ == "__main__":
    main()
