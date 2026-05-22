#!/usr/bin/env python3
"""
PPL Diagnostic: Find the right energy scale hierarchy.

The recall energy E = log₂(1/P) * scale is the CORRECT Boltzmann energy.
With β = ln2/scale, it reproduces n-gram probabilities exactly.

All other layers must be SMALL PERTURBATIONS on recall, not dominant forces.

This script tests multiple scale configurations to find the optimal balance.
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.model import IsingLMModel


def quick_ppl(model, n_samples=30):
    """Quick PPL evaluation with minimal output."""
    gen = model.generator
    sampler = gen.word_sampler
    
    total_log_prob = 0.0
    total_tokens = 0
    n_target_missing = 0
    n_target_low_prob = 0  # log_prob < -10
    
    # Energy layer diagnostics
    layer_contributions = {
        'recall': [], 'graded': [], 'knowledge': [],
        'category': [], 'logic': [], 'field': [], 'penalty': []
    }
    
    eval_seqs = model.test_sequences[:n_samples]
    
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
            
            # Top-500 filtering
            if len(candidate_words) > 500:
                if gen.graded_couplings is not None and gen.graded_couplings.word_counts is not None:
                    counts = gen.graded_couplings.word_counts[candidate_words]
                else:
                    counts = -gen.h[candidate_words]
                top_k = np.argsort(counts)[-499:]
                candidate_words = candidate_words[top_k]
                if int(target_word) not in set(candidate_words.tolist()):
                    candidate_words = np.append(candidate_words, target_word)
            
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
                lp = float(log_probs[target_idx[0]])
                total_log_prob += lp
                if lp < -10:
                    n_target_low_prob += 1
            else:
                total_log_prob += -15.0
            
            total_tokens += 1
    
    if total_tokens == 0:
        return float('inf'), {}
    
    avg_log_prob = total_log_prob / total_tokens
    perplexity = math.exp(-avg_log_prob)
    
    diagnostics = {
        'ppl': perplexity,
        'avg_log_prob': avg_log_prob,
        'n_tokens': total_tokens,
        'n_target_missing': n_target_missing,
        'n_target_low_prob': n_target_low_prob,
        'missing_rate': n_target_missing / total_tokens,
        'low_prob_rate': n_target_low_prob / total_tokens,
    }
    
    return perplexity, diagnostics


def main():
    print("=" * 70)
    print("PPL DIAGNOSTIC: Energy Scale Hierarchy")
    print("=" * 70)
    print()
    print("Recall energy E = log₂(1/P) * scale is the CORRECT Boltzmann energy.")
    print("With β = ln2/scale, it reproduces n-gram probabilities exactly.")
    print("All other layers must be SMALL PERTURBATIONS.")
    print()
    
    # Step 1: Train the model ONCE with maximal settings
    print("Training model (20K samples)...")
    t0 = time.time()
    
    model = IsingLMModel(
        vocab_min_freq=3,
        vocab_max_size=8000,
        ngram_max_n=5,
        ngram_min_count=1,
        pmi_window=5,
        pmi_min_count=2,
        pmi_cap=10,
        
        # Start with moderate recall scale
        recall_scale=800,
        pmi_weight=5,
        field_weight=1,
        
        # Knowledge layers (will be scaled down)
        knowledge_scale=15000,
        spin3_scale=50000,
        category_scale=800,
        logic_rule_scale=2000,
        logic_hard_scale=50000,
        
        beta_type=0.001,
        beta_word=0.001,
        copy_enabled=False,  # Disable copy for PPL testing
        copy_min_context=2,
        copy_min_confidence=0.25,
        same_word_penalty=50000,
        max_closed_class_run=2,
        ising_enabled=True,
        skip_pmi_max_dist=5,
        mcmc_refine_steps=2,
        use_conceptnet=True,
        
        walsh_enabled=False,
        graded_couplings_enabled=True,
        coupling_scale=1000,
        trigram_scale=2000,
        auto_calibrate_beta=True,
    )
    
    model.train(n_samples=20000)
    t_train = time.time() - t0
    print(f"Training time: {t_train:.1f}s")
    
    gen = model.generator
    kl = model.knowledge_layer
    gc = model.graded_couplings
    
    # Print recall statistics
    print(f"\nModel stats:")
    print(f"  Vocab size: {model.vocab_size}")
    print(f"  Test sequences: {len(model.test_sequences)}")
    print(f"  J₂ non-zero: {gc.J2.nnz:,}" if gc else "  No graded couplings")
    print(f"  Knowledge triples: {kl.n_triples}")
    
    # ====================================================================
    # TEST 1: Pure Recall (all other layers OFF)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Pure Recall Energy (all other layers OFF)")
    print("=" * 70)
    
    # Save original settings
    orig_gc = gen.graded_couplings
    orig_knowledge = gen.knowledge_layer
    orig_category = gen.category_layer
    orig_logic = gen.markov_logic_layer
    
    # Disable all layers except recall
    gen.graded_couplings = None
    gen.knowledge_layer = None
    gen.category_layer = None
    gen.markov_logic_layer = None
    gen.ising_enabled = False  # Disable PMI fallback too
    
    # Set β correctly for recall-only
    # β = ln2 / recall_scale = 0.693 / 800 ≈ 0.000866
    beta_recall = math.log(2) / model.recall_scale
    gen.word_sampler = type(gen.word_sampler)(beta=beta_recall, max_delta=5000, scale=1 << 30)
    gen.beta_word = beta_recall
    
    ppl_recall, diag_recall = quick_ppl(model, n_samples=30)
    print(f"  β = ln2/recall_scale = {beta_recall:.6f}")
    print(f"  PPL (recall only) = {ppl_recall:.2f}")
    print(f"  Avg log-prob = {diag_recall['avg_log_prob']:.4f}")
    print(f"  Target missing rate = {diag_recall['missing_rate']:.1%}")
    print(f"  Target low-prob rate = {diag_recall['low_prob_rate']:.1%}")
    
    # ====================================================================
    # TEST 2: Recall + Field (unigram frequency bias)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Recall + Local Field")
    print("=" * 70)
    
    gen.graded_couplings = None
    gen.knowledge_layer = None
    gen.category_layer = None
    gen.markov_logic_layer = None
    gen.ising_enabled = False
    
    # Auto-calibrate β for this config
    from ising_spin.model import GradedCouplings
    gen.word_sampler = type(gen.word_sampler)(beta=beta_recall, max_delta=5000, scale=1 << 30)
    gen.beta_word = beta_recall
    
    ppl_recall_field, diag_rf = quick_ppl(model, n_samples=30)
    print(f"  PPL (recall + field) = {ppl_recall_field:.2f}")
    
    # ====================================================================
    # TEST 3: Recall + Graded Couplings (small scale)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Recall + Graded Couplings at VARIOUS scales")
    print("=" * 70)
    
    # Test different coupling scales
    for cs in [10, 50, 100, 200, 500, 1000]:
        # Temporarily modify the graded coupling scale
        gen.graded_couplings = orig_gc
        gen.knowledge_layer = None
        gen.category_layer = None
        gen.markov_logic_layer = None
        gen.ising_enabled = False
        
        # Rebuild J₂ with new scale
        # We can't easily rebuild, so just scale the energy contribution
        # by adjusting the β or the coupling_scale effect
        # Actually, we can modify the coupling_scale and rebuild
        orig_coupling_scale = orig_gc.coupling_scale
        orig_trigram_scale = orig_gc.trigram_scale
        
        # Scale the J2 and J3 data proportionally
        scale_factor = cs / orig_coupling_scale
        orig_gc.J2.data = (orig_gc.J2.data * scale_factor).astype(np.int64)
        # Can't easily scale J3, skip for now
        
        # Auto-calibrate β for this scale
        # Use the recall-calibrated β as starting point
        gen.word_sampler = type(gen.word_sampler)(beta=beta_recall, max_delta=5000, scale=1 << 30)
        gen.beta_word = beta_recall
        
        ppl, diag = quick_ppl(model, n_samples=30)
        print(f"  coupling_scale={cs:>5}: PPL = {ppl:.2f}  (missing={diag['missing_rate']:.1%}, low_prob={diag['low_prob_rate']:.1%})")
        
        # Restore original J2
        orig_gc.J2.data = (orig_gc.J2.data / scale_factor).astype(np.int64)
    
    # Restore J2 to original
    orig_gc.J2.data = (orig_gc.J2.data * (orig_coupling_scale / orig_coupling_scale)).astype(np.int64)
    
    # ====================================================================
    # TEST 4: Recall + Knowledge at various scales
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Recall + Knowledge at VARIOUS scales")
    print("=" * 70)
    
    for ks in [0, 50, 100, 200, 400, 800, 1500, 5000, 15000]:
        gen.graded_couplings = None
        gen.knowledge_layer = orig_knowledge
        gen.category_layer = None
        gen.markov_logic_layer = None
        gen.ising_enabled = False
        
        # Scale knowledge
        if ks > 0:
            scale_factor = ks / 15000
            orig_knowledge.h_knowledge = (orig_knowledge.h_knowledge * scale_factor).astype(np.int64)
            # Scale J3
            for key in orig_knowledge.J3:
                orig_knowledge.J3[key] = [(w, int(c * scale_factor)) for w, c in orig_knowledge.J3[key]]
            orig_knowledge.spin3_scale = int(orig_knowledge.spin3_scale * scale_factor) if hasattr(orig_knowledge, 'spin3_scale') else ks
        
        gen.word_sampler = type(gen.word_sampler)(beta=beta_recall, max_delta=5000, scale=1 << 30)
        gen.beta_word = beta_recall
        
        ppl, diag = quick_ppl(model, n_samples=30)
        print(f"  knowledge_scale={ks:>6}: PPL = {ppl:.2f}  (missing={diag['missing_rate']:.1%}, low_prob={diag['low_prob_rate']:.1%})")
        
        # Restore knowledge
        if ks > 0:
            restore_factor = 15000 / ks
            orig_knowledge.h_knowledge = (orig_knowledge.h_knowledge * restore_factor).astype(np.int64)
            for key in orig_knowledge.J3:
                orig_knowledge.J3[key] = [(w, int(c * restore_factor)) for w, c in orig_knowledge.J3[key]]
    
    # ====================================================================
    # TEST 5: Auto-calibrate β for recall-only
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 5: β sensitivity for recall-only model")
    print("=" * 70)
    
    gen.graded_couplings = None
    gen.knowledge_layer = None
    gen.category_layer = None
    gen.markov_logic_layer = None
    gen.ising_enabled = False
    
    for beta_mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]:
        beta = beta_recall * beta_mult
        gen.word_sampler = type(gen.word_sampler)(beta=beta, max_delta=5000, scale=1 << 30)
        gen.beta_word = beta
        
        ppl, diag = quick_ppl(model, n_samples=30)
        print(f"  β={beta:.6f} ({beta_mult:.1f}×): PPL = {ppl:.2f}")
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"\n  Recall-only PPL (β=ln2/scale): {ppl_recall:.2f}")
    print(f"  v7.0 PPL (6-layer, current scales): 3.2e22")
    print(f"  Uniform random PPL: ~{model.vocab_size}")
    print(f"\n  The recall-only model should be the BASELINE.")
    print(f"  Any layer addition that INCREASES PPL over baseline is HURTING.")
    
    # Restore original settings
    gen.graded_couplings = orig_gc
    gen.knowledge_layer = orig_knowledge
    gen.category_layer = orig_category
    gen.markov_logic_layer = orig_logic
    gen.ising_enabled = True
    gen.word_sampler = type(gen.word_sampler)(beta=model.beta_word, max_delta=5000, scale=1 << 30)


if __name__ == "__main__":
    main()
