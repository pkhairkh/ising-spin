#!/usr/bin/env python3
"""
PoC Runner for V13 Fixed Ising Language Model.

V13 fixes V12's regressions:
  R1: Restore full recall strength (no interpolation dilution)
  R2: Copy mechanism reform (type check, same-word block, rate cap, min_context=3)
  R3: V11-style strong recall + KN fallback only
  R4: Improved anti-repetition (proportional same-word penalty, n-gram tracking)
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.v13_fixed_model import FixedV13Model
from ising_spin.type_system import IDX2POS, POS2IDX, N_POS


def run_v13_poc():
    print("=" * 70)
    print("V13 FIXED ISING + EXACT RECALL — PoC")
    print("=" * 70)
    print()
    print("V12 REGRESSION DIAGNOSIS:")
    print("  - Interpolation diluted recall from 100% to 70%")
    print("  - Copy at 60% rate bypassed ALL fixes")
    print("  - KN divided by total, weakening bonuses")
    print("  - min_context=2 made copy too aggressive")
    print()
    print("V13 FIXES:")
    print("  R1: Full recall strength + PMI as supplement (not replacement)")
    print("  R2: Copy reform: type check, same-word block, rate cap, min_ctx=3")
    print("  R3: V11-style recall + KN fallback (not KN for everything)")
    print("  R4: Proportional same-word penalty, n-gram repetition tracking")
    print()

    t0 = time.time()

    model = FixedV13Model(
        # === V13-specific fixes ===
        pmi_supplement_weight=5,           # R1: PMI supplement (increased for more diversity)
        unigram_supplement_weight=2,       # R1: Unigram supplement (increased)
        max_consecutive_copies=3,          # R2: Cap consecutive copies
        copy_type_check=True,             # R2: Check type compat for copy
        copy_same_word_block=True,         # R2: Block same-word copy
        same_word_penalty_strength=2,      # R4: Penalty = 2× max_recall
        ngram_repetition_penalty=500,      # R4: Penalty for repeated n-grams (raised significantly)
        max_closed_class_run=2,           # F1: Max consecutive closed-class
        closed_class_loop_penalty=300,     # F1: Penalty for closed-class loops
        use_fixed_index=True,             # R3: V11 recall + KN fallback

        # === V11 exact recall ===
        recall_scale=300,                  # Balanced — not too dominant, not too weak
        context_weight_factor=3,           # Moderate exponential scaling (3^k not 4^k)
        copy_min_context=3,               # R2: Raised from 2 to 3
        copy_min_confidence=0.4,          # Raised from 0.3 — be more selective about copy
        copy_enabled=True,
        ngram_max_n=5,
        ngram_min_count=1,
        
        # === V10 autoregressive ===
        beta_type=0.01,
        beta_word=0.1,                    # Increased for more deterministic recall-following
        top_k_words=300,
        use_idf_coupling=True,
        idf_scale=8,
        field_weight=0.0005,
        coupling_weight=5,
        
        # === V8 training ===
        vocab_min_freq=5,
        vocab_max_size=3000,
        seq_len=20,
        window=5,
        pmi_cap=10,
        min_cooc=2,
        pmi_weight=3,
        hebbian_weight=1,
        semantic_weight=1,
        dep_weight=2,
        grammar_penalty=60,
        emission_bonus=100,
        emission_penalty=500,
        phase1_beta=200,
        phase2_beta=500,
        phase3_beta=1000,
        total_sweeps=60,
        use_caldera=True,
        nmf_factors=64,
        nmf_iterations=30,
        nmf_n_top=10,
        use_spacy=True,
        spacy_max_texts=5000,
        transition_min_count=20,
        sw_cluster_enabled=True,
        sw_wolff_variant=True,
        n_replicas=4,
        pt_swap_interval=5,
        entropy_T_ent=50,
        entropy_delta_E_window=500,
        entropy_precision=100,
        demon_initial_energy=1000,
        use_demon=True,
        wl_warmup_sweeps=30,
        wl_e_min=-50000,
        wl_e_max=50000,
        wl_scale=10000,
        use_lifted=True,
        marginal_entropy_scale=10000,
        use_marginal_entropy=True,
    )

    model.train(n_samples=20000)
    
    t_train = time.time() - t0
    print(f"\nTraining time: {t_train:.1f}s")

    gen = model.v13_generator
    
    print(f"\n  Vocab size: {gen.vocab_size}")
    print(f"  Fixed index: {gen.use_fixed_index}")
    
    # === Phase 1: Quick generation test ===
    print("\n" + "=" * 70)
    print("PHASE 1: QUICK GENERATION TEST")
    print("=" * 70)

    prompts = ["the", "a", "in", "science", "research", "students", "he", "they",
               "to", "of", "for", "education", "we", "this", "that"]
    
    for prompt in prompts:
        try:
            result = gen.generate(prompt=prompt, length=15)
            text = result['text']
            types_str = ' '.join(result['type_names'])
            n_copies = sum(1 for d in result['diagnostics'] if d.get('copy_used'))
            n_recall_hits = sum(1 for d in result['diagnostics'] if d.get('recall_hit'))
            
            has_of_the_loop = "of the of the" in text
            has_double_det = any(
                result['type_names'][i] == 'DET' and result['type_names'][i+1] == 'DET'
                for i in range(len(result['type_names']) - 1)
            )
            
            flag = ""
            if has_of_the_loop:
                flag += " LOOP"
            if has_double_det:
                flag += " DOUBLE-DET"
            
            print(f"  '{prompt}' -> {text}{flag}")
            print(f"           [{types_str}] copies={n_copies} recalls={n_recall_hits}")
        except Exception as e:
            print(f"  '{prompt}' -> ERROR: {e}")
            import traceback
            traceback.print_exc()

    # === Phase 2: V11 vs V13 comparison ===
    print("\n" + "=" * 70)
    print("V11 vs V13 SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    
    from ising_spin.exact_recall_v11_model import ExactRecallV11Generator
    v11_gen = ExactRecallV11Generator(
        sampler=model.v8_model.sampler,
        vocab=model.v8_model.vocab,
        ngram_index=model.ngram_index,
        recall_scale=500,
        context_weight_factor=4,
        copy_min_context=3,
        copy_min_confidence=0.3,
        copy_enabled=True,
        beta_type=0.01,
        beta_word=0.05,
        top_k_words=300,
        use_idf_coupling=True,
        idf_scale=8,
        field_weight=0.0005,
        coupling_weight=5,
    )
    
    comparison_prompts = ["the", "science", "research", "students", "education",
                          "to", "of", "in", "for", "he"]
    
    for prompt in comparison_prompts:
        try:
            v11_result = v11_gen.generate(prompt=prompt, length=20)
            v11_text = v11_result['text']
            
            v13_result = gen.generate(prompt=prompt, length=20)
            v13_text = v13_result['text']
            
            v11_flags = []
            if "of the of the" in v11_text:
                v11_flags.append("LOOP")
            v11_words = v11_text.split()
            for i in range(len(v11_words) - 1):
                if v11_words[i] in {"the", "a"} and v11_words[i+1] in {"the", "a"}:
                    v11_flags.append("DOUBLE-DET")
            
            v13_flags = []
            if "of the of the" in v13_text:
                v13_flags.append("LOOP")
            v13_words = v13_text.split()
            for i in range(len(v13_words) - 1):
                if v13_words[i] in {"the", "a"} and v13_words[i+1] in {"the", "a"}:
                    v13_flags.append("DOUBLE-DET")
            
            v11_flag_str = f" [{','.join(v11_flags)}]" if v11_flags else ""
            v13_flag_str = f" [{','.join(v13_flags)}]" if v13_flags else ""
            
            print(f"\n  Prompt: '{prompt}'")
            print(f"  V11: {v11_text}{v11_flag_str}")
            print(f"  V13: {v13_text}{v13_flag_str}")
        except Exception as e:
            print(f"  '{prompt}' -> ERROR: {e}")

    # === Phase 3: Full evaluation ===
    print("\n" + "=" * 70)
    print("PHASE 3: FULL EVALUATION")
    print("=" * 70)
    
    all_text = []
    all_metrics = {}
    n_eval = 30
    
    for i in range(n_eval):
        try:
            words, types = model.generate_raw(length=20)
            metrics = model.evaluate_grammar(words, types)
            text = model.vocab.decode(words)
            all_text.append(text)
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0) + v
        except Exception as e:
            print(f"  Sample {i+1}: ERROR: {e}")
    
    print(f"\nGrammar patterns across {n_eval} samples:")
    for k, v in sorted(all_metrics.items()):
        per_sample = v / n_eval
        print(f"  {k}: {v} total ({per_sample:.1f}/sample)")
    
    rep_count = all_metrics.get("repeated_words", 0)
    total_words = n_eval * 20
    rep_rate = rep_count / max(1, total_words - n_eval)
    print(f"\n  Repetition rate: {rep_count}/{total_words - n_eval} ({rep_rate:.3f})")
    
    det_noun = all_metrics.get("det_noun", 0)
    det_non = all_metrics.get("det_non_noun", 0)
    det_total = det_noun + det_non
    if det_total > 0:
        print(f"  DET->NOUN accuracy: {det_noun}/{det_total} ({100*det_noun/det_total:.1f}%)")
    
    # Pathology check
    print("\n--- PATHOLOGY CHECK ---")
    n_of_the_loops = 0
    n_double_dets = 0
    
    for text in all_text:
        if "of the of the" in text:
            n_of_the_loops += 1
        words_list = text.split()
        for i in range(len(words_list) - 1):
            if words_list[i] in {"the", "a", "an", "this", "that"} and \
               words_list[i+1] in {"the", "a", "an", "this", "that"}:
                n_double_dets += 1
    
    print(f"  'of the of the' loops: {n_of_the_loops}/{n_eval}")
    print(f"  Double DET patterns: {n_double_dets}/{n_eval}")
    
    # All generated texts
    print("\n--- ALL GENERATED TEXT (V13) ---")
    for i, text in enumerate(all_text):
        print(f"  {i+1}. {text}")
    
    # V13 mechanism stats
    v13_stats = gen.get_v13_stats()
    recall_stats = gen.get_recall_stats()
    
    print("\n" + "=" * 70)
    print("V13 MECHANISM STATISTICS")
    print("=" * 70)
    print(f"\n  R1: Recall Strength")
    print(f"    Recall hit rate: {v13_stats.get('recall_hit_rate', 0):.1%}")
    print(f"    PMI supplement weight: {gen.pmi_supplement_weight}/10")
    print(f"    Unigram supplement weight: {gen.unigram_supplement_weight}/10")
    
    print(f"\n  R2: Copy Reform")
    print(f"    Copy rate: {v13_stats.get('copy_rate', 0):.1%}")
    print(f"    Copy blocked (type): {v13_stats.get('copy_blocked_type', 0)}")
    print(f"    Copy blocked (same-word): {v13_stats.get('copy_blocked_same_word', 0)}")
    print(f"    Copy blocked (rate cap): {v13_stats.get('copy_blocked_rate_cap', 0)}")
    print(f"    Copy blocked (loop): {v13_stats.get('copy_blocked_loop', 0)}")
    print(f"    Max consecutive copies: {gen.max_consecutive_copies}")
    
    print(f"\n  R4: Anti-Repetition")
    print(f"    Same-word blocks: {v13_stats.get('same_word_blocked', 0)}")
    print(f"    Same-word penalty strength: {gen.same_word_penalty_strength}× max_recall")
    print(f"    N-gram rep penalties: {v13_stats.get('ngram_rep_penalty', 0)}")
    
    print(f"\n  F1: Type-Compatible Recall")
    print(f"    Type repair rate: {v13_stats.get('type_repair_rate', 0):.1%}")
    print(f"    Closed-loop blocks: {v13_stats.get('closed_loop_blocked', 0)}")
    
    print(f"\n  Overall Recall Stats")
    print(f"    Recall hit rate: {recall_stats.get('recall_hit_rate', 0):.1%}")
    print(f"    Copy rate: {recall_stats.get('copy_rate', 0):.1%}")
    print(f"    Avg recall bonus: {recall_stats.get('avg_recall_bonus', 0):.0f}")
    print(f"    Max recall bonus: {recall_stats.get('max_recall_bonus', 0):.0f}")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.1f}s")

    return model


if __name__ == "__main__":
    run_v13_poc()
