#!/usr/bin/env python3
"""
PoC Runner for V12 Coherent Ising Language Model.

V12 = V11 exact recall + THREE TARGETED FIXES:

  F1: Type-Compatible Recall + Function-Word Anti-Loop
    - Fixes "to thanks" (POS/recall conflict)
    - Fixes "of the of the" (closed-class loops)

  F2: Bridge Stitching + Copy-Fade
    - Smooth transitions between copied and generated segments
    - No more jarring boundaries

  F3: Kneser-Ney Backoff + Interpolation + Enhanced Fallback
    - Graceful fallback when no recall match
    - Continuation counts for better lower-order estimates
    - Adaptive interpolation weights

COMPARISON:
  V8 (Joint MCMC):           Gibberish (PMI drowned by grammar)
  V9 (Energy Landscape):     Gibberish (landscape mods too weak)
  V10 (Autoregressive):      Better grammar, still incoherent (PMI = bigram)
  V11 (Exact Recall):        Coherent phrases! But fragments & loops
  V12 (Coherent):            V11 + F1+F2+F3 fixes → clean coherent text
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.v12_coherent_model import CoherentV12Model
from ising_spin.type_system import IDX2POS, POS2IDX, N_POS


def run_v12_poc():
    print("=" * 70)
    print("V12 COHERENT ISING + EXACT RECALL — PoC")
    print("=" * 70)
    print()
    print("V11 BREAKTHROUGH: Exact n-gram recall produced coherent text!")
    print("V12 FIXES THREE REMAINING PATHOLOGIES:")
    print()
    print("  F1: Type-Compatible Recall")
    print("      - Recall candidates FILTERED by grammar-chosen POS type")
    print("      - Function-word anti-loop: max 2 closed-class words in a row")
    print("      - Same-POS-at-distance-2 penalty for DET/PREP/PART")
    print()
    print("  F2: Bridge Stitching + Copy-Fade")
    print("      - Boosted PMI coupling at copy→generate boundaries")
    print("      - Copy-fade: gradual transition from copy to generate mode")
    print()
    print("  F3: Kneser-Ney Backoff + Interpolation")
    print("      - Continuation counts for proper backoff hierarchy")
    print("      - Adaptive interpolation: recall=7/10 + PMI=2/10 + uni=1/10")
    print("      - Enhanced unigram fallback with type compatibility")
    print()

    t0 = time.time()

    # Train V12 model
    model = CoherentV12Model(
        # === V12-specific: Three Fixes ===
        max_closed_class_run=2,             # F1: max consecutive closed-class words
        closed_class_loop_penalty=300,      # F1: energy penalty for DET/PREP/PART loops
        copy_fade_strength=0.5,             # F2: recall bonus reduction at copy→gen boundary
        bridge_context_boost=3,             # F2: PMI coupling boost at copy→gen boundary
        lambda_recall_hit=7,               # F3: recall weight when hit (7/10)
        lambda_pmi_hit=2,                  # F3: PMI weight when recall hit (2/10)
        lambda_unigram_hit=1,              # F3: unigram weight when recall hit (1/10)
        lambda_pmi_miss=6,                 # F3: PMI weight when no recall (6/10)
        lambda_unigram_miss=4,             # F3: unigram weight when no recall (4/10)
        use_kneser_ney=True,               # F3: use Kneser-Ney backoff index

        # === V11-specific: Exact Recall ===
        recall_scale=500,
        context_weight_factor=4,
        copy_min_context=2,
        copy_min_confidence=0.3,
        copy_enabled=True,
        ngram_max_n=5,
        ngram_min_count=1,
        
        # === V10 autoregressive ===
        beta_type=0.01,
        beta_word=0.05,
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

    gen = model.v12_generator

    print(f"\n  Vocab size: {gen.vocab_size}")
    print(f"  N-gram index type: {'KneserNey' if gen.use_kneser_ney else 'Standard'}")

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
            
            # Check for V11 pathologies
            has_of_the_loop = "of the of the" in text
            has_double_det = any(
                result['type_names'][i] == 'DET' and result['type_names'][i+1] == 'DET'
                for i in range(len(result['type_names']) - 1)
            )
            
            flag = ""
            if has_of_the_loop:
                flag += " ⚠️ OF-THE-LOOP"
            if has_double_det:
                flag += " ⚠️ DOUBLE-DET"
            
            print(f"  '{prompt}' → {text}{flag}")
            print(f"           [{types_str}] copies={n_copies} recalls={n_recall_hits}")
        except Exception as e:
            print(f"  '{prompt}' → ERROR: {e}")
            import traceback
            traceback.print_exc()

    # === Phase 2: Recall scale tuning ===
    print("\n" + "=" * 70)
    print("PHASE 2: RECALL SCALE TUNING")
    print("=" * 70)

    for scale in [300, 500, 1000, 2000]:
        gen.recall_scale = scale
        print(f"\n--- recall_scale = {scale} ---")
        
        for prompt in ["the", "science", "research", "education"][:3]:
            try:
                result = gen.generate(prompt=prompt, length=20)
                text = result['text']
                n_copies = sum(1 for d in result['diagnostics'] if d.get('copy_used'))
                n_recalls = sum(1 for d in result['diagnostics'] if d.get('recall_hit'))
                print(f"  '{prompt}' → {text}")
                print(f"           copies={n_copies} recalls={n_recalls}")
            except Exception as e:
                print(f"  '{prompt}' → ERROR: {e}")

    # === Phase 3: Full evaluation at best scale ===
    best_scale = 1000
    gen.recall_scale = best_scale

    print("\n" + "=" * 70)
    print(f"PHASE 3: FULL EVALUATION AT recall_scale = {best_scale}")
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

    # Grammar stats
    print(f"\nGrammar patterns across {n_eval} samples:")
    for k, v in sorted(all_metrics.items()):
        per_sample = v / n_eval
        print(f"  {k}: {v} total ({per_sample:.1f}/sample)")

    # Repetition analysis
    rep_count = all_metrics.get("repeated_words", 0)
    total_words = n_eval * 20
    rep_rate = rep_count / max(1, total_words - n_eval)
    print(f"\n  Repetition rate: {rep_count}/{total_words - n_eval} ({rep_rate:.3f})")

    # DET→NOUN accuracy
    det_noun = all_metrics.get("det_noun", 0)
    det_non = all_metrics.get("det_non_noun", 0)
    det_total = det_noun + det_non
    if det_total > 0:
        print(f"  DET->NOUN accuracy: {det_noun}/{det_total} ({100*det_noun/det_total:.1f}%)")

    # === Check for V11 pathologies ===
    print("\n" + "=" * 70)
    print("PATHOLOGY CHECK (V12 vs V11)")
    print("=" * 70)
    
    n_of_the_loops = 0
    n_double_dets = 0
    n_to_thanks = 0
    n_fragments = 0
    
    for text in all_text:
        if "of the of the" in text:
            n_of_the_loops += 1
        # Check for double determiners
        words_list = text.split()
        for i in range(len(words_list) - 1):
            if words_list[i] in {"the", "a", "an", "this", "that"} and \
               words_list[i+1] in {"the", "a", "an", "this", "that"}:
                n_double_dets += 1
        # Check for "to + NOUN" fragments (to thanks, to research used as noun)
        for i in range(len(words_list) - 1):
            if words_list[i] == "to" and words_list[i+1].endswith("s") and \
               not words_list[i+1].endswith("es"):
                n_to_thanks += 1
    
    print(f"  'of the of the' loops: {n_of_the_loops}/{n_eval}")
    print(f"  Double DET patterns: {n_double_dets}/{n_eval}")
    print(f"  'to + NOUN' fragments: {n_to_thanks}/{n_eval}")

    # All generated texts
    print("\n" + "=" * 70)
    print("ALL GENERATED TEXT (V12)")
    print("=" * 70)
    for i, text in enumerate(all_text):
        print(f"  {i+1}. {text}")

    # V12 mechanism stats
    v12_stats = gen.get_v12_stats()
    recall_stats = gen.get_recall_stats()
    
    print("\n" + "=" * 70)
    print("V12 MECHANISM STATISTICS")
    print("=" * 70)
    print("\n  F1: Type-Compatible Recall")
    print(f"    Type repair rate: {v12_stats.get('type_repair_rate', 0):.1%} of positions")
    print(f"    Closed-loop blocks: {v12_stats.get('closed_loop_blocked', 0)}")
    print(f"    Closed-loop block rate: {v12_stats.get('closed_loop_block_rate', 0):.1%}")
    
    print("\n  F2: Bridge Stitching + Copy-Fade")
    print(f"    Copy-fade uses: {v12_stats.get('copy_fade_used', 0)}")
    print(f"    Copy-fade rate: {v12_stats.get('copy_fade_rate', 0):.1%}")
    print(f"    Bridge uses: {v12_stats.get('bridge_used', 0)}")
    print(f"    Bridge rate: {v12_stats.get('bridge_rate', 0):.1%}")
    
    print("\n  F3: Interpolation + Backoff")
    print(f"    Recall hit rate: {v12_stats.get('recall_hit_rate', 0):.1%}")
    print(f"    Recall miss rate: {1 - v12_stats.get('recall_hit_rate', 0):.1%}")
    print(f"    Kneser-Ney enabled: {gen.use_kneser_ney}")
    
    print("\n  Overall Recall Stats")
    print(f"    Recall hit rate: {recall_stats.get('recall_hit_rate', 0):.1%}")
    print(f"    Copy rate: {recall_stats.get('copy_rate', 0):.1%}")
    print(f"    Avg recall bonus: {recall_stats.get('avg_recall_bonus', 0):.0f}")
    print(f"    Max recall bonus: {recall_stats.get('max_recall_bonus', 0):.0f}")

    # === Phase 4: V11 vs V12 side-by-side ===
    print("\n" + "=" * 70)
    print("V11 vs V12 SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    
    # Build V11 generator for comparison
    from ising_spin.exact_recall_v11_model import ExactRecallV11Generator
    v11_gen = ExactRecallV11Generator(
        sampler=model.v8_model.sampler,
        vocab=model.v8_model.vocab,
        ngram_index=model.ngram_index,
        recall_scale=best_scale,
        context_weight_factor=4,
        copy_min_context=2,
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
    
    # Reset V12 stats
    gen.recall_scale = best_scale
    gen._recall_stats = {
        'total_positions': 0, 'recall_hits': 0,
        'copy_used': 0, 'avg_recall_bonus': 0, 'max_recall_bonus': 0,
    }
    gen._v12_stats = {
        'total_positions': 0, 'recall_hit': 0, 'recall_miss': 0,
        'type_repair': 0, 'closed_loop_blocked': 0,
        'copy_fade_used': 0, 'bridge_used': 0, 'backoff_used': 0,
        'copy_loop_broken': 0, 'same_word_blocked': 0,
    }
    
    comparison_prompts = ["the", "science", "research", "students", "education",
                          "to", "of", "in", "for", "he"]
    
    for prompt in comparison_prompts:
        try:
            v11_result = v11_gen.generate(prompt=prompt, length=20)
            v11_text = v11_result['text']
            v11_copies = sum(1 for d in v11_result['diagnostics'] if d.get('copy_used'))
            
            v12_result = gen.generate(prompt=prompt, length=20)
            v12_text = v12_result['text']
            v12_copies = sum(1 for d in v12_result['diagnostics'] if d.get('copy_used'))
            
            # Flag V11 pathologies
            v11_flags = []
            if "of the of the" in v11_text:
                v11_flags.append("LOOP")
            v11_words = v11_text.split()
            for i in range(len(v11_words) - 1):
                if v11_words[i] in {"the", "a"} and v11_words[i+1] in {"the", "a"}:
                    v11_flags.append("DOUBLE-DET")
            
            v12_flags = []
            if "of the of the" in v12_text:
                v12_flags.append("LOOP")
            v12_words = v12_text.split()
            for i in range(len(v12_words) - 1):
                if v12_words[i] in {"the", "a"} and v12_words[i+1] in {"the", "a"}:
                    v12_flags.append("DOUBLE-DET")
            
            v11_flag_str = f" ⚠️ {','.join(v11_flags)}" if v11_flags else ""
            v12_flag_str = f" ⚠️ {','.join(v12_flags)}" if v12_flags else ""
            
            print(f"\n  Prompt: '{prompt}'")
            print(f"  V11: {v11_text}{v11_flag_str}")
            print(f"  V12: {v12_text}{v12_flag_str}")
        except Exception as e:
            print(f"  '{prompt}' → ERROR: {e}")
            import traceback
            traceback.print_exc()

    # === Phase 5: Summary ===
    print("\n" + "=" * 70)
    print("V11 → V12 IMPROVEMENT SUMMARY")
    print("=" * 70)
    print()
    print("  V11 Pathology 1: POS-Recall Conflicts ('to thanks')")
    print(f"    V12 Fix (F1): Type-compatible recall + function-word anti-loop")
    print(f"    Type repairs: {v12_stats.get('type_repair', 0)} positions")
    print(f"    Loop blocks: {v12_stats.get('closed_loop_blocked', 0)} positions")
    print()
    print("  V11 Pathology 2: Verbatim Echoing / Stitching Gaps")
    print(f"    V12 Fix (F2): Copy-fade + bridge context boosting")
    print(f"    Copy-fade uses: {v12_stats.get('copy_fade_used', 0)}")
    print(f"    Bridge uses: {v12_stats.get('bridge_used', 0)}")
    print()
    print("  V11 Pathology 3: Weak Fallback (random words)")
    print(f"    V12 Fix (F3): Kneser-Ney backoff + interpolation")
    print(f"    Recall hit rate: {v12_stats.get('recall_hit_rate', 0):.1%}")
    print(f"    (When miss: PMI=6/10 + unigram=4/10 instead of raw PMI)")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.1f}s")

    return model


if __name__ == "__main__":
    run_v12_poc()
