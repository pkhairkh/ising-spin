#!/usr/bin/env python3
"""
PoC Runner for v8 Ising Spin Language Model — Marginal MaxEnt Anti-Repetition.

Builds on v7 (Demon algorithm, Wang-Landau DOS, Lifted MCMC) and adds:

  P4d: Marginal Maximum Entropy on Per-Position Consecutive Counts
     - Replaces coarse total-energy Boltzmann entropy with PER-POSITION
       marginal surprisal S_i = log2(consecutive_count[i] + 1) * SCALE
     - The bonus for LEAVING a stuck position:
       bonus = T_ent * log2(consecutive_count[i] + 1) * SCALE / 1000
     - This is SUBTRACTED from delta_E, making acceptance easier
     - Derived from Jaynes' MaxEnt: stuck position has low marginal entropy
     - Detailed balance preserved via snapshot (Berg & Neuhaus 1991)

The KEY v8 improvement: v7 uses Boltzmann entropy at the TOTAL energy level,
which is too coarse. Many different word configurations map to the same energy
bin, so "the the the" and "the big dog" may have similar total energies.
v8 applies MaxEnt at the PER-POSITION level, directly penalizing positions
that have been stuck on the same word for many consecutive sweeps.
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.enhanced_v8_model import EnhancedV8Model
from ising_spin.type_system import IDX2POS, POS2IDX, N_POS


def run_v8_poc():
    print("=" * 70)
    print("ISING SPIN LANGUAGE MODEL v8 — MARGINAL MAXENT ANTI-REPETITION PoC")
    print("=" * 70)
    print()
    print("v8 Adds P4d on top of v7:")
    print("  P4d: Marginal Maximum Entropy on Per-Position Consecutive Counts")
    print("       - Replaces coarse total-energy Boltzmann entropy with")
    print("         PER-POSITION marginal surprisal")
    print("       - S_i = log2(consecutive_count[i] + 1) * SCALE")
    print("       - Bonus for LEAVING stuck position: T_ent * S_i / 1000")
    print("       - Subtracted from delta_E → easier to accept change")
    print("       - Derived from Jaynes' MaxEnt principle on P(X_i)")
    print("       - Detailed balance preserved via snapshot mechanism")
    print("         (Berg & Neuhaus 1991 — multicanonical approach)")
    print()
    print("  KEY INSIGHT: v7's total-energy Boltzmann entropy is too coarse.")
    print("  'the the the' and 'the big dog' may have similar total energies,")
    print("  so the penalty doesn't distinguish them. v8 applies MaxEnt at the")
    print("  PER-POSITION level, directly penalizing stuck positions.")
    print()
    print("Retained from v7:")
    print("  P4a: Demon Algorithm (Creutz 1983) — integer-only acceptance, no exp()")
    print("  P4b: Wang-Landau Density of States — true Boltzmann entropy S(E) = ln g(E)")
    print("  P4c: Lifted MCMC (Turitsyn et al. 2011) — direction bit for O(N) faster mixing")
    print("  P3a: Swendsen-Wang cluster moves (Wolff variant)")
    print("  P3b: Proper parallel tempering (geometric ladder, 8 replicas)")
    print("  P3c: Entropy-regularized free energy (fallback when P4d off)")
    print("  P0:  Locally-balanced proposals (Zanella 2017)")
    print("  P0:  Hard POS transition constraints (min_count=20)")
    print("  P1:  CALDERA NMF, strengthened emission, implicational couplings")
    print()

    model = EnhancedV8Model(
        # Vocabulary
        vocab_min_freq=5,
        vocab_max_size=3000,
        # Sequence
        seq_len=20,
        # Coupling
        window=5,
        pmi_cap=10,
        min_cooc=2,
        # Weights
        pmi_weight=3,
        hebbian_weight=1,
        semantic_weight=1,
        dep_weight=2,
        # Grammar
        grammar_penalty=60,
        # Emission
        emission_bonus=100,
        emission_penalty=500,
        # Annealing
        phase1_beta=200,
        phase2_beta=500,
        phase3_beta=1000,
        total_sweeps=120,
        # CALDERA NMF
        use_caldera=True,
        nmf_factors=64,
        nmf_iterations=30,
        nmf_n_top=10,
        # SpaCy
        use_spacy=True,
        spacy_max_texts=5000,
        # Transition constraints
        transition_min_count=20,
        # P3a: Swendsen-Wang cluster
        sw_cluster_enabled=True,
        sw_wolff_variant=True,
        # P3b: Proper parallel tempering
        n_replicas=8,
        pt_swap_interval=5,
        # P3c: Entropy regularization (fallback when P4d off)
        entropy_T_ent=50,
        entropy_delta_E_window=500,
        entropy_precision=100,
        # P4a: Demon algorithm
        demon_initial_energy=1000,
        use_demon=True,
        # P4b: Wang-Landau
        wl_warmup_sweeps=50,
        wl_e_min=-50000,
        wl_e_max=50000,
        wl_scale=10000,
        # P4c: Lifted MCMC
        use_lifted=True,
        # P4d: Marginal MaxEnt
        marginal_entropy_scale=10000,
        use_marginal_entropy=True,
    )

    model.train(n_samples=20000)
    model.save("data/v8_model")

    # Generate samples
    print("\n" + "=" * 70)
    print("GENERATION SAMPLES (v8: Marginal MaxEnt Anti-Repetition)")
    print("=" * 70)

    prompts = ["the", "in", "a", "science", "research", "education", "students"]

    all_results = []
    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        for trial in range(3):
            try:
                result = model.generate_with_trace(prompt=prompt, length=15)
                text = result['text']
                types = ' '.join(result['types'][:15])
                print(f"  Trial {trial+1}: {text}")
                print(f"           [{types}]")

                # Show v8-specific diagnostics
                demon_info = result.get('demon_stats', {})
                wl_info = result.get('wl_stats', {})
                marginal_info = result.get('marginal_stats', {})
                if demon_info:
                    print(f"           Demon: E_demon={demon_info.get('demon_energy', '?')}, "
                          f"T_demon={demon_info.get('temperature', 0):.4f}, "
                          f"accept_rate={demon_info.get('acceptance_rate', 0):.3f}")
                if wl_info:
                    print(f"           WL: bins={wl_info.get('energy_bins_visited', '?')}, "
                          f"records={wl_info.get('energy_records', '?')}")
                if marginal_info:
                    print(f"           Marginal: avg_count={marginal_info.get('avg_consecutive_count', 0):.1f}, "
                          f"max_count={marginal_info.get('max_consecutive_count', 0)}, "
                          f"stuck5+={marginal_info.get('stuck_5plus', 0)}, "
                          f"stuck10+={marginal_info.get('stuck_10plus', 0)}, "
                          f"flip_rate={marginal_info.get('flip_rate', 0):.3f}")

                all_results.append({
                    "prompt": prompt,
                    "trial": trial + 1,
                    "text": text,
                    "types": result['types'],
                    "energy": result['energy'],
                    "pt_stats": result.get('pt_stats', {}),
                    "demon_stats": result.get('demon_stats', {}),
                    "wl_stats": result.get('wl_stats', {}),
                    "marginal_stats": result.get('marginal_stats', {}),
                })
            except Exception as e:
                print(f"  Trial {trial+1}: ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Grammar evaluation
    print("\n" + "=" * 70)
    print("GRAMMAR EVALUATION (20 samples)")
    print("=" * 70)

    all_metrics = {}
    n_eval = 20
    all_text = []
    for i in range(n_eval):
        try:
            words, types = model.generate_raw(length=15)
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

    # Type distribution
    print("\n" + "=" * 70)
    print("POS TYPE DISTRIBUTION")
    print("=" * 70)

    all_types = []
    for i in range(15):
        try:
            words, types = model.generate_raw(length=15)
            all_types.extend(types)
        except:
            pass

    type_counts = {}
    for t in all_types:
        name = IDX2POS.get(t, "UNK")
        type_counts[name] = type_counts.get(name, 0) + 1

    total = sum(type_counts.values())
    if total > 0:
        for name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = count * 100 / total
            bar = "#" * int(pct / 2)
            print(f"  {name:8s}: {count:4d} ({pct:5.1f}%) {bar}")

    # P4a: Demon algorithm statistics
    print("\n" + "=" * 70)
    print("P4a: DEMON ALGORITHM STATISTICS (Creutz 1983)")
    print("=" * 70)
    if hasattr(model.sampler, 'demon'):
        demon = model.sampler.demon
        stats = demon.get_stats()
        print(f"  Final demon energy: {stats['demon_energy']}")
        print(f"  Acceptance rate: {stats['acceptance_rate']:.4f}")
        print(f"  Temperature reading: T_demon = {stats['temperature']:.6f}")
        print(f"  n_accepted: {stats['n_accepted']}")
        print(f"  n_rejected: {stats['n_rejected']}")
        print(f"  Initial demon energy: {model.sampler.demon_initial_energy}")
        print()
        print("  Key advantage: NO exp() calls in the hot path!")
        print("  All acceptance decisions are pure integer comparison.")

    # P4b: Wang-Landau / Energy histogram statistics
    print("\n" + "=" * 70)
    print("P4b: ENERGY HISTOGRAM STATISTICS (Running Boltzmann Entropy)")
    print("=" * 70)
    if hasattr(model.sampler, '_energy_histogram') and model.sampler._energy_histogram is not None:
        visited = int((model.sampler._energy_histogram > 0).sum())
        total_records = int(model.sampler._energy_histogram.sum())
        bin_width = getattr(model.sampler, '_energy_bin_width', 0)
        print(f"  Energy bins visited: {visited}")
        print(f"  Total energy records: {total_records}")
        print(f"  Bin width: {bin_width}")
        print(f"  (Note: v8 uses marginal entropy as primary; this is fallback/diagnostics)")

    # P4c: Lifted MCMC statistics
    print("\n" + "=" * 70)
    print("P4c: LIFTED MCMC STATISTICS (Turitsyn et al. 2011)")
    print("=" * 70)
    print(f"  Lifted MCMC enabled: {model.use_lifted}")
    if model.use_lifted:
        print("  Direction bit: sweep positions with momentum")
        print("  Accept → continue in same direction (momentum)")
        print("  Reject → reverse direction (bounce)")
        print("  Proven O(N) faster mixing on chain structures")
        print("  Preserves lifted detailed balance: pi(x,d) = pi(x)/2")

    # P4d: Marginal entropy statistics
    print("\n" + "=" * 70)
    print("P4d: MARGINAL MAXENT STATISTICS (Per-Position Anti-Repetition)")
    print("=" * 70)
    print(f"  Marginal MaxEnt enabled: {model.use_marginal_entropy}")
    if model.use_marginal_entropy and model.sampler.marginal_entropy is not None:
        me = model.sampler.marginal_entropy
        stats = me.get_stats()
        print(f"  Total sweeps tracked: {stats['total_sweeps']}")
        print(f"  Total position-updates: {stats['total_updates']}")
        print(f"  Total flips (word changes): {stats['total_flips']}")
        print(f"  Flip rate: {stats['flip_rate']:.4f}")
        print(f"  Average consecutive count: {stats['avg_consecutive_count']:.2f}")
        print(f"  Maximum consecutive count: {stats['max_consecutive_count']}")
        print(f"  Positions stuck >= 5 sweeps: {stats['stuck_5plus']}")
        print(f"  Positions stuck >= 10 sweeps: {stats['stuck_10plus']}")
        print(f"  Positions stuck >= 20 sweeps: {stats['stuck_20plus']}")
        print(f"  SCALE: {stats['SCALE']}")
        print()
        print("  Mathematical derivation:")
        print("    S_i = log2(consecutive_count[i] + 1) * SCALE")
        print("    bonus = T_ent * S_i / 1000")
        print("    delta_F = delta_E - bonus")
        print()
        print("  This is PRINCIPLED from Jaynes' MaxEnt on P(X_i):")
        print("    - Stuck position: P(w_old|i) ≈ 1, H(X_i) ≈ 0")
        print("    - Surprisal of STAYING ≈ 0 (no surprise)")
        print("    - Surprisal of LEAVING → ∞ (very surprising)")
        print("    - We HELP escape by subtracting bonus from delta_E")
        print()
        print("  Detailed balance preserved via snapshot:")
        print("    - snapshot() freezes counts at start of each sweep")
        print("    - All acceptance decisions use the snapshot")
        print("    - update_all() updates counts after each sweep")
        print("    - Same as multicanonical (Berg & Neuhaus 1991)")
        print()
        print("  Example bonuses (T_ent=50, SCALE=10000):")
        for k in [1, 3, 5, 10, 20, 50]:
            n = k + 1
            s_int = n.bit_length() - 1
            if n > (1 << s_int):
                frac = ((n - (1 << s_int)) * 10000) // (1 << s_int)
            else:
                frac = 0
            S_i = s_int * 10000 + frac
            bonus = (50 * S_i) // 1000
            import math as _math
            S_exact = _math.log2(k + 1)
            bonus_exact = 50 * S_exact * 10
            print(f"    consecutive_count={k:2d}: S_i={S_i:6d}, "
                  f"bonus≈{bonus:5d} (exact: {bonus_exact:.0f})")

    # PT swap statistics
    print("\n" + "=" * 70)
    print("PARALLEL TEMPERING STATISTICS (Proper: Geometric Ladder + Precomputed)")
    print("=" * 70)
    if hasattr(model.sampler, 'pt_swap_counts'):
        for i in range(len(model.sampler.pt_swap_counts)):
            swaps = model.sampler.pt_swap_counts[i]
            attempts = model.sampler.pt_attempt_counts[i]
            rate = swaps / max(1, attempts)
            beta_lo = model.sampler.pt_betas[i]
            beta_hi = model.sampler.pt_betas[i + 1] if i + 1 < len(model.sampler.pt_betas) else "?"
            print(f"  Swap {i}<->{i+1} (B={beta_lo}<->{beta_hi}): "
                  f"{swaps}/{attempts} ({rate:.3f})")

    # Repetition analysis
    print("\n" + "=" * 70)
    print("REPETITION ANALYSIS")
    print("=" * 70)
    rep_count = all_metrics.get("repeated_words", 0)
    total_words = n_eval * 15
    rep_rate = rep_count / max(1, total_words - n_eval)
    print(f"  Repeated adjacent words: {rep_count}/{total_words - n_eval} "
          f"({rep_rate:.3f})")

    # DET→NOUN accuracy
    det_noun = all_metrics.get("det_noun", 0)
    det_non = all_metrics.get("det_non_noun", 0)
    det_total = det_noun + det_non
    if det_total > 0:
        print(f"  DET->NOUN accuracy: {det_noun}/{det_total} ({100*det_noun/det_total:.1f}%)")

    # v7 → v8 comparison
    print("\n" + "=" * 70)
    print("v7 → v8 CHANGE SUMMARY")
    print("=" * 70)
    print("  ADDED: P4d Marginal Maximum Entropy on Per-Position Consecutive Counts")
    print("         - Replaces coarse total-energy Boltzmann entropy with PER-POSITION")
    print("           marginal surprisal S_i = log2(consecutive_count[i] + 1) * SCALE")
    print("         - The bonus for LEAVING a stuck position:")
    print("           bonus = T_ent * log2(consecutive_count[i] + 1) * SCALE / 1000")
    print("         - This is SUBTRACTED from delta_E, making acceptance easier")
    print("         - Derived from Jaynes' MaxEnt principle on P(X_i):")
    print("           * Stuck position: P(w_old|i) ≈ 1, H(X_i) ≈ 0")
    print("           * We help escape by lowering the effective barrier")
    print("         - Detailed balance preserved via snapshot mechanism:")
    print("           * snapshot() freezes counts at start of each sweep")
    print("           * All acceptance decisions use the snapshot")
    print("           * update_all() updates counts after the sweep")
    print("           * Same approach as multicanonical (Berg & Neuhaus 1991)")
    print()
    print("  KEY IMPROVEMENT: Fine-grained anti-repetition")
    print("  - v7: Boltzmann entropy at TOTAL energy level → too coarse")
    print("    'the the the' and 'the big dog' may have similar total energies")
    print("  - v8: MaxEnt at PER-POSITION level → directly targets stuck words")
    print("    Position stuck for K sweeps gets bonus proportional to log2(K+1)")
    print("    The longer a word is stuck, the easier it is to change")
    print()
    print("  KEY IMPROVEMENT: Principled from MaxEnt, not a hack")
    print("  - The marginal surprisal is derived from Jaynes' maximum entropy")
    print("    principle applied to the marginal distribution P(X_i)")
    print("  - When P(X_i) is peaked (stuck), the marginal entropy is low")
    print("  - The bonus helps the chain escape low-entropy configurations")
    print("  - This is the SAME principle as the multicanonical algorithm:")
    print("    modify the sampling weights to flatten the distribution")

    # Sample output
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT (10 examples)")
    print("=" * 70)
    for i, text in enumerate(all_text[:10]):
        print(f"  {i+1}. {text}")

    # Save results
    results = {
        "grammar_metrics": all_metrics,
        "n_samples": n_eval,
        "per_sample": {k: v/n_eval for k, v in all_metrics.items()},
        "repetition_rate": rep_rate,
        "det_noun_accuracy": det_noun / max(1, det_total),
        "sample_texts": all_text[:10],
        "type_distribution": type_counts,
        "transition_min_count": 20,
        "model_version": "v8",
        "v8_features": {
            "swendsen_wang": True,
            "wolff_variant": True,
            "proper_pt": True,
            "n_replicas": 8,
            "entropy_regularized": True,
            "T_ent": 50,
            "demon_algorithm": True,
            "demon_initial_energy": 1000,
            "wang_landau": True,
            "wl_warmup_sweeps": 50,
            "wl_e_range": [-50000, 50000],
            "lifted_mcmc": True,
            "marginal_maxent": True,
            "marginal_entropy_scale": 10000,
        },
    }
    with open("data/v8_model/poc_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to data/v8_model/poc_results.json")

    return model


if __name__ == "__main__":
    run_v8_poc()
