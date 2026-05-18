#!/usr/bin/env python3
"""
PoC Runner for v7 Ising Spin Language Model — Principled MCMC + Microcanonical Sampling.

Builds on v6 (SW clusters, proper PT, entropy regularization) and adds THREE
new principled mathematical solutions:

  P4a: Demon Algorithm (Creutz 1983) — integer-only acceptance, no exp()
  P4b: Wang-Landau Density of States — true Boltzmann entropy S(E) = ln g(E)
  P4c: Lifted MCMC (Turitsyn et al. 2011) — direction bit for O(N) faster mixing
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.enhanced_v7_model import EnhancedV7Model
from ising_spin.type_system import IDX2POS, POS2IDX, N_POS


def run_v7_poc():
    print("=" * 70)
    print("ISING SPIN LANGUAGE MODEL v7 — PRINCIPLED MCMC + MICROCANONICAL PoC")
    print("=" * 70)
    print()
    print("v7 Adds THREE principled solutions on top of v6:")
    print("  P4a: Demon Algorithm (Creutz 1983)")
    print("       - Integer-only acceptance: NO exp() in hot path")
    print("       - Microcanonical sampling with energy reservoir")
    print("       - Temperature reading from demon energy distribution")
    print()
    print("  P4b: Wang-Landau Density of States")
    print("       - Warm-up phase maps g(E) with flat-histogram method")
    print("       - TRUE Boltzmann entropy S(E) = ln g(E) replaces crude local entropy")
    print("       - STATE FUNCTION — preserves detailed balance")
    print()
    print("  P4c: Lifted MCMC (Turitsyn et al. 2011)")
    print("       - Direction bit (+1/-1) gives momentum to sweep ordering")
    print("       - Accept → continue, Reject → bounce")
    print("       - Proven O(N) faster mixing on chain structures")
    print()
    print("Retained from v6:")
    print("  P3a: Swendsen-Wang cluster moves (Wolff variant)")
    print("  P3b: Proper parallel tempering (geometric ladder, 8 replicas)")
    print("  P3c: Entropy-regularized free energy (now with WL Boltzmann entropy)")
    print("  P0:  Locally-balanced proposals (Zanella 2017)")
    print("  P0:  Hard POS transition constraints (min_count=20)")
    print("  P1:  CALDERA NMF, strengthened emission, implicational couplings")
    print()

    model = EnhancedV7Model(
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
        # P3c: Entropy regularization
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
    )

    model.train(n_samples=20000)
    model.save("data/v7_model")

    # Generate samples
    print("\n" + "=" * 70)
    print("GENERATION SAMPLES (v7: Principled MCMC + Microcanonical)")
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

                # Show v7-specific diagnostics
                demon_info = result.get('demon_stats', {})
                wl_info = result.get('wl_stats', {})
                if demon_info:
                    print(f"           Demon: E_demon={demon_info.get('demon_energy', '?')}, "
                          f"T_demon={demon_info.get('temperature', 0):.4f}, "
                          f"accept_rate={demon_info.get('acceptance_rate', 0):.3f}")
                if wl_info:
                    print(f"           WL: coverage={wl_info.get('coverage', 0):.3f}, "
                          f"ln_f={wl_info.get('ln_f', '?')}, "
                          f"converged={wl_info.get('converged', False)}")

                all_results.append({
                    "prompt": prompt,
                    "trial": trial + 1,
                    "text": text,
                    "types": result['types'],
                    "energy": result['energy'],
                    "pt_stats": result.get('pt_stats', {}),
                    "demon_stats": result.get('demon_stats', {}),
                    "wl_stats": result.get('wl_stats', {}),
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

    # P4b: Wang-Landau statistics
    print("\n" + "=" * 70)
    print("P4b: WANG-LANDAU DENSITY OF STATES STATISTICS")
    print("=" * 70)
    if hasattr(model.sampler, 'wl'):
        wl = model.sampler.wl
        stats = wl.get_stats()
        print(f"  Total WL updates: {stats['n_updates']}")
        print(f"  Energy bins: {stats['n_bins']}")
        print(f"  Visited bins: {stats['n_visited']} ({stats['coverage']:.3f} coverage)")
        print(f"  Current ln_f: {stats['ln_f']}")
        print(f"  Converged: {stats['converged']}")
        print(f"  ln_f reductions: {stats['n_reductions']}")
        print(f"  max log_g: {stats['max_log_g']}")
        print(f"  min log_g (positive): {stats['min_log_g_positive']}")
        print()

        # Show entropy profile at a few energy levels
        print("  Entropy profile S(E) = log_g(E) / SCALE at selected energies:")
        sample_energies = [-40000, -30000, -20000, -10000, 0, 10000, 20000, 30000, 40000]
        for E in sample_energies:
            S = wl.get_entropy(E)
            S_real = S / wl.SCALE if S > 0 else 0
            print(f"    E={E:+6d}: S(E)={S:8d} (S_real={S_real:.4f})")

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

    # v6 → v7 comparison
    print("\n" + "=" * 70)
    print("v6 → v7 CHANGE SUMMARY")
    print("=" * 70)
    print("  ADDED: P4a Demon Algorithm (Creutz 1983)")
    print("         - Replaces Metropolis acceptance with integer-only comparison")
    print("         - Eliminates ALL exp() calls from the hot path")
    print("         - Natural microcanonical sampling with energy reservoir")
    print("         - Temperature reading from demon energy: T = 1/ln(1 + 1/<E_demon>)")
    print()
    print("  ADDED: P4b Wang-Landau Density of States")
    print("         - Warm-up phase estimates g(E) with flat-histogram method")
    print("         - TRUE Boltzmann entropy S(E) = ln g(E) replaces crude local entropy")
    print("         - F = E - T_ent * S(E) is a state function (preserves DB)")
    print("         - log_g(E) stored as scaled int64 (SCALE=10000), all integer arithmetic")
    print()
    print("  ADDED: P4c Lifted MCMC (Turitsyn et al. 2011)")
    print("         - Direction bit (+1/-1) gives momentum to sweep ordering")
    print("         - Accept → continue in direction, Reject → reverse (bounce)")
    print("         - Proven O(N) faster mixing on chain structures")
    print("         - Satisfies lifted detailed balance: pi(x,d) = pi(x)/2")
    print()
    print("  KEY IMPROVEMENT: Zero floating-point in generation hot path")
    print("  - v6: Metropolis uses precomputed lookup table (still involves table lookup)")
    print("  - v7: Demon uses pure integer comparison (no table needed)")
    print()
    print("  KEY IMPROVEMENT: True thermodynamic entropy")
    print("  - v6: Crude local entropy (log2 of accessible neighbors within dE window)")
    print("  - v7: Wang-Landau Boltzmann entropy S(E) = ln g(E) from density of states")
    print()
    print("  KEY IMPROVEMENT: Faster mixing")
    print("  - v6: Sequential sweep (random or deterministic order)")
    print("  - v7: Lifted chain with momentum (proven O(N) faster)")

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
        "model_version": "v7",
        "v7_features": {
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
        },
    }
    with open("data/v7_model/poc_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to data/v7_model/poc_results.json")

    return model


if __name__ == "__main__":
    run_v7_poc()
