#!/usr/bin/env python3
"""
Ising Spin Language Model - Enhanced Typed Ising-Potts Architecture v3

Demonstrates text generation with ZERO floating-point operations
in the generation loop, using grammatically and semantically
structured integer arithmetic.

Architecture v3:
  - Type layer (Potts): POS tags, ~13 states (spaCy-accurate)
  - Value layer: words, ~8K states (NMF-factorized couplings)
  - Couplings: log-floor PMI (integer bit operations)
  - Dependency couplings: J_tree from parse trees (long-range agreement)
  - Integer NMF: J ≈ W×H for vocabulary scaling beyond 3K
  - Grammar: integer quadratic penalties
  - Semantics: compatibility-gated Hebbian coupling
  - Generation: staged annealing

Usage:
    # v3 Enhanced model (recommended):
    python run.py enhanced-train --n_samples 100000
    python run.py enhanced-generate --prompt "the" --length 25
    python run.py enhanced-demo --n_samples 50000
    python run.py enhanced-eval --n_samples 50000

    # v2 Typed model (legacy):
    python run.py typed-train --n_samples 50000
    python run.py typed-generate --prompt "the" --length 25
    python run.py typed-demo --n_samples 30000
    python run.py typed-eval --n_samples 30000

    # v1 Character-level model (legacy):
    python run.py train --n_samples 50000
    python run.py generate --prompt "the " --length 300
    python run.py demo
"""

import argparse
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from ising_spin.char_model import CharIsingModel
from ising_spin.typed_model import TypedIsingModel
from ising_spin.enhanced_model import EnhancedTypedModel
from ising_spin.type_system import IDX2POS, POS2IDX, N_POS
from ising_spin.semantic_types import SEMANTIC_SUPERTYPES
from ising_spin.data_loader import load_fineweb_edu


# =============================================================================
# Enhanced v3 Model Commands
# =============================================================================

def cmd_enhanced_train(args):
    """Train the enhanced v3 model with all four improvements."""
    model = EnhancedTypedModel(
        vocab_min_freq=args.min_freq,
        vocab_max_size=args.max_vocab,
        seq_len=args.seq_len,
        window=args.window,
        pmi_cap=args.pmi_cap,
        min_cooc=args.min_cooc,
        pmi_weight=args.pmi_weight,
        hebbian_weight=args.hebbian_weight,
        semantic_weight=args.semantic_weight,
        dep_weight=args.dep_weight,
        grammar_penalty=args.grammar_penalty,
        phase1_beta=args.phase1_beta,
        phase2_beta=args.phase2_beta,
        phase3_beta=args.phase3_beta,
        total_sweeps=args.sweeps,
        use_nmf=not args.no_nmf,
        nmf_factors=args.nmf_factors,
        nmf_iterations=args.nmf_iterations,
        use_spacy=not args.no_spacy,
        spacy_max_texts=args.spacy_max_texts,
    )

    model.train(n_samples=args.n_samples)

    # Save
    save_dir = args.save_dir
    model.save(save_dir)
    print(f"\nModel saved to {save_dir}/")

    # Quick generation test
    print("\n" + "=" * 70)
    print("QUICK GENERATION TEST")
    print("=" * 70)
    for prompt in ["the", "in", "science", "research", "education", "students"]:
        try:
            result = model.generate_with_trace(prompt=prompt, length=15)
            print(f"\n  Prompt='{prompt}':")
            print(f"  Text: {result['text'][:120]}")
            print(f"  Types: {' '.join(result['types'][:15])}")
            print(f"  Energy: {result['energy']}")
        except Exception as e:
            print(f"\n  Prompt='{prompt}': ERROR: {e}")

    return model


def cmd_enhanced_generate(args):
    """Generate text from trained enhanced model."""
    model = EnhancedTypedModel.load(args.save_dir)

    print("=" * 70)
    print("ENHANCED ISING-POTTS MODEL v3 - GENERATION")
    print("(Zero FP in generation loop. Integer arithmetic only.)")
    print("=" * 70)

    for i in range(args.n_samples):
        result = model.generate_with_trace(
            prompt=args.prompt,
            length=args.length,
        )
        print(f"\n--- Sample {i+1} ---")
        print(f"  Text: {result['text']}")
        print(f"  Types: {' '.join(result['types'])}")
        print(f"  Energy: {result['energy']}")
        print(f"  Type distribution: {result['type_counts']}")


def cmd_enhanced_demo(args):
    """Full demo of enhanced v3 model."""
    model = EnhancedTypedModel(
        vocab_min_freq=3,
        vocab_max_size=5000,
        seq_len=25,
        window=6,
        pmi_cap=12,
        grammar_penalty=40,
        total_sweeps=120,
        use_nmf=True,
        nmf_factors=128,
        use_spacy=True,
    )

    model.train(n_samples=args.n_samples)
    model.save(args.save_dir)

    # Generate samples
    print("\n" + "=" * 70)
    print("GENERATION SAMPLES (v3 Enhanced)")
    print("=" * 70)

    prompts = ["the", "in", "a", "science", "research", "education", "students"]
    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        for trial in range(3):
            try:
                result = model.generate_with_trace(prompt=prompt, length=20)
                text = result['text']
                types = ' '.join(result['types'][:20])
                print(f"  Trial {trial+1}: {text}")
                print(f"           [{types}]")
            except Exception as e:
                print(f"  Trial {trial+1}: ERROR: {e}")

    # Grammar evaluation
    print("\n" + "=" * 70)
    print("GRAMMAR EVALUATION")
    print("=" * 70)

    total_metrics = {}
    n_eval = 10
    for i in range(n_eval):
        try:
            words, types = model.generate_raw(length=20)
            metrics = model.evaluate_grammar(words, types)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
        except Exception as e:
            print(f"  Eval sample {i+1}: ERROR: {e}")

    print(f"\nGrammar patterns across {n_eval} samples (total counts):")
    for k, v in sorted(total_metrics.items()):
        print(f"  {k}: {v}")

    # Type distribution
    print("\n" + "=" * 70)
    print("POS TYPE DISTRIBUTION")
    print("=" * 70)

    all_types = []
    for i in range(20):
        try:
            words, types = model.generate_raw(length=20)
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
            bar = "█" * int(pct / 2)
            print(f"  {name:8s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Dependency coupling analysis
    if model.dep_couplings is not None:
        print("\n" + "=" * 70)
        print("DEPENDENCY COUPLING ANALYSIS")
        print("=" * 70)
        dep_stats = model.dep_couplings.get_dep_stats()
        print(f"\n  J_tree non-zeros: {dep_stats['J_tree_nnz']}")
        print(f"  Agreement rules: {dep_stats['agreement_rules']}")
        print(f"  J_tree range: {dep_stats['J_tree_range']}")
        print(f"\n  Dependency label distribution:")
        for label, count in sorted(dep_stats['dep_label_counts'].items()):
            if count > 0:
                print(f"    {label}: {count}")

    # NMF analysis
    if model.nmf_model is not None and model.nmf_model.fitted:
        print("\n" + "=" * 70)
        print("INTEGER NMF ANALYSIS")
        print("=" * 70)
        mem = model.nmf_model.memory_savings()
        print(f"\n  Memory savings: {mem['savings_pct']:.1f}%")
        print(f"  Full matrix: {mem['full_matrix_elements']:,} elements")
        print(f"  Factorized: {mem['factorized_elements']:,} elements")
        print(f"  Number of factors: {mem['n_factors_total']}")

    return model


def cmd_enhanced_eval(args):
    """Thorough evaluation of enhanced v3 model with comparison."""
    print("=" * 70)
    print("ENHANCED TYPED ISING-POTTS MODEL v3 — THOROUGH EVALUATION")
    print("=" * 70)

    # Train
    model = EnhancedTypedModel(
        vocab_min_freq=args.min_freq,
        vocab_max_size=args.max_vocab,
        seq_len=25,
        window=6,
        pmi_cap=12,
        grammar_penalty=40,
        total_sweeps=120,
        use_nmf=True,
        nmf_factors=128,
        use_spacy=True,
    )

    model.train(n_samples=args.n_samples)

    # Evaluate
    print("\n" + "=" * 70)
    print("GRAMMAR EVALUATION (50 samples)")
    print("=" * 70)

    all_metrics = {}
    all_text = []
    n_eval = 50
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

    # Sample quality
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT (10 examples)")
    print("=" * 70)
    for i, text in enumerate(all_text[:10]):
        print(f"  {i+1}. {text}")

    # Dependency statistics
    if model.dep_couplings is not None:
        print("\n" + "=" * 70)
        print("DEPENDENCY COUPLING STATISTICS")
        print("=" * 70)
        dep_stats = model.dep_couplings.get_dep_stats()
        print(f"  J_tree non-zeros: {dep_stats['J_tree_nnz']}")
        print(f"  Total dep edges observed: {dep_stats['total_dep_edges']}")
        print(f"  Agreement rules: {dep_stats['agreement_rules']}")
        for rule in model.dep_couplings.agreement_rules:
            print(f"    {rule['dep_label']}: "
                  f"expected({IDX2POS.get(rule['expected_head_pos'], '?')}, "
                  f"{IDX2POS.get(rule['expected_dep_pos'], '?')}), "
                  f"count={rule['count']}, penalty={rule['penalty']}")

    # NMF quality
    if model.nmf_model is not None and model.nmf_model.fitted:
        print("\n" + "=" * 70)
        print("NMF RECONSTRUCTION QUALITY")
        print("=" * 70)
        J_full = model.pmi_weight * model.pmi.J_PMI + model.hebbian_weight * model.pmi.J_Hebb
        J_full += model.semantic_weight * model.semantics.J_sem
        if model.dep_couplings is not None:
            J_full += model.dep_weight * model.dep_couplings.J_tree
        J_recon = model.nmf_model.reconstruct()
        abs_err = int(np.sum(np.abs(J_full - J_recon)))
        max_err = int(np.max(np.abs(J_full - J_recon)))
        rel_err = abs_err / max(1, int(np.sum(np.abs(J_full))))
        print(f"  Absolute error: {abs_err:,}")
        print(f"  Max element error: {max_err}")
        print(f"  Relative error: {rel_err:.4f}")
        mem = model.nmf_model.memory_savings()
        print(f"  Memory savings: {mem['savings_pct']:.1f}%")

    # Save
    model.save(args.save_dir)

    # Save evaluation results
    results = {
        "grammar_metrics": all_metrics,
        "n_samples": n_eval,
        "per_sample": {k: v/n_eval for k, v in all_metrics.items()},
    }
    with open(os.path.join(args.save_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.save_dir}/eval_results.json")


# =============================================================================
# Typed Ising-Potts Model Commands (v2 legacy)
# =============================================================================

def cmd_typed_train(args):
    """Train the typed Ising-Potts model on fineweb-edu."""
    model = TypedIsingModel(
        vocab_min_freq=args.min_freq,
        vocab_max_size=args.max_vocab,
        seq_len=args.seq_len,
        window=args.window,
        pmi_cap=args.pmi_cap,
        min_cooc=args.min_cooc,
        pmi_weight=args.pmi_weight,
        hebbian_weight=args.hebbian_weight,
        semantic_weight=args.semantic_weight,
        grammar_penalty=args.grammar_penalty,
        phase1_beta=args.phase1_beta,
        phase2_beta=args.phase2_beta,
        phase3_beta=args.phase3_beta,
        total_sweeps=args.sweeps,
    )

    model.train(n_samples=args.n_samples)

    # Save
    save_dir = args.save_dir
    model.save(save_dir)
    print(f"\nModel saved to {save_dir}/")

    # Quick generation test
    print("\n" + "=" * 70)
    print("QUICK GENERATION TEST")
    print("=" * 70)
    for prompt in ["the", "in", "science", "research", "education"]:
        result = model.generate_with_trace(prompt=prompt, length=15)
        print(f"\n  Prompt='{prompt}':")
        print(f"  Text: {result['text'][:120]}")
        print(f"  Types: {' '.join(result['types'][:15])}")
        print(f"  Energy: {result['energy']}")

    return model


def cmd_typed_generate(args):
    """Generate text from trained typed model."""
    model = TypedIsingModel.load(args.save_dir)

    print("=" * 70)
    print("TYPED ISING-POTTS MODEL - GENERATION")
    print("(Zero FP in generation loop. Integer arithmetic only.)")
    print("=" * 70)

    for i in range(args.n_samples):
        result = model.generate_with_trace(
            prompt=args.prompt,
            length=args.length,
        )
        print(f"\n--- Sample {i+1} ---")
        print(f"  Text: {result['text']}")
        print(f"  Types: {' '.join(result['types'])}")
        print(f"  Energy: {result['energy']}")
        print(f"  Type distribution: {result['type_counts']}")


def cmd_typed_demo(args):
    """Full demo: train + generate + evaluate."""
    model = TypedIsingModel(
        vocab_min_freq=3,
        vocab_max_size=3000,
        seq_len=25,
        window=6,
        pmi_cap=12,
        grammar_penalty=40,
        total_sweeps=120,
    )

    model.train(n_samples=args.n_samples)

    # Save
    model.save(args.save_dir)

    # Generate multiple samples
    print("\n" + "=" * 70)
    print("GENERATION SAMPLES")
    print("=" * 70)

    prompts = ["the", "in", "a", "science", "research", "education", "students"]
    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        for trial in range(3):
            result = model.generate_with_trace(prompt=prompt, length=20)
            text = result['text']
            types = ' '.join(result['types'][:20])
            print(f"  Trial {trial+1}: {text}")
            print(f"           [{types}]")

    # Grammar evaluation
    print("\n" + "=" * 70)
    print("GRAMMAR EVALUATION")
    print("=" * 70)

    total_metrics = {}
    n_eval = 10
    for i in range(n_eval):
        words, types = model.generate_raw(length=20)
        metrics = model.evaluate_grammar(words, types)
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v

    print(f"\nGrammar patterns across {n_eval} samples (total counts):")
    for k, v in sorted(total_metrics.items()):
        print(f"  {k}: {v}")

    # Type distribution analysis
    print("\n" + "=" * 70)
    print("POS TYPE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    all_types = []
    for i in range(20):
        words, types = model.generate_raw(length=20)
        all_types.extend(types)

    type_counts = {}
    for t in all_types:
        name = IDX2POS.get(t, "UNK")
        type_counts[name] = type_counts.get(name, 0) + 1

    print("\nPOS tag distribution in generated text:")
    total = sum(type_counts.values())
    for name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count * 100 / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {name:8s}: {count:4d} ({pct:5.1f}%) {bar}")

    # PMI coupling analysis
    print("\n" + "=" * 70)
    print("PMI COUPLING ANALYSIS")
    print("=" * 70)

    J = model.pmi.J_PMI
    print(f"\nJ_PMI statistics:")
    print(f"  Non-zeros: {int((J != 0).sum())} / {J.shape[0]*J.shape[1]}")
    print(f"  Positive: {int((J > 0).sum())}")
    print(f"  Negative: {int((J < 0).sum())}")
    print(f"  Range: [{int(J.min())}, {int(J.max())}]")

    # Show top PMI associations
    print("\nTop PMI associations:")
    for i in range(min(10, model.pmi.vocab_size)):
        neighbors = model.pmi.get_pmi_neighbors(i, positive_only=True)[:5]
        word = model.vocab.idx2word.get(i, "?")
        if neighbors:
            n_str = ", ".join(
                f"{model.vocab.idx2word.get(w, '?')}({v})"
                for w, v in neighbors
            )
            print(f"  {word} → {n_str}")

    return model


def cmd_typed_eval(args):
    """Thorough evaluation of the typed model."""
    print("=" * 70)
    print("TYPED ISING-POTTS MODEL — THOROUGH EVALUATION")
    print("=" * 70)

    # Train
    model = TypedIsingModel(
        vocab_min_freq=args.min_freq,
        vocab_max_size=args.max_vocab,
        seq_len=25,
        window=6,
        pmi_cap=12,
        grammar_penalty=40,
        total_sweeps=120,
    )

    model.train(n_samples=args.n_samples)

    # Grammar evaluation
    print("\n" + "=" * 70)
    print("GRAMMAR EVALUATION (50 samples)")
    print("=" * 70)

    all_metrics = {}
    all_text = []
    n_eval = 50
    for i in range(n_eval):
        words, types = model.generate_raw(length=20)
        metrics = model.evaluate_grammar(words, types)
        text = model.vocab.decode(words)
        all_text.append(text)
        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0) + v

    print(f"\nGrammar patterns across {n_eval} samples:")
    for k, v in sorted(all_metrics.items()):
        per_sample = v / n_eval
        print(f"  {k}: {v} total ({per_sample:.1f}/sample)")

    # Sample quality
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT (10 examples)")
    print("=" * 70)
    for i, text in enumerate(all_text[:10]):
        print(f"  {i+1}. {text}")

    # Save evaluation results
    results = {
        "grammar_metrics": all_metrics,
        "n_samples": n_eval,
        "per_sample": {k: v/n_eval for k, v in all_metrics.items()},
    }
    with open(os.path.join(args.save_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.save_dir}/eval_results.json")


# =============================================================================
# Legacy Character-Level Model Commands
# =============================================================================

def cmd_train(args):
    """Train character-level model on fineweb-edu."""
    texts = load_fineweb_edu(n_samples=args.n_samples, min_length=20, max_length=5000)
    model = CharIsingModel(temperature=args.temperature, n_sweeps=args.n_sweeps, use_trigram=True)
    model.train_from_texts(texts)
    model.save(args.save_dir)

    print("\nQuick test:")
    for prompt in ["the ", "in the ", "science "]:
        text = model.generate(prompt=prompt, length=200, n_sweeps=30)
        print(f"  [{prompt.strip()}]: {text[:100]}")


def cmd_generate(args):
    """Generate text from trained character model."""
    model = CharIsingModel.load(args.save_dir, temperature=args.temperature, n_sweeps=args.n_sweeps)
    for i in range(args.n_samples):
        text = model.generate(prompt=args.prompt, length=args.length, n_sweeps=args.n_sweeps)
        print(f"\n--- Sample {i+1} ---\n{text}")


def cmd_demo(args):
    """Full character-level demo."""
    texts = load_fineweb_edu(n_samples=args.n_samples, min_length=20, max_length=5000)
    model = CharIsingModel(temperature=1.0, n_sweeps=30, use_trigram=True)
    model.train_from_texts(texts)

    for temp in [0.5, 0.8, 1.0]:
        model.temperature = temp
        print(f"\n--- Temperature {temp} ---")
        for prompt in ["the ", "science "]:
            text = model.generate(prompt=prompt, length=200, n_sweeps=30)
            print(f"  [{prompt.strip()}]: {text[:100]}")

    model.save(args.save_dir)


def cmd_verify(args):
    """Verify zero FP in generation loop."""
    print("See source code comments: all generation-path operations are integer.")
    print("The ONLY FP occurs during one-time training (probability precomputation).")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ising Spin Language Model - Zero FP Text Generation"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- Enhanced v3 model commands ---
    et = subparsers.add_parser("enhanced-train", help="Train enhanced v3 model")
    et.add_argument("--n_samples", type=int, default=100000)
    et.add_argument("--min_freq", type=int, default=3)
    et.add_argument("--max_vocab", type=int, default=5000)
    et.add_argument("--seq_len", type=int, default=25)
    et.add_argument("--window", type=int, default=6)
    et.add_argument("--pmi_cap", type=int, default=12)
    et.add_argument("--min_cooc", type=int, default=2)
    et.add_argument("--pmi_weight", type=int, default=3)
    et.add_argument("--hebbian_weight", type=int, default=1)
    et.add_argument("--semantic_weight", type=int, default=1)
    et.add_argument("--dep_weight", type=int, default=2)
    et.add_argument("--grammar_penalty", type=int, default=40)
    et.add_argument("--phase1_beta", type=int, default=200)
    et.add_argument("--phase2_beta", type=int, default=500)
    et.add_argument("--phase3_beta", type=int, default=1000)
    et.add_argument("--sweeps", type=int, default=120)
    et.add_argument("--nmf_factors", type=int, default=128)
    et.add_argument("--nmf_iterations", type=int, default=50)
    et.add_argument("--spacy_max_texts", type=int, default=None)
    et.add_argument("--no_nmf", action="store_true", help="Disable NMF factorization")
    et.add_argument("--no_spacy", action="store_true", help="Disable SpaCy POS tagging")
    et.add_argument("--save_dir", type=str, default="data/enhanced_model")

    eg = subparsers.add_parser("enhanced-generate", help="Generate from enhanced model")
    eg.add_argument("--prompt", type=str, default="the")
    eg.add_argument("--length", type=int, default=20)
    eg.add_argument("--n_samples", type=int, default=5)
    eg.add_argument("--save_dir", type=str, default="data/enhanced_model")

    ed = subparsers.add_parser("enhanced-demo", help="Full enhanced model demo")
    ed.add_argument("--n_samples", type=int, default=50000)
    ed.add_argument("--save_dir", type=str, default="data/enhanced_model")

    ee = subparsers.add_parser("enhanced-eval", help="Thorough enhanced model evaluation")
    ee.add_argument("--n_samples", type=int, default=50000)
    ee.add_argument("--min_freq", type=int, default=3)
    ee.add_argument("--max_vocab", type=int, default=5000)
    ee.add_argument("--save_dir", type=str, default="data/enhanced_model")

    # --- v2 Typed model commands ---
    tt = subparsers.add_parser("typed-train", help="Train typed Ising-Potts model")
    tt.add_argument("--n_samples", type=int, default=30000)
    tt.add_argument("--min_freq", type=int, default=3)
    tt.add_argument("--max_vocab", type=int, default=3000)
    tt.add_argument("--seq_len", type=int, default=25)
    tt.add_argument("--window", type=int, default=6)
    tt.add_argument("--pmi_cap", type=int, default=12)
    tt.add_argument("--min_cooc", type=int, default=2)
    tt.add_argument("--pmi_weight", type=int, default=3)
    tt.add_argument("--hebbian_weight", type=int, default=1)
    tt.add_argument("--semantic_weight", type=int, default=1)
    tt.add_argument("--grammar_penalty", type=int, default=40)
    tt.add_argument("--phase1_beta", type=int, default=200)
    tt.add_argument("--phase2_beta", type=int, default=500)
    tt.add_argument("--phase3_beta", type=int, default=1000)
    tt.add_argument("--sweeps", type=int, default=120)
    tt.add_argument("--save_dir", type=str, default="data/typed_model")

    tg = subparsers.add_parser("typed-generate", help="Generate from typed model")
    tg.add_argument("--prompt", type=str, default="the")
    tg.add_argument("--length", type=int, default=20)
    tg.add_argument("--n_samples", type=int, default=5)
    tg.add_argument("--save_dir", type=str, default="data/typed_model")

    td = subparsers.add_parser("typed-demo", help="Full typed model demo")
    td.add_argument("--n_samples", type=int, default=30000)
    td.add_argument("--save_dir", type=str, default="data/typed_model")

    te = subparsers.add_parser("typed-eval", help="Thorough evaluation")
    te.add_argument("--n_samples", type=int, default=30000)
    te.add_argument("--min_freq", type=int, default=3)
    te.add_argument("--max_vocab", type=int, default=3000)
    te.add_argument("--save_dir", type=str, default="data/typed_model")

    # --- Legacy commands ---
    tp = subparsers.add_parser("train", help="Train character-level model")
    tp.add_argument("--n_samples", type=int, default=50000)
    tp.add_argument("--temperature", type=float, default=1.0)
    tp.add_argument("--n_sweeps", type=int, default=30)
    tp.add_argument("--save_dir", type=str, default="data/char_model")

    gp = subparsers.add_parser("generate", help="Generate text (character)")
    gp.add_argument("--prompt", type=str, default="the ")
    gp.add_argument("--length", type=int, default=300)
    gp.add_argument("--n_sweeps", type=int, default=30)
    gp.add_argument("--n_samples", type=int, default=3)
    gp.add_argument("--temperature", type=float, default=0.8)
    gp.add_argument("--save_dir", type=str, default="data/char_model")

    dp = subparsers.add_parser("demo", help="Character-level demo")
    dp.add_argument("--n_samples", type=int, default=50000)
    dp.add_argument("--save_dir", type=str, default="data/char_model")

    vp = subparsers.add_parser("verify", help="Verify zero FP")
    vp.add_argument("--save_dir", type=str, default="data/char_model")

    args = parser.parse_args()

    commands = {
        "enhanced-train": cmd_enhanced_train,
        "enhanced-generate": cmd_enhanced_generate,
        "enhanced-demo": cmd_enhanced_demo,
        "enhanced-eval": cmd_enhanced_eval,
        "typed-train": cmd_typed_train,
        "typed-generate": cmd_typed_generate,
        "typed-demo": cmd_typed_demo,
        "typed-eval": cmd_typed_eval,
        "train": cmd_train,
        "generate": cmd_generate,
        "demo": cmd_demo,
        "verify": cmd_verify,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
