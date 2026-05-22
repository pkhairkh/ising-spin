#!/usr/bin/env python3
"""
v12 Training Script — Memory-Efficient + PPL-Optimized

Key improvements over v11:
  1. Memory-efficient batched n-gram index building (handles 3M+ texts on 16GB Pi)
  2. Auto-scaling ngram_min_count for large corpora
  3. Proper cache file handling (picks correct cache for requested sample count)
  4. Topic Spin coherence engine enabled
  5. Kneser-Ney backoff enabled
  6. Interpolated n-gram smoothing enabled
  7. Larger vocab (4000) for better coverage

Target: PPL ~20 on Pi (16GB RAM, 4 cores)

Usage:
  python train_v12.py                    # Default: 500K samples
  python train_v12.py --samples 1000000  # 1M samples
  python train_v12.py --samples 3000000  # 3M samples (uses batched build)
  python train_v12.py --samples 5000000  # 5M samples (uses batched build)
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────

DEFAULT_SAMPLES = 500000
DEFAULT_VOCAB = 4000
DEFAULT_RECALL_SCALE = 1600
DEFAULT_PMI_WEIGHT = 5
DEFAULT_SAME_WORD_PENALTY = 5000

# Cache directory
CACHE_DIR = Path(__file__).parent
OUTPUT_DIR = CACHE_DIR / "output"


def find_cache_file(n_samples: int) -> str:
    """
    Find the best cache file for the requested number of samples.
    
    Rules:
      - Look for exact match first (e.g., cached_fineweb_500k.json for 500K)
      - Then look for a cache with ENOUGH data (>= n_samples)
      - Prefer the smallest sufficient cache to minimize loading time
      - Never use a cache with fewer texts than requested
    """
    cache_files = {}
    for f in CACHE_DIR.glob("cached_fineweb_*.json"):
        # Parse size from filename: cached_fineweb_500k.json -> 500000
        name = f.stem  # cached_fineweb_500k
        size_str = name.split("_")[-1]  # 500k
        try:
            if size_str.endswith("k"):
                size = int(size_str[:-1]) * 1000
            elif size_str.endswith("m"):
                size = int(size_str[:-1]) * 1000000
            else:
                size = int(size_str)
            cache_files[size] = str(f)
        except (ValueError, IndexError):
            continue
    
    # Exact match first
    if n_samples in cache_files:
        return cache_files[n_samples]
    
    # Find smallest cache with enough data
    sufficient = {s: p for s, p in cache_files.items() if s >= n_samples}
    if sufficient:
        best_size = min(sufficient.keys())
        return sufficient[best_size]
    
    # No cache has enough data
    return None


def load_data(n_samples: int) -> list:
    """
    Load or download FineWeb-Edu data with proper cache handling.
    
    This fixes the bug where the wrong cache file was loaded:
    - Old behavior: picked any cache file, even if too small
    - New behavior: only uses cache if it has enough data
    """
    cache_path = find_cache_file(n_samples)
    
    if cache_path:
        print(f"Loading: {cache_path}")
        with open(cache_path, "r") as f:
            texts = json.load(f)
        print(f"  {len(texts)} texts loaded")
        
        if len(texts) >= n_samples:
            # Use only the requested number
            return texts[:n_samples]
        else:
            # Cache doesn't have enough — need to download more
            print(f"  Cache has {len(texts)} texts but need {n_samples}")
            print(f"  Downloading {n_samples} texts from FineWeb-Edu...")
            # Fall through to download
    
    # Download from HuggingFace
    print("No cached data found. Downloading from HuggingFace...")
    print("(This may take a while on first run)")
    
    from ising_spin.model import load_fineweb_edu
    t0 = time.time()
    texts = load_fineweb_edu(n_samples=n_samples)
    print(f"  Downloaded {len(texts)} texts in {time.time()-t0:.1f}s")
    
    # Save cache with size-based naming
    if n_samples >= 1000000 and n_samples % 1000000 == 0:
        cache_name = f"cached_fineweb_{n_samples // 1000000}m.json"
    elif n_samples >= 1000 and n_samples % 1000 == 0:
        cache_name = f"cached_fineweb_{n_samples // 1000}k.json"
    else:
        cache_name = f"cached_fineweb_{n_samples}.json"
    
    cache_file = CACHE_DIR / cache_name
    print(f"  Saving cache to: {cache_file}")
    with open(cache_file, "w") as f:
        json.dump(texts, f)
    
    return texts


def beta_sweep_ppl(model, beta_factors=None, n_seqs=10):
    """
    Sweep beta_factor to find optimal PPL.
    
    beta_factor scales the word_sampler's beta:
    effective_beta = beta_factor * (0.5 * ln(2) / recall_scale)
    
    The model's compute_perplexity uses the word_sampler's beta directly,
    so we temporarily modify it for each sweep point.
    """
    if beta_factors is None:
        beta_factors = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.50]
    
    import numpy as np
    from ising_spin.model import LN2_NUM, LN2_DEN
    
    # Compute the theoretical beta: beta = 0.5 * ln(2) / recall_scale
    recall_scale = model.recall_scale
    # beta = 0.5 * ln(2) / scale, but in fixed-point: 
    # The IntegerBoltzmannSampler uses beta directly
    base_beta = 0.5 * float(LN2_NUM) / float(LN2_DEN) / recall_scale
    
    best_ppl = float('inf')
    best_f = 1.0
    original_beta = model.generator.word_sampler.beta
    
    print("\nBeta sweep:")
    for f in beta_factors:
        # Temporarily set beta
        model.generator.word_sampler.beta = base_beta * f
        try:
            ppl = model.compute_perplexity(n_samples=n_seqs)
            marker = " <-- BEST" if ppl < best_ppl else ""
            print(f"    f={f:.2f}: PPL={ppl:.1f}{marker}")
            if ppl < best_ppl:
                best_ppl = ppl
                best_f = f
        except Exception as e:
            print(f"    f={f:.2f}: Error: {e}")
    
    # Restore best beta
    model.generator.word_sampler.beta = base_beta * best_f
    print(f"\nBest: f={best_f:.2f}, PPL={best_ppl:.1f}")
    
    return best_f, best_ppl


def main():
    parser = argparse.ArgumentParser(description="v12 Training — Memory-Efficient + PPL-Optimized")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="Number of training samples (default: 500K)")
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB,
                        help="Vocabulary size (default: 4000)")
    parser.add_argument("--recall-scale", type=int, default=DEFAULT_RECALL_SCALE,
                        help="Recall energy scale (default: 1600)")
    parser.add_argument("--pmi-weight", type=int, default=DEFAULT_PMI_WEIGHT,
                        help="PMI coupling weight (default: 5)")
    parser.add_argument("--same-word-penalty", type=int, default=DEFAULT_SAME_WORD_PENALTY,
                        help="Same-word repetition penalty (default: 200)")
    parser.add_argument("--no-topic-spin", action="store_true",
                        help="Disable Topic Spin coherence engine")
    parser.add_argument("--no-kn-backoff", action="store_true",
                        help="Disable Kneser-Ney backoff")
    parser.add_argument("--no-interpolated", action="store_true",
                        help="Disable interpolated n-gram smoothing")
    parser.add_argument("--ngram-min-count", type=int, default=2,
                        help="N-gram minimum count (default: 2, auto-scaled for large corpora)")
    parser.add_argument("--max-seq-len", type=int, default=30,
                        help="Maximum sequence length for training (default: 30)")
    args = parser.parse_args()
    
    # ─── Header ─────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"v12_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ISING SPIN GLASS LANGUAGE MODEL — v12 TRAINING")
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"Output: {output_dir}")
    print(f"Workers: {os.cpu_count()} (CPU count: {os.cpu_count()})")
    print("=" * 70)
    
    # ─── Load Data ──────────────────────────────────────────
    texts = load_data(args.samples)
    n_texts = len(texts)
    print(f"Using {n_texts} of {n_texts} available texts")
    
    # ─── Config ─────────────────────────────────────────────
    topic_spin = not args.no_topic_spin
    kn_backoff = not args.no_kn_backoff
    interpolated = not args.no_interpolated
    
    print(f"\n{'=' * 70}")
    print(f"CONFIG: v12")
    print(f"  vocab_max_size={args.vocab}")
    print(f"  kn_backoff={kn_backoff}")
    print(f"  interpolated={interpolated}")
    print(f"  topic_spin={topic_spin}")
    print(f"  ngram_min_count={args.ngram_min_count}")
    print(f"  recall_scale={args.recall_scale}")
    print(f"  pmi_weight={args.pmi_weight}")
    print(f"  same_word_penalty={args.same_word_penalty}")
    print(f"  max_seq_len={args.max_seq_len}")
    print(f"  n_samples={n_texts}")
    print(f"{'=' * 70}")
    
    # ─── Train ──────────────────────────────────────────────
    from ising_spin.model import IsingLMModel
    
    model = IsingLMModel(
        # Vocabulary
        vocab_min_freq=25,
        vocab_max_size=args.vocab,
        # N-gram
        ngram_max_n=5,
        ngram_min_count=args.ngram_min_count,
        # PMI
        pmi_window=5,
        pmi_min_count=2,
        pmi_cap=10,
        # Energy scales
        recall_scale=args.recall_scale,
        pmi_weight=args.pmi_weight,
        field_weight=1,
        same_word_penalty=args.same_word_penalty,
        # Beta — use auto-calibration from recall energies
        beta_type=0.001,
        beta_word=0.001,
        # Copy mechanism
        copy_enabled=True,
        copy_min_context=2,
        copy_min_confidence=0.25,
        # Ising
        ising_enabled=True,
        skip_pmi_max_dist=5,
        # v8.0: Recall-primary mode (keeps other layers as small perturbations)
        recall_primary_mode=True,
        # v8.2: Topic Spin (Potts coherence layer)
        topic_spin_enabled=topic_spin,
        topic_n_topics=16,
        topic_coherence_penalty=400,
        topic_spin_flip_interval=20,
        topic_context_window=30,
        topic_coupling_scale=100,
        # v9.0: Interpolated n-gram smoothing (product of experts)
        interpolated=interpolated,
        # v10.0: Kneser-Ney backoff (continuation counts)
        kn_backoff=kn_backoff,
        # Disabled layers (recall-primary mode)
        knowledge_scale=0,
        spin3_scale=0,
        category_scale=0,
        logic_rule_scale=0,
        # v12: Enable auto-calibrate beta (fixed formula now with energy bug fix)
        auto_calibrate_beta=True,
        use_conceptnet=False,
        max_closed_class_run=2,
    )
    
    # Monkey-patch truncate_sequences to use configurable max_len
    import ising_spin.model as model_module
    _original_truncate = model_module.truncate_sequences
    
    def _custom_truncate(sequences, max_len=args.max_seq_len):
        return _original_truncate(sequences, max_len=max_len)
    
    model_module.truncate_sequences = _custom_truncate
    
    t_start = time.time()
    
    try:
        model.train(n_samples=n_texts, texts=texts)
    except MemoryError:
        print("\n!!! OUT OF MEMORY during training !!!")
        print("Try reducing --samples or increasing --ngram-min-count")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! TRAINING ERROR: {e} !!!")
        traceback.print_exc()
        sys.exit(1)
    
    t_train = time.time() - t_start
    print(f"\nTraining complete: {t_train:.1f}s ({t_train/60:.1f}min)")
    print(f"  Throughput: {n_texts / t_train:.0f} samples/sec")
    print(f"  Vocab size: {len(model.vocab)}")
    
    # Restore original truncate
    model_module.truncate_sequences = _original_truncate
    
    # ─── Evaluation ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("EVALUATION")
    print(f"{'=' * 70}")
    
    # Quick PPL at default beta
    print("\nQuick PPL (default beta, 10 seqs):", end=" ")
    try:
        quick_ppl = model.compute_perplexity(n_samples=10)
        print(f"{quick_ppl:.1f}")
    except Exception as e:
        print(f"Error: {e}")
        quick_ppl = 999
    
    # Beta sweep
    best_f, best_sweep_ppl = beta_sweep_ppl(model, n_seqs=10)
    
    # Full PPL evaluation at best beta
    print(f"\nFull PPL evaluation...")
    try:
        full_ppl = model.compute_perplexity(n_samples=100)
        print(f"  Perplexity: {full_ppl:.2f}")
    except Exception as e:
        print(f"  Error: {e}")
        full_ppl = best_sweep_ppl
    
    print(f"\nPPL (full, 100 seqs): {full_ppl:.2f}")
    
    # ─── Generation ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"GENERATION (PPL={full_ppl:.2f})")
    print(f"{'=' * 70}")
    
    prompts = ["the history of", "science and technology", "research shows that"]
    generated_texts = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n  --- '{prompt}' (400 words) ---")
        try:
            result = model.generator.generate(prompt=prompt, length=400)
            text = result.get("text", str(result))
            if isinstance(text, list):
                text = " ".join(text)
            print(f"  {text[:200]}...")
            
            # Count recalls and copies from stats
            stats = model.generator.get_stats()
            recalls = stats.get("recall_hits", 0)
            copies = stats.get("copy_count", 0)
            print(f"  recalls={recalls} copies={copies}")
            
            generated_texts.append(text)
            
            # Save generated text
            gen_file = output_dir / f"generated_{i}.txt"
            with open(gen_file, "w") as f:
                f.write(text)
        except Exception as e:
            print(f"  Generation error: {e}")
            generated_texts.append("")
    
    # ─── Save Results ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SAVING RESULTS")
    print(f"{'=' * 70}")
    
    results = {
        "version": "v12",
        "timestamp": timestamp,
        "config": {
            "vocab_max_size": args.vocab,
            "ngram_min_count": args.ngram_min_count,
            "recall_scale": args.recall_scale,
            "pmi_weight": args.pmi_weight,
            "same_word_penalty": args.same_word_penalty,
            "kn_backoff": kn_backoff,
            "interpolated": interpolated,
            "topic_spin": topic_spin,
            "n_samples": n_texts,
            "max_seq_len": args.max_seq_len,
        },
        "results": {
            "training_time_sec": t_train,
            "quick_ppl": quick_ppl,
            "best_beta_factor": best_f,
            "best_sweep_ppl": best_sweep_ppl,
            "full_ppl": full_ppl,
            "vocab_size": len(model.vocab),
        },
    }
    
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_file}")
    
    # Also save to root for easy access
    root_results = CACHE_DIR / "training_results.json"
    with open(root_results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {root_results}")
    
    t_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"DONE")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
