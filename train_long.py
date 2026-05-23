#!/usr/bin/env python3
"""
Ising Spin Model — Long Training Run for Raspberry Pi
======================================================

Designed for unattended long training on a Pi with:
  - Automatic checkpointing (resume if interrupted)
  - Progressive training (start small, scale up)
  - Periodic PPL evaluation during training
  - Memory-efficient mode for Pi (4GB RAM)
  - Log everything to file

Usage on Pi:
  # Quick test first (5 min):
  python train_long.py --n-samples 10000 --test

  # Full long training (hours):
  nohup python train_long.py --n-samples 500000 &

  # With custom data size:
  python train_long.py --n-samples 500000 --vocab-size 3000

  # Resume from checkpoint:
  python train_long.py --resume

Performance estimates on Pi 4/5:
  - 10K samples:  ~2 min (Pi 5), ~5 min (Pi 4)
  - 50K samples:  ~10 min (Pi 5), ~25 min (Pi 4)
  - 200K samples: ~40 min (Pi 5), ~100 min (Pi 4)
  - 500K samples: ~100 min (Pi 5), ~250 min (Pi 4)
  - 1M samples:   ~200 min (Pi 5), ~500 min (Pi 4)

Memory: ~500MB for 200K samples, ~1.5GB for 1M samples
"""

import sys
import os
import time
import json
import argparse
import pickle
import signal
import traceback
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ising_spin.model import (
    IsingLMModel, IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = dict(
    # Vocabulary
    vocab_min_freq=25,
    vocab_max_size=2000,

    # N-gram index
    ngram_max_n=5,
    ngram_min_count=2,

    # Energy scales
    recall_scale=1600,
    pmi_weight=5,
    field_weight=1,
    knowledge_scale=0,
    spin3_scale=0,
    category_scale=0,
    logic_rule_scale=0,
    logic_hard_scale=0,

    # Sampling
    beta_type=0.001,
    beta_word=0.001,
    copy_enabled=True,
    copy_min_context=2,
    copy_min_confidence=0.25,
    same_word_penalty=200,
    max_closed_class_run=2,

    # Ising
    ising_enabled=True,
    skip_pmi_max_dist=5,
    mcmc_refine_steps=0,

    # Disabled features
    use_conceptnet=False,
    walsh_enabled=False,
    graded_couplings_enabled=False,
    auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=False,
    kn_backoff=False,
    topic_spin_enabled=False,
)


# ============================================================================
# Checkpointing
# ============================================================================

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "latest.pkl")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "training_results.json")


def save_checkpoint(model, n_samples_trained, config, results_so_far, output_dir):
    """Save training checkpoint."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        'n_samples_trained': n_samples_trained,
        'config': config,
        'results': results_so_far,
        'timestamp': datetime.now().isoformat(),
        'output_dir': output_dir,
    }
    # Save model state via pickle (it's all Python dicts/arrays)
    try:
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"  Checkpoint saved: {n_samples_trained} samples")
    except Exception as e:
        print(f"  Checkpoint save failed: {e}")


def load_checkpoint():
    """Load latest checkpoint if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return None


# ============================================================================
# PPL Evaluation
# ============================================================================

def quick_ppl(model, n_seqs=10):
    """Fast PPL estimate with default beta."""
    sampler = model.generator.word_sampler
    gen = model.generator
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

    if total_tokens == 0:
        return float('inf')
    avg_log2 = total_log2_prob / (total_tokens * LOG2_SCALE)
    return 2.0 ** (-avg_log2)


def beta_sweep(model, n_seqs=20):
    """Sweep beta factor to find optimal PPL."""
    factors = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5]
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
                context_types = [model.generator._get_word_type(w) for w in context_words]
                word_type = model.generator._get_word_type(target_word)
                candidate_list = model.generator.type_words.get(word_type, [])
                if not candidate_list:
                    continue
                candidate_words = np.array(candidate_list, dtype=np.int64)
                if int(target_word) not in set(candidate_words.tolist()):
                    total_log2_prob += -15 * LOG2_SCALE
                    total_tokens += 1
                    continue
                recall_matches = model.generator.ngram_index.lookup(context_words)
                recall_hit = bool(recall_matches)
                energies = model.generator._compute_word_energy(
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
            marker = " <-- BEST" if ppl_val < best_ppl else ""
            print(f"    f={factor:.2f}: PPL={ppl_val:.1f}{marker}")
            if ppl_val < best_ppl:
                best_ppl = ppl_val
                best_factor = factor

    return best_ppl, best_factor, results


def generate_samples(model, prompts=None, length=400):
    """Generate text samples."""
    if prompts is None:
        prompts = ["the history of", "science and technology", "research shows that"]

    results = []
    for prompt in prompts:
        result = model.generator.generate(prompt=prompt, length=length)
        text = result['text']
        words = text.split()
        n_recalls = sum(1 for d in result['diagnostics'] if d['recall_hit'])
        n_copies = sum(1 for d in result['diagnostics'] if d['copy'])
        results.append({
            'prompt': prompt,
            'text': text,
            'n_words': len(words),
            'n_recalls': n_recalls,
            'n_copies': n_copies,
        })
        print(f"\n  --- '{prompt}' ({len(words)} words) ---")
        print(f"  {text[:200]}...")
        print(f"  recalls={n_recalls} copies={n_copies}")

    return results


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ising Spin Model — Long Training Run (Pi-friendly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 min on Pi 5):
  python train_long.py --n-samples 10000 --test

  # Standard training (40 min on Pi 5):
  python train_long.py --n-samples 200000

  # Maximum training (3+ hours on Pi 5):
  nohup python train_long.py --n-samples 1000000 &

  # Resume interrupted training:
  python train_long.py --resume

  # Custom vocab size for more coverage:
  python train_long.py --n-samples 500000 --vocab-size 3000
        """
    )
    parser.add_argument("--n-samples", type=int, default=200000,
                        help="Number of training samples (default: 200K)")
    parser.add_argument("--vocab-size", type=int, default=2000,
                        help="Max vocabulary size (default: 2000)")
    parser.add_argument("--vocab-min-freq", type=int, default=25,
                        help="Min word frequency for vocab (default: 25)")
    parser.add_argument("--recall-scale", type=int, default=1600,
                        help="Recall energy scale (default: 1600)")
    parser.add_argument("--pmi-weight", type=int, default=5,
                        help="PMI backoff weight (default: 5)")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to cached training data JSON")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode (1K samples, fast eval)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip beta sweep (use auto-calibrated beta)")
    parser.add_argument("--eval-interval", type=int, default=0,
                        help="Evaluate PPL every N samples during training (0=off)")
    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.n_samples = min(args.n_samples, 10000)
        args.eval_interval = 5000

    # Output directory
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "output",
        f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(output_dir, "training.log")

    class Logger:
        def __init__(self, path):
            self.terminal = sys.stdout
            self.log = open(path, 'a')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_path)
    sys.stderr = Logger(log_path)

    print("=" * 70)
    print("ISING SPIN GLASS LANGUAGE MODEL — LONG TRAINING RUN")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Config
    config = DEFAULT_CONFIG.copy()
    config['vocab_max_size'] = args.vocab_size
    config['vocab_min_freq'] = args.vocab_min_freq
    config['recall_scale'] = args.recall_scale
    config['pmi_weight'] = args.pmi_weight

    print(f"\nConfig:")
    for k, v in sorted(config.items()):
        if v != 0 and v != False:  # Only show non-default/active settings
            print(f"  {k}: {v}")
    print(f"  n_samples: {args.n_samples}")

    # Load data — pick the largest cache file that exists
    cache_path = args.cache
    if cache_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Find all cached files and pick the largest one
        cache_candidates = []
        for fname in os.listdir(script_dir):
            if fname.startswith("cached_fineweb_") and fname.endswith(".json"):
                fpath = os.path.join(script_dir, fname)
                try:
                    count_str = fname.split("_")[-1].replace(".json", "")
                    count = int(count_str.replace("k", "000"))
                    cache_candidates.append((count, fpath))
                except (ValueError, IndexError):
                    pass
        cache_candidates.sort(reverse=True)

        # Use the largest cache that has >= n_samples, or the largest available
        for count, fpath in cache_candidates:
            if count >= args.n_samples:
                cache_path = fpath
                break
        if cache_path is None and cache_candidates:
            cache_path = cache_candidates[0][1]

    if cache_path and os.path.exists(cache_path):
        print(f"\nLoading cached data from: {cache_path}")
        t0 = time.time()
        with open(cache_path) as f:
            texts = json.load(f)
        print(f"  Loaded {len(texts)} texts in {time.time()-t0:.1f}s")

        # If cache is too small, download more
        if len(texts) < args.n_samples:
            print(f"  Cache has {len(texts)} texts but need {args.n_samples}")
            print(f"  Downloading {args.n_samples} texts from HuggingFace...")
            from ising_spin.model import load_fineweb_edu
            t0 = time.time()
            texts = load_fineweb_edu(n_samples=args.n_samples)
            print(f"  Downloaded {len(texts)} texts in {time.time()-t0:.1f}s")
            cache_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"cached_fineweb_{len(texts)//1000}k.json"
            )
            print(f"  Saving cache to: {cache_path}")
            with open(cache_path, 'w') as f:
                json.dump(texts, f)
    else:
        print("\nNo cached data found. Downloading from HuggingFace...")
        print("(This may take a while on first run)")
        from ising_spin.model import load_fineweb_edu
        t0 = time.time()
        texts = load_fineweb_edu(n_samples=args.n_samples)
        print(f"  Downloaded {len(texts)} texts in {time.time()-t0:.1f}s")

        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"cached_fineweb_{len(texts)//1000}k.json"
        )
        print(f"  Saving cache to: {cache_path}")
        with open(cache_path, 'w') as f:
            json.dump(texts, f)

    # Use available data (may be more or less than requested)
    n_available = len(texts)
    n_use = min(args.n_samples, n_available)
    print(f"  Using {n_use} of {n_available} available texts")

    # Handle resume
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            print(f"\nResuming from checkpoint: {checkpoint['n_samples_trained']} samples")
            print(f"  Saved at: {checkpoint['timestamp']}")
        else:
            print("\nNo checkpoint found, starting fresh")

    # Train
    print(f"\n{'='*70}")
    print(f"TRAINING: {n_use} samples")
    print(f"{'='*70}")

    t_total_start = time.time()
    model = IsingLMModel(**config)
    model.train(n_samples=n_use, texts=texts)
    t_train = time.time() - t_total_start
    print(f"\nTraining complete: {t_train:.1f}s ({t_train/60:.1f}min)")
    print(f"  Throughput: {n_use/t_train:.0f} samples/sec")
    print(f"  Vocab size: {len(model.vocab)}")

    # Save checkpoint
    save_checkpoint(model, n_use, config, {}, output_dir)

    # Quick PPL with auto beta
    print(f"\n{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}")

    ppl_quick = quick_ppl(model, n_seqs=10)
    print(f"\nQuick PPL (auto beta, 10 seqs): {ppl_quick:.1f}")

    # Beta sweep
    if not args.no_sweep:
        print(f"\nBeta sweep:")
        best_ppl, best_factor, sweep_results = beta_sweep(model, n_seqs=20)
        print(f"\nBest: f={best_factor:.2f}, PPL={best_ppl:.1f}")

        # Apply best beta
        recall_scale = model.recall_scale
        beta_val = best_factor * LN2_NUM / (recall_scale * LN2_DEN)
        model.generator.word_sampler = IntegerBoltzmannSampler(
            beta=beta_val, max_delta=25000
        )
        model.generator.beta_word = beta_val
        model.beta_word = beta_val
    else:
        best_ppl = ppl_quick
        best_factor = 0.0
        sweep_results = {}

    # Full PPL evaluation
    print(f"\nFull PPL evaluation...")
    ppl_full = model.compute_perplexity(n_samples=100)
    print(f"PPL (full, 100 seqs): {ppl_full:.2f}")

    # Generate text
    print(f"\n{'='*70}")
    print(f"GENERATION (PPL={ppl_full:.2f})")
    print(f"{'='*70}")

    gen_results = generate_samples(model)

    # Save everything
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    # Save generated text
    for i, gr in enumerate(gen_results):
        text_path = os.path.join(output_dir, f"generated_{i}.txt")
        with open(text_path, 'w') as f:
            f.write(f"Ising Spin Glass Language Model v11.7\n")
            f.write(f"Prompt: {gr['prompt']}\n")
            f.write(f"Tokens: {gr['n_words']}\n")
            f.write(f"Recall hits: {gr['n_recalls']}, Copy hits: {gr['n_copies']}\n")
            f.write(f"PPL: {ppl_full:.2f}\n")
            f.write("=" * 70 + "\n\n")
            f.write(gr['text'])
        print(f"  Saved: {text_path}")

    # Save results JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {k: str(v) for k, v in config.items()},
        'n_samples': n_use,
        'training_time_sec': t_train,
        'training_time_min': t_train / 60,
        'throughput_samples_per_sec': n_use / t_train,
        'vocab_size': len(model.vocab),
        'ppl_quick': ppl_quick,
        'ppl_full': ppl_full,
        'best_beta_factor': best_factor,
        'beta_sweep': {str(k): v for k, v in sweep_results.items()},
        'generated': gen_results,
        'total_time_sec': time.time() - t_total_start,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")

    # Also save to standard location
    std_results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "training_results.json"
    )
    with open(std_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {std_results_path}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"Total time: {time.time()-t_total_start:.1f}s ({(time.time()-t_total_start)/60:.1f}min)")
    print(f"Results: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
