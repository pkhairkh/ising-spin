#!/usr/bin/env python3
"""
v13 Training Script — Genuine Spin Glass (Strong Couplings)

PHILOSOPHY: v12's "recall-primary" mode kept PMI couplings at 0.3% of
recall_scale — essentially a smoothed n-gram model with cosmetic Ising
decorations. The PPL ceiling was ~50, determined entirely by the n-gram
model quality.

v13 makes the model a GENUINE spin glass where couplings compete with
the recall field. In a real Ising model, J ~ h (coupling comparable to
external field). When J << h, you get independent spins in a field (n-gram).
When J ~ h, you get collective behavior (spin glass) with long-range
correlations that n-grams can't capture.

Key changes from v12:
  1. PMI weight: 5 → 200 (12.5% of recall_scale=1600)
     - PMI captures co-occurrence beyond exact n-gram matches
     - Skip-gram PMI at distances 1-5 adds long-range context
     - Maximum PMI contribution per context word: 200*10=2000
     - Over 5-word window: up to 10000 (comparable to recall energy)
  2. recall_primary_mode: False — allows strong couplings to compete
     - v12 capped all non-recall layers to <10% of recall_scale
     - v13 lets PMI couplings be a genuine part of the energy landscape
  3. auto_calibrate_beta: still enabled but will find different optimal
     because the energy landscape now includes strong couplings

Strategy: Start with PMI-only coupling (knowledge_scale=0, topic_spin=400).
If PMI helps, iterate by adding knowledge layer and stronger topic spin.
If PMI hurts, it means the PMI signal is too noisy at this scale.

Usage:
  python -u train_v13.py                    # Default: 1M samples
  python -u train_v13.py --samples 500000   # 500K (faster iteration)
  python -u train_v13.py --pmi-weight 800   # Try even stronger couplings
  python -u train_v13.py --pmi-weight 0     # Ablation: no PMI (pure n-gram)
"""

# ─── UNBUFFERED OUTPUT — prevents silent nohup hangs ─────────────────
import os
os.environ["PYTHONUNBUFFERED"] = "1"

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────

DEFAULT_SAMPLES = 1000000
DEFAULT_VOCAB = 4000
DEFAULT_RECALL_SCALE = 1600
DEFAULT_PMI_WEIGHT = 200  # v13: 12.5% of recall_scale (was 5 = 0.3%)
DEFAULT_SAME_WORD_PENALTY = 200
DEFAULT_NGRAM_MAX_SEQS = 1000000
DEFAULT_KNOWLEDGE_SCALE = 0    # v13: start with 0 (can enable later)
DEFAULT_TOPIC_PENALTY = 400    # v13: keep same (topic spin is marginal)

# Cache directory
CACHE_DIR = Path(__file__).parent
OUTPUT_DIR = CACHE_DIR / "output"


def get_rss_mb() -> int:
    """Get current process RSS in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024  # KB -> MB
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        except Exception:
            return 0
    return 0


def find_cache_file(n_samples: int) -> str:
    """Find the best cache file for the requested number of samples."""
    cache_files = {}
    for f in CACHE_DIR.glob("cached_fineweb_*.json"):
        name = f.stem
        size_str = name.split("_")[-1]
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

    if n_samples in cache_files:
        return cache_files[n_samples]

    sufficient = {s: p for s, p in cache_files.items() if s >= n_samples}
    if sufficient:
        best_size = min(sufficient.keys())
        return sufficient[best_size]

    return None


def load_data(n_samples: int) -> list:
    """Load or download FineWeb-Edu data with proper cache handling."""
    cache_path = find_cache_file(n_samples)

    if cache_path:
        print(f"Loading cache: {cache_path}")
        t0 = time.time()
        with open(cache_path, "r") as f:
            texts = json.load(f)
        t_load = time.time() - t0
        print(f"  {len(texts):,} texts loaded in {t_load:.1f}s")

        if len(texts) >= n_samples:
            return texts[:n_samples]
        else:
            print(f"  Cache has {len(texts):,} texts but need {n_samples:,}")

    print("No cached data found. Downloading from HuggingFace...")
    from ising_spin.model import load_fineweb_edu
    t0 = time.time()
    texts = load_fineweb_edu(n_samples=n_samples)
    print(f"  Downloaded {len(texts):,} texts in {time.time()-t0:.1f}s")

    if n_samples >= 1000000 and n_samples % 1000000 == 0:
        cache_name = f"cached_fineweb_{n_samples // 1000000}m.json"
    elif n_samples >= 1000 and n_samples % 1000 == 0:
        cache_name = f"cached_fineweb_{n_samples // 1000}k.json"
    else:
        cache_name = f"cached_fineweb_{n_samples}.json"

    cache_file = CACHE_DIR / cache_name
    print(f"  Saving cache to: {cache_file}")
    t0 = time.time()
    with open(cache_file, "w") as f:
        json.dump(texts, f)
    print(f"  Cache saved in {time.time()-t0:.1f}s")

    return texts


def beta_sweep_ppl(model, beta_factors=None, n_seqs=10):
    """Two-phase beta sweep to find optimal PPL."""
    import numpy as np
    from ising_spin.model import LN2_NUM, LN2_DEN, IntegerBoltzmannSampler

    recall_scale = model.recall_scale
    base_beta = 0.5 * float(LN2_NUM) / float(LN2_DEN) / recall_scale

    if beta_factors is None:
        beta_factors = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00]

    best_ppl = float('inf')
    best_f = 1.0

    print("\nBeta sweep (Phase 1 — coarse):")
    for f in beta_factors:
        new_beta = base_beta * f
        model.generator.word_sampler = IntegerBoltzmannSampler(
            beta=new_beta, max_delta=35000
        )
        try:
            ppl = model.compute_perplexity(n_samples=n_seqs)
            marker = " <-- BEST" if ppl < best_ppl else ""
            print(f"    f={f:.2f} (β={new_beta:.6f}): PPL={ppl:.1f}{marker}")
            if ppl < best_ppl:
                best_ppl = ppl
                best_f = f
        except Exception as e:
            print(f"    f={f:.2f}: Error: {e}")

    # Phase 2: Fine sweep around best coarse result
    fine_lo = max(0.5, best_f - 0.5)
    fine_hi = best_f + 0.5
    fine_factors = np.arange(fine_lo, fine_hi + 0.1, 0.1).tolist()
    fine_factors = [f for f in fine_factors if abs(f - best_f) > 0.05 or f == best_f]

    if len(fine_factors) > 1:
        print(f"\nBeta sweep (Phase 2 — fine around f={best_f:.2f}):")
        for f in fine_factors:
            new_beta = base_beta * f
            model.generator.word_sampler = IntegerBoltzmannSampler(
                beta=new_beta, max_delta=35000
            )
            try:
                ppl = model.compute_perplexity(n_samples=n_seqs)
                marker = " <-- BEST" if ppl < best_ppl else ""
                print(f"    f={f:.2f} (β={new_beta:.6f}): PPL={ppl:.1f}{marker}")
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_f = f
            except Exception as e:
                print(f"    f={f:.2f}: Error: {e}")

    model.generator.word_sampler = IntegerBoltzmannSampler(
        beta=base_beta * best_f, max_delta=35000
    )
    print(f"\nBest: f={best_f:.2f} (β={base_beta * best_f:.6f}), PPL={best_ppl:.1f}")

    return best_f, best_ppl


def main():
    parser = argparse.ArgumentParser(description="v13 Training — Genuine Spin Glass with Strong Couplings")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"Number of training samples (default: {DEFAULT_SAMPLES:,})")
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB,
                        help=f"Vocabulary size (default: {DEFAULT_VOCAB})")
    parser.add_argument("--recall-scale", type=int, default=DEFAULT_RECALL_SCALE,
                        help=f"Recall energy scale (default: {DEFAULT_RECALL_SCALE})")
    parser.add_argument("--pmi-weight", type=int, default=DEFAULT_PMI_WEIGHT,
                        help=f"PMI coupling weight (default: {DEFAULT_PMI_WEIGHT}, 25% of recall_scale)")
    parser.add_argument("--knowledge-scale", type=int, default=DEFAULT_KNOWLEDGE_SCALE,
                        help=f"Knowledge layer scale (default: {DEFAULT_KNOWLEDGE_SCALE})")
    parser.add_argument("--topic-penalty", type=int, default=DEFAULT_TOPIC_PENALTY,
                        help=f"Topic spin penalty (default: {DEFAULT_TOPIC_PENALTY})")
    parser.add_argument("--same-word-penalty", type=int, default=DEFAULT_SAME_WORD_PENALTY,
                        help=f"Same-word repetition penalty (default: {DEFAULT_SAME_WORD_PENALTY})")
    parser.add_argument("--no-topic-spin", action="store_true",
                        help="Disable Topic Spin coherence engine")
    parser.add_argument("--no-kn-backoff", action="store_true",
                        help="Disable Kneser-Ney backoff")
    parser.add_argument("--no-interpolated", action="store_true",
                        help="Disable interpolated n-gram smoothing")
    parser.add_argument("--ngram-min-count", type=int, default=2,
                        help="N-gram minimum count (default: 2)")
    parser.add_argument("--ngram-max-seqs", type=int, default=DEFAULT_NGRAM_MAX_SEQS,
                        help=f"Max sequences for n-gram index (default: {DEFAULT_NGRAM_MAX_SEQS:,})")
    parser.add_argument("--max-seq-len", type=int, default=30,
                        help="Maximum sequence length for training (default: 30)")
    parser.add_argument("--recall-primary", action="store_true",
                        help="Force recall-primary mode (v12 behavior — couplings capped)")
    args = parser.parse_args()

    # ─── Header ─────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"v13_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("ISING SPIN GLASS LANGUAGE MODEL — v13 TRAINING", flush=True)
    print("  *** GENUINE SPIN GLASS: J ~ h (couplings compete with field) ***", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    rss = get_rss_mb()
    if rss > 0:
        print(f"Memory (RSS): {rss:,} MB", flush=True)
    print("=" * 70, flush=True)

    # ─── Load Data ──────────────────────────────────────────
    texts = load_data(args.samples)
    n_texts = len(texts)
    print(f"Using {n_texts:,} texts for training")
    rss = get_rss_mb()
    if rss > 0:
        print(f"Memory after load (RSS): {rss:,} MB")

    # ─── Config ─────────────────────────────────────────────
    topic_spin = not args.no_topic_spin
    kn_backoff = not args.no_kn_backoff
    interpolated = not args.no_interpolated
    recall_primary = args.recall_primary

    print(f"\n{'=' * 70}")
    print(f"CONFIG: v13 — Genuine Spin Glass")
    print(f"  vocab_max_size={args.vocab}")
    print(f"  kn_backoff={kn_backoff}")
    print(f"  interpolated={interpolated}")
    print(f"  topic_spin={topic_spin}")
    print(f"  ngram_min_count={args.ngram_min_count}")
    print(f"  ngram_max_sequences={args.ngram_max_seqs:,}")
    print(f"  recall_scale={args.recall_scale}")
    print(f"  pmi_weight={args.pmi_weight}  ({100*args.pmi_weight/args.recall_scale:.0f}% of recall_scale)")
    print(f"  knowledge_scale={args.knowledge_scale}  ({100*args.knowledge_scale/args.recall_scale:.0f}% of recall_scale)")
    print(f"  topic_penalty={args.topic_penalty}  ({100*args.topic_penalty/args.recall_scale:.0f}% of recall_scale)")
    print(f"  same_word_penalty={args.same_word_penalty}")
    print(f"  recall_primary_mode={recall_primary}")
    print(f"  max_seq_len={args.max_seq_len}")
    print(f"  n_samples={n_texts:,}")
    print(f"{'=' * 70}")

    if not recall_primary:
        print("\n  *** v13: Couplings are STRONG — J ~ h ***")
        print("  PMI captures long-range co-occurrence beyond n-gram window")
        print("  Topic spin enforces coherence across 30-word spans")
        print("  Knowledge triples add semantic constraints")
        print("  If this works, PPL should drop below n-gram ceiling (~50)")

    # ─── Train ──────────────────────────────────────────────
    from ising_spin.model import IsingLMModel

    model = IsingLMModel(
        # Vocabulary
        vocab_min_freq=25,
        vocab_max_size=args.vocab,
        # N-gram
        ngram_max_n=5,
        ngram_min_count=args.ngram_min_count,
        ngram_max_sequences=args.ngram_max_seqs,
        # PMI
        pmi_window=5,
        pmi_min_count=2,
        pmi_cap=10,
        # Energy scales — v13: STRONG COUPLINGS
        recall_scale=args.recall_scale,
        pmi_weight=args.pmi_weight,       # 200 (12.5% of recall, was 5 = 0.3%)
        field_weight=1,
        same_word_penalty=args.same_word_penalty,
        knowledge_scale=args.knowledge_scale,  # 0 (start without, add later)
        spin3_scale=args.knowledge_scale,      # same as knowledge
        category_scale=0,
        logic_rule_scale=0,
        # Beta — use auto-calibration
        beta_type=0.001,
        beta_word=0.001,
        # Copy mechanism
        copy_enabled=True,
        copy_min_context=2,
        copy_min_confidence=0.25,
        # Ising
        ising_enabled=True,
        skip_pmi_max_dist=5,
        # v13: recall_primary_mode=False allows strong couplings
        recall_primary_mode=recall_primary,
        # v13: Disable Walsh (slow, not useful with strong PMI)
        walsh_enabled=False,
        # Topic Spin
        topic_spin_enabled=topic_spin,
        topic_n_topics=16,
        topic_coherence_penalty=args.topic_penalty,  # 400
        topic_spin_flip_interval=20,
        topic_context_window=30,
        topic_coupling_scale=100,
        # N-gram smoothing
        interpolated=interpolated,
        kn_backoff=kn_backoff,
        # Auto-calibrate beta
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
        rss = get_rss_mb()
        print(f"\n!!! OUT OF MEMORY during training !!! (RSS: {rss:,} MB)")
        print("Suggestions:")
        print(f"  1. Reduce --samples (currently {n_texts:,})")
        print(f"  2. Reduce --ngram-max-seqs (currently {args.ngram_max_seqs:,})")
        print(f"  3. Reduce --vocab (currently {args.vocab})")
        sys.exit(1)
    except Exception as e:
        rss = get_rss_mb()
        print(f"\n!!! TRAINING ERROR: {e} !!! (RSS: {rss:,} MB)")
        traceback.print_exc()
        sys.exit(1)

    t_train = time.time() - t_start
    rss = get_rss_mb()
    print(f"\nTraining complete: {t_train:.1f}s ({t_train/60:.1f}min)")
    print(f"  Throughput: {n_texts / t_train:.0f} samples/sec")
    print(f"  Vocab size: {len(model.vocab)}")
    if rss > 0:
        print(f"  Peak memory (RSS): {rss:,} MB")

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

            stats = model.generator.get_stats()
            recalls = stats.get("recall_hit", 0)
            copies = stats.get("copy_used", 0)
            print(f"  recalls={recalls} copies={copies}")

            generated_texts.append(text)

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
        "version": "v13",
        "timestamp": timestamp,
        "config": {
            "vocab_max_size": args.vocab,
            "ngram_min_count": args.ngram_min_count,
            "ngram_max_sequences": args.ngram_max_seqs,
            "recall_scale": args.recall_scale,
            "pmi_weight": args.pmi_weight,
            "knowledge_scale": args.knowledge_scale,
            "topic_penalty": args.topic_penalty,
            "same_word_penalty": args.same_word_penalty,
            "recall_primary_mode": recall_primary,
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
            "peak_rss_mb": get_rss_mb(),
        },
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_file}")

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
