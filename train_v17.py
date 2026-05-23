#!/usr/bin/env python3
"""
v17.4 Training Script — Tokenizer & Vocabulary Fixes + Sentence Boundaries

v17.4 CHANGES (from v17.3 — PPL=21.11 but incoherent generation):
  - CRITICAL FIX: TopicAssigner used text.split() instead of vocab._tokenize().
    Capitalized words ("The") never matched vocab ("the"), corrupting topic clustering.
  - CRITICAL FIX: Generator prompt used text.split() instead of vocab.encode().
    Contractions and punctuation in prompts didn't resolve correctly.
  - CRITICAL FIX: Each word was assigned to only ONE POS type bucket (primary).
    Words like "run" (NOUN+VERB) could only be generated in one context.
    Now words appear in ALL their allowed type buckets.
  - FIX: Exponential context_weight_factor scaling capped at 16 (was 2^(k-1)=512
    for 10-gram matches, completely dominating all other energy signals).
  - NEW: Sentence boundary marker <S> inserted after '.', '!', '?'.
    Prevents cross-sentence n-gram contamination.
  - INCREASED: Vocab from 4000 to 8000, min_freq from 25 to 15.

ARCHITECTURE:
  Word n-gram (5) + POS n-gram (10) + Topic n-gram (10) + Document State (7 vars)

Usage:
  python -u train_v17.py                          # Default: 500K samples
  python -u train_v17.py --samples 1000000         # 1M samples
  python -u train_v17.py --vocab 8000               # Custom vocab size
  python -u train_v17.py --no-pos-recall            # Ablation: without POS n-gram
  python -u train_v17.py --no-topic-recall          # Ablation: without topic n-gram
  python -u train_v17.py --no-state                 # Ablation: without document state
"""

# --- UNBUFFERED OUTPUT ---
import os
import sys
os.environ["PYTHONUNBUFFERED"] = "1"

# Ensure src/ is on the Python path
from pathlib import Path
_src = str(Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import argparse
import json
import time
import traceback
from pathlib import Path

# --- Configuration ---

DEFAULT_SAMPLES = 500000
DEFAULT_VOCAB = 8000
DEFAULT_MIN_FREQ = 15  # v17.4: Lowered from 25 to improve vocab coverage with 8000 words
DEFAULT_RECALL_SCALE = 1600
DEFAULT_POS_RECALL_SCALE = 800   # v17.2: increased from 400
DEFAULT_TOPIC_RECALL_SCALE = 400  # v17.2: increased from 200
DEFAULT_STATE_SCALE = 50          # v17.2: reduced from 200
DEFAULT_POS_NGRAM_MAX_N = 10
DEFAULT_TOPIC_NGRAM_MAX_N = 10

# Cache directory
CACHE_DIR = Path(__file__).parent
OUTPUT_DIR = CACHE_DIR / "output"


def get_rss_mb() -> int:
    """Get current process RSS in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
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
    from ising_spin.model_v17 import _load_fineweb_edu
    t0 = time.time()
    texts = _load_fineweb_edu(n_samples=n_samples)
    print(f"  Downloaded {len(texts):,} texts in {time.time()-t0:.1f}s")

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
    """Two-phase beta sweep to find optimal PPL."""
    import numpy as np
    from ising_spin.sampling import IntegerBoltzmannSampler, LN2_NUM, LN2_DEN

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
            beta=new_beta, max_delta=50000
        )
        try:
            ppl = model.compute_perplexity(n_samples=n_seqs)
            marker = " <-- BEST" if ppl < best_ppl else ""
            print(f"    f={f:.2f} (beta={new_beta:.6f}): PPL={ppl:.1f}{marker}")
            if ppl < best_ppl:
                best_ppl = ppl
                best_f = f
        except Exception as e:
            print(f"    f={f:.2f}: Error: {e}")

    # Phase 2: Fine sweep
    fine_lo = max(0.5, best_f - 0.5)
    fine_hi = best_f + 0.5
    fine_factors = np.arange(fine_lo, fine_hi + 0.1, 0.1).tolist()
    fine_factors = [f for f in fine_factors if abs(f - best_f) > 0.05 or f == best_f]

    if len(fine_factors) > 1:
        print(f"\nBeta sweep (Phase 2 — fine around f={best_f:.2f}):")
        for f in fine_factors:
            new_beta = base_beta * f
            model.generator.word_sampler = IntegerBoltzmannSampler(
                beta=new_beta, max_delta=50000
            )
            try:
                ppl = model.compute_perplexity(n_samples=n_seqs)
                marker = " <-- BEST" if ppl < best_ppl else ""
                print(f"    f={f:.2f} (beta={new_beta:.6f}): PPL={ppl:.1f}{marker}")
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_f = f
            except Exception as e:
                print(f"    f={f:.2f}: Error: {e}")

    model.generator.word_sampler = IntegerBoltzmannSampler(
        beta=base_beta * best_f, max_delta=50000
    )
    print(f"\nBest: f={best_f:.2f} (beta={base_beta * best_f:.6f}), PPL={best_ppl:.1f}")

    return best_f, best_ppl


def print_recall_diagnostics(model):
    """Print per-scale recall diagnostics from the multi-scale recall layer."""
    print(f"\nMulti-Scale Recall Diagnostics:")

    # Word recall diagnostics
    if model.word_index is not None:
        word_stats = model.word_index.get_stats() if hasattr(model.word_index, 'get_stats') else {}
        n_word_entries = sum(
            len(model.word_index.index[k])
            for k in range(1, model.word_index.max_n + 1)
        )
        n_word_sequences = sum(
            sum(len(v) for v in model.word_index.index[k].values())
            for k in range(1, model.word_index.max_n + 1)
        )
        print(f"  WORD n-gram index:")
        print(f"    contexts={n_word_entries:,}, continuations={n_word_sequences:,}")
        if word_stats:
            for k, v in word_stats.items():
                print(f"    {k}={v}")

    # POS recall diagnostics
    if model.pos_index is not None:
        pos_stats = model.pos_index.get_stats() if hasattr(model.pos_index, 'get_stats') else {}
        n_pos_entries = sum(
            len(model.pos_index.index[k])
            for k in range(1, model.pos_index.max_n + 1)
        )
        n_pos_sequences = sum(
            sum(len(v) for v in model.pos_index.index[k].values())
            for k in range(1, model.pos_index.max_n + 1)
        )
        print(f"  POS n-gram index:")
        print(f"    contexts={n_pos_entries:,}, continuations={n_pos_sequences:,}")
        if pos_stats:
            for k, v in pos_stats.items():
                print(f"    {k}={v}")

    # Topic recall diagnostics
    if model.topic_index is not None:
        topic_stats = model.topic_index.get_stats() if hasattr(model.topic_index, 'get_stats') else {}
        n_topic_entries = sum(
            len(model.topic_index.index[k])
            for k in range(1, model.topic_index.max_n + 1)
        )
        n_topic_sequences = sum(
            sum(len(v) for v in model.topic_index.index[k].values())
            for k in range(1, model.topic_index.max_n + 1)
        )
        print(f"  TOPIC n-gram index:")
        print(f"    contexts={n_topic_entries:,}, continuations={n_topic_sequences:,}")
        if topic_stats:
            for k, v in topic_stats.items():
                print(f"    {k}={v}")

    # Multi-scale recall summary
    if model.multiscale_recall is not None:
        summary = model.multiscale_recall.summary()
        print(f"  Multi-scale recall summary: {summary}")

    # Document state diagnostics
    if model.document_state is not None:
        ds = model.document_state
        n_state_vars = getattr(ds, 'n_state_vars', 7)
        print(f"  DOCUMENT STATE:")
        print(f"    n_state_vars={n_state_vars}, state_scale={model.state_scale}")
        if hasattr(ds, 'get_diagnostics'):
            ds_diag = ds.get_diagnostics()
            for k, v in ds_diag.items():
                print(f"    {k}={v}")

    # Generator stats (if generator exists)
    if model.generator is not None and hasattr(model.generator, 'get_stats'):
        gen_stats = model.generator.get_stats()
        print(f"  GENERATOR STATS:")
        word_hits = gen_stats.get("recall_hit", 0)
        pos_hits = gen_stats.get("pos_recall_hit", 0)
        topic_hits = gen_stats.get("topic_recall_hit", 0)
        state_energy = gen_stats.get("state_energy_sum", 0)
        total_pos = gen_stats.get("total_positions", 0)
        print(f"    word_recall_hits={word_hits}")
        print(f"    pos_recall_hits={pos_hits}")
        print(f"    topic_recall_hits={topic_hits}")
        print(f"    state_energy_sum={state_energy}")
        if total_pos > 0:
            avg_state_e = state_energy / total_pos
            print(f"    avg_state_energy_per_pos={avg_state_e:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="v17.0 Training — Multi-Scale Abstract Recall + Document State"
    )

    # Core parameters
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--recall-scale", type=int, default=DEFAULT_RECALL_SCALE,
                        help="Word n-gram recall energy scale (default: 1600)")
    parser.add_argument("--pos-recall-scale", type=int, default=DEFAULT_POS_RECALL_SCALE,
                        help="POS n-gram recall energy scale (default: 800)")
    parser.add_argument("--topic-recall-scale", type=int, default=DEFAULT_TOPIC_RECALL_SCALE,
                        help="Topic n-gram recall energy scale (default: 400)")
    parser.add_argument("--state-scale", type=int, default=DEFAULT_STATE_SCALE,
                        help="Document state energy scale (default: 200)")

    # POS and Topic n-gram sizes
    parser.add_argument("--pos-ngram-max-n", type=int, default=DEFAULT_POS_NGRAM_MAX_N,
                        help="Maximum POS n-gram order (default: 15)")
    parser.add_argument("--topic-ngram-max-n", type=int, default=DEFAULT_TOPIC_NGRAM_MAX_N,
                        help="Maximum topic n-gram order (default: 10)")

    # Ablation flags
    parser.add_argument("--no-pos-recall", action="store_true",
                        help="Ablation: disable POS n-gram recall (set scale=0)")
    parser.add_argument("--no-topic-recall", action="store_true",
                        help="Ablation: disable topic n-gram recall (set scale=0)")
    parser.add_argument("--no-state", action="store_true",
                        help="Ablation: disable document state (set scale=0)")

    # Standard n-gram parameters
    parser.add_argument("--no-kn-backoff", action="store_true")
    parser.add_argument("--no-interpolated", action="store_true")
    parser.add_argument("--ngram-min-count", type=int, default=2)
    parser.add_argument("--ngram-max-seqs", type=int, default=1000000)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--same-word-penalty", type=int, default=200)
    parser.add_argument("--n-topics", type=int, default=16)

    args = parser.parse_args()

    # --- Apply ablation flags ---
    pos_recall_scale = 0 if args.no_pos_recall else args.pos_recall_scale
    topic_recall_scale = 0 if args.no_topic_recall else args.topic_recall_scale
    state_scale = 0 if args.no_state else args.state_scale

    kn_backoff = not args.no_kn_backoff
    interpolated = not args.no_interpolated

    # --- Header ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"v17_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("ISING SPIN GLASS LANGUAGE MODEL — v17.4 TOKENIZER + VOCAB FIXES", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Workers: {os.cpu_count()}", flush=True)
    rss = get_rss_mb()
    if rss > 0:
        print(f"Memory (RSS): {rss:,} MB", flush=True)
    print("=" * 70, flush=True)

    # --- Load Data ---
    texts = load_data(args.samples)
    n_texts = len(texts)
    print(f"Using {n_texts:,} texts for training")

    # --- Config ---
    print(f"\n{'=' * 70}")
    print(f"CONFIG: v17.4 — Tokenizer & Vocabulary Fixes + Sentence Boundaries")
    print(f"  WORD RECALL:")
    print(f"    ngram_max_n=5, recall_scale={args.recall_scale}")
    print(f"  POS RECALL:")
    print(f"    pos_ngram_max_n={args.pos_ngram_max_n}, pos_recall_scale={pos_recall_scale}"
          f"{' [DISABLED]' if args.no_pos_recall else ''}")
    print(f"  TOPIC RECALL:")
    print(f"    topic_ngram_max_n={args.topic_ngram_max_n}, topic_recall_scale={topic_recall_scale}"
          f"{' [DISABLED]' if args.no_topic_recall else ''}")
    print(f"  DOCUMENT STATE:")
    print(f"    state_scale={state_scale}"
          f"{' [DISABLED]' if args.no_state else ''}")
    print(f"  STANDARD:")
    print(f"    vocab_max_size={args.vocab}")
    print(f"    kn_backoff={kn_backoff} {'[v17.3 FIX: NOW FORWARDED TO ENERGY]' if kn_backoff else ''}")
    print(f"    interpolated={interpolated} {'[v17.3 FIX: NOW FORWARDED TO ENERGY]' if interpolated else ''}")
    print(f"    same_word_penalty={args.same_word_penalty}")
    print(f"    n_topics={args.n_topics}")
    print(f"    n_samples={n_texts:,}")
    print(f"  v17.4 FIXES:")
    print(f"    tokenizer: TopicAssigner now uses vocab._tokenize() (was text.split())")
    print(f"    tokenizer: Generator prompt now uses vocab.encode() (was text.split())")
    print(f"    candidates: Multi-type words in ALL allowed POS buckets (was primary-only)")
    print(f"    context_weight: Capped at 16 (was exponential 2^(k-1))")
    print(f"    sentence_boundaries: <S> token prevents cross-sentence n-grams")
    print(f"    vocab: 8000 words, min_freq=15 (was 4000, 25)")
    print(f"    no_freq_filter=True (all type words as candidates)")
    print(f"    no_recent5_penalty_in_PPL=True")
    print(f"{'=' * 70}")

    # --- Train ---
    from ising_spin.model_v17 import IsingLMModel

    model = IsingLMModel(
        # Vocabulary
        vocab_min_freq=15,
        vocab_max_size=args.vocab,
        # Word N-gram
        ngram_max_n=5,
        ngram_min_count=args.ngram_min_count,
        ngram_max_sequences=args.ngram_max_seqs,
        # POS N-gram
        pos_ngram_max_n=args.pos_ngram_max_n,
        pos_ngram_min_count=2,
        # Topic
        n_topics=args.n_topics,
        topic_ngram_max_n=args.topic_ngram_max_n,
        topic_ngram_min_count=3,
        # Energy scales
        recall_scale=args.recall_scale,
        pos_recall_scale=pos_recall_scale,
        topic_recall_scale=topic_recall_scale,
        state_scale=state_scale,
        # Hard constraints
        same_word_penalty=args.same_word_penalty,
        max_closed_class_run=2,
        # Beta
        auto_calibrate_beta=True,
        # Interpolation
        interpolated=interpolated,
        kn_backoff=kn_backoff,
        # Copy mechanism
        copy_enabled=True,
        copy_min_context=3,
        copy_min_confidence=0.4,
        # Misc
        max_seq_len=args.max_seq_len,
    )

    t_start = time.time()

    try:
        model.train(n_samples=n_texts, texts=texts)
    except MemoryError:
        rss = get_rss_mb()
        print(f"\n!!! OUT OF MEMORY during training !!! (RSS: {rss:,} MB)")
        sys.exit(1)
    except Exception as e:
        rss = get_rss_mb()
        print(f"\n!!! TRAINING ERROR: {e} !!! (RSS: {rss:,} MB)")
        traceback.print_exc()
        sys.exit(1)

    t_train = time.time() - t_start
    rss = get_rss_mb()
    print(f"\nTraining complete: {t_train:.1f}s ({t_train/60:.1f}min)")
    print(f"  Vocab size: {len(model.vocab)}")
    if rss > 0:
        print(f"  Peak memory (RSS): {rss:,} MB")

    # --- Evaluation ---
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

    # --- Generation ---
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
            print(f"  {text[:300]}...")

            stats = model.generator.get_stats()
            word_hits = stats.get("recall_hit", 0)
            pos_hits = stats.get("pos_recall_hit", 0)
            topic_hits = stats.get("topic_recall_hit", 0)
            copies = stats.get("copy_used", 0)
            state_e = stats.get("state_energy_sum", 0)
            word_max_n = stats.get("word_max_n", 0)
            pos_max_n = stats.get("pos_max_n", 0)
            topic_max_n = stats.get("topic_max_n", 0)
            total_pos_gen = stats.get("total_positions", 1)
            avg_state_e = state_e / max(1, total_pos_gen)
            print(f"  word_hits={word_hits} pos_hits={pos_hits} "
                  f"topic_hits={topic_hits} copies={copies}")
            print(f"  state_energy={state_e} (avg={avg_state_e:.0f}/pos)")
            print(f"  max_n_grams: word={word_max_n} pos={pos_max_n} topic={topic_max_n}")

            generated_texts.append(text)

            gen_file = output_dir / f"generated_{i}.txt"
            with open(gen_file, "w") as f:
                f.write(text)
        except Exception as e:
            print(f"  Generation error: {e}")
            traceback.print_exc()
            generated_texts.append("")

    # --- Multi-Scale Recall Diagnostics (AFTER generation so stats are populated) ---
    print(f"\n{'=' * 70}")
    print("RECALL DIAGNOSTICS")
    print(f"{'=' * 70}")
    print_recall_diagnostics(model)

    # --- Save Results ---
    results = {
        "version": "v17.4",
        "architecture": "Multi-Scale Abstract Recall + Evolving Document State",
        "timestamp": timestamp,
        "config": {
            "recall_scale": args.recall_scale,
            "pos_recall_scale": pos_recall_scale,
            "topic_recall_scale": topic_recall_scale,
            "state_scale": state_scale,
            "pos_ngram_max_n": args.pos_ngram_max_n,
            "topic_ngram_max_n": args.topic_ngram_max_n,
            "no_pos_recall": args.no_pos_recall,
            "no_topic_recall": args.no_topic_recall,
            "no_state": args.no_state,
            "vocab_max_size": args.vocab,
            "kn_backoff": kn_backoff,
            "interpolated": interpolated,
            "same_word_penalty": args.same_word_penalty,
            "n_topics": args.n_topics,
            "n_samples": n_texts,
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
    print(f"DONE — v17.4 Tokenizer & Vocabulary Fixes + Sentence Boundaries")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"PPL: {full_ppl:.2f}")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
