#!/usr/bin/env python3
"""
Training Script — Multi-Scale Abstract Recall + Evolving Document State

ARCHITECTURE:
  Word n-gram (5) + POS n-gram (15) + Topic n-gram (10) + Document State (7 vars)

KEY INSIGHT:
  When the 5-word n-gram is unseen, the POS 10-gram IS seen.
  When POS is ambiguous, topic disambiguates.
  When all n-grams miss, document state carries discourse coherence.

Usage:
  python -u train.py                          # Default: 500K samples
  python -u train.py --samples 1000000         # 1M samples
  python -u train.py --no-pos-recall            # Ablation: without POS n-gram
  python -u train.py --no-topic-recall          # Ablation: without topic n-gram
  python -u train.py --no-state                 # Ablation: without document state
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
DEFAULT_VOCAB = 2000  # TinyStories vocabulary ~2K; was 4000 for FineWeb-Edu
DEFAULT_DATASET = "tinystories"
DEFAULT_RECALL_SCALE = 1600
DEFAULT_POS_RECALL_SCALE = 800
DEFAULT_TOPIC_RECALL_SCALE = 400
DEFAULT_STATE_SCALE = 200
DEFAULT_POS_NGRAM_MAX_N = 15
DEFAULT_TOPIC_NGRAM_MAX_N = 10
DEFAULT_MEMORY_BUDGET = 0  # 0 = unlimited; set to MB for constrained devices (e.g. 14000 for 16GB Pi)

# Cache directory
CACHE_DIR = Path(__file__).parent
OUTPUT_DIR = CACHE_DIR / "output"


def get_rss_mb() -> int:
    """Get current process RSS in MB."""
    from ising_spin.utils import get_rss_mb as _get_rss_mb
    return _get_rss_mb()


def find_cache_file(n_samples: int, dataset_name: str = "fineweb") -> str:
    """Find the best cache file for the requested number of samples."""
    cache_files = {}
    glob_pattern = f"cached_{dataset_name}_*.json"
    for f in CACHE_DIR.glob(glob_pattern):
        name = f.stem
        # Extract size from filename: cached_tinystories_500k.json → 500k
        parts = name.split("_")
        size_str = parts[-1]
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


def load_data(n_samples: int, dataset_name: str = DEFAULT_DATASET) -> list:
    """Load or download data with proper cache handling.

    Supported datasets: tinystories, tiny-textbooks, writingprompts, fineweb-edu
    """
    from ising_spin.utils import DATASET_LOADERS, DEFAULT_DATASET as _DEF

    # Resolve dataset name
    dataset_name = dataset_name or _DEF
    if dataset_name not in DATASET_LOADERS:
        print(f"  Unknown dataset '{dataset_name}', falling back to {_DEF}")
        dataset_name = _DEF

    # Sanitize dataset name for cache filenames
    cache_dataset = dataset_name.replace("-", "_")
    cache_path = find_cache_file(n_samples, cache_dataset)

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

    print(f"No cached data found. Downloading {dataset_name} from HuggingFace...")
    loader = DATASET_LOADERS[dataset_name]
    t0 = time.time()
    texts = loader(n_samples=n_samples)
    print(f"  Downloaded {len(texts):,} texts in {time.time()-t0:.1f}s")

    if n_samples >= 1000000 and n_samples % 1000000 == 0:
        cache_name = f"cached_{cache_dataset}_{n_samples // 1000000}m.json"
    elif n_samples >= 1000 and n_samples % 1000 == 0:
        cache_name = f"cached_{cache_dataset}_{n_samples // 1000}k.json"
    else:
        cache_name = f"cached_{cache_dataset}_{n_samples}.json"

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
        n_word_entries = getattr(model.word_index, 'n_entries', 0)
        n_word_sequences = getattr(model.word_index, 'n_sequences', 0)
        print(f"  WORD n-gram index:")
        print(f"    entries={n_word_entries:,}, sequences={n_word_sequences:,}")
        if word_stats:
            for k, v in word_stats.items():
                print(f"    {k}={v}")

    # POS recall diagnostics
    if model.pos_index is not None:
        pos_stats = model.pos_index.get_stats() if hasattr(model.pos_index, 'get_stats') else {}
        n_pos_entries = getattr(model.pos_index, 'n_entries', 0)
        n_pos_sequences = getattr(model.pos_index, 'n_sequences', 0)
        print(f"  POS n-gram index:")
        print(f"    entries={n_pos_entries:,}, sequences={n_pos_sequences:,}")
        if pos_stats:
            for k, v in pos_stats.items():
                print(f"    {k}={v}")

    # Topic recall diagnostics
    if model.topic_index is not None:
        topic_stats = model.topic_index.get_stats() if hasattr(model.topic_index, 'get_stats') else {}
        n_topic_entries = getattr(model.topic_index, 'n_entries', 0)
        n_topic_sequences = getattr(model.topic_index, 'n_sequences', 0)
        print(f"  TOPIC n-gram index:")
        print(f"    entries={n_topic_entries:,}, sequences={n_topic_sequences:,}")
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
        print(f"    word_recall_hits={word_hits}")
        print(f"    pos_recall_hits={pos_hits}")
        print(f"    topic_recall_hits={topic_hits}")
        print(f"    state_energy_sum={state_energy}")


def auto_tune_for_memory(
    budget_mb: int,
    n_samples: int,
    word_ngram_max_n: int,
    pos_ngram_max_n: int,
    topic_ngram_max_n: int,
    ngram_max_seqs: int,
    reservoir_dim: int,
    vsa_dim: int,
) -> dict:
    """Auto-tune parameters for a memory budget (MB).

    Based on empirical memory profiling on 500K TinyStories:
      - Tokenized data + vocab: ~2.6 GB for 500K
      - Word 5-gram index: +6.5 GB (2M contexts, 3.2M continuations)
      - POS 15-gram: +3-4 GB projected
      - Topic 10-gram: +1-2 GB projected
      - v18 modules: +0.5-1 GB

    Strategy: reduce n-gram orders and cap sequences to fit budget.
    Target RSS = budget_mb * 0.85 (leave 15% headroom for OS).

    Returns dict of adjusted parameters.
    """
    target_mb = int(budget_mb * 0.85)
    adjustments = {}

    # Reserve for tokenization, vocab, POS system, topic, etc.
    base_mb = 2500  # ~2.5 GB for data structures
    if n_samples > 200000:
        base_mb += (n_samples - 200000) // 200000 * 500  # ~500 MB per 200K extra
    v18_mb = 0

    # Estimate v18 module memory
    if reservoir_dim > 0:
        v18_mb += reservoir_dim * 2 // 1000  # rough estimate
    if vsa_dim > 0:
        v18_mb += vsa_dim * 2 // 1000

    available_mb = target_mb - base_mb - v18_mb
    if available_mb < 1000:
        print(f"  MEMORY WARNING: Only {available_mb} MB available for indexes!")
        available_mb = max(1000, available_mb)

    # Allocate index budget: word gets 50%, POS 30%, topic 20%
    word_budget = int(available_mb * 0.50)
    pos_budget = int(available_mb * 0.30)
    topic_budget = int(available_mb * 0.20)

    # Word n-gram: each additional order roughly doubles contexts
    # 5-gram on 500K = ~6.5 GB, 4-gram = ~4 GB, 3-gram = ~2 GB
    if word_budget < 3000:
        adjustments['ngram_max_n'] = 3
    elif word_budget < 5000:
        adjustments['ngram_max_n'] = 4
    else:
        adjustments['ngram_max_n'] = word_ngram_max_n  # keep original

    # POS n-gram: 13 POS types, so contexts are smaller but max_n=15 is huge
    # POS 7-gram: ~1.5 GB, POS 10-gram: ~2.5 GB, POS 15-gram: ~4 GB
    if pos_budget < 1500:
        adjustments['pos_ngram_max_n'] = 5
    elif pos_budget < 2500:
        adjustments['pos_ngram_max_n'] = 7
    elif pos_budget < 3500:
        adjustments['pos_ngram_max_n'] = 10
    else:
        adjustments['pos_ngram_max_n'] = pos_ngram_max_n

    # Topic n-gram: 16 topics, very compact contexts
    # Topic 5-gram: ~0.5 GB, Topic 8-gram: ~1 GB, Topic 10-gram: ~1.5 GB
    if topic_budget < 800:
        adjustments['topic_ngram_max_n'] = 4
    elif topic_budget < 1200:
        adjustments['topic_ngram_max_n'] = 6
    else:
        adjustments['topic_ngram_max_n'] = topic_ngram_max_n

    # Cap n-gram sequences: fewer sequences = smaller indexes
    # For 16GB Pi, 200K sequences is safe; 100K for 8GB
    if budget_mb <= 8000:
        adjustments['ngram_max_seqs'] = min(ngram_max_seqs, 100000)
    elif budget_mb <= 16000:
        adjustments['ngram_max_seqs'] = min(ngram_max_seqs, 250000)
    elif budget_mb <= 32000:
        adjustments['ngram_max_seqs'] = min(ngram_max_seqs, 400000)
    # else: keep original

    # Reduce reservoir/VSA dims for very constrained devices
    if budget_mb <= 8000:
        adjustments['reservoir_dim'] = min(reservoir_dim, 128)
        adjustments['vsa_dim'] = min(vsa_dim, 256)
    elif budget_mb <= 16000:
        adjustments['reservoir_dim'] = min(reservoir_dim, 256)
        adjustments['vsa_dim'] = min(vsa_dim, 512)

    return adjustments


def main():
    parser = argparse.ArgumentParser(
        description="v18.0 Training — Multi-Scale Abstract Recall + v18 Extensions"
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

    # v18 module flags
    parser.add_argument("--enable-reservoir", action="store_true",
                        help="Enable Integer ESN Reservoir")
    parser.add_argument("--enable-coupling", action="store_true",
                        help="Enable Factorial State Coupling")
    parser.add_argument("--enable-vsa", action="store_true",
                        help="Enable VSA/qFHRR compositional binding")
    parser.add_argument("--enable-all-v18", action="store_true",
                        help="Enable all v18 modules (reservoir + coupling + VSA)")
    parser.add_argument("--reservoir-dim", type=int, default=512,
                        help="ESN reservoir dimension (default: 512)")
    parser.add_argument("--reservoir-scale", type=int, default=800,
                        help="ESN reservoir energy scale (default: 800)")
    parser.add_argument("--coupling-scale", type=int, default=200,
                        help="Factorial coupling energy scale (default: 200)")
    parser.add_argument("--vsa-scale", type=int, default=800,
                        help="VSA energy scale (default: 800)")
    parser.add_argument("--vsa-dim", type=int, default=512,
                        help="VSA phase vector dimension (default: 512)")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=["tinystories", "tiny-textbooks", "writingprompts", "fineweb-edu"],
                        help=f"Dataset to train on (default: {DEFAULT_DATASET})")
    parser.add_argument("--curriculum", action="store_true",
                        help="Curriculum: TinyStories → tiny-textbooks → WritingPrompts")
    parser.add_argument("--vocab-min-freq", type=int, default=5,
                        help="Min word frequency for vocab (default: 5, was 25 for FineWeb)")

    # Memory budget
    parser.add_argument("--memory-budget", type=int, default=DEFAULT_MEMORY_BUDGET,
                        help="Memory budget in MB (0=unlimited; 14000=16GB Pi, 7000=8GB Pi). "
                             "Auto-adjusts n-gram orders, sequence caps, and v18 dims.")

    args = parser.parse_args()

    # --- Apply ablation flags ---
    pos_recall_scale = 0 if args.no_pos_recall else args.pos_recall_scale
    topic_recall_scale = 0 if args.no_topic_recall else args.topic_recall_scale
    state_scale = 0 if args.no_state else args.state_scale

    kn_backoff = not args.no_kn_backoff
    interpolated = not args.no_interpolated

    # --- v18 flags ---
    enable_reservoir = args.enable_reservoir or args.enable_all_v18
    enable_coupling = args.enable_coupling or args.enable_all_v18
    enable_vsa = args.enable_vsa or args.enable_all_v18

    # --- Memory budget auto-tuning ---
    mem_adjustments = {}
    if args.memory_budget > 0:
        mem_adjustments = auto_tune_for_memory(
            budget_mb=args.memory_budget,
            n_samples=args.samples,
            word_ngram_max_n=5,
            pos_ngram_max_n=args.pos_ngram_max_n,
            topic_ngram_max_n=args.topic_ngram_max_n,
            ngram_max_seqs=args.ngram_max_seqs,
            reservoir_dim=args.reservoir_dim,
            vsa_dim=args.vsa_dim,
        )
        print(f"\n{'=' * 70}")
        print(f"MEMORY BUDGET: {args.memory_budget:,} MB (target RSS: {int(args.memory_budget * 0.85):,} MB)")
        print(f"{'=' * 70}")
        for key, val in mem_adjustments.items():
            orig = {
                'ngram_max_n': 5, 'pos_ngram_max_n': args.pos_ngram_max_n,
                'topic_ngram_max_n': args.topic_ngram_max_n,
                'ngram_max_seqs': args.ngram_max_seqs,
                'reservoir_dim': args.reservoir_dim, 'vsa_dim': args.vsa_dim,
            }.get(key, '?')
            print(f"  {key}: {orig} → {val}")
        print(f"{'=' * 70}")

    # Apply memory adjustments
    effective_ngram_max_n = mem_adjustments.get('ngram_max_n', 5)
    effective_pos_ngram_max_n = mem_adjustments.get('pos_ngram_max_n', args.pos_ngram_max_n)
    effective_topic_ngram_max_n = mem_adjustments.get('topic_ngram_max_n', args.topic_ngram_max_n)
    effective_ngram_max_seqs = mem_adjustments.get('ngram_max_seqs', args.ngram_max_seqs)
    effective_reservoir_dim = mem_adjustments.get('reservoir_dim', args.reservoir_dim)
    effective_vsa_dim = mem_adjustments.get('vsa_dim', args.vsa_dim)

    # --- Header ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("ISING SPIN GLASS LANGUAGE MODEL — MULTI-SCALE ABSTRACT RECALL", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Workers: {os.cpu_count()}", flush=True)
    rss = get_rss_mb()
    if rss > 0:
        print(f"Memory (RSS): {rss:,} MB", flush=True)
    print("=" * 70, flush=True)

    # --- Load Data ---
    if args.curriculum:
        # Curriculum learning: TinyStories → tiny-textbooks → WritingPrompts
        print("\n" + "=" * 70, flush=True)
        print("CURRICULUM LEARNING: Phase 1 — TinyStories (grammar)", flush=True)
        print("=" * 70, flush=True)
        texts = load_data(args.samples, dataset_name="tinystories")
    else:
        texts = load_data(args.samples, dataset_name=args.dataset)
    n_texts = len(texts)
    print(f"Using {n_texts:,} texts for training")

    # --- Config ---
    print(f"\n{'=' * 70}")
    print(f"CONFIG: Multi-Scale Abstract Recall + Document State")
    print(f"  WORD RECALL:")
    print(f"    ngram_max_n={effective_ngram_max_n}, recall_scale={args.recall_scale}")
    print(f"  POS RECALL:")
    print(f"    pos_ngram_max_n={effective_pos_ngram_max_n}, pos_recall_scale={pos_recall_scale}"
          f"{' [DISABLED]' if args.no_pos_recall else ''}")
    print(f"  TOPIC RECALL:")
    print(f"    topic_ngram_max_n={effective_topic_ngram_max_n}, topic_recall_scale={topic_recall_scale}"
          f"{' [DISABLED]' if args.no_topic_recall else ''}")
    print(f"  DOCUMENT STATE:")
    print(f"    state_scale={state_scale}"
          f"{' [DISABLED]' if args.no_state else ''}")
    print(f"  STANDARD:")
    print(f"    dataset={args.dataset}")
    print(f"    curriculum={args.curriculum}")
    print(f"    vocab_max_size={args.vocab}")
    print(f"    vocab_min_freq={args.vocab_min_freq}")
    print(f"    kn_backoff={kn_backoff}")
    print(f"    interpolated={interpolated}")
    print(f"    same_word_penalty={args.same_word_penalty}")
    print(f"    n_topics={args.n_topics}")
    print(f"    n_samples={n_texts:,}")
    print(f"    ngram_max_seqs={effective_ngram_max_seqs:,}")
    if args.memory_budget > 0:
        print(f"    memory_budget={args.memory_budget:,} MB")
    print(f"  v18 EXTENSIONS:")
    print(f"    reservoir={'ENABLED' if enable_reservoir else 'DISABLED'}"
          f" (dim={effective_reservoir_dim}, scale={args.reservoir_scale})")
    print(f"    coupling={'ENABLED' if enable_coupling else 'DISABLED'}"
          f" (scale={args.coupling_scale})")
    print(f"    vsa={'ENABLED' if enable_vsa else 'DISABLED'}"
          f" (dim={effective_vsa_dim}, scale={args.vsa_scale})")
    print(f"{'=' * 70}")

    # --- Train ---
    from ising_spin.orchestrator import IsingLMModel

    model = IsingLMModel(
        # Vocabulary
        vocab_min_freq=args.vocab_min_freq,
        vocab_max_size=args.vocab,
        # Word N-gram
        ngram_max_n=effective_ngram_max_n,
        ngram_min_count=args.ngram_min_count,
        ngram_max_sequences=effective_ngram_max_seqs,
        # POS N-gram
        pos_ngram_max_n=effective_pos_ngram_max_n,
        pos_ngram_min_count=2,
        # Topic
        n_topics=args.n_topics,
        topic_ngram_max_n=effective_topic_ngram_max_n,
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
        # v18 modules
        enable_reservoir=enable_reservoir,
        enable_coupling=enable_coupling,
        enable_vsa=enable_vsa,
        reservoir_dim=effective_reservoir_dim,
        reservoir_scale=args.reservoir_scale,
        coupling_scale=args.coupling_scale,
        vsa_scale=args.vsa_scale,
        vsa_dim=effective_vsa_dim,
        # Memory budget
        memory_budget_mb=args.memory_budget,
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

    # --- Multi-Scale Recall Diagnostics ---
    print(f"\n{'=' * 70}")
    print("RECALL DIAGNOSTICS")
    print(f"{'=' * 70}")
    print_recall_diagnostics(model)

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
            print(f"  word_hits={word_hits} pos_hits={pos_hits} "
                  f"topic_hits={topic_hits} copies={copies} state_energy={state_e}")

            generated_texts.append(text)

            gen_file = output_dir / f"generated_{i}.txt"
            with open(gen_file, "w") as f:
                f.write(text)
        except Exception as e:
            print(f"  Generation error: {e}")
            traceback.print_exc()
            generated_texts.append("")

    # --- Save Results ---
    results = {
        "version": "18.0.0",
        "architecture": "Multi-Scale Abstract Recall + Document State + v18 Extensions",
        "dataset": args.dataset,
        "curriculum": args.curriculum,
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
    print(f"DONE — Multi-Scale Abstract Recall + Document State")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"PPL: {full_ppl:.2f}")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
