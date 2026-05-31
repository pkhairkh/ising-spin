#!/usr/bin/env python3
"""
Integer Language Model — Training Script

Pure integer language model. No neural nets. No torch dependency.
Runs on a Pi 5. Produces grammatically coherent text.

Usage:
  python -u train.py                           # Full run (50K texts)
  python -u train.py --samples 5000 --vocab 1000  # Quick test
"""

import os, sys, argparse, json, time, traceback
os.environ["PYTHONUNBUFFERED"] = "1"

from pathlib import Path
import numpy as np

# Ensure src/ is on path
_src = str(Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Clear stale __pycache__
import shutil
for root, dirs, files in os.walk(Path(__file__).parent / "src"):
    if "__pycache__" in dirs:
        shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)
        dirs.remove("__pycache__")

from ising_spin import IntegerLM, Vocabulary, IDX2POS
from ising_spin.utils import get_rss_mb

CACHE_DIR = Path(__file__).parent
OUTPUT_DIR = CACHE_DIR / "output"


def load_data(n_samples):
    """Load TinyStories dataset."""
    cache_files = {}
    for f in CACHE_DIR.glob("cached_tiny_stories_*.json"):
        parts = f.stem.split("_")
        size_str = parts[-1]
        try:
            size = int(size_str[:-1]) * (1000 if size_str.endswith("k") else 1000000)
            cache_files[size] = str(f)
        except (ValueError, IndexError):
            continue

    cache_path = cache_files.get(n_samples) or min(
        (s for s in cache_files if s >= n_samples), default=None, key=lambda s: s
    )
    if cache_path is None:
        sufficient = {s: p for s, p in cache_files.items() if s >= n_samples}
        if sufficient:
            cache_path = sufficient[min(sufficient.keys())]

    if cache_path:
        print(f"Loading cache: {cache_path}")
        with open(cache_path) as f:
            texts = json.load(f)
        if len(texts) >= n_samples:
            return texts[:n_samples]

    print("Downloading TinyStories from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=f"train[:{n_samples}]")
    texts = [s["text"] for s in ds]
    cache_name = f"cached_tiny_stories_{n_samples // 1000}k.json"
    with open(CACHE_DIR / cache_name, "w") as f:
        json.dump(texts, f)
    return texts


def main():
    parser = argparse.ArgumentParser(description="Integer Language Model")

    # Data
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--vocab", type=int, default=2000)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--vocab-min-freq", type=int, default=5)

    # POS energy tables
    parser.add_argument("--n-pos-hashes", type=int, default=2)
    parser.add_argument("--pos-table-size", type=int, default=1009)
    parser.add_argument("--pos-eta", type=int, default=3)
    parser.add_argument("--pos-clip", type=int, default=500)

    # Lexical energy tables
    parser.add_argument("--n-lex-hashes", type=int, default=3)
    parser.add_argument("--lex-table-size", type=int, default=65537)
    parser.add_argument("--lex-eta", type=int, default=1)
    parser.add_argument("--lex-clip", type=int, default=1000)

    # Skip-gram
    parser.add_argument("--use-skip", action="store_true", default=True)
    parser.add_argument("--no-skip", action="store_true", default=False)
    parser.add_argument("--n-skip-hashes", type=int, default=2)
    parser.add_argument("--skip-table-size", type=int, default=65537)
    parser.add_argument("--skip-eta", type=int, default=1)
    parser.add_argument("--skip-clip", type=int, default=800)

    # Trigram
    parser.add_argument("--use-trigram", action="store_true", default=True)
    parser.add_argument("--no-trigram", action="store_true", default=False)

    # Weights
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--lex-weight", type=float, default=1.0)
    parser.add_argument("--skip-weight", type=float, default=0.5)

    # NCE
    parser.add_argument("--nce-epochs", type=int, default=3)
    parser.add_argument("--nce-negatives", type=int, default=3)

    # Generation
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rep-penalty", type=float, default=3.0)
    parser.add_argument("--rep-window", type=int, default=5)
    parser.add_argument("--gen-length", type=int, default=100)

    args = parser.parse_args()
    use_skip = args.use_skip and not args.no_skip
    use_trigram = args.use_trigram and not args.no_trigram

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"ilm_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("INTEGER LANGUAGE MODEL — Pure Integer, No Neural Nets", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"  POS:   {args.n_pos_hashes} hashes, size={args.pos_table_size}", flush=True)
    print(f"  Lex:   {args.n_lex_hashes} hashes, size={args.lex_table_size}", flush=True)
    print(f"  Skip:  {'ON' if use_skip else 'OFF'}, {args.n_skip_hashes} hashes, size={args.skip_table_size}", flush=True)
    print(f"  Tri:   {'ON' if use_trigram else 'OFF'}", flush=True)
    print(f"  Weights: pos={args.pos_weight}, lex={args.lex_weight}, skip={args.skip_weight}", flush=True)
    print(f"  Gen:   alpha={args.alpha}, T={args.temperature}, rep={args.rep_penalty}", flush=True)
    rss = get_rss_mb()
    if rss:
        print(f"Memory (RSS): {rss} MB", flush=True)
    print("=" * 70, flush=True)

    # [1] Load data
    print("\n[1/6] Loading data...", flush=True)
    texts = load_data(args.samples)
    print(f"  {len(texts):,} texts", flush=True)

    # [2] Build vocabulary + POS
    print("\n[2/6] Building vocabulary + POS types...", flush=True)
    vocab = Vocabulary(max_size=args.vocab, min_freq=args.vocab_min_freq, max_seq_len=args.max_seq_len)
    vocab.build(texts)
    print(f"  {vocab.V} words", flush=True)
    for name, count in sorted(vocab.pos_distribution().items(), key=lambda x: -x[1]):
        print(f"    {name}: {count}", flush=True)

    # [3] Tokenize
    print("\n[3/6] Tokenizing...", flush=True)
    sequences = vocab.tokenize(texts)
    n_train = int(0.9 * len(sequences))
    train_seqs, test_seqs = sequences[:n_train], sequences[n_train:]
    print(f"  Train: {len(train_seqs):,}, Test: {len(test_seqs):,}", flush=True)

    # [4] Build model
    print("\n[4/6] Building Integer Language Model...", flush=True)
    model = IntegerLM(
        vocab=vocab,
        n_pos_hashes=args.n_pos_hashes,
        pos_table_size=args.pos_table_size,
        pos_eta=args.pos_eta,
        pos_clip=args.pos_clip,
        n_lex_hashes=args.n_lex_hashes,
        lex_table_size=args.lex_table_size,
        lex_eta=args.lex_eta,
        lex_clip=args.lex_clip,
        use_skip=use_skip,
        n_skip_hashes=args.n_skip_hashes,
        skip_table_size=args.skip_table_size,
        skip_eta=args.skip_eta,
        skip_clip=args.skip_clip,
        use_trigram=use_trigram,
        pos_weight=args.pos_weight,
        lex_weight=args.lex_weight,
        skip_weight=args.skip_weight,
        top_k=args.top_k,
        alpha=args.alpha,
        temperature=args.temperature,
        rep_penalty=args.rep_penalty,
        rep_window=args.rep_window,
        seed=42,
    )
    print(f"  Energy memory: {model.energy.memory_mb():.2f} MB", flush=True)
    print(f"  Bigram memory: {model.bigram.statistics().get('memory_mb', 0):.1f} MB", flush=True)

    # [5] Train + Calibrate
    print("\n[5/6] Training + calibrating...", flush=True)
    t0 = time.time()

    train_stats = model.train(train_seqs, n_epochs=args.nce_epochs, n_negatives=args.nce_negatives)
    cal_stats = model.calibrate(train_seqs)
    t_train = time.time() - t0

    # Show POS transition matrix
    print("\n  === POS Transition Matrix ===", flush=True)
    pos_matrix = model.pos_transition_matrix()
    header = "        " + " ".join(f"{IDX2POS.get(j, 'X'):>6s}" for j in range(13))
    print(header, flush=True)
    for i in range(13):
        row = f"  {IDX2POS.get(i, 'X'):>6s} "
        for j in range(13):
            row += f"{pos_matrix[i,j]:>6.0f}"
        print(row, flush=True)

    # [6] Evaluate
    print("\n[6/6] Evaluating...", flush=True)

    # Discriminative accuracy
    print(f"\n{'='*70}", flush=True)
    print("DISCRIMINATIVE ACCURACY", flush=True)
    disc = model.discriminative_accuracy(test_seqs, n_samples=500)
    print(f"  {disc['accuracy']:.3f} ({disc['comparisons']} comparisons)", flush=True)

    # Perplexity
    print(f"\n{'='*70}", flush=True)
    print("PERPLEXITY", flush=True)
    ppl = model.perplexity(test_seqs, n_samples=min(100, len(test_seqs)))
    print(f"  Base PPL: {ppl['base_ppl']:.2f}", flush=True)
    print(f"  LEGD PPL: {ppl['legd_ppl']:.2f}", flush=True)
    delta = ppl['base_ppl'] - ppl['legd_ppl']
    print(f"  {'IMPROVEMENT' if delta > 0 else 'REGRESSION'}: {abs(delta):.2f} PPL", flush=True)

    # Generation
    print(f"\n{'='*70}", flush=True)
    print("GENERATION", flush=True)
    for prompt in ["once upon a time", "there was a little", "the little girl"]:
        text = model.generate_text(prompt, length=args.gen_length)
        print(f"\n  '{prompt}':", flush=True)
        print(f"  {text[:300]}", flush=True)

    # Save results
    diag = model.diagnostics()
    t_total = time.time() - t0
    results = {
        "version": "1.1.0",
        "architecture": "Integer Language Model — LEGD + balanced POS negatives",
        "timestamp": timestamp,
        "config": vars(args),
        "results": {
            "training_time_s": t_train,
            "total_time_s": t_total,
            "disc_accuracy": disc['accuracy'],
            "base_ppl": ppl['base_ppl'],
            "legd_ppl": ppl['legd_ppl'],
            "vocab_size": vocab.V,
            "alpha": diag['alpha'],
            "temperature": diag['temperature'],
            "metropolis_threshold": diag['metropolis_threshold'],
            "pos_weight": diag['pos_weight'],
            "lex_weight": diag['lex_weight'],
            "skip_weight": diag['skip_weight'],
        },
    }
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save POS matrix
    pos_data = {}
    for i in range(13):
        for j in range(13):
            pos_data[f"{IDX2POS.get(i,'X')}->{IDX2POS.get(j,'X')}"] = float(pos_matrix[i,j])
    with open(output_dir / "pos_matrix.json", "w") as f:
        json.dump(pos_data, f, indent=2)

    print(f"\n{'='*70}", flush=True)
    print(f"DONE — Integer Language Model")
    print(f"  Time: {t_total:.1f}s | Disc: {disc['accuracy']:.3f} | "
          f"Base PPL: {ppl['base_ppl']:.2f} | LEGD PPL: {ppl['legd_ppl']:.2f}")
    print(f"  Alpha: {diag['alpha']:.1f} | T: {diag['temperature']} | "
          f"POS: {diag['pos_weight']} | LEX: {diag['lex_weight']} | SKIP: {diag['skip_weight']}")
    print(f"  Results: {output_dir}")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
