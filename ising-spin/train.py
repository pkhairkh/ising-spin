#!/usr/bin/env python3
"""
Integer EBM Re-ranker v76g — Training Script

v76g ARCHITECTURAL FIXES:
  1. Z-score energy normalization: DAM energies are normalized to match
     base model log-prob scale, eliminating the bit-shift hack.
  2. Bigram energy table: Explicit word-order model replaces the DAM's
     failed attempt to learn word swaps via NCE.
  3. No word_swap in NCE: Removed the impossible corruption type that
     was wasting J-matrix capacity.

Pipeline:
  1. Load base model (GPT-2 or DummyBaseLM)
  2. Load corpus, build vocabulary, tokenize
  3. Build SDR encoder
  4. Build DAM discriminator (NCE-trained, 3 corruption types only)
  5. Build bigram log-prob table
  6. NCE training: Hebbian with contrastive signal
  7. Calibrate re-ranking (z-score normalization + alpha/beta search)
  8. Evaluate: discriminative accuracy, PPL, generation

Usage:
  python -u train.py --samples 5000 --vocab 1000 --no-base-model --nce-epochs 1
  python -u train.py  # Default: 50K samples, GPT-2 base model
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

# Clear stale __pycache__
import shutil
for root, dirs, files in os.walk(Path(__file__).parent / "src"):
    if "__pycache__" in dirs:
        shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)
        dirs.remove("__pycache__")

import argparse
import json
import time
import traceback
import numpy as np
from pathlib import Path

# --- Configuration ---
DEFAULT_SAMPLES = 50000
DEFAULT_VOCAB = 2000
DEFAULT_DATASET = "tinystories"
DEFAULT_SDR_DIM = 512
DEFAULT_SDR_SPARSITY = 0.02

# Cache directory
CACHE_DIR = Path(__file__).parent
OUTPUT_DIR = CACHE_DIR / "output"


def get_rss_mb() -> int:
    """Get current process RSS in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except Exception:
        return 0


def find_cache_file(n_samples: int, dataset_name: str = "tinystories") -> str:
    """Find the best cache file for the requested number of samples."""
    cache_files = {}
    glob_pattern = f"cached_{dataset_name}_*.json"
    for f in CACHE_DIR.glob(glob_pattern):
        name = f.stem
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
    """Load or download data with proper cache handling."""
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

    print(f"Downloading {dataset_name} from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{n_samples}]")
        texts = [s["text"] for s in ds]
        cache_name = f"cached_{cache_dataset}_{n_samples // 1000}k.json"
        with open(CACHE_DIR / cache_name, "w") as f:
            json.dump(texts, f)
        print(f"  Downloaded and cached {len(texts):,} texts")
        return texts
    except Exception as e:
        print(f"  Error downloading: {e}")
        return []


def build_vocabulary(texts: list, max_vocab: int = 2000, min_freq: int = 5):
    """Build vocabulary from texts."""
    from collections import Counter
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)

    filtered = {w: c for w, c in word_counts.items() if c >= min_freq}
    sorted_words = sorted(filtered.items(), key=lambda x: -x[1])
    vocab_words = ["<pad>", "<unk>", "<bos>", "<eos>"] + [w for w, _ in sorted_words[:max_vocab-4]]

    word2idx = {w: i for i, w in enumerate(vocab_words)}
    idx2word = {i: w for i, w in enumerate(vocab_words)}
    word_freq = np.zeros(len(vocab_words), dtype=np.int32)
    for w, c in sorted_words[:max_vocab-4]:
        if w in word2idx:
            word_freq[word2idx[w]] = c

    return vocab_words, word2idx, idx2word, word_freq


def tokenize_texts(texts: list, word2idx: dict, max_seq_len: int = 30):
    """Tokenize texts into word ID sequences."""
    sequences = []
    for text in texts:
        words = text.lower().split()
        ids = [word2idx.get(w, 1) for w in words]
        ids = [id for id in ids if id >= 4]
        if len(ids) >= 2:
            sequences.append(ids[:max_seq_len])
    return sequences


def build_pos_types(vocab_words, word_freq):
    """Build simple POS type system based on word suffixes."""
    n_types = 13
    pos_types = np.zeros(len(vocab_words), dtype=np.int32)

    for i, word in enumerate(vocab_words):
        if i < 4:
            pos_types[i] = 0
        elif word.endswith(("ed",)):
            pos_types[i] = 1
        elif word.endswith(("ing",)):
            pos_types[i] = 2
        elif word.endswith(("ly",)):
            pos_types[i] = 3
        elif word.endswith(("tion", "ment", "ness", "ity")):
            pos_types[i] = 4
        elif word.endswith(("ous", "ful", "less", "ive", "able")):
            pos_types[i] = 5
        elif word in ("the", "a", "an"):
            pos_types[i] = 6
        elif word in ("is", "was", "were", "are", "am", "be", "been", "being"):
            pos_types[i] = 7
        elif word in ("he", "she", "it", "they", "we", "i", "you"):
            pos_types[i] = 8
        elif word in ("and", "but", "or", "so", "because", "when", "if"):
            pos_types[i] = 9
        elif word in ("to", "from", "in", "on", "at", "with", "by", "for"):
            pos_types[i] = 10
        elif word in (".", ",", "!", "?", ";", ":"):
            pos_types[i] = 11
        else:
            pos_types[i] = 12

    return pos_types


def main():
    parser = argparse.ArgumentParser(
        description="Integer EBM Re-ranker v76g — Z-score Normalization + Bigram Energy"
    )

    # Core parameters
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)

    # Base model
    parser.add_argument("--base-model", type=str, default="gpt2")
    parser.add_argument("--no-base-model", action="store_true", default=False,
                        help="Use DummyBaseLM (no torch required)")

    # Re-ranking
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--sdr-dim", type=int, default=DEFAULT_SDR_DIM)
    parser.add_argument("--sdr-sparsity", type=float, default=DEFAULT_SDR_SPARSITY)
    parser.add_argument("--dam-scale", type=int, default=1600)
    parser.add_argument("--j-clip", type=int, default=32000)
    parser.add_argument("--log-prob-scale", type=int, default=100)
    parser.add_argument("--dam-alpha", type=float, default=1.0,
                        help="v76g: Mixing weight for z-score normalized DAM energy. "
                             "0=pure base model, 1=equal DAM+base, >1=DAM dominates")
    parser.add_argument("--bigram-weight", type=int, default=10,
                        help="v76g: Weight for bigram log-prob energy (replaces word_swap NCE)")

    # NCE
    parser.add_argument("--nce-eta", type=int, default=10,
                        help="NCE learning rate")
    parser.add_argument("--nce-epochs", type=int, default=3,
                        help="NCE training epochs")
    parser.add_argument("--nce-negatives", type=int, default=3,
                        help="v76g: Number of NCE negatives (3 types: random_sub, pos_violate, topic_violate)")

    # Spin and episodic
    parser.add_argument("--spin-weight", type=int, default=5)
    parser.add_argument("--episodic-weight", type=int, default=2)
    parser.add_argument("--bind-weight", type=int, default=5)
    parser.add_argument("--max-episodes", type=int, default=10000)

    # Generation
    parser.add_argument("--rerank-beta", type=float, default=0.01)
    parser.add_argument("--gen-length", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--context-window", type=int, default=10)
    parser.add_argument("--vocab-min-freq", type=int, default=5)

    # Advanced
    parser.add_argument("--exp-temperature", type=int, default=100)

    args = parser.parse_args()

    # --- Header ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"reranker_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    use_dummy = args.no_base_model or args.base_model == "dummy"

    print("=" * 70, flush=True)
    print("INTEGER EBM RE-RANKER v76g — Z-score Normalization + Bigram Energy", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Key changes from v76e/f:", flush=True)
    print(f"  1. Z-score energy normalization (replaces bit-shift hack)", flush=True)
    print(f"  2. Bigram energy table (replaces word_swap NCE)", flush=True)
    print(f"  3. Only 3 NCE corruption types (no word_swap)", flush=True)
    rss = get_rss_mb()
    if rss > 0:
        print(f"Memory (RSS): {rss:,} MB", flush=True)
    print("=" * 70, flush=True)

    # --- [1] Load Data ---
    print("\n[1/7] Loading data...", flush=True)
    texts = load_data(args.samples, dataset_name=args.dataset)
    n_texts = len(texts)
    print(f"  Using {n_texts:,} texts for training", flush=True)

    # --- [2] Build Vocabulary ---
    print("\n[2/7] Building vocabulary...", flush=True)
    vocab_words, word2idx, idx2word, word_freq = build_vocabulary(
        texts, max_vocab=args.vocab, min_freq=args.vocab_min_freq
    )
    V = len(vocab_words)
    print(f"  Vocabulary: {V} words", flush=True)

    # --- [3] Tokenize ---
    print("\n[3/7] Tokenizing...", flush=True)
    sequences = tokenize_texts(texts, word2idx, max_seq_len=args.max_seq_len)
    # Split train/test
    n_train = int(0.9 * len(sequences))
    train_seqs = sequences[:n_train]
    test_seqs = sequences[n_train:]
    print(f"  Train: {len(train_seqs):,}, Test: {len(test_seqs):,}", flush=True)

    # --- [4] Build POS types ---
    print("\n[4/7] Building POS type system...", flush=True)
    pos_types = build_pos_types(vocab_words, word_freq)
    print(f"  POS system: {len(set(pos_types))} types, {V} words typed", flush=True)

    # --- [5] Build SDR Encoder ---
    print("\n[5/7] Building SDR encoder...", flush=True)
    from ising_spin.attractor.sdr import SDREncoder
    sdr_encoder = SDREncoder(
        vocab_size=V,
        D=args.sdr_dim,
        sparsity=args.sdr_sparsity,
    )
    sdr_encoder.build(word_freq=word_freq)
    k = sdr_encoder.k
    print(f"  SDR: D={args.sdr_dim}, k={k} ({args.sdr_sparsity*100:.1f}% sparse)", flush=True)

    # --- [6] Build Base Model ---
    print("\n[6/7] Building base model...", flush=True)
    base_model = None
    if use_dummy:
        from ising_spin.attractor.base_model import DummyBaseLM
        base_model = DummyBaseLM(
            vocab_words=vocab_words,
            word_freq=word_freq,
            seed=42,
        )
        base_model.build_bigrams(train_seqs)
        print(f"  DummyBaseLM: {V} words, bigrams built", flush=True)
    else:
        try:
            from ising_spin.attractor.base_model import BaseLMInterface
            base_model = BaseLMInterface(model_name=args.base_model)
            print(f"  Base model: {args.base_model} loaded", flush=True)
        except (ImportError, Exception) as e:
            print(f"  Cannot load GPT-2: {e}", flush=True)
            print(f"  Falling back to DummyBaseLM", flush=True)
            from ising_spin.attractor.base_model import DummyBaseLM
            base_model = DummyBaseLM(
                vocab_words=vocab_words,
                word_freq=word_freq,
                seed=42,
            )
            base_model.build_bigrams(train_seqs)
            use_dummy = True

    # --- [7] Build ReRankerEngine and Train ---
    print("\n[7/7] Building ReRankerEngine and training...", flush=True)
    from ising_spin.attractor.reranker_engine import ReRankerEngine

    engine = ReRankerEngine(
        base_model=base_model,
        sdr_encoder=sdr_encoder,
        vocab_words=vocab_words,
        word2idx=word2idx,
        idx2word=idx2word,
        pos_types=pos_types,
        word_freq=word_freq,
        dam_scale=args.dam_scale,
        j_clip=args.j_clip,
        f_type=2,  # F_EXP_APPROX
        exp_temperature=args.exp_temperature,
        nce_eta=args.nce_eta,
        nce_negatives=args.nce_negatives,
        nce_epochs=args.nce_epochs,
        context_window=args.context_window,
        dam_alpha=args.dam_alpha,
        top_k=args.top_k,
        log_prob_scale=args.log_prob_scale,
        spin_weight=args.spin_weight,
        episodic_weight=args.episodic_weight,
        bind_weight=args.bind_weight,
        bigram_weight=args.bigram_weight,
        max_episodes=args.max_episodes,
        seed=42,
    )

    # Train
    t_start = time.time()
    train_stats = engine.train(train_seqs)
    t_train = time.time() - t_start
    print(f"\n  Training complete: {t_train:.1f}s", flush=True)

    # --- Discriminative Accuracy ---
    print(f"\n{'=' * 70}", flush=True)
    print("DISCRIMINATIVE ACCURACY (KEY METRIC)", flush=True)
    print(f"{'=' * 70}", flush=True)

    try:
        disc_acc = engine.compute_discriminative_accuracy(
            test_seqs, n_samples=500
        )
        print(f"  Overall: {disc_acc['overall_accuracy']:.3f}", flush=True)
        for key in sorted(disc_acc.keys()):
            if key.endswith("_accuracy"):
                print(f"    {key}: {disc_acc[key]:.3f}", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)
        traceback.print_exc()
        disc_acc = {"overall_accuracy": 0.0}

    # --- Perplexity ---
    print(f"\n{'=' * 70}", flush=True)
    print("PERPLEXITY", flush=True)
    print(f"{'=' * 70}", flush=True)

    try:
        ppl_stats = engine.compute_perplexity(test_seqs, n_samples=min(100, len(test_seqs)))
        print(f"  Base PPL:      {ppl_stats['base_ppl']:.2f}", flush=True)
        print(f"  Re-ranked PPL: {ppl_stats['reranked_ppl']:.2f}", flush=True)
        print(f"  Evaluated on:  {ppl_stats['n_tokens']} tokens", flush=True)
        improvement = ppl_stats['base_ppl'] - ppl_stats['reranked_ppl']
        if improvement > 0:
            print(f"  DAM IMPROVES the base model by {improvement:.2f} PPL", flush=True)
        else:
            print(f"  DAM HURTS the base model by {-improvement:.2f} PPL", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)
        traceback.print_exc()
        ppl_stats = {"base_ppl": float("inf"), "reranked_ppl": float("inf"), "n_tokens": 0}

    # --- Generation ---
    print(f"\n{'=' * 70}", flush=True)
    print("GENERATION", flush=True)
    print(f"{'=' * 70}", flush=True)

    prompts = ["once upon a time", "there was a little", "the little girl"]

    for i, prompt in enumerate(prompts):
        print(f"\n  --- '{prompt}' ({args.gen_length} words) ---", flush=True)
        try:
            prompt_words = prompt.lower().split()
            prompt_ids = [word2idx.get(w, 1) for w in prompt_words]
            prompt_ids = [pid for pid in prompt_ids if pid >= 4]

            if len(prompt_ids) == 0:
                print("  (prompt words not in vocab)", flush=True)
                continue

            generated_ids = engine.generate(
                prompt_ids, length=args.gen_length
            )

            text = " ".join(idx2word.get(wid, "<unk>") for wid in generated_ids)
            print(f"  {text[:300]}", flush=True)

            gen_file = output_dir / f"generated_{i}.txt"
            with open(gen_file, "w") as f:
                f.write(text)

            # Also generate WITHOUT re-ranking for comparison
            base_ids = list(prompt_ids)
            for step in range(args.gen_length):
                candidates, log_probs = base_model.get_top_k(
                    base_ids, k=min(args.top_k, V)
                )
                best = int(np.argmax(log_probs))
                base_ids.append(int(candidates[best]))
            base_text = " ".join(idx2word.get(wid, "<unk>") for wid in base_ids)
            print(f"\n  --- Base model only (no re-ranking) ---", flush=True)
            print(f"  {base_text[:300]}", flush=True)

        except Exception as e:
            print(f"  Generation error: {e}", flush=True)
            traceback.print_exc()

    # --- Diagnostics ---
    print(f"\n{'=' * 70}", flush=True)
    print("DIAGNOSTICS", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"  DAM: J_nnz={int(np.count_nonzero(engine.dam.J))}, "
          f"J_max={int(np.max(np.abs(engine.dam.J)))}, "
          f"h_nnz={int(np.count_nonzero(engine.dam.h))}", flush=True)
    print(f"  Normalization: DAM_mean={engine._dam_energy_mean:.1f}, "
          f"DAM_std={engine._dam_energy_std:.1f}", flush=True)
    print(f"  Base: base_mean={engine._base_energy_mean:.1f}, "
          f"base_std={engine._base_energy_std:.1f}", flush=True)
    print(f"  Alpha={engine.dam_alpha:.1f}, Beta={engine.rerank_beta}", flush=True)
    print(f"  Bigram: {'built' if engine._bigram_logprob is not None else 'none'}", flush=True)
    print(f"  Spin: z_active={int(np.sum(engine.three_band.m_z > 0))}, "
          f"x_active={int(np.sum(engine.three_band.m_x > 0))}, "
          f"y_active={int(np.sum(engine.three_band.m_y > 0))}", flush=True)
    print(f"  Episodic: {len(engine.episodic.episodes)} episodes", flush=True)
    print(f"  Binding: {len(engine.binding._recent_words)} recent words", flush=True)

    # --- Save Results ---
    t_total = time.time() - t_start
    results = {
        "version": "76g.0.0",
        "architecture": "Integer EBM Re-ranker v76g (z-score normalization + bigram energy)",
        "timestamp": timestamp,
        "config": vars(args),
        "results": {
            "training_time_s": t_train,
            "total_time_s": t_total,
            "discriminative_accuracy": disc_acc.get("overall_accuracy", 0.0),
            "base_ppl": ppl_stats.get("base_ppl", float("inf")),
            "reranked_ppl": ppl_stats.get("reranked_ppl", float("inf")),
            "vocab_size": V,
            "J_nnz": int(np.count_nonzero(engine.dam.J)),
            "J_max": int(np.max(np.abs(engine.dam.J))),
            "base_model": "dummy" if use_dummy else args.base_model,
            "dam_alpha": engine.dam_alpha,
            "rerank_beta": engine.rerank_beta,
            "dam_energy_mean": engine._dam_energy_mean,
            "dam_energy_std": engine._dam_energy_std,
            "base_energy_mean": engine._base_energy_mean,
            "base_energy_std": engine._base_energy_std,
        },
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {results_file}")

    root_results = CACHE_DIR / "training_results.json"
    with open(root_results, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {root_results}")

    print(f"\n{'=' * 70}", flush=True)
    print(f"DONE — Integer EBM Re-ranker v76g")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"Discriminative accuracy: {disc_acc.get('overall_accuracy', 0.0):.3f}")
    print(f"Base PPL: {ppl_stats.get('base_ppl', float('inf')):.2f}")
    print(f"Re-ranked PPL: {ppl_stats.get('reranked_ppl', float('inf')):.2f}")
    print(f"Alpha: {engine.dam_alpha:.1f}, Beta: {engine.rerank_beta}")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
