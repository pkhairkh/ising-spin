# Integer Language Model

A pure integer language model with no neural networks and no torch dependency. Runs on a Pi 5. Produces grammatically coherent text for simple domains like children's stories.

## What It Is

An integer-only language model with **multi-class dynamic features**:

1. **Bigram counting** — base probabilities P(word | previous) from integer counts
2. **Multi-class word system** — frequency buckets + distributional clusters running simultaneously
3. **Dynamic feature registry** — add/remove energy features at runtime, each declares which class system it uses
4. **Balanced NCE training** — equal representation across all class types
5. **N-gram blocking** — prevent repeated bigrams in generation
6. **LEGD** — P(c) proportional to P_base(c) * exp(-alpha * E_norm(c))

**Memory**: ~25 MB for a 2000-word vocabulary. No GPU needed.

## What Changed in v82

**Multi-class architecture.** v81 replaced static POS with frequency buckets (PPL went from 12M to 13.89 — huge win). But the class transition matrix was nearly uniform because frequency buckets group words by how OFTEN they appear, not how they BEHAVE. v82 introduces distributional clusters that capture syntactic role from data, and runs BOTH class systems simultaneously.

**Distributional clusters.** Words that appear in similar contexts (similar left/right neighbors) get the same cluster. "the" and "a" cluster together (similar followers). "was" and "is" cluster together. This gives non-uniform class transition matrices with real syntactic patterns — something frequency buckets alone can't provide.

**N-gram blocking.** Generation now prevents repeating recent bigrams, not just unigrams. This breaks repetition loops like "the little girl named lily... the little girl named lily...".

**Adaptive clipping.** Energy tables use percentile-based clipping instead of fixed limits, preventing saturation while preserving learned distribution shape.

## Word Class Systems

| System | Key | K | Captures | Example |
|--------|-----|---|----------|---------|
| Frequency buckets | `freq` | 20 | Importance gradient | "the"→bucket 1, "cat"→bucket 8 |
| Distributional clusters | `dist` | 30 | Syntactic role | "the"→cluster 5, "was"→cluster 3 |

Both systems are DATA-DRIVEN — computed from corpus statistics, not hardcoded rules.
POS tags are kept for diagnostics only — never used in features.

## Available Features

### Lexical features (no class dependency)

| Feature | Hash | Default? |
|---------|------|----------|
| `LexBigramFeature` | hash(prev_word, cand_word) | Yes |
| `LexSkipFeature` | hash(prev2_word, cand_word) | Yes |
| `LexTrigramFeature` | hash(prev2_word, prev_word, cand_word) | Yes |

### Class features (class_key selects which class system)

| Feature | Hash | class_key | Default? |
|---------|------|-----------|----------|
| `WordClassBigramFeature` | hash(prev_word, cand_class) | "freq" | Yes |
| `ClassWordBigramFeature` | hash(prev_class, cand_word) | "freq" | Yes |
| `WordClassBigramFeature` | hash(prev_word, cand_class) | "dist" | Yes |
| `ClassWordBigramFeature` | hash(prev_class, cand_word) | "dist" | Yes |
| `ClassTrigramFeature` | hash(prev2_class, prev_class, cand_class) | "dist" | Yes |

## Quick Start

```bash
# Full training run (50K TinyStories texts, 8 features with dist clusters)
python -u train.py

# Quick test with smaller data
python -u train.py --samples 5000 --vocab 1000 --nce-epochs 1

# All feature variants
python -u train.py --features all

# Disable distributional clusters (freq-only mode)
python -u train.py --no-dist

# More distributional clusters
python -u train.py --n-clusters 40

# Custom feature set
python -u train.py --features lex_bi,word_cls_bi_freq,cls_word_bi_freq
```

## Adding Custom Features

```python
from ising_spin import FeatureSpec, IntegerLM

class MyFeature(FeatureSpec):
    """Custom feature using the 'dist' class system."""

    def __init__(self, **kwargs):
        super().__init__("my_feature", n_hashes=2, table_size=65537,
                         eta=1, clip=50, weight=0.5, class_key="dist", **kwargs)

    def get_hash_args_batch(self, context, candidates, word_class):
        if not context:
            return None
        K = len(candidates)
        prev_class = np.full(K, int(word_class[context[-1]]), dtype=np.int64)
        return (prev_class, candidates.astype(np.int64))

    def get_hash_args_nce(self, prev_words, prev_class, right_words, right_class,
                          prev2_words=None, prev2_class=None, mask=None):
        return (prev_class, right_words)

# Register it
model = IntegerLM(vocab=vocab)
model.add_feature(MyFeature())
model.train(sequences)
```

## Project Structure

```
src/ising_spin/
├── __init__.py              # Package exports
├── integer_lm.py            # Main class: IntegerLM
├── bigram_model.py          # Pure integer bigram base model
├── feature_hash_energy.py   # FeatureSpec base + concrete features + multi-class registry
├── boltzmann.py             # Integer-only Boltzmann sampler
├── vocabulary.py            # Vocabulary + multi-class word system (freq + dist)
└── utils.py                 # Utility functions
train.py                     # Training script
```

## How It Works

### Training

1. Count bigrams from text (integer counting, no gradients)
2. Build frequency buckets (words ranked by frequency, binned into K groups)
3. Build distributional clusters (words grouped by context similarity via min-hash)
4. Train all feature tables via Noise-Contrastive Estimation (NCE):
   - Real (prev, target) pairs: table[hash] -= eta (lower energy = more likely)
   - Fake (prev, random) pairs: table[hash] += eta (higher energy = less likely)
   - Negatives are BALANCED across all class systems
5. Calibrate alpha and feature weights via grid search

### Generation

At each step:
1. Bigram model proposes top-K candidates with probabilities
2. Each feature computes its energy contribution using its class array
3. LEGD combines: P(c) proportional to P_base(c) * exp(-alpha * E_norm(c))
4. Metropolis gate hard-rejects candidates above threshold
5. Repetition penalty for recently-used words AND bigrams
6. Sample from distribution

### Why Multi-Class Works

Frequency buckets capture the functional/content word gradient but can't distinguish "the" from "was" (both high-frequency). Distributional clusters capture syntactic role but miss the importance gradient. Using BOTH gives complementary signals:

- `hash(word, freq_bucket)`: "the"→bucket 1 predicts high-freq followers
- `hash(word, dist_cluster)`: "the"→cluster 5 predicts noun followers

Together, these give the model both importance AND syntactic information.

## Dependencies

- Python 3.8+
- NumPy
- (Optional) `datasets` for downloading TinyStories

No torch. No GPU. No neural nets.

## License

MIT
