# Integer Language Model

A pure integer language model with no neural networks and no torch dependency. Runs on a Pi 5. Produces grammatically coherent text for simple domains like children's stories.

## What It Is

An integer-only language model with **dynamic features**:

1. **Bigram counting** — base probabilities P(word | previous) from integer counts
2. **Dynamic feature registry** — add/remove energy features at runtime
3. **Mixed word-POS features** — hash(word, POS) with 26000+ keys (not the old static 13x13 POS matrix!)
4. **Balanced NCE training** — equal representation across POS types
5. **Metropolis gate** — hard rejection for grammatically invalid tokens
6. **LEGD** — P(c) proportional to P_base(c) * exp(-alpha * E_norm(c))

**Memory**: ~20 MB for a 2000-word vocabulary. No GPU needed.

## What Changed in v80

**The POS disaster is fixed.** The old system used a hardcoded 13x13 POS matrix with only 169 unique keys. Every slot saturated at +/-500, the gradient went to zero, and POS discrimination accuracy was below random (0.46). The new system uses **mixed word-POS features** like hash(prev_word, cand_pos) with V*13 = 26000+ unique keys. Hash collisions in a 65537-slot table create smooth generalization without saturation.

**Variable features.** The old system hardcoded exactly 4 table types. The new system uses a `FeatureSpec` base class with `add_feature()` / `remove_feature()`. You can register any number of features, including custom ones you define yourself.

**Fixed energy scaling.** The old system had clip values of 500-1000, producing energy std of 3600+ that overwhelmed the base model (std 246). With clips of 30-100 and proper z-score normalization, the energy acts as a gentle correction.

## Available Features

### Bigram features (need context >= 1)

| Feature | Hash | Keys | Default? |
|---------|------|------|----------|
| `LexBigramFeature` | hash(prev_word, cand_word) | V^2 = 4M | Yes |
| `WordPosBigramFeature` | hash(prev_word, cand_pos) | V*13 = 26K | Yes |
| `PosWordBigramFeature` | hash(prev_pos, cand_word) | 13*V = 26K | Yes |
| `PosBigramFeature` | hash(prev_pos, cand_pos) | 169 | No* |

*PosBigramFeature is excluded by default — 169 keys causes saturation.

### Skip-gram features (need context >= 2)

| Feature | Hash | Keys | Default? |
|---------|------|------|----------|
| `LexSkipFeature` | hash(prev2_word, cand_word) | V^2 = 4M | Yes |
| `WordPosSkipFeature` | hash(prev2_word, cand_pos) | V*13 = 26K | No |
| `PosWordSkipFeature` | hash(prev2_pos, cand_word) | 13*V = 26K | No |
| `PosSkipFeature` | hash(prev2_pos, cand_pos) | 169 | No* |

### Trigram features (need context >= 2)

| Feature | Hash | Keys | Default? |
|---------|------|------|----------|
| `PosTrigramFeature` | hash(prev2_pos, prev_pos, cand_pos) | 2197 | Yes |
| `LexTrigramFeature` | hash(prev2_word, prev_word, cand_word) | V^3 = 8B | Yes |

## Quick Start

```bash
# Full training run (50K TinyStories texts, default 6 features)
python -u train.py

# Quick test with smaller data
python -u train.py --samples 5000 --vocab 1000 --nce-epochs 1

# All 10 features
python -u train.py --features all

# Custom feature set
python -u train.py --features lex_bi,word_pos_bi,pos_word_bi

# Adjust energy scale
python -u train.py --lex-clip 50 --pos-clip 20
```

## Adding Custom Features

```python
from ising_spin import FeatureSpec, IntegerLM

class MyFeature(FeatureSpec):
    """Custom feature: hash(my_thing, cand_word)"""

    def __init__(self, **kwargs):
        super().__init__("my_feature", n_hashes=2, table_size=65537,
                         eta=1, clip=50, weight=0.5, **kwargs)

    def get_hash_args_batch(self, context, candidates, word_pos):
        if not context:
            return None
        K = len(candidates)
        # Your custom extraction logic here
        my_thing = np.full(K, some_value, dtype=np.int64)
        return (my_thing, candidates.astype(np.int64))

    def get_hash_args_nce(self, prev_words, prev_pos, right_words, right_pos,
                          prev2_words=None, prev2_pos=None, mask=None):
        # Same extraction for NCE training
        return (some_array, right_words)

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
├── feature_hash_energy.py   # FeatureSpec base + 10 concrete features + registry
├── boltzmann.py             # Integer-only Boltzmann sampler
├── vocabulary.py            # Vocabulary + POS type system
└── utils.py                 # Utility functions
train.py                     # Training script
```

## How It Works

### Training

1. Count bigrams from text (integer counting, no gradients)
2. Train all feature tables via Noise-Contrastive Estimation (NCE):
   - Real (prev, target) pairs: table[hash] -= eta (lower energy = more likely)
   - Fake (prev, random) pairs: table[hash] += eta (higher energy = less likely)
   - Negatives are BALANCED across POS types (not 88% NOUN)
3. Calibrate alpha via grid search (range [0.001, 0.5])

### Generation

At each step:
1. Bigram model proposes top-K candidates with probabilities
2. Each feature computes its energy contribution (O(1) per candidate per feature)
3. LEGD combines: P(c) proportional to P_base(c) * exp(-alpha * E_norm(c))
4. Metropolis gate hard-rejects candidates above threshold
5. Repetition penalty for recently-used words
6. Sample from distribution

### The Key Insight: Word-POS Features

The old approach: hash(prev_pos, cand_pos) produces 13*13 = 169 unique keys. Each key independently accumulates NCE updates and saturates at the clip limit. Once saturated, the gradient is zero — the table can't learn.

The new approach: hash(prev_word, cand_pos) produces V*13 = 26000+ unique keys. With a 65537-slot table and 2 hash functions:
- "the"->NOUN and "a"->NOUN hash to DIFFERENT slots (specificity)
- But nearby hash slots create smooth generalization (like "the" and "a" both being determiners)
- "the"->VERB hashes to a completely different slot (no contamination)
- Saturation is impossible: with 26000 keys and clip=50, the table converges smoothly

## Dependencies

- Python 3.8+
- NumPy
- (Optional) `datasets` for downloading TinyStories

No torch. No GPU. No neural nets.

## License

MIT
