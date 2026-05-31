# Integer Language Model

A pure integer language model with no neural networks and no torch dependency. Runs on a Pi 5. Produces grammatically coherent text for simple domains like children's stories.

## What It Is

An integer-only language model that combines:

1. **Bigram counting** — base probabilities P(word | previous) from integer counts
2. **POS energy rules** — category-level syntax that generalizes across words ("the cat" improves "a dog" because both are DET→NOUN)
3. **Lexical hash tables** — token-specific knowledge for fine-grained distinctions
4. **Skip-gram patterns** — structural dependencies beyond adjacent tokens (subject-verb, determiner-verb)
5. **Metropolis gate** — hard rejection for grammatically invalid tokens
6. **Boltzmann sampling** — stochastic generation from energy distribution

**Memory**: ~20 MB for a 2000-word vocabulary. No GPU needed.

## What It Can Do

- Generate grammatically coherent text for simple domains
- Score text for grammaticality (energy-based discrimination)
- Provide an interpretable 13×13 POS transition matrix showing what rules it learned
- Run on any device with Python and numpy — no torch, no GPU

## What It Cannot Do

- Generate creative or surprising text (it's a bigram model with syntactic corrections)
- Handle long-range semantic coherence beyond a few tokens
- Compete with neural language models on perplexity
- Understand meaning — it knows syntax, not semantics

## How It Works

### Training

1. Count bigrams from text (integer counting, no gradients)
2. Train energy tables via Noise-Contrastive Estimation (NCE):
   - For real (prev, target) pairs: table[hash] -= eta (lower energy = more likely)
   - For fake (prev, random) pairs: table[hash] += eta (higher energy = less likely)
3. Calibrate mixing weights via grid search

### Generation

At each step:
1. Bigram model proposes top-K candidates with probabilities
2. Feature hash energy computes ΔE for each candidate (O(1) per candidate)
3. Metropolis gate hard-rejects candidates above energy threshold
4. Repetition penalty added for recently-used words
5. Combined energy = base_energy + α × hash_energy + rep_penalty
6. Boltzmann sample from adjusted distribution

### The Key Insight: POS Generalization

With 13 POS categories, there are only 13×13 = 169 possible POS bigram pairs. Every DET→NOUN transition ("the cat", "a dog", "this house", ...) maps to the **same** energy table slot. Training on "the cat" automatically improves the score for "a dog" — without ever seeing that pair. This is generalization through shared structure, not through learned representations.

The POS table learns rules like:
- DET→NOUN = strongly negative energy (good transition)
- NOUN→DET = strongly positive energy (bad transition)
- AUX→VERB = negative energy (good)
- PUNCT→PUNCT = positive energy (bad)

These rules apply to **all** words with those POS tags, including words the model has never seen together.

### Skip-grams

Skip-gram tables hash (context[-2], candidate), skipping the immediately preceding word. This captures patterns like:
- "the cat **sat**" → skip2("the", "sat") captures DET→VERB at distance 2
- POS version: hash(POS("the"), POS("sat")) = hash(DET, VERB)

This provides structural awareness beyond what adjacent bigrams can capture.

## Quick Start

```bash
# Full training run (50K TinyStories texts)
python -u train.py

# Quick test with smaller data
python -u train.py --samples 5000 --vocab 1000 --nce-epochs 1

# Custom configuration
python -u train.py --nce-epochs 5 --lex-table-size 131071 --pos-eta 5
```

## Project Structure

```
src/ising_spin/
├── __init__.py              # Package exports
├── integer_lm.py            # Main class: IntegerLM
├── bigram_model.py          # Pure integer bigram base model
├── feature_hash_energy.py   # POS + lexical + skip-gram energy tables
├── boltzmann.py             # Integer-only Boltzmann sampler
├── vocabulary.py            # Vocabulary + POS type system
└── utils.py                 # Utility functions
train.py                     # Training script
```

## Architecture Details

### BigramModel (base model)

- Dense 2000×2000 int32 count matrix (16 MB)
- Laplace-smoothed probabilities: P(j|i) = (count[i][j] + α) / (total[i] + αV)
- Returns top-K candidates with log-probabilities

### FeatureHashEnergyTable (energy correction)

| Table | Size | What It Captures |
|-------|------|-----------------|
| POS bigram | 1009 × 2 hashes | DET→NOUN, NOUN→VERB rules (169 possible pairs) |
| POS trigram | 1265 × 2 hashes | Three-word POS patterns |
| Lexical bigram | 65537 × 3 hashes | Token-specific "the cat" vs "the dog" |
| Lexical trigram | 65569 × 3 hashes | Three-word token patterns |
| Skip-gram lexical | 65537 × 2 hashes | Long-range token dependencies |
| Skip-gram POS | 1009 × 2 hashes | Long-range POS dependencies |

Total memory: ~3 MB for the energy tables + 16 MB for bigrams = ~19 MB.

### IntegerBoltzmannSampler

Pre-computes a lookup table via integer geometric recurrence (no `math.exp`):
- Table construction: integer Taylor expansion of exp(-β)
- Sampling: integer array lookup + cumulative sum + binary search
- Zero floating-point operations in the hot path

## Dependencies

- Python 3.8+
- NumPy
- (Optional) `datasets` for downloading TinyStories

No torch. No GPU. No neural nets.

## License

MIT
