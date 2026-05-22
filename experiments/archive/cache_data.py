#!/usr/bin/env python3
"""Cache FineWeb-Edu data to disk for reliable training."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.model import load_fineweb_edu

N = 50000
cache_path = "/home/z/my-project/ising-spin/cached_fineweb_50k.json"

if os.path.exists(cache_path):
    print(f"Cache already exists: {cache_path}")
    with open(cache_path) as f:
        texts = json.load(f)
    print(f"Loaded {len(texts)} texts from cache")
else:
    print(f"Loading {N} texts from FineWeb-Edu...")
    texts = load_fineweb_edu(n_samples=N)
    print(f"Got {len(texts)} texts, saving to cache...")
    with open(cache_path, 'w') as f:
        json.dump(texts, f)
    print(f"Saved {len(texts)} texts to {cache_path}")

print(f"Sample: {texts[0][:200]}")
