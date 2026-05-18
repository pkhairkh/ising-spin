"""
Main pipeline for the Ising Spin Language Model.

Orchestrates: data loading -> vocabulary building -> coupling computation -> generation.
All generation-path computation uses integer arithmetic only.
"""

import os
import time
from typing import List, Optional

import numpy as np

from .vocabulary import Vocabulary
from .couplings import IsingCouplings
from .sampler import IsingSampler
from .data_loader import load_fineweb_edu, tokenize_texts, truncate_sequences


class IsingLanguageModel:
    """
    End-to-end Ising Spin Language Model.

    Training phase (one-time, may use FP for data loading):
        1. Load corpus from fineweb-edu
        2. Build integer vocabulary (frequency counts)
        3. Compute integer couplings (co-occurrence counts)

    Generation phase (ZERO FP operations):
        1. Initialize state from prompt or randomly
        2. Run Gibbs sampling with integer-only arithmetic
        3. Decode state to text
    """

    def __init__(
        self,
        vocab_min_freq: int = 5,
        vocab_max_size: Optional[int] = 5000,
        seq_len: int = 30,
        window: int = 5,
        temperature: int = 800,
        n_sweeps: int = 100,
    ):
        self.vocab_min_freq = vocab_min_freq
        self.vocab_max_size = vocab_max_size
        self.seq_len = seq_len
        self.window = window
        self.temperature = temperature
        self.n_sweeps = n_sweeps

        self.vocab: Optional[Vocabulary] = None
        self.couplings: Optional[IsingCouplings] = None
        self.sampler: Optional[IsingSampler] = None

    def train(
        self,
        n_samples: int = 50000,
        min_count: int = 2,
        scaling: int = 1,
    ) -> "IsingLanguageModel":
        """
        Train the model: build vocabulary and compute couplings.

        This is the only phase that may involve FP (for data loading).
        The resulting couplings are pure integer.

        Args:
            n_samples: Number of texts to load from fineweb-edu.
            min_count: Minimum co-occurrence count for storing a coupling.
            scaling: Integer multiplier for all coupling strengths.

        Returns:
            self (for chaining)
        """
        print("=" * 60)
        print("ISING SPIN LANGUAGE MODEL - TRAINING")
        print("=" * 60)

        # Step 1: Load data
        t0 = time.time()
        texts = load_fineweb_edu(n_samples=n_samples)
        print(f"Data loading: {time.time() - t0:.1f}s")

        # Step 2: Build vocabulary (integer counting only)
        t0 = time.time()
        self.vocab = Vocabulary(
            min_freq=self.vocab_min_freq,
            max_size=self.vocab_max_size,
        )
        self.vocab.build(texts)
        print(f"Vocabulary: {len(self.vocab)} words ({time.time() - t0:.1f}s)")

        # Step 3: Tokenize (integer encoding)
        t0 = time.time()
        sequences = tokenize_texts(texts, self.vocab)
        sequences = truncate_sequences(sequences, max_len=self.seq_len)
        print(f"Tokenization: {len(sequences)} sequences ({time.time() - t0:.1f}s)")

        # Step 4: Compute couplings (integer counting only)
        t0 = time.time()
        self.couplings = IsingCouplings(
            vocab_size=len(self.vocab),
            seq_len=self.seq_len,
            window=self.window,
        )
        self.couplings.compute_from_sequences(
            sequences, min_count=min_count, scaling=scaling
        )
        print(f"Coupling computation: {time.time() - t0:.1f}s")

        # Report coupling statistics (all integers)
        J_global_nnz = int(np.count_nonzero(self.couplings.J_global))
        n_dist_levels = len(self.couplings.J_by_dist)
        total_sparse_couplings = sum(
            len(c) for c in self.couplings.J_by_dist.values()
        )
        print(f"Global coupling non-zeros: {J_global_nnz}")
        print(f"Distance-specific coupling levels: {n_dist_levels}")
        print(f"Total sparse distance couplings: {total_sparse_couplings}")
        print(f"Max coupling value: {int(self.couplings.J_global.max())}")
        print(f"Max field value: {int(self.couplings.h.max())}")

        # Step 5: Build sampler
        self.sampler = IsingSampler(
            couplings=self.couplings,
            vocab=self.vocab,
            temperature=self.temperature,
            n_sweeps=self.n_sweeps,
        )

        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        return self

    def generate(
        self,
        prompt: Optional[str] = None,
        length: int = 20,
        n_sweeps: Optional[int] = None,
        verbose: bool = False,
    ) -> str:
        """
        Generate text. ZERO FP operations in the generation loop.

        Args:
            prompt: Optional conditioning prompt.
            length: Number of tokens to generate.
            n_sweeps: Override default number of sweeps.
            verbose: Print intermediate states.

        Returns:
            Generated text string.
        """
        if self.sampler is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.sampler.generate(
            length=length,
            prompt=prompt,
            n_sweeps=n_sweeps,
            verbose=verbose,
        )

    def generate_batch(
        self,
        n_samples: int = 5,
        prompt: Optional[str] = None,
        length: int = 20,
        n_sweeps: Optional[int] = None,
    ) -> List[str]:
        """Generate multiple samples."""
        if self.sampler is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.sampler.generate_multiple(
            n_samples=n_samples,
            length=length,
            prompt=prompt,
            n_sweeps=n_sweeps,
        )

    def save(self, directory: str):
        """Save model to directory."""
        os.makedirs(directory, exist_ok=True)
        self.vocab.save(os.path.join(directory, "vocab.json"))
        self.couplings.save(os.path.join(directory, "couplings"))

    @classmethod
    def load(cls, directory: str, **kwargs) -> "IsingLanguageModel":
        """Load model from directory."""
        model = cls(**kwargs)
        model.vocab = Vocabulary.load(os.path.join(directory, "vocab.json"))
        model.couplings = IsingCouplings.load(os.path.join(directory, "couplings"))
        model.sampler = IsingSampler(
            couplings=model.couplings,
            vocab=model.vocab,
            temperature=kwargs.get("temperature", 800),
            n_sweeps=kwargs.get("n_sweeps", 100),
        )
        return model
