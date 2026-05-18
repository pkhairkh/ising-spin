"""
Gibbs sampler for the Ising Spin Language Model.

Generation loop uses ZERO floating-point operations:
  - Energy computation: integer addition only
  - Probability conversion: precomputed integer threshold tables
  - Random sampling: integer comparison against thresholds
  - State update: integer assignment

No exp(), no softmax(), no floating-point arithmetic whatsoever
in the generation loop.
"""

import random
from typing import List, Optional

import numpy as np

from .couplings import IsingCouplings
from .vocabulary import Vocabulary


class ProbabilityTable:
    """
    Precomputed integer threshold table for Boltzmann sampling.

    Instead of computing P(w) = exp(E(w) / T) / Z (which requires FP),
    we precompute cumulative integer thresholds for each energy level.

    The probability of accepting a change with energy difference delta_E is:
        P(accept) = exp(-delta_E * beta) when delta_E > 0
        P(accept) = 1 when delta_E <= 0

    We approximate this with integer thresholds precomputed in a lookup table.
    The table is computed ONCE at initialization — not during generation.
    """

    def __init__(self, max_delta_e: int = 2000, beta_int: int = 1000,
                 rand_max: int = 2**31 - 1):
        self.rand_max = rand_max
        self.beta_int = beta_int
        self.max_delta_e = max_delta_e

        # Build threshold table
        # This is the ONLY place where FP is used — one-time precomputation.
        # In production, this table would be a compile-time constant.
        self.thresholds = [0] * (2 * max_delta_e + 1)

        for delta_e in range(-max_delta_e, max_delta_e + 1):
            idx = delta_e + max_delta_e
            if delta_e <= 0:
                self.thresholds[idx] = rand_max  # Always accept
            else:
                import math
                beta = beta_int / 1000.0
                prob = math.exp(-delta_e * beta)
                threshold = int(rand_max * prob)
                self.thresholds[idx] = max(0, min(rand_max, threshold))

    def accept(self, delta_e: int, rand_val: int) -> bool:
        """
        Decide whether to accept a transition.

        PURE INTEGER OPERATION: one table lookup + one comparison.
        Zero floating-point in the generation loop.
        """
        if delta_e <= -self.max_delta_e:
            return True
        if delta_e >= self.max_delta_e:
            return False

        idx = delta_e + self.max_delta_e
        return rand_val < self.thresholds[idx]


class IsingSampler:
    """
    Gibbs sampler for text generation using the Ising spin model.

    Generation loop:
        1. Initialize state with random words (or prompt)
        2. For each sweep:
            a. Pick a position
            b. Propose candidate words from coupling neighborhood
            c. Compute energy difference (integer addition)
            d. Accept/reject using integer threshold lookup
            e. Update state
        3. After sufficient sweeps, read out state as generated text

    ZERO FLOATING-POINT OPERATIONS IN THE GENERATION LOOP.
    """

    def __init__(
        self,
        couplings: IsingCouplings,
        vocab: Vocabulary,
        temperature: int = 1000,
        n_sweeps: int = 100,
        proposal_top_k: int = 30,
    ):
        self.couplings = couplings
        self.vocab = vocab
        self.n_sweeps = n_sweeps
        self.proposal_top_k = proposal_top_k

        self.couplings.beta_int = temperature

        # Precompute probability table (one-time, NOT in generation loop)
        self.prob_table = ProbabilityTable(
            max_delta_e=2000,
            beta_int=temperature,
            rand_max=2**31 - 1,
        )

        # Precompute proposal sets for each word (integer lookup)
        print("  Precomputing proposal sets...")
        self.proposal_cache = {}
        vocab_size = len(self.vocab)
        # For each word, find its top-k neighbors from J_global
        for w in range(vocab_size):
            neighbors = self.couplings.get_neighbor_words(w, top_k=proposal_top_k)
            # Always include the word itself and common words
            if w not in neighbors:
                neighbors.append(w)
            self.proposal_cache[w] = neighbors

        # Precompute field-weighted proposal distribution for each position
        # (integer weights only)
        self.field_weights = {}
        for i in range(min(couplings.seq_len, 30)):
            w = couplings.h[i]
            if w.sum() > 0:
                self.field_weights[i] = np.cumsum(w)
            else:
                self.field_weights[i] = None

        print(f"  Proposal sets ready (top-{proposal_top_k} neighbors per word)")

    def _propose_word(self, pos: int, current_word: int) -> int:
        """
        Propose a candidate word for position pos.

        Strategy: mix field-based proposals (from h) with
        coupling-based proposals (from J_global neighborhood).

        Pure integer operations: cumulative sum search.
        """
        r = random.randint(0, 99)

        if r < 40:
            # 40%: sample from field distribution at this position
            cumsum = self.field_weights.get(pos % self.couplings.seq_len)
            if cumsum is not None:
                total = int(cumsum[-1])
                if total > 0:
                    rv = random.randint(1, total)
                    idx = int(np.searchsorted(cumsum, rv))
                    return min(idx, len(self.vocab) - 1)

        if r < 80:
            # 40%: sample from coupling neighborhood of current word
            neighbors = self.proposal_cache.get(current_word, [])
            if neighbors:
                return random.choice(neighbors)

        # 20%: uniform random from vocabulary
        return random.randint(0, len(self.vocab) - 1)

    def _init_state(self, length: int, prompt: Optional[List[int]] = None) -> List[int]:
        """Initialize state from prompt or field-weighted random."""
        state = []
        for i in range(length):
            if prompt and i < len(prompt):
                state.append(prompt[i])
            else:
                pos = i % self.couplings.seq_len
                cumsum = self.field_weights.get(pos)
                if cumsum is not None:
                    total = int(cumsum[-1])
                    if total > 0:
                        rv = random.randint(1, total)
                        idx = int(np.searchsorted(cumsum, rv))
                        state.append(min(idx, len(self.vocab) - 1))
                    else:
                        state.append(random.randint(0, len(self.vocab) - 1))
                else:
                    state.append(random.randint(0, len(self.vocab) - 1))
        return state

    def generate(
        self,
        length: int = 20,
        prompt: Optional[str] = None,
        n_sweeps: Optional[int] = None,
        verbose: bool = False,
    ) -> str:
        """
        Generate text using Gibbs sampling.

        THE GENERATION LOOP CONTAINS ZERO FLOATING-POINT OPERATIONS.
        All arithmetic is integer addition, comparison, and lookup.
        """
        sweeps = n_sweeps or self.n_sweeps

        # Encode prompt
        prompt_tokens = None
        if prompt:
            prompt_tokens = self.vocab.encode(prompt)

        # Initialize state
        state = self._init_state(length, prompt_tokens)

        if verbose:
            print(f"  Init: {self.vocab.decode(state)}")

        # Gibbs sampling loop — ZERO FLOATING-POINT
        for sweep in range(sweeps):
            for pos in range(length):
                # Skip prompt positions
                if prompt_tokens and pos < len(prompt_tokens):
                    continue

                current_word = state[pos]
                current_energy = self.couplings.get_local_energy(state, pos, current_word)

                # Propose a new word (integer operations)
                proposed_word = self._propose_word(pos, current_word)

                if proposed_word == current_word:
                    continue

                # Compute energy difference (INTEGER SUBTRACTION)
                proposed_energy = self.couplings.get_local_energy(state, pos, proposed_word)
                delta_e = proposed_energy - current_energy  # integer

                # Accept/reject using precomputed threshold table
                # PURE INTEGER: one table lookup + one comparison
                rand_val = random.randint(0, 2**31 - 2)
                if self.prob_table.accept(delta_e, rand_val):
                    state[pos] = proposed_word

            if verbose and (sweep + 1) % 25 == 0:
                print(f"  Sweep {sweep + 1}: {self.vocab.decode(state)}")

        return self.vocab.decode(state)

    def generate_multiple(
        self,
        n_samples: int = 5,
        length: int = 20,
        prompt: Optional[str] = None,
        n_sweeps: Optional[int] = None,
    ) -> List[str]:
        """Generate multiple independent samples."""
        return [
            self.generate(length=length, prompt=prompt, n_sweeps=n_sweeps)
            for _ in range(n_samples)
        ]

    def compute_energy_trace(
        self,
        length: int = 20,
        prompt: Optional[str] = None,
        n_sweeps: Optional[int] = None,
    ) -> List[int]:
        """
        Run generation and record energy at each sweep.
        Integer trace only.
        """
        sweeps = n_sweeps or self.n_sweeps
        prompt_tokens = self.vocab.encode(prompt) if prompt else None
        state = self._init_state(length, prompt_tokens)

        energies = []
        for sweep in range(sweeps):
            for pos in range(length):
                if prompt_tokens and pos < len(prompt_tokens):
                    continue

                current_word = state[pos]
                current_energy = self.couplings.get_local_energy(state, pos, current_word)
                proposed_word = self._propose_word(pos, current_word)

                if proposed_word != current_word:
                    proposed_energy = self.couplings.get_local_energy(state, pos, proposed_word)
                    delta_e = proposed_energy - current_energy
                    rand_val = random.randint(0, 2**31 - 2)
                    if self.prob_table.accept(delta_e, rand_val):
                        state[pos] = proposed_word

            energies.append(self.couplings.get_energy(state))

        return energies
