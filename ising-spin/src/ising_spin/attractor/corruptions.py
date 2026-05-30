"""
NCE Corruption Types for DAM Discriminator Training.

Instead of PCD (which relies on fantasy particles that don't work for language),
we use Noise Contrastive Estimation with EXPLICIT negative examples.

Each corruption type teaches the DAM a different aspect of language competence:
  1. RANDOM_SUB → lexical coherence (are these words semantically related?)
  2. POS_VIOLATE → grammatical structure (does this word fit the POS pattern?)
  3. TOPIC_VIOLATE → semantic coherence (is this word on-topic?)
  4. WORD_SWAP → word order sensitivity (does order matter here?)

The DAM learns to assign LOW energy to correct (context, next_word) pairs
and HIGH energy to corrupted pairs. This is exactly a discriminator.
"""

import numpy as np
from typing import List, Tuple, Optional


# Corruption type constants
RANDOM_SUB = 0       # Replace next_word with random vocab word
POS_VIOLATE = 1      # Replace with word of DIFFERENT POS type
TOPIC_VIOLATE = 2    # Replace with word from different frequency cluster
WORD_SWAP = 3        # Swap two adjacent words in context

CORRUPTION_NAMES = {
    RANDOM_SUB: "random_sub",
    POS_VIOLATE: "pos_violate",
    TOPIC_VIOLATE: "topic_violate",
    WORD_SWAP: "word_swap",
}


class Corruptor:
    """
    Generate NCE negative samples via 4 corruption types.

    For each (context_words, next_word) pair, generates n_negatives
    corrupted pairs. The DAM should learn to assign higher energy
    (lower probability) to corrupted pairs than to the original.
    """

    def __init__(
        self,
        vocab_words: List[str],
        word2idx: dict,
        idx2word: dict,
        pos_types: Optional[np.ndarray] = None,
        word_freq: Optional[np.ndarray] = None,
        n_pos_types: int = 13,
        seed: int = 42,
    ):
        """
        Args:
            vocab_words: List of vocabulary words (index = word ID).
            word2idx: Dict mapping word -> index.
            idx2word: Dict mapping index -> word.
            pos_types: Array (V,) of POS type IDs per word (0..n_pos_types-1).
                       None if POS system not available.
            word_freq: Array (V,) of word frequencies. Used for topic violation
                       (replace with word from different frequency cluster).
            n_pos_types: Number of distinct POS types.
            seed: Random seed.
        """
        self.vocab_words = vocab_words
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.V = len(vocab_words)
        self.pos_types = pos_types
        self.word_freq = word_freq
        self.n_pos_types = n_pos_types
        self._rng = np.random.RandomState(seed)

        # Precompute POS-type word lists for fast POS_VIOLATE sampling
        self._pos_word_lists = {}
        if pos_types is not None:
            for pt in range(n_pos_types):
                mask = pos_types == pt
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    self._pos_word_lists[pt] = indices

        # Precompute frequency clusters for TOPIC_VIOLATE
        # Split vocab into 5 frequency clusters (very common, common, medium, rare, very rare)
        self._freq_clusters = {}
        if word_freq is not None:
            freq_vals = word_freq.astype(np.float64)
            percentiles = np.percentile(freq_vals[freq_vals > 0],
                                        [20, 40, 60, 80])
            for i, word_idx in enumerate(range(self.V)):
                f = freq_vals[word_idx]
                if f <= 0:
                    cluster = 0
                elif f <= percentiles[0]:
                    cluster = 1
                elif f <= percentiles[1]:
                    cluster = 2
                elif f <= percentiles[2]:
                    cluster = 3
                elif f <= percentiles[3]:
                    cluster = 4
                else:
                    cluster = 5
                if cluster not in self._freq_clusters:
                    self._freq_clusters[cluster] = []
                self._freq_clusters[cluster].append(word_idx)

            # Convert to arrays for fast sampling
            for k in self._freq_clusters:
                self._freq_clusters[k] = np.array(self._freq_clusters[k])

    def generate_negatives(
        self,
        context_word_ids: List[int],
        next_word_id: int,
        n_negatives: int = 4,
    ) -> List[Tuple[List[int], int, int]]:
        """
        Generate n_negatives corrupted (context, candidate) pairs.

        Args:
            context_word_ids: List of word IDs in the context.
            next_word_id: The correct next word ID.
            n_negatives: Number of negative examples to generate.

        Returns:
            List of (corrupted_context_ids, corrupted_candidate_id, corruption_type)
            tuples. For WORD_SWAP, the context is modified and candidate stays same.
            For other types, context stays same and candidate is modified.
        """
        negatives = []
        types = [RANDOM_SUB, POS_VIOLATE, TOPIC_VIOLATE, WORD_SWAP]

        for i in range(n_negatives):
            ctype = types[i % len(types)]
            neg = self._corrupt(context_word_ids, next_word_id, ctype)
            if neg is not None:
                negatives.append(neg)
            else:
                # Fallback to random substitution if the preferred type fails
                neg = self._corrupt(context_word_ids, next_word_id, RANDOM_SUB)
                if neg is not None:
                    negatives.append(neg)

        return negatives

    def _corrupt(
        self,
        context_word_ids: List[int],
        next_word_id: int,
        ctype: int,
    ) -> Optional[Tuple[List[int], int, int]]:
        """Generate a single corrupted pair."""

        if ctype == RANDOM_SUB:
            # Replace next_word with a random word from vocab
            # Exclude the correct word
            candidates = list(range(self.V))
            if next_word_id in candidates and len(candidates) > 1:
                candidates.remove(next_word_id)
            if len(candidates) == 0:
                return None
            neg_cand = self._rng.choice(candidates)
            return (context_word_ids, int(neg_cand), ctype)

        elif ctype == POS_VIOLATE:
            # Replace next_word with a word of DIFFERENT POS type
            if self.pos_types is None or next_word_id >= len(self.pos_types):
                return self._corrupt(context_word_ids, next_word_id, RANDOM_SUB)

            target_pos = self.pos_types[next_word_id]
            # Collect all words with a DIFFERENT POS type
            other_indices = []
            for pt, indices in self._pos_word_lists.items():
                if pt != target_pos:
                    other_indices.append(indices)
            if not other_indices:
                return self._corrupt(context_word_ids, next_word_id, RANDOM_SUB)
            other_indices = np.concatenate(other_indices)
            # Exclude the correct word
            other_indices = other_indices[other_indices != next_word_id]
            if len(other_indices) == 0:
                return self._corrupt(context_word_ids, next_word_id, RANDOM_SUB)
            neg_cand = self._rng.choice(other_indices)
            return (context_word_ids, int(neg_cand), ctype)

        elif ctype == TOPIC_VIOLATE:
            # Replace next_word with a word from a DIFFERENT frequency cluster
            if not self._freq_clusters or self.word_freq is None:
                return self._corrupt(context_word_ids, next_word_id, RANDOM_SUB)

            # Find the cluster of the correct word
            freq_vals = self.word_freq.astype(np.float64)
            f = freq_vals[next_word_id]
            if f <= 0:
                target_cluster = 0
            else:
                percentiles = np.percentile(freq_vals[freq_vals > 0], [20, 40, 60, 80])
                if f <= percentiles[0]:
                    target_cluster = 1
                elif f <= percentiles[1]:
                    target_cluster = 2
                elif f <= percentiles[2]:
                    target_cluster = 3
                elif f <= percentiles[3]:
                    target_cluster = 4
                else:
                    target_cluster = 5

            # Pick from a DIFFERENT cluster
            other_clusters = [k for k in self._freq_clusters if k != target_cluster]
            if not other_clusters:
                return self._corrupt(context_word_ids, next_word_id, RANDOM_SUB)
            chosen_cluster = self._rng.choice(other_clusters)
            indices = self._freq_clusters[chosen_cluster]
            indices = indices[indices != next_word_id]
            if len(indices) == 0:
                return self._corrupt(context_word_ids, next_word_id, RANDOM_SUB)
            neg_cand = self._rng.choice(indices)
            return (context_word_ids, int(neg_cand), ctype)

        elif ctype == WORD_SWAP:
            # Swap two adjacent words in the context
            if len(context_word_ids) < 2:
                return self._corrupt(context_word_ids, next_word_id, RANDOM_SUB)

            # Pick a random adjacent pair to swap
            swap_pos = self._rng.randint(0, len(context_word_ids) - 1)
            neg_ctx = list(context_word_ids)
            neg_ctx[swap_pos], neg_ctx[swap_pos + 1] = \
                neg_ctx[swap_pos + 1], neg_ctx[swap_pos]
            return (neg_ctx, next_word_id, ctype)

        return None

    def generate_batch_negatives(
        self,
        batch_context_ids: List[List[int]],
        batch_next_ids: np.ndarray,
        n_negatives: int = 4,
    ) -> List[List[Tuple[List[int], int, int]]]:
        """
        Generate negatives for a batch of (context, next_word) pairs.

        Args:
            batch_context_ids: List of context word ID lists.
            batch_next_ids: Array of correct next word IDs, shape (N,).
            n_negatives: Number of negatives per positive.

        Returns:
            List (length N) of lists (length n_negatives) of
            (corrupted_context_ids, corrupted_candidate_id, corruption_type).
        """
        batch_negs = []
        for ctx, nxt in zip(batch_context_ids, batch_next_ids):
            negs = self.generate_negatives(ctx, nxt, n_negatives)
            batch_negs.append(negs)
        return batch_negs
