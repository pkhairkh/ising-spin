"""
Topic assignment from training data using integer K-means.

This module is ONLY for computing topic assignments from training data.
Runtime spin-flip logic (sigma_T, coherence penalty, etc.) belongs in
the state module.
"""

import math
from typing import List

import numpy as np


class TopicAssigner:
    """
    Compute word-topic assignments from training data using integer K-means.

    Provides:
      - build(): runs integer K-means on training texts, computes word_topics
      - word_topics: (vocab_size,) int8 array of dominant topic per word
      - n_topics: number of topics
      - topic_word_counts: (n_topics, vocab_size) int64 array
    """

    def __init__(self, n_topics: int = 16):
        self.n_topics = n_topics
        self._word_topics: np.ndarray | None = None   # (vocab_size,), dtype=int8
        self._topic_word_counts: np.ndarray | None = None  # (n_topics, vocab_size), dtype=int64
        self._built = False

    @property
    def word_topics(self) -> np.ndarray:
        """(vocab_size,) int8 array of topic assignments per word."""
        if self._word_topics is None:
            raise RuntimeError("TopicAssigner has not been built yet. Call build() first.")
        return self._word_topics

    @property
    def topic_word_counts(self) -> np.ndarray:
        """(n_topics, vocab_size) int64 array of word counts per topic."""
        if self._topic_word_counts is None:
            raise RuntimeError("TopicAssigner has not been built yet. Call build() first.")
        return self._topic_word_counts

    def build(self, texts: List[str], vocab) -> "TopicAssigner":
        """
        Build topic assignments from training corpus — ALL INTEGER.

        Runs integer K-means with cosine-similarity-like assignment (dot product
        with integer L2-norm scaling) to cluster documents, then assigns each
        word its dominant topic based on aggregate counts.

        Args:
            texts: list of training text strings
            vocab: Vocabulary object with word2idx attribute

        Returns:
            self
        """
        K = self.n_topics
        vocab_size = len(vocab)

        # Step 1: Build document-term matrix (integer counts) — subsample for speed
        MAX_CLUSTER_DOCS = 5000
        cluster_texts = texts[:MAX_CLUSTER_DOCS] if len(texts) > MAX_CLUSTER_DOCS else texts
        n_docs = len(cluster_texts)

        print(f"  Building Topic Assigner (K={K})")

        if n_docs == 0:
            print(f"    No documents — skipping topic assignment")
            return self

        print(f"    [1/4] Building document-term matrix ({n_docs} docs, {vocab_size} vocab)...")
        doc_vectors = np.zeros((n_docs, vocab_size), dtype=np.int32)
        for d, text in enumerate(cluster_texts):
            # v17.4 FIX: Use vocab._tokenize() instead of text.split()
            # text.split() misses lowercasing, punctuation stripping, and contraction splitting.
            # This caused massive lookup failures (e.g., "The" never matched "the").
            tokens = vocab._tokenize(text)
            for w in tokens:
                idx = vocab.word2idx.get(w)
                if idx is not None:
                    doc_vectors[d, idx] += 1

        # Step 2: Initialize centroids from evenly-spaced documents
        print(f"    [2/4] Initializing {K} topic centroids...")
        centroids = np.zeros((K, vocab_size), dtype=np.int64)
        step = max(1, n_docs // K)
        for k in range(K):
            centroids[k] = doc_vectors[(k * step) % n_docs].astype(np.int64)

        # Step 3: Iterative hard clustering — vectorized for speed
        print(f"    [3/4] Running integer K-means ({K} topics, 5 iters)...")
        assignments = np.zeros(n_docs, dtype=np.int32)

        for iteration in range(5):
            # INTEGER-ONLY K-means via normalized dot product similarity.
            # Instead of float64 cosine similarity, use integer dot product
            # with L2-norm scaling via integer square root approximation.
            doc_sq = (doc_vectors.astype(np.int64) ** 2).sum(axis=1)  # L2² per doc
            doc_norms_int = np.array([max(1, int(math.isqrt(int(s)))) for s in doc_sq], dtype=np.int64)
            cent_sq = (centroids ** 2).sum(axis=1)  # L2² per centroid
            cent_norms_int = np.array([max(1, int(math.isqrt(int(s)))) for s in cent_sq], dtype=np.int64)

            # Compute similarity = dot(d, c) / (|d| * |c|) as integer fixed-point
            FP_SCALE = 1 << 30
            dot_products = doc_vectors.astype(np.int64) @ centroids.T  # (n_docs, K)
            norm_products = doc_norms_int[:, None] * cent_norms_int[None, :]  # (n_docs, K)
            norm_products = np.maximum(norm_products, 1)  # avoid div/0
            similarities = (dot_products * FP_SCALE) // norm_products  # (n_docs, K) int64
            new_assignments = np.argmax(similarities, axis=1).astype(np.int32)

            changed = int((new_assignments != assignments).sum())
            assignments = new_assignments

            # Recompute centroids
            for k in range(K):
                mask = assignments == k
                if mask.any():
                    centroids[k] = doc_vectors[mask].sum(axis=0).astype(np.int64)
                else:
                    centroids[k] = doc_vectors[np.random.randint(n_docs)].astype(np.int64)

            sizes = [int((assignments == k).sum()) for k in range(K)]
            print(f"      Iter {iteration + 1}: {changed} reassigned, sizes={sizes}")

            if changed == 0:
                break

        # Step 4: Compute word-topic assignments from ALL texts
        print(f"    [4/4] Computing word-topic assignments (all {len(texts)} texts)...")
        topic_word_counts = np.zeros((K, vocab_size), dtype=np.int64)

        # Use cluster assignments for the clustered subset
        for d in range(n_docs):
            topic_word_counts[assignments[d]] += doc_vectors[d]

        # Recompute centroid norms for chunk assignment (after final K-means iteration)
        cent_sq_final = (centroids ** 2).sum(axis=1)
        cent_norms_final = np.array([max(1, int(math.isqrt(int(s)))) for s in cent_sq_final], dtype=np.int64)
        FP_SCALE_FINAL = 1 << 30

        # For remaining texts, batch into chunks and vectorize
        remaining = texts[n_docs:]
        if remaining:
            CHUNK = 2000
            for chunk_start in range(0, len(remaining), CHUNK):
                chunk = remaining[chunk_start:chunk_start + CHUNK]
                chunk_vecs = np.zeros((len(chunk), vocab_size), dtype=np.int64)
                for d, text in enumerate(chunk):
                    # v17.4 FIX: Use vocab._tokenize() instead of text.split()
                    tokens = vocab._tokenize(text)
                    for w in tokens:
                        idx = vocab.word2idx.get(w)
                        if idx is not None:
                            chunk_vecs[d, idx] += 1
                # INTEGER-ONLY assignment via normalized dot product
                c_sq = (chunk_vecs ** 2).sum(axis=1)
                c_norms_int = np.array([max(1, int(math.isqrt(int(s)))) for s in c_sq], dtype=np.int64)
                dot_prods = chunk_vecs @ centroids.T  # (chunk_size, K)
                norm_prods = c_norms_int[:, None] * cent_norms_final[None, :]
                norm_prods = np.maximum(norm_prods, 1)
                sims = (dot_prods * FP_SCALE_FINAL) // norm_prods
                chunk_assignments = np.argmax(sims, axis=1)
                for d in range(len(chunk)):
                    topic_word_counts[chunk_assignments[d]] += chunk_vecs[d]

        self._word_topics = np.argmax(topic_word_counts, axis=0).astype(np.int8)
        self._topic_word_counts = topic_word_counts

        n_unique = len(set(self._word_topics.tolist()))
        topic_sizes = [int((self._word_topics == k).sum()) for k in range(K)]
        print(f"    Topic assignments: {n_unique} topics used, sizes={topic_sizes}")

        self._built = True
        return self
