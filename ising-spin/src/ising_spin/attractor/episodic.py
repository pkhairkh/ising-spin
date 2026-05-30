"""
Content-Addressable Episodic Memory — Sparse pattern storage and retrieval.

WHY EPISODIC MEMORY:
  The DAM captures statistical regularities (what tends to follow what),
  but it doesn't remember specific episodes. For coherent text generation,
  we need BOTH:
    - Statistical: "the" is usually followed by a noun
    - Episodic: "the dragon" appeared earlier, so "dragon" should be
      accessible when we need a specific noun

  Episodic memory stores individual patterns (not averages) and retrieves
  them via content-based addressing (Hamming overlap). This provides:
    - Long-range coherence: earlier content can influence later predictions
    - Entity tracking: "dragon" → "it" → "the dragon" consistency
    - Narrative continuity: story elements persist across the document

CONTENT-ADDRESSABLE RETRIEVAL:
  Query: current context SDR
  Compare: Hamming overlap with all stored episodes
  Retrieve: top-k most similar episodes
  Inject: as external field into DAM dynamics

  This is EXACTLY how biological episodic memory works (hippocampus):
    - Store: pattern separation (sparse, decorrelated)
    - Retrieve: pattern completion (content-addressable)
    - Inject: as external field (reinstatement in neocortex)

SPARSE STORAGE:
  Each episode is stored as a sparse binary vector (the SDR of the
  current document state). We store up to max_episodes patterns.

  Retrieval uses Hamming overlap, which is extremely fast for sparse
  vectors: only k active bits to check per pattern.

INTEGER-ONLY:
  All overlap computations are integer. No floating point anywhere.
  Storage is compact: (max_episodes, k) int16 indices ≈ 200 KB for
  10000 episodes with k=10.
"""

import numpy as np
from typing import List, Optional, Tuple


class EpisodicMemory:
    """
    Content-addressable episodic memory for sparse SDR patterns.

    Stores episodes as sparse binary patterns (lists of active bit indices).
    Retrieves via Hamming overlap. Injects retrieved patterns as external
    field into DAM dynamics.
    """

    def __init__(
        self,
        D: int = 512,
        k: int = 10,
        max_episodes: int = 10000,
        retrieval_top_k: int = 5,
        field_scale: int = 500,
        decay_rate: int = 1,
        seed: int = 42,
    ):
        """
        Args:
            D: SDR dimension (must match L0 dimension).
            k: Number of active bits per episode.
            max_episodes: Maximum number of stored episodes.
            retrieval_top_k: Number of top episodes to retrieve.
            field_scale: Scale for external field injection.
            decay_rate: Decay rate for older episodes (0 = no decay).
            seed: Random seed.
        """
        self.D = D
        self.k = k
        self.max_episodes = max_episodes
        self.retrieval_top_k = retrieval_top_k
        self.field_scale = field_scale
        self.decay_rate = decay_rate
        self.seed = seed

        # Storage: list of active bit indices per episode
        # Using sparse format for efficiency
        self.episodes: List[np.ndarray] = []  # Each element: array of k active indices
        self.episode_ages: List[int] = []  # Age of each episode (for decay)
        self._step = 0

        # Dense accumulator for fast overlap computation
        # episode_counts[d] = number of episodes where bit d is active
        self.episode_counts = np.zeros(D, dtype=np.int32)

        self._rng = np.random.RandomState(seed)

    def store(self, episode_sdr: np.ndarray) -> None:
        """
        Store a new episode.

        If memory is full, the oldest episode is removed (FIFO).
        The episode is stored in sparse format (active bit indices only).

        Args:
            episode_sdr: Binary vector (D,) uint8 with k active bits.
        """
        active = np.where(episode_sdr > 0)[0].astype(np.int16)

        if len(self.episodes) >= self.max_episodes:
            # Remove oldest episode
            old_active = self.episodes.pop(0)
            self.episode_ages.pop(0)
            # Update counts
            for idx in old_active:
                if 0 <= idx < self.D:
                    self.episode_counts[idx] = max(0, self.episode_counts[idx] - 1)

        self.episodes.append(active)
        self.episode_ages.append(self._step)

        # Update counts
        for idx in active:
            if 0 <= idx < self.D:
                self.episode_counts[idx] += 1

        self._step += 1

    def retrieve(
        self,
        query_sdr: np.ndarray,
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Retrieve the most similar episodes by Hamming overlap.

        Hamming overlap between sparse vectors = count of shared active bits.
        For two k-sparse vectors, overlap ∈ [0, k].

        Args:
            query_sdr: Query SDR (D,) uint8.
            top_k: Number of episodes to retrieve (default: self.retrieval_top_k).

        Returns:
            List of (episode_index, overlap) tuples, sorted by overlap descending.
        """
        if top_k is None:
            top_k = self.retrieval_top_k

        if not self.episodes:
            return []

        query_active = np.where(query_sdr > 0)[0]
        if len(query_active) == 0:
            return []

        # Compute overlaps using sparse intersection
        overlaps = []
        for i, ep_active in enumerate(self.episodes):
            overlap = len(np.intersect1d(ep_active, query_active, assume_unique=True))
            if overlap > 0:
                # Apply age decay: older episodes have weaker signal
                age = self._step - self.episode_ages[i]
                decay = max(1, self.decay_rate * age // 1000)
                effective_overlap = overlap * 100 // max(1, decay)
                overlaps.append((i, effective_overlap))

        if not overlaps:
            return []

        # Sort by overlap (descending) and return top-k
        overlaps.sort(key=lambda x: x[1], reverse=True)
        return overlaps[:top_k]

    def compute_field(
        self,
        query_sdr: np.ndarray,
        scale: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute external field from retrieved episodes.

        The field is a weighted combination of the top-k retrieved episodes.
        Each episode contributes proportionally to its overlap with the query.

        This is the "reinstatement" mechanism: retrieved episodic memories
        inject their content as external field into the DAM dynamics,
        biasing the attractor landscape toward coherent continuations.

        Args:
            query_sdr: Query SDR (D,) uint8.
            scale: Field scale (default: self.field_scale).

        Returns:
            External field vector (D,) int32.
        """
        if scale is None:
            scale = self.field_scale

        field = np.zeros(self.D, dtype=np.int32)

        if not self.episodes:
            return field

        retrieved = self.retrieve(query_sdr)
        if not retrieved:
            # No good matches — use the aggregate field from episode_counts
            # This provides a "priors" field based on what's been stored
            query_active = np.where(query_sdr > 0)[0]
            if len(query_active) > 0:
                # Boost dimensions that co-occur with query in stored episodes
                for d in query_active:
                    # Neighbors of d in stored episodes
                    pass  # Skip for now — use simple count-based field

            # Simple fallback: use episode_counts as a frequency-based field
            field = self.episode_counts.astype(np.int32) * scale // max(1, len(self.episodes))
            return field

        # Weighted combination of retrieved episodes
        total_weight = 0
        for ep_idx, overlap in retrieved:
            weight = overlap  # Overlap as weight
            total_weight += weight
            ep_active = self.episodes[ep_idx]
            for d in ep_active:
                field[d] += weight * scale

        # Normalize by total weight
        if total_weight > 0:
            field = field * 100 // total_weight

        return field

    def compute_word_episodic_energy(
        self,
        candidate_words: np.ndarray,
        sdr_encoder,
        scale: int = 500,
    ) -> np.ndarray:
        """
        Compute episodic energy contribution for candidate words.

        Words whose SDRs overlap with recently stored episodes get
        lower energy (more likely). This provides long-range coherence.

        Uses the episode_counts for efficient batch computation:
        each candidate's energy is based on how many of its active
        bits appear in stored episodes.

        Args:
            candidate_words: Array of candidate word IDs.
            sdr_encoder: SDR encoder.
            scale: Energy scale.

        Returns:
            Energy array (n_candidates,) int64. Lower = more likely.
        """
        n_cand = len(candidate_words)
        energies = np.zeros(n_cand, dtype=np.int64)

        if not self.episodes:
            return energies

        # For each candidate, compute overlap with episode_counts
        for i, w in enumerate(candidate_words):
            w = int(w)
            if w < 0 or w >= sdr_encoder.vocab_size:
                energies[i] = scale  # High energy for OOV
                continue

            # Sum of episode_counts at candidate's active positions
            active_bits = sdr_encoder.word_active_bits[w]
            if len(active_bits) > 0:
                # Higher count = more co-occurrence with stored episodes = lower energy
                count_sum = int(np.sum(self.episode_counts[active_bits]))
                energies[i] = -count_sum * scale // max(1, len(self.episodes) * self.k)

        return energies

    def reset(self) -> None:
        """Reset episodic memory for a new document."""
        self.episodes = []
        self.episode_ages = []
        self.episode_counts = np.zeros(self.D, dtype=np.int32)
        self._step = 0

    def get_diagnostics(self) -> dict:
        """Return diagnostics."""
        return {
            'D': self.D,
            'k': self.k,
            'n_episodes': len(self.episodes),
            'max_episodes': self.max_episodes,
            'memory_kb': len(self.episodes) * self.k * 2 / 1024,
        }
