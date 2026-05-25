"""
Scene Tracker — Macro-spin for scene/location coherence.

In the Ising spin glass, this implements persistent macro-spins for
scene/location tracking.  Like entity spins, scene spins persist
across the entire document with energy barriers preventing random
flipping.

Architecture:
  - N_SCENE_SLOTS scene slots (default 4), each with:
      scene_hash: int32 hash of the scene/location name
      activation: int32 energy level
      last_seen: int32 position of last mention
  - scene_word_affinity: (N_BUCKETS, vocab_size) int16 matrix
  - Scene vocabulary: when "park" is active, words like "swings",
    "grass", "slide", "playground" get energy bonuses.

Energy contribution:
  E_scene(w) = -sum_active_slots(
      (activation * affinity[scene_bucket, w]) >> SHIFT
  ) * scene_scale / MAX_ACTIVATION

Memory budget (V=2000):
  - scene_word_affinity: 64 × 2000 × 2 bytes = 256 KB
  - Scene slots: negligible
  Total: ~256 KB
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# --- Location/scene keywords ---
# Maps common location words to scene categories
SCENE_KEYWORDS = {
    # Outdoor nature
    "forest": "nature", "woods": "nature", "garden": "nature",
    "field": "nature", "meadow": "nature", "mountain": "nature",
    "river": "nature", "lake": "nature", "ocean": "nature",
    "beach": "nature", "sea": "nature", "hill": "nature",

    # Park/playground
    "park": "park", "playground": "park", "swings": "park",
    "slide": "park", "sandbox": "park",

    # Home/indoor
    "house": "home", "home": "home", "room": "home", "kitchen": "home",
    "bedroom": "home", "bathroom": "home", "bed": "home", "door": "home",
    "window": "home", "table": "home", "chair": "home",

    # School
    "school": "school", "classroom": "school", "teacher": "school",
    "student": "school", "desk": "school", "book": "school",
    "library": "school", "lesson": "school",

    # City/town
    "city": "city", "town": "city", "street": "city", "road": "city",
    "store": "city", "shop": "city", "market": "city", "restaurant": "city",

    # Animal/farm
    "farm": "farm", "barn": "farm", "stable": "farm",
    "chicken": "farm", "cow": "farm", "horse": "farm", "pig": "farm",
}

# Prepositions that often introduce locations
LOCATION_PREPOSITIONS = frozenset({
    "in", "at", "on", "to", "into", "through", "across",
    "near", "by", "around", "from",
})


class SceneTracker:
    """
    Persistent macro-spins for scene/location coherence.

    Scene slots maintain activation with energy barriers, similar to
    entity slots.  When a scene is established (e.g., "went to the park"),
    the scene spin activates and persists, biasing subsequent word
    selection toward scene-appropriate vocabulary.

    All arithmetic is integer-only.
    """

    # Activation dynamics
    MAX_ACTIVATION = 800
    DECAY_RATE = 1       # Very slow decay — scenes persist longer than entities
    THRESHOLD = 30
    SCENE_MENTION_SET = 800
    SCENE_REFRESH = 600

    # Affinity parameters
    N_BUCKETS = 64
    AFFINITY_WINDOW = 25   # Words within 25 tokens of scene mention
    AFFINITY_Q = 8

    N_SCENE_SLOTS = 4

    def __init__(
        self,
        vocab_size: int,
        n_slots: int = 4,
        n_buckets: int = 64,
        scene_scale: int = 400,
        idx2word: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize SceneTracker.

        Args:
            vocab_size: Vocabulary size V.
            n_slots: Number of scene slots (default 4).
            n_buckets: Hash buckets for affinity (default 64).
            scene_scale: Energy scale for scene coupling (default 400).
            idx2word: Mapping from word ID to word string.
        """
        self.vocab_size = vocab_size
        self.n_slots = n_slots
        self.n_buckets = n_buckets
        self.scene_scale = scene_scale
        self.idx2word = idx2word

        # Scene slots
        self.scene_hash = np.zeros(n_slots, dtype=np.int32)
        self.activation = np.zeros(n_slots, dtype=np.int32)
        self.last_seen = np.zeros(n_slots, dtype=np.int32)
        self.slot_used = np.zeros(n_slots, dtype=bool)

        # Scene-word affinity matrix: (N_BUCKETS, V) int16
        self.affinity: Optional[np.ndarray] = None
        self._bucket_counts: Optional[np.ndarray] = None

        # Current position
        self._position: int = 0

        # Whether affinity has been built
        self._built = False

        # Scene name → slot mapping
        self._scene_to_slot: Dict[int, int] = {}

        # Track recent prepositions for scene detection
        self._prev_was_prep: bool = False

        # Diagnostics
        self._stats = {
            'scene_activations': 0,
            'active_scenes_avg': 0.0,
            'total_positions': 0,
        }

    def reset(self) -> None:
        """Reset scene state for a new document."""
        self.scene_hash.fill(0)
        self.activation.fill(0)
        self.last_seen.fill(0)
        self.slot_used.fill(False)
        self._position = 0
        self._scene_to_slot.clear()
        self._prev_was_prep = False
        self._stats = {
            'scene_activations': 0,
            'active_scenes_avg': 0.0,
            'total_positions': 0,
        }

    def _hash_scene(self, name: str) -> int:
        """Hash a scene name to a positive integer fitting in int32."""
        h = 0
        for ch in name:
            h = (h * 37 + ord(ch)) & 0x7FFFFFFF  # Keep within int31 (positive int32)
        return h

    def _scene_bucket(self, scene_hash: int) -> int:
        """Map scene hash to affinity bucket."""
        return scene_hash % self.n_buckets

    def _is_location_word(self, word_str: str) -> bool:
        """Check if a word is a location/scene keyword."""
        return word_str.lower() in SCENE_KEYWORDS

    def _find_slot(self, scene_hash: int) -> int:
        """Find slot for a scene, evicting if necessary."""
        if scene_hash in self._scene_to_slot:
            return self._scene_to_slot[scene_hash]

        for i in range(self.n_slots):
            if not self.slot_used[i]:
                return i

        # Evict lowest activation
        min_idx = int(np.argmin(self.activation))
        old_hash = int(self.scene_hash[min_idx])
        if old_hash in self._scene_to_slot:
            del self._scene_to_slot[old_hash]
        return min_idx

    def activate_scene(self, name: str, position: Optional[int] = None) -> int:
        """
        Activate a scene by name.

        Args:
            name: Scene name string.
            position: Current token position.

        Returns:
            Slot index.
        """
        if position is None:
            position = self._position

        scene_hash = self._hash_scene(name)
        slot = self._find_slot(scene_hash)

        self.scene_hash[slot] = scene_hash
        self.activation[slot] = self.MAX_ACTIVATION
        self.last_seen[slot] = position
        self.slot_used[slot] = True

        self._scene_to_slot[scene_hash] = slot
        self._stats['scene_activations'] += 1

        return slot

    def update(self, word_id: int, word_str: Optional[str] = None) -> None:
        """
        Update scene state based on the current word.

        Detects scene/location mentions and activates scene spins.
        Also refreshes scene activation when location-appropriate
        vocabulary appears.

        Args:
            word_id: Integer token ID.
            word_str: Optional string form of the word.
        """
        self._position += 1

        # Decay active scenes
        active_mask = self.slot_used & (self.activation > 0)
        self.activation[active_mask] -= self.DECAY_RATE

        # Free expired slots
        expired = self.slot_used & (self.activation < self.THRESHOLD)
        for i in np.where(expired)[0]:
            old_hash = int(self.scene_hash[i])
            if old_hash in self._scene_to_slot:
                del self._scene_to_slot[old_hash]
            self.slot_used[i] = False
            self.activation[i] = 0

        if word_str is None:
            self._prev_was_prep = False
            return

        w = word_str.strip().lower()

        # Detect location words (possibly preceded by a preposition)
        if self._is_location_word(w):
            self.activate_scene(w, position=self._position - 1)
        elif self._prev_was_prep and len(w) > 2:
            # After a preposition, a noun might be a location
            # Heuristic: capitalize check or common location patterns
            if word_str and word_str[0].isupper() and len(word_str) > 2:
                # Could be a named location
                self.activate_scene(w, position=self._position - 1)

        # Track prepositions for next word
        self._prev_was_prep = w in LOCATION_PREPOSITIONS

        # Update diagnostics
        n_active = int(np.sum(self.slot_used & (self.activation >= self.THRESHOLD)))
        self._stats['active_scenes_avg'] = (
            self._stats['active_scenes_avg'] * self._stats['total_positions'] + n_active
        ) / max(1, self._stats['total_positions'] + 1)
        self._stats['total_positions'] += 1

    # ===================================================================
    # BUILD: Learn scene-word affinity from training data
    # ===================================================================

    def build(
        self,
        sequences: List[List[int]],
        idx2word: Optional[Dict[int, str]] = None,
    ) -> "SceneTracker":
        """
        Build scene-word affinity matrix from training data.

        For each scene keyword in the training data, accumulate all
        words that appear within AFFINITY_WINDOW tokens.

        Args:
            sequences: List of training sequences.
            idx2word: Mapping from word ID to word string.

        Returns:
            self
        """
        if idx2word is not None:
            self.idx2word = idx2word

        V = self.vocab_size
        B = self.n_buckets
        W = self.AFFINITY_WINDOW

        cooc = np.zeros((B, V), dtype=np.int32)
        bucket_totals = np.zeros(B, dtype=np.int32)
        n_scenes = 0

        for seq_idx, seq in enumerate(sequences):
            if len(seq) < 3:
                continue

            recent_scenes: List[Tuple[int, int]] = []

            for pos, word_id in enumerate(seq):
                if word_id < 0 or word_id >= V:
                    continue

                word_str = None
                if self.idx2word is not None:
                    word_str = self.idx2word.get(word_id)
                if word_str is None:
                    continue

                w = word_str.strip().lower()

                # Check if this is a scene keyword
                if self._is_location_word(w):
                    scene_hash = self._hash_scene(w)
                    bucket = self._scene_bucket(scene_hash)
                    recent_scenes.append((scene_hash, pos))
                    n_scenes += 1

                # Accumulate word-scene co-occurrence
                for sh, sp in recent_scenes:
                    if pos - sp > W:
                        continue
                    if word_id == sp:
                        continue
                    bucket = self._scene_bucket(sh)
                    cooc[bucket, word_id] += 1
                    bucket_totals[bucket] += 1

        # Normalize: Q8 * count / total
        self.affinity = np.zeros((B, V), dtype=np.int16)
        for b in range(B):
            total = max(1, int(bucket_totals[b]))
            normalized = (cooc[b].astype(np.int64) * (1 << self.AFFINITY_Q)) // total
            self.affinity[b] = np.clip(normalized, -32768, 32767).astype(np.int16)

        self._bucket_counts = bucket_totals
        self._built = True

        n_nonzero = int(np.sum(bucket_totals > 0))
        mem_kb = self.affinity.nbytes / 1024
        print(f"    SceneTracker.build(): {n_scenes} scene mentions, "
              f"{n_nonzero}/{B} active buckets, memory={mem_kb:.1f} KB")

        return self

    # ===================================================================
    # ENERGY: Compute scene macro-spin energy
    # ===================================================================

    def compute_energy(
        self,
        candidate_words: np.ndarray,
        scene_scale: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute scene macro-spin energy for candidate words.

        E_scene(w) = -sum_active_slots(
            (activation * affinity[scene_bucket, w]) >> SHIFT
        ) * scene_scale / MAX_ACTIVATION

        Args:
            candidate_words: Array of candidate word IDs.
            scene_scale: Override energy scale.

        Returns:
            np.ndarray of int64 energies.
        """
        n_candidates = len(candidate_words)
        if not self._built or self.affinity is None:
            return np.zeros(n_candidates, dtype=np.int64)

        scale = scene_scale if scene_scale is not None else self.scene_scale

        active_mask = self.slot_used & (self.activation >= self.THRESHOLD)
        active_slots = np.where(active_mask)[0]

        if len(active_slots) == 0:
            return np.zeros(n_candidates, dtype=np.int64)

        total_energy = np.zeros(n_candidates, dtype=np.int64)
        safe_candidates = np.clip(candidate_words, 0, self.vocab_size - 1)

        for slot_idx in active_slots:
            scene_hash_val = int(self.scene_hash[slot_idx])
            bucket = self._scene_bucket(scene_hash_val)
            act = int(self.activation[slot_idx])

            aff = self.affinity[bucket, safe_candidates].astype(np.int64)
            weighted = (act * aff) >> self.AFFINITY_Q
            total_energy += weighted

        if self.MAX_ACTIVATION > 0:
            total_energy = (total_energy * scale) // self.MAX_ACTIVATION

        return -total_energy

    @property
    def built(self) -> bool:
        """Whether the affinity matrix has been built."""
        return self._built

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information."""
        active = []
        for i in range(self.n_slots):
            if self.slot_used[i] and self.activation[i] >= self.THRESHOLD:
                active.append({
                    'slot': i,
                    'activation': int(self.activation[i]),
                })

        return {
            'n_active_scenes': len(active),
            'active_scenes': active,
            'position': self._position,
            'stats': self._stats.copy(),
        }
