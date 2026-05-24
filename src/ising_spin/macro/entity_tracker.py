"""
Entity Tracker — Persistent macro-spins for entity coherence.

In the Ising spin glass, this implements SLOW spins with energy barriers.
Unlike the reservoir (exponential decay, no barrier), entity spins persist
indefinitely until explicitly "flipped" by a new entity mention.

Architecture:
  - N_SLOTS entity slots (default 8), each with:
      entity_hash: int32 hash of the entity name
      activation: int32 energy level (MAX_ACT on mention, decays by DECAY_RATE/token)
      last_seen: int32 position of last mention
      gender: int8 (0=unknown, 1=masc, 2=fem, 3=neuter, 4=plural)
  - entity_word_affinity: (N_BUCKETS, vocab_size) int16 matrix
      Learned from training: for each word within WINDOW tokens of an entity,
      increment the affinity.  Normalized by entity count in Q8.
  - Pronoun resolution: map pronouns to most active entity with matching gender.
      Refreshes entity activation without requiring explicit name mention.

Energy contribution:
  E_entity(w) = -sum_active_slots(
      (activation * affinity[entity_bucket, w]) >> SHIFT
  ) * entity_scale / MAX_ACTIVATION

  Only ACTIVE entities contribute (activation > THRESHOLD).
  This creates a LONG-RANGE BIAS: "she" is selected because Lily's entity
  spin is still active 50+ tokens after her last mention.

Memory budget (V=2000, TinyStories):
  - entity_word_affinity: 128 × 2000 × 2 bytes = 512 KB
  - Entity slots: 8 × 4 fields × 4 bytes = 128 bytes
  - Pronoun maps: ~200 bytes
  Total: ~513 KB (negligible on Pi 5)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# --- Pronoun-gender mapping for entity resolution ---
# Maps pronouns to (gender, number) tuples
# gender: 0=unknown, 1=masc, 2=fem, 3=neuter, 4=plural
# number: 0=unknown, 1=singular, 2=plural
PRONOUN_GENDER = {
    # Subject pronouns
    "he":   (1, 1),  # masculine singular
    "she":  (2, 1),  # feminine singular
    "it":   (3, 1),  # neuter singular
    "they": (4, 2),  # plural (also singular they, but treat as plural)
    # Object pronouns
    "him":  (1, 1),
    "her":  (2, 1),
    "it":   (3, 1),
    "them": (4, 2),
    # Possessive pronouns
    "his":  (1, 1),
    "her":  (2, 1),  # same as object — ambiguous but consistent
    "its":  (3, 1),
    "their": (4, 2),
    # Possessive determiners
    "my":   (0, 1),
    "your": (0, 2),  # can be singular or plural
    "our":  (4, 2),
    # Reflexive
    "himself": (1, 1),
    "herself": (2, 1),
    "itself":  (3, 1),
    "themselves": (4, 2),
}

# Common proper nouns by gender (for initial gender assignment)
# This is a small heuristic set — the main gender detection is from
# co-occurrence with gendered pronouns in training data
FEMALE_NAMES = frozenset({
    "lily", "emma", "olivia", "ava", "sophia", "isabella", "mia",
    "charlotte", "amelia", "harper", "eve", "anna", "bella", "clara",
    "daisy", "ella", "fiona", "grace", "hannah", "iris", "julia",
    "kate", "luna", "mary", "nora", "olive", "penny", "rose", "sara",
    "lucy", "alice", "sarah", "molly", "zoe", "lily", "stella",
})
MALE_NAMES = frozenset({
    "jack", "oliver", "liam", "noah", "william", "james", "benjamin",
    "lucas", "henry", "alexander", "daniel", "michael", "ethan",
    "david", "joseph", "samuel", "ryan", "nathan", "leo", "max",
    "tom", "tim", "bob", "sam", "ben", "max", "finn", "oscar",
    "george", "arthur", "harry", "charlie", "theo", "owen",
})


class EntityTracker:
    """
    Persistent macro-spins for entity coherence.

    Entity slots maintain activation with energy barriers — they persist
    across the entire document, not just the last few tokens.  When a
    pronoun resolves to an entity, the activation is refreshed, creating
    a feedback loop that maintains entity coherence over 400+ tokens.

    All arithmetic is integer-only.
    """

    # Activation dynamics (all integers)
    MAX_ACTIVATION = 1000       # Maximum entity activation energy
    DECAY_RATE = 2              # Activation decay per token
    THRESHOLD = 50              # Minimum activation to be "active"
    PRONOUN_REFRESH = 800       # Activation refresh on pronoun resolution
    MENTION_SET = 1000          # Activation on direct entity mention

    # Entity-word affinity
    N_BUCKETS = 128             # Hash buckets for entity-word affinity
    AFFINITY_WINDOW = 20        # Tokens within which word co-occurs with entity
    AFFINITY_Q = 8              # Q8 fixed-point for affinity normalization

    # Number of entity slots
    N_SLOTS = 8

    # Gender constants
    GENDER_UNKNOWN = 0
    GENDER_MASC = 1
    GENDER_FEM = 2
    GENDER_NEUTER = 3
    GENDER_PLURAL = 4

    def __init__(
        self,
        vocab_size: int,
        n_slots: int = 8,
        n_buckets: int = 128,
        max_activation: int = 1000,
        decay_rate: int = 2,
        threshold: int = 50,
        entity_scale: int = 800,
        idx2word: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize EntityTracker.

        Args:
            vocab_size: Vocabulary size V.
            n_slots: Number of entity slots (default 8).
            n_buckets: Number of hash buckets for affinity matrix (default 128).
            max_activation: Maximum entity activation (default 1000).
            decay_rate: Activation decay per token (default 2).
            threshold: Minimum activation to be "active" (default 50).
            entity_scale: Energy scale for entity coupling (default 800).
            idx2word: Optional mapping from word ID to word string.
        """
        self.vocab_size = vocab_size
        self.n_slots = n_slots
        self.n_buckets = n_buckets
        self.max_activation = max_activation
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.entity_scale = entity_scale
        self.idx2word = idx2word

        # Entity slots: parallel arrays for vectorized operations
        self.entity_hash = np.zeros(n_slots, dtype=np.int32)    # Hash of entity name
        self.activation = np.zeros(n_slots, dtype=np.int32)     # Current activation energy
        self.last_seen = np.zeros(n_slots, dtype=np.int32)      # Position of last mention
        self.gender = np.zeros(n_slots, dtype=np.int8)          # Gender: 0-4
        self.slot_used = np.zeros(n_slots, dtype=bool)          # Whether slot is occupied

        # Entity-word affinity matrix: (N_BUCKETS, V) int16
        # affinity[bucket, w] = Q8 * (count of w within WINDOW tokens of entity in bucket)
        self.affinity: Optional[np.ndarray] = None
        # Count of entities per bucket (for normalization)
        self._bucket_counts: Optional[np.ndarray] = None

        # Current position counter
        self._position: int = 0

        # Whether affinity matrix has been built
        self._built = False

        # Entity name → slot index mapping for quick lookup
        self._entity_name_to_slot: Dict[int, int] = {}

        # Known entity names (lowercase) — populated during training from capitalized words
        self.known_entities: set = set()

        # Word ID → entity name mapping (for generation-time detection)
        self._word_to_entity: Dict[int, str] = {}

        # Diagnostics
        self._stats = {
            'entity_mentions': 0,
            'pronoun_resolutions': 0,
            'active_entities_avg': 0.0,
            'total_positions': 0,
        }

    def reset(self) -> None:
        """Reset all entity slots for a new document."""
        self.entity_hash.fill(0)
        self.activation.fill(0)
        self.last_seen.fill(0)
        self.gender.fill(0)
        self.slot_used.fill(False)
        self._position = 0
        self._entity_name_to_slot.clear()
        self._stats = {
            'entity_mentions': 0,
            'pronoun_resolutions': 0,
            'active_entities_avg': 0.0,
            'total_positions': 0,
        }

    def _hash_entity(self, name: str) -> int:
        """Hash an entity name to a positive integer fitting in int32."""
        h = 0
        for ch in name:
            h = (h * 31 + ord(ch)) & 0x7FFFFFFF  # Keep within int31 (positive int32)
        return h

    def _entity_bucket(self, entity_hash: int) -> int:
        """Map entity hash to affinity bucket (0..N_BUCKETS-1)."""
        return entity_hash % self.n_buckets

    def _extract_known_entities(self, raw_texts: List[str]) -> None:
        """
        Extract known entity names from raw (pre-tokenization) training text.

        Looks for capitalized words that are NOT sentence starters.
        These are likely proper nouns (character names, place names).

        Args:
            raw_texts: List of original text strings from the training corpus.
        """
        import re

        # Sentence-starting words to exclude
        sentence_starters_lower = {
            "the", "a", "an", "this", "that", "these", "those",
            "it", "he", "she", "we", "they", "there", "here",
            "when", "where", "how", "why", "what", "which",
            "if", "but", "and", "or", "so", "yet", "for",
            "in", "on", "at", "to", "from", "with", "by",
            "as", "not", "no", "each", "every", "all",
            "both", "neither", "either", "some", "any",
            "my", "your", "his", "her", "its", "our", "their",
            "is", "are", "was", "were", "has", "have", "had",
            "do", "does", "did", "can", "could", "will", "would",
            "should", "may", "might", "must", "shall",
            "one", "two", "three", "once", "then", "now",
            "just", "very", "really", "also", "still",
        }

        # Count capitalized words that appear mid-sentence
        entity_counts: Dict[str, int] = {}
        for text in raw_texts:
            if not text:
                continue
            # Split into sentences (rough heuristic)
            sentences = re.split(r'[.!?]\s*', text)
            for sentence in sentences:
                words = sentence.strip().split()
                if len(words) < 2:
                    continue
                # Skip the first word of each sentence (likely capitalized for grammar)
                for w in words[1:]:
                    # Check if word is capitalized and not a common starter
                    clean = w.strip('.,;:!?\'"()-')
                    if (len(clean) > 1
                            and clean[0].isupper()
                            and clean.lower() not in sentence_starters_lower
                            and clean.isalpha()):
                        lower = clean.lower()
                        entity_counts[lower] = entity_counts.get(lower, 0) + 1

        # Keep entities that appear more than once (likely real names)
        # Also always include known male/female names
        for name in FEMALE_NAMES | MALE_NAMES:
            entity_counts[name] = entity_counts.get(name, 0) + 1

        # Filter: must appear at least 2 times
        self.known_entities = {
            name for name, count in entity_counts.items() if count >= 2
        }

        print(f"    Known entities: {len(self.known_entities)} names "
              f"(top: {sorted(entity_counts, key=entity_counts.get, reverse=True)[:10]})")

    def _infer_gender(self, name: str) -> int:
        """Infer gender from entity name using heuristics."""
        lower = name.lower()
        if lower in FEMALE_NAMES:
            return self.GENDER_FEM
        if lower in MALE_NAMES:
            return self.GENDER_MASC
        return self.GENDER_UNKNOWN

    def _find_slot(self, entity_hash: int) -> int:
        """Find slot for an entity. Returns slot index or -1 if no slot available."""
        # Check if entity already has a slot
        if entity_hash in self._entity_name_to_slot:
            return self._entity_name_to_slot[entity_hash]

        # Find an empty slot
        for i in range(self.n_slots):
            if not self.slot_used[i]:
                return i

        # All slots full: evict the one with lowest activation
        min_idx = int(np.argmin(self.activation))
        # Remove old entity from name mapping
        old_hash = int(self.entity_hash[min_idx])
        if old_hash in self._entity_name_to_slot:
            del self._entity_name_to_slot[old_hash]
        return min_idx

    def activate_entity(self, name: str, position: Optional[int] = None) -> int:
        """
        Activate an entity by name (direct mention).

        Sets activation to MAX_ACTIVATION and refreshes the slot.
        This is called when a proper noun is generated.

        Args:
            name: Entity name string (e.g., "Lily").
            position: Current token position (default: self._position).

        Returns:
            Slot index of the activated entity.
        """
        if position is None:
            position = self._position

        entity_hash = self._hash_entity(name)
        slot = self._find_slot(entity_hash)

        # Activate the slot
        self.entity_hash[slot] = entity_hash
        self.activation[slot] = self.max_activation
        self.last_seen[slot] = position
        self.slot_used[slot] = True

        # Set gender if not already set or if we can infer it
        if self.gender[slot] == self.GENDER_UNKNOWN:
            self.gender[slot] = self._infer_gender(name)

        # Update name mapping
        self._entity_name_to_slot[entity_hash] = slot

        self._stats['entity_mentions'] += 1

        return slot

    def resolve_pronoun(self, pronoun: str, position: Optional[int] = None) -> Optional[int]:
        """
        Resolve a pronoun to the most likely active entity.

        Uses gender matching and activation level to find the best
        entity slot.  If found, refreshes the entity's activation.

        Args:
            pronoun: The pronoun string (e.g., "she", "his").
            position: Current token position.

        Returns:
            Slot index of the resolved entity, or None if no match.
        """
        if position is None:
            position = self._position

        gender_info = PRONOUN_GENDER.get(pronoun.lower())
        if gender_info is None:
            return None

        target_gender, target_number = gender_info

        # Find active entities with matching gender
        best_slot = None
        best_activation = -1

        for i in range(self.n_slots):
            if not self.slot_used[i] or self.activation[i] < self.threshold:
                continue

            # Gender matching: if target gender is specified,
            # prefer matching gender but allow unknown
            slot_gender = int(self.gender[i])
            gender_match = (
                slot_gender == target_gender
                or slot_gender == self.GENDER_UNKNOWN
                or target_gender == self.GENDER_UNKNOWN
            )

            if not gender_match:
                continue

            if self.activation[i] > best_activation:
                best_activation = int(self.activation[i])
                best_slot = i

        if best_slot is not None:
            # Refresh activation on pronoun resolution
            self.activation[best_slot] = max(
                int(self.activation[best_slot]),
                self.PRONOUN_REFRESH
            )
            self.last_seen[best_slot] = position
            self._stats['pronoun_resolutions'] += 1

        return best_slot

    def step(self) -> None:
        """
        Advance one token: decay all entity activations.

        This is the SPIN GLASS DYNAMICS of the macro-spin layer.
        Unlike the reservoir (exponential decay), entity activation
        decays LINEARLY and has an energy barrier at THRESHOLD.

        The decay rate is deliberately slow: at DECAY_RATE=2 per token,
        an entity at MAX_ACTIVATION=1000 takes 475 tokens to reach
        THRESHOLD=50.  This gives correlation length ξ >> 400 tokens.
        """
        self._position += 1

        # Decay active entities
        active_mask = self.slot_used & (self.activation > 0)
        self.activation[active_mask] -= self.decay_rate

        # Free slots below threshold
        expired = self.slot_used & (self.activation < self.threshold)
        for i in np.where(expired)[0]:
            old_hash = int(self.entity_hash[i])
            if old_hash in self._entity_name_to_slot:
                del self._entity_name_to_slot[old_hash]
            self.slot_used[i] = False
            self.activation[i] = 0
            self.entity_hash[i] = 0
            self.gender[i] = 0

        # Track average active entities
        n_active = int(np.sum(self.slot_used & (self.activation >= self.threshold)))
        self._stats['active_entities_avg'] = (
            self._stats['active_entities_avg'] * self._stats['total_positions'] + n_active
        ) / max(1, self._stats['total_positions'] + 1)
        self._stats['total_positions'] += 1

    def update(self, word_id: int, word_str: Optional[str] = None) -> None:
        """
        Update entity state based on the current word.

        Detection strategy:
        1. If the word is in the known_entities set → activate entity
        2. If the word is a pronoun → resolve and refresh entity
        3. If the word is capitalized and not a sentence starter → activate entity

        Args:
            word_id: Integer token ID.
            word_str: Optional string form of the word.
        """
        # Advance position and decay
        self.step()

        if word_str is None:
            return

        w = word_str.strip()
        if not w:
            return

        w_lower = w.lower()

        # Strategy 1: Check known entities (populated during training)
        if w_lower in self.known_entities:
            self.activate_entity(w_lower, position=self._position - 1)
            return

        # Also check word_id mapping
        if word_id >= 0 and word_id in self._word_to_entity:
            entity_name = self._word_to_entity[word_id]
            self.activate_entity(entity_name, position=self._position - 1)
            return

        # Strategy 2: Check if this is a pronoun — resolve and refresh entity
        if w_lower in PRONOUN_GENDER:
            self.resolve_pronoun(w_lower)
            return

        # Strategy 3: Check if this is a proper noun (capitalized, not at sentence start)
        sentence_starters = {
            "The", "A", "An", "This", "That", "These", "Those",
            "It", "He", "She", "We", "They", "There", "Here",
            "When", "Where", "How", "Why", "What", "Which",
            "If", "But", "And", "Or", "So", "Yet", "For",
            "In", "On", "At", "To", "From", "With", "By",
            "As", "Not", "No", "Each", "Every", "All",
            "Both", "Neither", "Either", "Some", "Any",
            "My", "Your", "His", "Her", "Its", "Our", "Their",
            "Is", "Are", "Was", "Were", "Has", "Have", "Had",
            "Do", "Does", "Did", "Can", "Could", "Will", "Would",
            "Should", "May", "Might", "Must", "Shall",
        }

        if len(w) > 1 and w[0].isupper() and w not in sentence_starters:
            # Likely a proper noun — activate entity
            self.activate_entity(w_lower, position=self._position - 1)

    # ===================================================================
    # BUILD: Learn entity-word affinity from training data
    # ===================================================================

    def build(
        self,
        sequences: List[List[int]],
        idx2word: Optional[Dict[int, str]] = None,
        raw_texts: Optional[List[str]] = None,
    ) -> "EntityTracker":
        """
        Build entity-word affinity matrix from training data.

        For each entity mention in the training data, we track all words
        that appear within AFFINITY_WINDOW tokens of that entity.  The
        affinity matrix stores normalized co-occurrence counts:

            affinity[bucket, w] = Q8 * sum(entity_counts[bucket, w]) / entity_total[bucket]

        where bucket = entity_hash % N_BUCKETS.

        Also builds the known_entities set from capitalized words in the
        raw training text (before tokenization lowercases everything).
        This allows entity detection during generation even though the
        vocabulary stores lowercase words.

        All integer arithmetic.

        Args:
            sequences: List of training sequences (word ID lists).
            idx2word: Mapping from word ID to word string.
            raw_texts: Optional list of original (pre-tokenization) text strings.
                Used to extract entity names from capitalized words.

        Returns:
            self
        """
        if idx2word is not None:
            self.idx2word = idx2word

        # Build known_entities from raw text (capitalized words)
        if raw_texts is not None:
            self._extract_known_entities(raw_texts)

        # Also build word_id → entity mapping from idx2word
        if idx2word is not None:
            for word_id, word_str in idx2word.items():
                if isinstance(word_id, int) and word_str:
                    w_lower = word_str.lower()
                    if w_lower in self.known_entities:
                        self._word_to_entity[word_id] = w_lower

        V = self.vocab_size
        B = self.n_buckets
        W = self.AFFINITY_WINDOW

        # Accumulate co-occurrence counts
        # cooc[bucket, word_id] += 1 for each word within W tokens of entity in bucket
        cooc = np.zeros((B, V), dtype=np.int32)
        bucket_totals = np.zeros(B, dtype=np.int32)

        # Also accumulate gender information: for each entity, track
        # how often gendered pronouns follow it
        gender_cooc = np.zeros((B, 5), dtype=np.int32)  # 5 gender categories

        n_entities = 0
        n_sequences = len(sequences)

        for seq_idx, seq in enumerate(sequences):
            if len(seq) < 3:
                continue

            # Track recent entity mentions: list of (entity_hash, position)
            recent_entities: List[Tuple[int, int]] = []

            for pos, word_id in enumerate(seq):
                if word_id < 0 or word_id >= V:
                    continue

                # Get word string
                word_str = None
                if self.idx2word is not None:
                    word_str = self.idx2word.get(word_id)
                if word_str is None:
                    continue

                w = word_str.strip()
                if not w:
                    continue

                # Check if this is a proper noun
                sentence_starters = {
                    "The", "A", "An", "This", "That", "It", "He", "She",
                    "We", "They", "There", "Here", "When", "Where", "How",
                    "Why", "What", "Which", "If", "But", "And", "Or", "So",
                    "In", "On", "At", "To", "From", "With", "By", "As",
                }
                is_proper = (
                    len(w) > 1
                    and w[0].isupper()
                    and w not in sentence_starters
                )

                if is_proper:
                    entity_hash = self._hash_entity(w)
                    bucket = self._entity_bucket(entity_hash)
                    recent_entities.append((entity_hash, pos))

                    # Infer gender from name
                    gender = self._infer_gender(w)
                    if gender != self.GENDER_UNKNOWN:
                        gender_cooc[bucket, gender] += 1

                    n_entities += 1

                # Check if this is a gendered pronoun
                pronoun_info = PRONOUN_GENDER.get(w.lower())
                if pronoun_info is not None:
                    # Find most recent entity (within W tokens)
                    for eh, ep in reversed(recent_entities):
                        if pos - ep <= W:
                            bucket = self._entity_bucket(eh)
                            pronoun_gender = pronoun_info[0]
                            gender_cooc[bucket, pronoun_gender] += 1
                            break

                # Accumulate word-entity co-occurrence
                # For each recent entity, this word is within W tokens
                for eh, ep in recent_entities:
                    if pos - ep > W:
                        continue
                    if word_id == ep:  # Don't count the entity itself
                        continue
                    bucket = self._entity_bucket(eh)
                    cooc[bucket, word_id] += 1
                    bucket_totals[bucket] += 1

            # Progress reporting
            if (seq_idx + 1) % 50000 == 0:
                print(f"    EntityTracker.build(): {seq_idx+1}/{n_sequences} sequences, "
                      f"{n_entities} entities")

        # Normalize affinity matrix: Q8 * count / total
        # affinity[bucket, w] = Q8 * cooc[bucket, w] / max(1, bucket_totals[bucket])
        self.affinity = np.zeros((B, V), dtype=np.int16)
        for b in range(B):
            total = max(1, int(bucket_totals[b]))
            # Vectorized normalization
            normalized = (cooc[b].astype(np.int64) * (1 << self.AFFINITY_Q)) // total
            self.affinity[b] = np.clip(normalized, -32768, 32767).astype(np.int16)

        self._bucket_counts = bucket_totals
        self._built = True

        n_nonzero_buckets = int(np.sum(bucket_totals > 0))
        mem_mb = self.affinity.nbytes / (1024 * 1024)
        print(f"    EntityTracker.build(): {n_entities} entity mentions, "
              f"{n_nonzero_buckets}/{B} active buckets, memory={mem_mb:.2f} MB")

        return self

    # ===================================================================
    # ENERGY: Compute entity macro-spin energy for candidate words
    # ===================================================================

    def compute_energy(
        self,
        candidate_words: np.ndarray,
        entity_scale: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute entity macro-spin energy for candidate words.

        E_entity(w) = -sum_active_slots(
            (activation * affinity[entity_bucket, w]) >> SHIFT
        ) * entity_scale / MAX_ACTIVATION

        Only ACTIVE entities (activation >= THRESHOLD) contribute.
        Words affiliated with active entities get LOWER energy (more likely).
        Words NOT affiliated with any active entity get E=0 (no bias).

        This is the CROSS-SCALE COUPLING: macro-spin state (entity
        activation) biases micro-spin selection (word choice).

        Args:
            candidate_words: Array of candidate word IDs, shape (n_candidates,).
            entity_scale: Override energy scale (default: self.entity_scale).

        Returns:
            np.ndarray of int64 energies, shape (n_candidates,).
            LOWER = more likely (entity-affiliated words preferred).
        """
        n_candidates = len(candidate_words)
        if not self._built or self.affinity is None:
            return np.zeros(n_candidates, dtype=np.int64)

        scale = entity_scale if entity_scale is not None else self.entity_scale

        # Find active entity slots
        active_mask = self.slot_used & (self.activation >= self.threshold)
        active_slots = np.where(active_mask)[0]

        if len(active_slots) == 0:
            return np.zeros(n_candidates, dtype=np.int64)

        # Compute weighted affinity sum for each candidate
        # For each active entity slot i:
        #   bucket = entity_hash[i] % N_BUCKETS
        #   weight = activation[i]
        #   contribution = weight * affinity[bucket, candidate_words]
        total_energy = np.zeros(n_candidates, dtype=np.int64)

        for slot_idx in active_slots:
            entity_hash_val = int(self.entity_hash[slot_idx])
            bucket = self._entity_bucket(entity_hash_val)
            act = int(self.activation[slot_idx])

            # Look up affinity for all candidates at once
            safe_candidates = np.clip(candidate_words, 0, self.vocab_size - 1)
            aff = self.affinity[bucket, safe_candidates].astype(np.int64)  # (n,)

            # Weighted contribution: activation * affinity
            # affinity is in Q8, activation is in [0, MAX_ACTIVATION]
            # We want: (act * aff) >> AFFINITY_Q to get back to integer scale
            weighted = (act * aff) >> self.AFFINITY_Q  # (n,) int64

            total_energy += weighted

        # Final energy: negative of weighted sum (lower energy = more likely)
        # Scale by entity_scale / MAX_ACTIVATION
        # This normalizes the energy to the desired scale
        if self.max_activation > 0:
            total_energy = (total_energy * scale) // self.max_activation

        # NEGATE: higher affinity → lower energy (more likely)
        energies = -total_energy

        return energies

    @property
    def built(self) -> bool:
        """Whether the affinity matrix has been built."""
        return self._built

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information about entity state."""
        active = []
        for i in range(self.n_slots):
            if self.slot_used[i] and self.activation[i] >= self.threshold:
                active.append({
                    'slot': i,
                    'activation': int(self.activation[i]),
                    'last_seen': int(self.last_seen[i]),
                    'gender': int(self.gender[i]),
                })

        return {
            'n_active_entities': len(active),
            'active_entities': active,
            'position': self._position,
            'stats': self._stats.copy(),
        }
