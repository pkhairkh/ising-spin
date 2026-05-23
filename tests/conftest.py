"""
Shared test fixtures for ISG-LM v18.3 test suite.

Provides:
  - small_vocab:  Vocabulary with ~200 words from 50 short texts
  - small_model:  Fully trained v18 model with reduced parameters
  - vsa_encoder:  VSA encoder with D=128 for the small vocab
  - random_context: Random context of 10 word IDs
"""

import numpy as np
import pytest
from ising_spin.vocabulary import Vocabulary, POSTypeSystem, TopicAssigner
from ising_spin.vocabulary.pos import POS2IDX, N_POS
from ising_spin.vsa import VSAEncoder
from ising_spin.model_v18 import IsingLMModelV18


# ===================================================================
# Synthetic corpus for testing
# ===================================================================

_SYNTHETIC_TEXTS = [
    "the cat sat on the mat and looked at the dog",
    "a dog ran in the park and chased the ball",
    "the bird flew over the house and landed on the tree",
    "she read a book about science and nature",
    "the children played in the garden with the ball",
    "he studied mathematics at the university",
    "the weather was warm and sunny today",
    "they walked along the river and saw the fish",
    "the teacher explained the problem to the students",
    "we visited the museum and saw the paintings",
    "the scientist discovered a new species of plant",
    "the artist painted a picture of the mountain",
    "the engineer designed a bridge over the river",
    "the doctor treated the patient with medicine",
    "the farmer grew vegetables in the field",
    "the musician played a song on the piano",
    "the writer published a novel about history",
    "the chef cooked a meal for the guests",
    "the athlete won the race at the stadium",
    "the pilot flew the plane across the ocean",
    "the cat slept on the sofa all day long",
    "the dog barked at the mailman this morning",
    "the sun rose over the hills and the valley",
    "the rain fell on the roof and the garden",
    "the wind blew through the trees and the house",
    "the snow covered the mountain and the valley",
    "the fire burned in the fireplace all night",
    "the water flowed down the river to the sea",
    "the earth orbits around the sun each year",
    "the moon shone bright in the dark sky",
    "the stars twinkled in the night sky above",
    "the ocean waves crashed on the sandy shore",
    "the forest was full of tall green trees",
    "the desert stretched for miles in the heat",
    "the city was busy with people and cars",
    "the village was quiet and peaceful at night",
    "the mountain was covered in snow and ice",
    "the lake was calm and still in the morning",
    "the island was surrounded by clear blue water",
    "the garden was full of colorful flowers",
    "the library had many books on every subject",
    "the school was closed for the summer holiday",
    "the hospital was busy with patients and staff",
    "the market was full of fresh fruit and fish",
    "the station was crowded with people and trains",
    "the airport was busy with planes and people",
    "the road was long and winding through the hills",
    "the bridge crossed over the wide deep river",
    "the tunnel went through the mountain to the valley",
    "the park was a nice place for a walk",
    "the zoo had many animals from around the world",
    "the farm had cows and sheep and chickens",
    "the factory produced goods for the market",
    "the office was on the top floor of the building",
    "the restaurant served food from many countries",
    "the hotel had a pool and a garden",
    "the theater showed plays and musicals each week",
    "the cinema had new movies every friday night",
    "the beach was warm and sunny in the summer",
    "the forest path led to a small clear stream",
    "the old man sat on the bench in the park",
    "the young girl ran through the field of flowers",
    "the boy kicked the ball into the goal",
    "the woman walked her dog along the beach",
    "the man drove his car to work each day",
    "the baby cried in the middle of the night",
    "the student studied hard for the exam",
    "the teacher wrote on the board with chalk",
    "the doctor prescribed medicine for the cold",
    "the nurse helped the patient get better",
    "the police officer directed traffic at the corner",
    "the firefighter put out the fire in the building",
    "the baker made fresh bread every morning",
    "the butcher sold meat at the market",
    "the mechanic fixed the car in the garage",
    "the carpenter built a table from wood",
    "the plumber repaired the pipe in the kitchen",
    "the electrician installed new lights in the house",
    "the painter painted the walls of the room",
    "the singer performed a song on the stage",
    "the dancer moved gracefully across the floor",
    "the actor played a role in the movie",
    "the director guided the cast through the scene",
    "the author wrote a story about adventure",
    "the poet composed a verse about nature",
    "the journalist reported the news on television",
    "the editor reviewed the article for the paper",
    "the photographer took a picture of the sunset",
    "the scientist conducted an experiment in the lab",
    "the engineer solved the problem with mathematics",
    "the programmer wrote code for the computer",
    "the designer created a new style for the brand",
    "the architect planned a building for the city",
    "the judge made a decision in the court case",
    "the lawyer argued the case before the jury",
    "the soldier marched in the parade on memorial day",
    "the sailor navigated the ship through the storm",
    "the astronaut traveled to space in the rocket",
    "the explorer discovered a cave in the mountain",
    "the historian studied the events of the past",
    "the philosopher pondered the meaning of life",
    "the psychologist analyzed the behavior of the patient",
    "the economist studied the market and the economy",
]


@pytest.fixture(scope="session")
def small_vocab():
    """Vocabulary with ~200 words from 50 short texts."""
    vocab = Vocabulary(min_freq=1, max_size=200)
    vocab.build(_SYNTHETIC_TEXTS[:50])
    return vocab


@pytest.fixture(scope="session")
def small_model(small_vocab):
    """Fully trained v18 model with 200-word vocab on 100 texts."""
    model = IsingLMModelV18(
        vocab_min_freq=1,
        vocab_max_size=200,
        ngram_max_n=3,
        ngram_min_count=1,
        pos_ngram_max_n=5,
        pos_ngram_min_count=1,
        topic_ngram_max_n=5,
        topic_ngram_min_count=1,
        n_topics=8,
        dense_am_dim=32,
        dense_am_degree=2,
        dense_am_hash_dim=16,
        vsa_dimension=64,
        reservoir_dim=32,
        reservoir_alpha_q15=31130,
        rff_dim=32,
        rff_hash_dim=16,
        rff_scale=600,
        recall_scale=1600,
        pos_recall_scale=800,
        topic_recall_scale=400,
        state_scale=400,
        vsa_scale=800,
        dense_am_scale=1200,
        reservoir_scale=800,
        coupling_scale=200,
        same_word_penalty=200,
        max_closed_class_run=2,
        auto_calibrate_beta=False,
        beta_word=0.1,
        beta_type=0.01,
        interpolated=True,
        kn_backoff=True,
        max_seq_len=30,
    )
    # Train with the synthetic texts
    model.train(texts=_SYNTHETIC_TEXTS)
    return model


@pytest.fixture(scope="session")
def vsa_encoder_fixture(small_vocab):
    """VSA encoder with D=128 for the small vocab."""
    from ising_spin.vocabulary.pos import POS2IDX, N_POS

    # Build a POS system for the small vocab
    pos_system = POSTypeSystem(vocab_size=len(small_vocab), window=3)
    pos_system.build_from_vocabulary(small_vocab.word2idx, small_vocab.idx2word)

    # Build a topic assigner
    topic_assigner = TopicAssigner(n_topics=8)
    topic_assigner.build(_SYNTHETIC_TEXTS[:50], small_vocab)

    encoder = VSAEncoder(
        vocab_size=len(small_vocab),
        n_pos=N_POS,
        n_topics=8,
        dimension=128,
        seed=42,
    )
    encoder.build(
        pos_system=pos_system,
        word_topics=topic_assigner.word_topics,
    )
    return encoder


@pytest.fixture
def random_context():
    """Random context of 10 word IDs."""
    rng = np.random.RandomState(42)
    return rng.randint(5, 200, size=10).tolist()
