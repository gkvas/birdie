"""
Tests for birdie.core.retrieval - embedding and similarity primitives.
"""

import math

from birdie.core.retrieval import EMBED_DIM, cosine_similarity, embed


# ---------------------------------------------------------------------------
# EMBED_DIM constant
# ---------------------------------------------------------------------------

def test_embed_dim_is_positive_int():
    assert isinstance(EMBED_DIM, int)
    assert EMBED_DIM > 0


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------

def test_embed_returns_list_of_floats():
    vec = embed("hello world")
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)


def test_embed_length_matches_embed_dim():
    assert len(embed("hello world")) == EMBED_DIM


def test_embed_is_unit_length():
    vec = embed("some text to embed")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-9


def test_embed_is_deterministic():
    text = "the quick brown fox"
    assert embed(text) == embed(text)


def test_embed_different_texts_produce_different_vectors():
    assert embed("apple") != embed("zoology")


def test_embed_case_insensitive():
    assert embed("Hello World") == embed("hello world")


def test_embed_empty_string_returns_zero_vector():
    vec = embed("")
    assert len(vec) == EMBED_DIM
    assert all(x == 0.0 for x in vec)


def test_embed_single_token():
    vec = embed("python")
    assert len(vec) == EMBED_DIM
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# cosine_similarity()
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical_vectors_is_one():
    vec = embed("hello")
    assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = [1.0] + [0.0] * (EMBED_DIM - 1)
    b = [0.0, 1.0] + [0.0] * (EMBED_DIM - 2)
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_opposite_vectors_is_minus_one():
    vec = embed("hello")
    neg = [-x for x in vec]
    assert abs(cosine_similarity(vec, neg) - (-1.0)) < 1e-9


def test_cosine_similarity_is_symmetric():
    a = embed("first sentence")
    b = embed("second sentence")
    assert cosine_similarity(a, b) == cosine_similarity(b, a)


def test_cosine_similarity_range():
    pairs = [
        ("apple", "orange"),
        ("python", "snake"),
        ("car engine", "cooking recipes"),
    ]
    for t1, t2 in pairs:
        score = cosine_similarity(embed(t1), embed(t2))
        assert -1.0 <= score <= 1.0


def test_cosine_similarity_similar_texts_score_higher_than_unrelated():
    query = embed("python programming language")
    related = embed("python coding development")
    unrelated = embed("chocolate cake recipe ingredients")
    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_cosine_similarity_bigrams_improve_discrimination():
    # "new york" as a phrase should be closer to "new york city" than
    # two unrelated single words that happen to share a unigram.
    phrase = embed("new york")
    close = embed("new york city")
    distant = embed("new planet")
    assert cosine_similarity(phrase, close) > cosine_similarity(phrase, distant)
