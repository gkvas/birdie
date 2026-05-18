"""
Lightweight text retrieval primitives for Birdie's long-term memory.

This module provides dependency-free embedding and similarity functions used
to index and query LTM entries.  The algorithm is a hash-trick bag-of-ngrams
projection: each unigram and bigram is hashed into a fixed-size vector, then
the vector is L2-normalised so that dot-product equals cosine similarity.

Public API
----------
EMBED_DIM : int
    Dimensionality of the embedding vectors produced by ``embed()``.

embed(text) -> list[float]
    Map a plain-text string to a unit-length embedding vector.
    Deterministic and fast; requires no model files or network access.

cosine_similarity(a, b) -> float
    Return the cosine similarity of two L2-normalised vectors in [-1, 1].
    Assumes both vectors were produced by ``embed()`` (i.e. already unit-length),
    so the implementation is a plain dot product.

Typical usage::

    from birdie.core.retrieval import embed, cosine_similarity

    vec_a = embed("the quick brown fox")
    vec_b = embed("a fast reddish fox")
    score = cosine_similarity(vec_a, vec_b)  # closer to 1.0 = more similar
"""

from __future__ import annotations

import hashlib
import math
from typing import List

EMBED_DIM: int = 512
"""Number of dimensions in every vector returned by :func:`embed`."""


def embed(text: str) -> List[float]:
    """Return a unit-length embedding vector for *text*.

    The vector is computed via a hash-trick projection over unigrams and
    bigrams: each n-gram is hashed with SHA-256, mapped to a bucket in
    ``[0, EMBED_DIM)``, and accumulated with a sign derived from the hash.
    The result is L2-normalised, so :func:`cosine_similarity` reduces to a
    dot product.

    An empty string (or one that produces an all-zero vector) is returned
    as-is without normalisation; its dot product with any vector is 0.

    Args:
        text: Plain text to embed.  Case is lowercased before tokenisation.

    Returns:
        A list of ``EMBED_DIM`` floats representing the unit-length vector.
    """
    tokens = text.lower().split()
    ngrams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]

    vec = [0.0] * EMBED_DIM
    for ng in ngrams:
        digest = hashlib.sha256(ng.encode()).digest()
        idx = int.from_bytes(digest[:3], "little") % EMBED_DIM
        sign = 1 if digest[3] & 1 else -1
        vec[idx] += sign

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Return the cosine similarity of two embedding vectors.

    Both vectors must be L2-normalised (as produced by :func:`embed`), in
    which case the cosine similarity equals the dot product and lies in
    ``[-1.0, 1.0]``.  A value of 1.0 means identical direction; 0.0 means
    orthogonal; negative values indicate opposing directions.

    Args:
        a: First unit-length vector of length ``EMBED_DIM``.
        b: Second unit-length vector of length ``EMBED_DIM``.

    Returns:
        Cosine similarity as a float in ``[-1.0, 1.0]``.
    """
    return sum(x * y for x, y in zip(a, b))
