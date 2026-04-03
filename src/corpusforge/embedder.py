"""
CorpusForge Embedder
=====================
Wraps sentence-transformers to produce chunk embeddings.

Model choice: all-MiniLM-L6-v2
  - Fast and lightweight (~80MB), runs well on CPU.
  - Produces 384-dimensional float32 vectors.
  - Good general-purpose semantic similarity for English technical prose.

Design notes:
  - The model is lazy-loaded on first use (loading takes ~1s; embedding is fast).
    This avoids loading the model for commands that don't need it (cf status, cf files).
  - Heading path is prepended to chunk text before embedding. This significantly
    improves similarity quality for short sections whose topic is implied by context.
  - Embedding serialization (np.ndarray ↔ BLOB) is strictly the DB layer's concern.
    This class emits np.ndarray values; db.py handles tobytes() / frombuffer().
  - Cosine similarity lives here because it's a property of the vector space, not
    the storage format. It's used by search and cross-ref detection, not by db.py.
"""

import numpy as np
from typing import Any


class CorpusEmbedder:
    """
    Embed chunk dictionaries using a sentence-transformers model.

    Usage:
        embedder = CorpusEmbedder()
        chunks = embedder.embed_chunks(parsed_chunks)
        # Each chunk dict now has an 'embedding' key: np.ndarray of float32.

        sim = CorpusEmbedder.cosine_similarity(vec_a, vec_b)
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: "SentenceTransformer | None" = None

    @property
    def model(self) -> "SentenceTransformer":
        """Lazy-load the model on first access. Importing sentence_transformers
        pulls in PyTorch and the full HuggingFace stack — several seconds of
        startup cost. Deferring the import here means commands like cf status
        and cf files stay instant."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_chunks(
        self, chunks: list[dict[str, Any]], batch_size: int = 32
    ) -> list[dict[str, Any]]:
        """
        Compute embeddings for a list of chunk dicts and attach them in place.

        Each chunk dict must have a 'content' key. 'heading_path' is optional;
        if present and not 'Root', it is prepended to the content before embedding
        to provide richer semantic context for short sections.

        The 'embedding' key is added to each dict as a float32 np.ndarray.
        Serialization to bytes is deferred to db.py.

        Returns the same list with embeddings attached (mutates in place and
        also returns for convenience in pipeline chains).
        """
        if not chunks:
            return chunks

        texts = []
        for chunk in chunks:
            heading = chunk.get("heading_path", "")
            prefix = f"{heading}\n" if heading and heading != "Root" else ""
            texts.append(f"{prefix}{chunk['content']}")

        # sentence-transformers handles batching internally.
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.astype(np.float32)

        return chunks

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Cosine similarity between two vectors. Returns value in [-1.0, 1.0].

        Guards against zero-norm vectors (returns 0.0 rather than NaN/ZeroDivision).
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
