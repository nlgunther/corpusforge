"""
Tests for db.py

All tests use the memory_db fixture (CorpusDB(":memory:")) so they are
fast, isolated, and leave no files on disk.

Coverage targets:
  - Schema initialization and version check
  - File upsert: insert / unchanged / updated actions
  - Chunk batch insert and retrieval
  - Cascade delete: deleting a file removes its chunks
  - Embedding round-trip: BLOB serialization through _emb_to_blob / blob_to_embedding
  - assert_embedding_model: locks model on first call, raises on mismatch
  - get_corpus_stats: accurate counts across tables
  - set_state / get_state: key-value store
  - transaction() rollback on exception
  - Topic CRUD and chunk_topic assignment
"""

import numpy as np
import pytest
from corpusforge.db import CorpusDB, _SCHEMA_VERSION


# ═══════════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════════

class TestSchema:

    def test_schema_version_seeded(self, memory_db):
        assert memory_db.get_state("schema_version") == _SCHEMA_VERSION

    def test_fresh_db_has_no_files(self, memory_db):
        assert memory_db.get_all_files() == []

    def test_fresh_db_stats_are_zero(self, memory_db):
        stats = memory_db.get_corpus_stats()
        assert stats["file_count"] == 0
        assert stats["chunk_count"] == 0
        assert stats["topic_count"] == 0


# ═══════════════════════════════════════════════════════════════════════
# File upsert
# ═══════════════════════════════════════════════════════════════════════

class TestFileUpsert:

    def _insert(self, db, filepath="/docs/a.md", file_hash="hash1", title="A"):
        return db.upsert_file(
            filepath=filepath, filename="a.md", fmt="md",
            file_hash=file_hash, title=title,
        )

    def test_insert_returns_id_and_inserted(self, memory_db):
        file_id, action = self._insert(memory_db)
        assert isinstance(file_id, int)
        assert action == "inserted"

    def test_same_hash_returns_unchanged(self, memory_db):
        self._insert(memory_db)
        _, action = self._insert(memory_db)
        assert action == "unchanged"

    def test_changed_hash_returns_updated(self, memory_db):
        file_id, _ = self._insert(memory_db)
        _, action = self._insert(memory_db, file_hash="hash2")
        assert action == "updated"

    def test_updated_returns_same_id(self, memory_db):
        file_id, _ = self._insert(memory_db)
        new_id, _ = self._insert(memory_db, file_hash="hash2")
        assert file_id == new_id

    def test_get_file_by_path(self, memory_db):
        file_id, _ = self._insert(memory_db)
        row = memory_db.get_file_by_path("/docs/a.md")
        assert row is not None
        assert row["id"] == file_id
        assert row["title"] == "A"

    def test_get_file_by_id(self, memory_db):
        file_id, _ = self._insert(memory_db)
        row = memory_db.get_file(file_id)
        assert row["filename"] == "a.md"

    def test_get_file_missing_returns_none(self, memory_db):
        assert memory_db.get_file(999) is None

    def test_get_all_files_lists_all(self, memory_db):
        self._insert(memory_db, filepath="/a.md")
        self._insert(memory_db, filepath="/b.md", file_hash="h2")
        assert len(memory_db.get_all_files()) == 2

    def test_update_clears_auto_summary(self, memory_db):
        """On hash change, auto_summary must be reset to NULL."""
        file_id, _ = self._insert(memory_db)
        memory_db.update_file_summary(file_id, "Old summary")
        self._insert(memory_db, file_hash="hash2")  # triggers update
        row = memory_db.get_file(file_id)
        assert row["auto_summary"] is None

    def test_delete_file(self, memory_db):
        file_id, _ = self._insert(memory_db)
        deleted = memory_db.delete_file(file_id)
        assert deleted is True
        assert memory_db.get_file(file_id) is None

    def test_delete_nonexistent_returns_false(self, memory_db):
        assert memory_db.delete_file(999) is False


# ═══════════════════════════════════════════════════════════════════════
# Chunks
# ═══════════════════════════════════════════════════════════════════════

class TestChunks:

    def _file(self, db):
        file_id, _ = db.upsert_file("/f.md", "f.md", "md", "h1")
        return file_id

    def _make_chunks(self, n=3):
        return [
            {
                "chunk_index": i,
                "chunk_type": "text",
                "heading_path": f"Section {i}",
                "content": f"Content of chunk number {i} with some words.",
                "token_count": 8,
                "embedding": np.zeros(384, dtype=np.float32),
            }
            for i in range(n)
        ]

    def test_insert_batch_returns_ids(self, memory_db):
        file_id = self._file(memory_db)
        ids = memory_db.insert_chunks_batch(file_id, self._make_chunks(3))
        assert len(ids) == 3
        assert all(isinstance(i, int) for i in ids)

    def test_get_chunks_for_file(self, memory_db):
        file_id = self._file(memory_db)
        memory_db.insert_chunks_batch(file_id, self._make_chunks(3))
        chunks = memory_db.get_chunks_for_file(file_id)
        assert len(chunks) == 3

    def test_chunks_ordered_by_index(self, memory_db):
        file_id = self._file(memory_db)
        memory_db.insert_chunks_batch(file_id, self._make_chunks(5))
        chunks = memory_db.get_chunks_for_file(file_id)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == sorted(indices)

    def test_get_chunk_by_id(self, memory_db):
        file_id = self._file(memory_db)
        ids = memory_db.insert_chunks_batch(file_id, self._make_chunks(1))
        chunk = memory_db.get_chunk(ids[0])
        assert chunk["content"] == "Content of chunk number 0 with some words."

    def test_empty_batch_returns_empty_list(self, memory_db):
        file_id = self._file(memory_db)
        assert memory_db.insert_chunks_batch(file_id, []) == []

    def test_chunks_without_embedding(self, memory_db):
        """Chunks may be inserted with embedding=None."""
        file_id = self._file(memory_db)
        chunks = [{"chunk_index": 0, "content": "No embedding here at all.",
                   "token_count": 5, "chunk_type": "text", "heading_path": "Root"}]
        ids = memory_db.insert_chunks_batch(file_id, chunks)
        assert len(ids) == 1
        all_embeddings = memory_db.get_all_embeddings()
        assert len(all_embeddings) == 0  # NULL embeddings are excluded


# ═══════════════════════════════════════════════════════════════════════
# Cascade delete
# ═══════════════════════════════════════════════════════════════════════

class TestCascade:

    def test_delete_file_cascades_to_chunks(self, memory_db):
        file_id, _ = memory_db.upsert_file("/f.md", "f.md", "md", "h1")
        memory_db.insert_chunks_batch(file_id, [
            {"chunk_index": 0, "content": "Some content here.", "token_count": 3,
             "chunk_type": "text", "heading_path": "Root"}
        ])
        memory_db.delete_file(file_id)
        assert memory_db.get_chunks_for_file(file_id) == []

    def test_delete_chunks_for_file(self, memory_db):
        file_id, _ = memory_db.upsert_file("/f.md", "f.md", "md", "h1")
        memory_db.insert_chunks_batch(file_id, [
            {"chunk_index": i, "content": f"Content {i} here.", "token_count": 3,
             "chunk_type": "text", "heading_path": "Root"}
            for i in range(4)
        ])
        deleted = memory_db.delete_chunks_for_file(file_id)
        assert deleted == 4
        assert memory_db.get_chunks_for_file(file_id) == []


# ═══════════════════════════════════════════════════════════════════════
# Embedding round-trip
# ═══════════════════════════════════════════════════════════════════════

class TestEmbeddingRoundTrip:

    def test_blob_round_trip(self, memory_db):
        """Embeddings stored as BLOB must come back bit-for-bit identical."""
        original = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        blob = memory_db._emb_to_blob(original)
        recovered = memory_db.blob_to_embedding(blob)
        np.testing.assert_array_equal(original, recovered)

    def test_blob_round_trip_384_dims(self, memory_db):
        """Full-size embedding (384 dims) must survive round-trip."""
        original = np.random.rand(384).astype(np.float32)
        recovered = memory_db.blob_to_embedding(memory_db._emb_to_blob(original))
        np.testing.assert_array_equal(original, recovered)

    def test_get_all_embeddings_returns_arrays(self, memory_db):
        file_id, _ = memory_db.upsert_file("/f.md", "f.md", "md", "h1")
        emb = np.ones(384, dtype=np.float32)
        memory_db.insert_chunks_batch(file_id, [
            {"chunk_index": 0, "content": "With embedding.", "token_count": 2,
             "chunk_type": "text", "heading_path": "Root", "embedding": emb}
        ])
        results = memory_db.get_all_embeddings()
        assert len(results) == 1
        chunk_id, recovered = results[0]
        assert isinstance(chunk_id, int)
        np.testing.assert_array_equal(recovered, emb)

    def test_get_all_embeddings_skips_null(self, memory_db):
        """Chunks with NULL embeddings must be excluded from get_all_embeddings."""
        file_id, _ = memory_db.upsert_file("/f.md", "f.md", "md", "h1")
        memory_db.insert_chunks_batch(file_id, [
            {"chunk_index": 0, "content": "No embedding.", "token_count": 2,
             "chunk_type": "text", "heading_path": "Root", "embedding": None}
        ])
        assert memory_db.get_all_embeddings() == []

    def test_blob_to_embedding_returns_copy(self, memory_db):
        """blob_to_embedding must return a writable copy, not a read-only buffer view."""
        blob = memory_db._emb_to_blob(np.zeros(4, dtype=np.float32))
        arr = memory_db.blob_to_embedding(blob)
        arr[0] = 99.0  # must not raise


# ═══════════════════════════════════════════════════════════════════════
# Embedding model guard
# ═══════════════════════════════════════════════════════════════════════

class TestEmbeddingModelGuard:

    def test_first_call_locks_model(self, memory_db):
        memory_db.assert_embedding_model("model-A")
        assert memory_db.get_state("embedding_model") == "model-A"

    def test_same_model_does_not_raise(self, memory_db):
        memory_db.assert_embedding_model("model-A")
        memory_db.assert_embedding_model("model-A")  # must not raise

    def test_different_model_raises(self, memory_db):
        memory_db.assert_embedding_model("model-A")
        with pytest.raises(RuntimeError, match="model-A"):
            memory_db.assert_embedding_model("model-B")


# ═══════════════════════════════════════════════════════════════════════
# Corpus state
# ═══════════════════════════════════════════════════════════════════════

class TestCorpusState:

    def test_get_missing_key_returns_none(self, memory_db):
        assert memory_db.get_state("nonexistent_key") is None

    def test_set_and_get(self, memory_db):
        memory_db.set_state("my_key", "my_value")
        assert memory_db.get_state("my_key") == "my_value"

    def test_set_overwrites(self, memory_db):
        memory_db.set_state("k", "v1")
        memory_db.set_state("k", "v2")
        assert memory_db.get_state("k") == "v2"


# ═══════════════════════════════════════════════════════════════════════
# Corpus statistics
# ═══════════════════════════════════════════════════════════════════════

class TestCorpusStats:

    def test_file_count(self, memory_db):
        memory_db.upsert_file("/a.md", "a.md", "md", "h1")
        memory_db.upsert_file("/b.md", "b.md", "md", "h2")
        assert memory_db.get_corpus_stats()["file_count"] == 2

    def test_chunk_count(self, memory_db):
        file_id, _ = memory_db.upsert_file("/a.md", "a.md", "md", "h1")
        memory_db.insert_chunks_batch(file_id, [
            {"chunk_index": i, "content": f"Content {i}.", "token_count": 2,
             "chunk_type": "text", "heading_path": "Root"}
            for i in range(5)
        ])
        assert memory_db.get_corpus_stats()["chunk_count"] == 5

    def test_embedded_chunk_count(self, memory_db):
        file_id, _ = memory_db.upsert_file("/a.md", "a.md", "md", "h1")
        emb = np.zeros(384, dtype=np.float32)
        memory_db.insert_chunks_batch(file_id, [
            {"chunk_index": 0, "content": "With embedding.", "token_count": 2,
             "chunk_type": "text", "heading_path": "Root", "embedding": emb},
            {"chunk_index": 1, "content": "Without embedding.", "token_count": 2,
             "chunk_type": "text", "heading_path": "Root", "embedding": None},
        ])
        stats = memory_db.get_corpus_stats()
        assert stats["chunk_count"] == 2
        assert stats["embedded_chunk_count"] == 1

    def test_corpus_summary_none_when_unset(self, memory_db):
        assert memory_db.get_corpus_stats()["corpus_summary"] is None

    def test_corpus_summary_when_set(self, memory_db):
        memory_db.set_state("corpus_summary", "Overview of the corpus.")
        assert memory_db.get_corpus_stats()["corpus_summary"] == "Overview of the corpus."


# ═══════════════════════════════════════════════════════════════════════
# Transaction
# ═══════════════════════════════════════════════════════════════════════

class TestTransaction:

    def test_transaction_commits_on_success(self, memory_db):
        with memory_db.transaction() as conn:
            conn.execute(
                "INSERT INTO corpus_state (key, value, updated_at) VALUES (?, ?, ?)",
                ("tx_test", "hello", memory_db._now()),
            )
        assert memory_db.get_state("tx_test") == "hello"

    def test_transaction_rolls_back_on_exception(self, memory_db):
        with pytest.raises(ValueError):
            with memory_db.transaction() as conn:
                conn.execute(
                    "INSERT INTO corpus_state (key, value, updated_at) VALUES (?, ?, ?)",
                    ("should_rollback", "value", memory_db._now()),
                )
                raise ValueError("Simulated failure mid-transaction")
        # The row must not have been committed
        assert memory_db.get_state("should_rollback") is None

    def test_multi_step_upsert_is_atomic(self, memory_db):
        """
        The canonical re-ingest pattern (upsert + delete_chunks + insert_chunks)
        must all commit together or not at all.
        """
        # First ingest
        file_id, _ = memory_db.upsert_file("/f.md", "f.md", "md", "hash1")
        memory_db.insert_chunks_batch(file_id, [
            {"chunk_index": 0, "content": "Old content here.", "token_count": 3,
             "chunk_type": "text", "heading_path": "Root",
             "embedding": np.zeros(384, dtype=np.float32)}
        ])

        # Second ingest (update) — simulate a mid-transaction crash
        with pytest.raises(RuntimeError):
            with memory_db.transaction() as conn:
                memory_db.upsert_file("/f.md", "f.md", "md", "hash2", conn=conn)
                memory_db.delete_chunks_for_file(file_id, conn=conn)
                raise RuntimeError("Crash before inserting new chunks")

        # Old chunks must still be there — the delete was rolled back
        assert len(memory_db.get_chunks_for_file(file_id)) == 1


# ═══════════════════════════════════════════════════════════════════════
# Topics
# ═══════════════════════════════════════════════════════════════════════

class TestTopics:

    def test_create_topic_returns_id(self, memory_db):
        topic_id = memory_db.create_topic("My Topic", "A description.")
        assert isinstance(topic_id, int)

    def test_get_all_topics(self, memory_db):
        memory_db.create_topic("Topic A")
        memory_db.create_topic("Topic B")
        topics = memory_db.get_all_topics()
        assert len(topics) == 2

    def test_roots_only_excludes_children(self, memory_db):
        parent_id = memory_db.create_topic("Parent")
        memory_db.create_topic("Child", parent_topic_id=parent_id)
        roots = memory_db.get_all_topics(roots_only=True)
        assert len(roots) == 1
        assert roots[0]["name"] == "Parent"

    def test_delete_all_topics(self, memory_db):
        memory_db.create_topic("To Delete")
        count = memory_db.delete_all_topics()
        assert count == 1
        assert memory_db.get_all_topics() == []

    def test_assign_chunks_to_topic(self, memory_db):
        file_id, _ = memory_db.upsert_file("/f.md", "f.md", "md", "h1")
        chunk_ids = memory_db.insert_chunks_batch(file_id, [
            {"chunk_index": 0, "content": "Chunk content here.", "token_count": 3,
             "chunk_type": "text", "heading_path": "Root"}
        ])
        topic_id = memory_db.create_topic("Test Topic")
        memory_db.assign_chunks_to_topic_batch([(chunk_ids[0], topic_id, 0.95)])
        chunks = memory_db.get_chunks_for_topic(topic_id)
        assert len(chunks) == 1
        assert chunks[0]["similarity"] == pytest.approx(0.95)
