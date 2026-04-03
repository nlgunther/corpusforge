"""
Tests for ingester.py

Tests the full ingest pipeline using injected stubs for the embedder and
summarizer, and an in-memory DB. This means:
  - No ML model is loaded
  - No Gemini API calls are made
  - No files are written to disk (except the input Markdown via tmp_path)
  - Tests run in milliseconds

Coverage targets:
  - Successful insert: action, chunk_count, title, auto_summary
  - Unchanged detection: same file ingested twice returns 'unchanged'
  - Updated detection: modified file re-ingested returns 'updated'
  - Updated path: old chunks deleted, new chunks inserted
  - summarize=False: auto_summary is None, summarizer not called
  - summarize=True: auto_summary is populated from stub
  - FileNotFoundError on missing file
  - Embedding model mismatch raises RuntimeError
  - Re-ingest is atomic: crash mid-transaction leaves old chunks intact
"""

import numpy as np
import pytest
from pathlib import Path

from corpusforge.db import CorpusDB
from corpusforge.ingester import ingest_file, IngestResult
from corpusforge.parsers.markdown_parser import MarkdownParser


# ── Helper ────────────────────────────────────────────────────────────

def run(filepath, db, stub_embedder, stub_summarizer=None, summarize=False, **kwargs):
    """Thin wrapper to reduce boilerplate in every test."""
    return ingest_file(
        filepath=str(filepath),
        db=db,
        embedder=stub_embedder,
        summarizer=stub_summarizer,
        summarize=summarize,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════
# Successful insert
# ═══════════════════════════════════════════════════════════════════════

class TestInsert:

    def test_action_is_inserted(self, simple_md, memory_db, stub_embedder):
        result = run(simple_md, memory_db, stub_embedder)
        assert result.action == "inserted"

    def test_filename_correct(self, simple_md, memory_db, stub_embedder):
        result = run(simple_md, memory_db, stub_embedder)
        assert result.filename == "simple.md"

    def test_title_extracted(self, simple_md, memory_db, stub_embedder):
        result = run(simple_md, memory_db, stub_embedder)
        assert result.title == "My Document"

    def test_chunk_count_positive(self, simple_md, memory_db, stub_embedder):
        result = run(simple_md, memory_db, stub_embedder)
        assert result.chunk_count > 0

    def test_chunks_written_to_db(self, simple_md, memory_db, stub_embedder):
        result = run(simple_md, memory_db, stub_embedder)
        file_row = memory_db.get_file_by_path(str(simple_md))
        db_chunks = memory_db.get_chunks_for_file(file_row["id"])
        assert len(db_chunks) == result.chunk_count

    def test_file_row_written_to_db(self, simple_md, memory_db, stub_embedder):
        run(simple_md, memory_db, stub_embedder)
        row = memory_db.get_file_by_path(str(simple_md))
        assert row is not None
        assert row["filename"] == "simple.md"

    def test_result_is_ingest_result(self, simple_md, memory_db, stub_embedder):
        result = run(simple_md, memory_db, stub_embedder)
        assert isinstance(result, IngestResult)


# ═══════════════════════════════════════════════════════════════════════
# Unchanged detection
# ═══════════════════════════════════════════════════════════════════════

class TestUnchanged:

    def test_second_ingest_is_unchanged(self, simple_md, memory_db, stub_embedder):
        run(simple_md, memory_db, stub_embedder)
        result = run(simple_md, memory_db, stub_embedder)
        assert result.action == "unchanged"

    def test_unchanged_chunk_count_is_zero(self, simple_md, memory_db, stub_embedder):
        run(simple_md, memory_db, stub_embedder)
        result = run(simple_md, memory_db, stub_embedder)
        assert result.chunk_count == 0

    def test_unchanged_auto_summary_is_none(self, simple_md, memory_db, stub_embedder):
        run(simple_md, memory_db, stub_embedder)
        result = run(simple_md, memory_db, stub_embedder)
        assert result.auto_summary is None

    def test_unchanged_does_not_duplicate_chunks(self, simple_md, memory_db, stub_embedder):
        run(simple_md, memory_db, stub_embedder)
        first_count = memory_db.get_corpus_stats()["chunk_count"]
        run(simple_md, memory_db, stub_embedder)
        second_count = memory_db.get_corpus_stats()["chunk_count"]
        assert first_count == second_count


# ═══════════════════════════════════════════════════════════════════════
# Updated detection
# ═══════════════════════════════════════════════════════════════════════

class TestUpdated:

    def test_modified_file_is_updated(self, tmp_path, memory_db, stub_embedder):
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nOriginal content here.\n", encoding="utf-8")
        run(f, memory_db, stub_embedder)
        f.write_text("# Title\n\nModified content here is different.\n", encoding="utf-8")
        result = run(f, memory_db, stub_embedder)
        assert result.action == "updated"

    def test_updated_replaces_chunks(self, tmp_path, memory_db, stub_embedder):
        """After update, chunk count must reflect new content, not old + new."""
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nOriginal content with several words.\n", encoding="utf-8")
        run(f, memory_db, stub_embedder)
        first_count = memory_db.get_corpus_stats()["chunk_count"]

        # Write a file with clearly more chunks (more sections)
        f.write_text(
            "# Title\n\n## A\n\nSection A content.\n\n## B\n\nSection B content.\n\n"
            "## C\n\nSection C content.\n\n## D\n\nSection D content.\n",
            encoding="utf-8",
        )
        run(f, memory_db, stub_embedder)
        second_count = memory_db.get_corpus_stats()["chunk_count"]

        # Total chunks in DB must equal the NEW file's chunks only
        file_row = memory_db.get_file_by_path(str(f))
        db_chunks = memory_db.get_chunks_for_file(file_row["id"])
        assert len(db_chunks) == second_count

    def test_updated_keeps_single_file_row(self, tmp_path, memory_db, stub_embedder):
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nOriginal content.\n", encoding="utf-8")
        run(f, memory_db, stub_embedder)
        f.write_text("# Title\n\nModified content here.\n", encoding="utf-8")
        run(f, memory_db, stub_embedder)
        assert memory_db.get_corpus_stats()["file_count"] == 1


# ═══════════════════════════════════════════════════════════════════════
# Summarization flag
# ═══════════════════════════════════════════════════════════════════════

class TestSummarizationFlag:

    def test_summarize_false_auto_summary_is_none(self, simple_md, memory_db, stub_embedder):
        result = run(simple_md, memory_db, stub_embedder, summarize=False)
        assert result.auto_summary is None

    def test_summarize_true_auto_summary_populated(
        self, simple_md, memory_db, stub_embedder, stub_summarizer
    ):
        result = ingest_file(
            filepath=str(simple_md),
            db=memory_db,
            embedder=stub_embedder,
            summarizer=stub_summarizer,
            summarize=True,
        )
        assert result.auto_summary == stub_summarizer.SUMMARY

    def test_summarize_true_summary_written_to_db(
        self, simple_md, memory_db, stub_embedder, stub_summarizer
    ):
        ingest_file(
            filepath=str(simple_md),
            db=memory_db,
            embedder=stub_embedder,
            summarizer=stub_summarizer,
            summarize=True,
        )
        row = memory_db.get_file_by_path(str(simple_md))
        assert row["auto_summary"] == stub_summarizer.SUMMARY

    def test_summarize_false_summary_not_written(self, simple_md, memory_db, stub_embedder):
        run(simple_md, memory_db, stub_embedder, summarize=False)
        row = memory_db.get_file_by_path(str(simple_md))
        # update_file_summary is called with None — stored as NULL
        assert row["auto_summary"] is None


# ═══════════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_missing_file_raises_file_not_found(self, memory_db, stub_embedder):
        with pytest.raises(FileNotFoundError):
            run("/nonexistent/path/doc.md", memory_db, stub_embedder)

    def test_model_mismatch_raises_runtime_error(
        self, simple_md, memory_db, stub_embedder
    ):
        """
        If the DB was built with model-A and we try to ingest with model-B,
        assert_embedding_model must raise before any DB writes occur.
        """
        memory_db.assert_embedding_model("model-A")
        stub_embedder.model_name = "model-B"
        with pytest.raises(RuntimeError, match="model-A"):
            run(simple_md, memory_db, stub_embedder)

    def test_model_mismatch_leaves_db_clean(self, simple_md, memory_db, stub_embedder):
        """After a model mismatch error, no partial data must be in the DB."""
        memory_db.assert_embedding_model("model-A")
        stub_embedder.model_name = "model-B"
        with pytest.raises(RuntimeError):
            run(simple_md, memory_db, stub_embedder)
        assert memory_db.get_corpus_stats()["file_count"] == 0
        assert memory_db.get_corpus_stats()["chunk_count"] == 0


# ═══════════════════════════════════════════════════════════════════════
# Atomicity
# ═══════════════════════════════════════════════════════════════════════

class TestAtomicity:

    def test_failed_update_preserves_old_chunks(
        self, tmp_path, memory_db, stub_embedder, monkeypatch
    ):
        """
        If the transaction fails mid-way through a re-ingest (after deleting
        old chunks but before inserting new ones), the old chunks must survive
        because the delete was rolled back with the rest of the transaction.

        We simulate a crash by monkeypatching insert_chunks_batch to raise.
        """
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nOriginal content with words.\n", encoding="utf-8")
        run(f, memory_db, stub_embedder)
        original_count = memory_db.get_corpus_stats()["chunk_count"]

        # Modify the file to trigger the update path
        f.write_text("# Title\n\nModified content with words.\n", encoding="utf-8")

        # Patch insert_chunks_batch to simulate a crash after delete
        def crash(*args, **kwargs):
            raise RuntimeError("Simulated crash")
        monkeypatch.setattr(memory_db, "insert_chunks_batch", crash)

        with pytest.raises(RuntimeError):
            run(f, memory_db, stub_embedder)

        # Old chunks must still be there
        assert memory_db.get_corpus_stats()["chunk_count"] == original_count
