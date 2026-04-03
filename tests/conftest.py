"""
Shared pytest fixtures for CorpusForge tests.

Key design decisions:
  - memory_db: uses CorpusDB(":memory:") so every test starts with a clean,
    isolated database. No temp files, no cleanup needed, no disk I/O.
  - stub_embedder: returns fixed-size deterministic zero vectors without loading
    PyTorch or sentence-transformers. Makes ingest tests instant and hermetic.
  - stub_summarizer: returns a fixed string without making any network calls.
    Isolates pipeline tests from Gemini availability/rate limits.
  - sample_md_file: writes a real file to a tmp_path so parse_file() has
    something to read. Uses pytest's built-in tmp_path fixture for auto-cleanup.

Usage in tests:
    def test_something(memory_db, stub_embedder):
        result = ingest_file("doc.md", db=memory_db, embedder=stub_embedder, ...)
"""

# import sys
import numpy as np
import pytest
from pathlib import Path

# Ensure the corpusforge package root is on the path when running from tests/
# sys.path.insert(0, str(Path(__file__).parent.parent))

from corpusforge.db import CorpusDB
from corpusforge.embedder import CorpusEmbedder


# ── Database ──────────────────────────────────────────────────────────

@pytest.fixture
def memory_db() -> CorpusDB:
    """
    A fresh in-memory CorpusDB for each test.

    IMPORTANT: SQLite ':memory:' databases are connection-scoped — each new
    connection gets a fresh empty database. CorpusDB opens a new connection
    per method call, which would mean every call sees an empty DB.

    The fix: use a named temporary file instead of ':memory:'. This gives us
    a real on-disk DB that is isolated per test (via tmp_path-equivalent) and
    automatically cleaned up. We use Python's tempfile for this.
    """
    import tempfile
    import os
    # Create a temp file and immediately close it — CorpusDB will open it.
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # Delete so CorpusDB._ensure_schema sees it as new
    db = CorpusDB(path)
    yield db
    # Cleanup after test
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ── Embedder stub ─────────────────────────────────────────────────────

class StubEmbedder(CorpusEmbedder):
    """
    Drop-in replacement for CorpusEmbedder that never loads the ML model.

    Returns deterministic 384-dimensional float32 zero vectors — the same
    dimensionality as all-MiniLM-L6-v2, so DB round-trip tests work correctly.
    The model_name is preserved so assert_embedding_model() passes normally.
    """
    DIMS = 384  # must match real model output dimensions

    def embed_chunks(self, chunks, batch_size=32):
        """Attach zero vectors to each chunk dict. No model loaded."""
        for chunk in chunks:
            chunk["embedding"] = np.zeros(self.DIMS, dtype=np.float32)
        return chunks


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    """A StubEmbedder instance with the default model name."""
    return StubEmbedder()


# ── Summarizer stub ───────────────────────────────────────────────────

class StubSummarizer:
    """
    Drop-in replacement for CorpusSummarizer that never calls the Gemini API.

    Returns a fixed string so tests can assert on auto_summary without
    requiring GEMINI_API_KEY or network access.
    """
    SUMMARY = "Stub summary: no LLM was called."

    def summarize_file(self, content_preview, title=None, user_summary=None):
        return self.SUMMARY

    def name_topic(self, chunk_texts):
        return {"name": "Stub Topic", "description": "Stub description."}

    def compile_topic(self, topic_name, topic_description, chunk_texts):
        return f"# {topic_name}\n\nStub compiled content."


@pytest.fixture
def stub_summarizer() -> StubSummarizer:
    """A StubSummarizer instance."""
    return StubSummarizer()


# ── Sample Markdown files ─────────────────────────────────────────────

SIMPLE_MD = """\
# My Document

This is the introduction paragraph. It has enough words to survive the minimum
chunk size filter and should be emitted as a single text chunk.

## Section One

Content for section one. This section discusses the first topic in detail.
It has multiple sentences to ensure it clears the minimum word threshold.

## Section Two

Content for section two. This section discusses the second topic in detail.
It also has multiple sentences for the same reason.
"""

CODE_BLOCK_MD = """\
# Code Example

Here is some prose before the code block.

```python
def hello():
    return "world"
```

And some prose after the code block.
"""

NO_H1_MD = """\
## Section Without H1

This document has no top-level heading. The title should be None.
It still has enough content to produce at least one chunk.
"""

CONSECUTIVE_HEADINGS_MD = """\
# Top

## Middle

### Bottom

Finally some content that actually has words in it. This should be the only chunk.
"""

UNCLOSED_FENCE_MD = """\
# Fenced

```python
def oops():
    # this fence is never closed
    pass
"""

LARGE_SECTION_MD = "# Big Section\n\n" + "\n\n".join(("word " * 15).strip() for _ in range(34))


@pytest.fixture
def simple_md(tmp_path: Path) -> Path:
    """A well-formed Markdown file with title, two sections, no code."""
    p = tmp_path / "simple.md"
    p.write_text(SIMPLE_MD, encoding="utf-8")
    return p


@pytest.fixture
def code_block_md(tmp_path: Path) -> Path:
    """A Markdown file containing a fenced code block."""
    p = tmp_path / "code_block.md"
    p.write_text(CODE_BLOCK_MD, encoding="utf-8")
    return p


@pytest.fixture
def no_h1_md(tmp_path: Path) -> Path:
    """A Markdown file with no H1 heading."""
    p = tmp_path / "no_h1.md"
    p.write_text(NO_H1_MD, encoding="utf-8")
    return p


@pytest.fixture
def consecutive_headings_md(tmp_path: Path) -> Path:
    """Headings with no content between them — tests the min-word-count guard."""
    p = tmp_path / "consecutive.md"
    p.write_text(CONSECUTIVE_HEADINGS_MD, encoding="utf-8")
    return p


@pytest.fixture
def unclosed_fence_md(tmp_path: Path) -> Path:
    """A Markdown file with an unclosed code fence at EOF."""
    p = tmp_path / "unclosed.md"
    p.write_text(UNCLOSED_FENCE_MD, encoding="utf-8")
    return p


@pytest.fixture
def large_section_md(tmp_path: Path) -> Path:
    """A single section with 500 words — triggers paragraph-boundary splitting."""
    p = tmp_path / "large.md"
    p.write_text(LARGE_SECTION_MD, encoding="utf-8")
    return p


@pytest.fixture
def empty_md(tmp_path: Path) -> Path:
    """A completely empty Markdown file."""
    p = tmp_path / "empty.md"
    p.write_text("", encoding="utf-8")
    return p
