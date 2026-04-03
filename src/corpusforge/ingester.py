"""
CorpusForge Ingest Pipeline
=============================
Orchestrates the full ingest workflow: parse → embed → summarize → commit.

This module owns the business logic of ingestion. cli.py calls it and handles
presentation (rich output, progress spinners). Keeping these concerns separate
means the pipeline can also be called programmatically or tested without a TTY.

Pipeline steps:
  1. Parse: MarkdownParser extracts chunks and file metadata.
  2. Hash check: if the file hash is unchanged, ingest is skipped.
  3. Embed: CorpusEmbedder attaches float32 vectors to each chunk dict.
  4. Summarize: CorpusSummarizer generates an LLM summary from the first ~1500 words.
     Failure here is non-fatal; a placeholder is stored instead.
  5. Commit: All DB writes happen in a single transaction for atomicity.
     On re-ingest, old chunks are deleted before new ones are inserted.

The IngestResult dataclass carries everything the CLI needs to display the outcome.
"""

from dataclasses import dataclass, field
from pathlib import Path

from .db import CorpusDB
from .embedder import CorpusEmbedder
from .parsers.markdown_parser import MarkdownParser
from .summarizer import CorpusSummarizer


@dataclass
class IngestResult:
    """
    Output of a single ingest operation.

    action:        'inserted', 'updated', or 'unchanged'
    filename:      basename of the ingested file
    title:         document title extracted from first H1, or None
    chunk_count:   number of chunks stored (0 if unchanged)
    auto_summary:  LLM-generated summary string (None if unchanged or skipped)
    """
    action: str
    filename: str
    title: str | None = None
    chunk_count: int = 0
    auto_summary: str | None = None


def ingest_file(
    filepath: str,
    db: CorpusDB | None = None,
    embedder: CorpusEmbedder | None = None,
    summarizer: CorpusSummarizer | None = None,
    parser: MarkdownParser | None = None,
    max_summary_words: int = 1500,
    summarize: bool = True,
) -> IngestResult:
    """
    Run the full ingest pipeline for a single Markdown file.

    Components (db, embedder, summarizer, parser) are accepted as optional
    arguments so callers can inject pre-constructed instances for efficiency
    (e.g., the embedder model is slow to load; reuse it across multiple files).
    If not provided, defaults are constructed here.

    Args:
        filepath:          path to the Markdown file to ingest.
        db:                CorpusDB instance (default: uses 'corpusforge.db').
        embedder:          CorpusEmbedder instance.
        summarizer:        CorpusSummarizer instance (only constructed if summarize=True).
        parser:            MarkdownParser instance.
        max_summary_words: how many words of content to pass to the summarizer.
        summarize:         if True, call the LLM summarizer after embedding.
                           Defaults to False — opt-in via `cf ingest --summarize`.

    Returns:
        IngestResult with action, filename, title, chunk_count, auto_summary.

    Raises:
        FileNotFoundError: if filepath does not exist.
        RuntimeError:      if the embedding model is incompatible with the DB
                           (propagated from db.assert_embedding_model).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Use provided instances or construct defaults (summarizer deferred — only needed if not skip_llm).
    db       = db      or CorpusDB()
    embedder = embedder or CorpusEmbedder()
    parser   = parser  or MarkdownParser(max_words_per_chunk=400)

    # Guard: ensure embedding model is consistent with the database.
    # Raises RuntimeError if a different model was used previously.
    db.assert_embedding_model(embedder.model_name)

    # ── Step 1: Parse ────────────────────────────────────────────────
    parsed = parser.parse_file(str(path))

    # ── Step 2: Hash check ───────────────────────────────────────────
    existing = db.get_file_by_path(str(path))
    if existing and existing["file_hash"] == parsed["file_hash"]:
        return IngestResult(action="unchanged", filename=parsed["filename"])

    # ── Step 3: Embed ────────────────────────────────────────────────
    chunks = embedder.embed_chunks(parsed["chunks"])

    # ── Step 4: Summarize ────────────────────────────────────────────
    # Build a content preview from the first max_summary_words words of chunk text.
    # Skip entirely if --skip-llm was passed (fast re-ingest, offline, testing).
    if not summarize:
        auto_summary = None
    else:
        summarizer = summarizer or CorpusSummarizer()
        preview_parts: list[str] = []
        word_total = 0
        for chunk in chunks:
            preview_parts.append(chunk["content"])
            word_total += chunk["token_count"]
            if word_total >= max_summary_words:
                break

        auto_summary = summarizer.summarize_file(
            content_preview="\n".join(preview_parts),
            title=parsed["title"],
        )

    # ── Step 5: Commit (single atomic transaction) ───────────────────
    with db.transaction() as conn:
        file_id, action = db.upsert_file(
            filepath=str(path),
            filename=parsed["filename"],
            fmt=path.suffix.lstrip("."),
            file_hash=parsed["file_hash"],
            title=parsed["title"],
            conn=conn,
        )

        if action == "updated":
            # Wipe stale chunks; CASCADE removes orphaned chunk_topics and cross_refs.
            db.delete_chunks_for_file(file_id, conn=conn)

        db.insert_chunks_batch(file_id, chunks, conn=conn)
        db.update_file_summary(file_id, auto_summary, conn=conn)
        # Single commit happens when the `with` block exits successfully.

    return IngestResult(
        action=action,
        filename=parsed["filename"],
        title=parsed["title"],
        chunk_count=len(chunks),
        auto_summary=auto_summary,
    )
