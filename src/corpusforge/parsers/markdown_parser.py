"""
CorpusForge Markdown Parser
============================
Parses Markdown files into structured chunks ready for embedding.

Chunking strategy:
  - Chunk boundaries are set at heading transitions, not fixed token counts.
  - If a section exceeds max_words_per_chunk, it is split at the next blank line
    (paragraph boundary). This preserves semantic coherence better than hard splits.
  - Each heading line is prepended to the NEXT chunk's content buffer so that the
    stored chunk text is self-contained and human-readable. The heading_path field
    provides the full breadcrumb hierarchy as metadata independently.
  - Code blocks are fenced and emitted as chunk_type='code'. The opening and closing
    fence lines are included in the chunk content for context.
  - Chunks with fewer than 3 words are discarded (avoids emitting heading-only noise).
    This guard is bypassed for code blocks to preserve intentionally minimal snippets.

KNOWN LIMITATION: If a section is a massive wall of text with no blank lines and
no sub-headings, it will grow unbounded until the next heading or EOF. This is
acceptable for well-structured technical documents. Add forced mid-paragraph splits
if you need to support prose-heavy documents.
"""

import hashlib
import re
from pathlib import Path
from typing import Any


class MarkdownParser:
    """
    Parse a Markdown file into metadata + a list of chunk dicts.

    Usage:
        parser = MarkdownParser(max_words_per_chunk=400)
        result = parser.parse_file("/path/to/doc.md")
        # result: {"filepath", "filename", "file_hash", "title", "chunks"}
        # Each chunk: {"chunk_index", "chunk_type", "heading_path", "content", "token_count"}
    """

    # Word count approximation: 1 word ≈ 1.3 tokens for English technical prose.
    # Replace len(text.split()) with tiktoken if exact context-window management matters.
    MIN_CHUNK_WORDS = 3

    def __init__(self, max_words_per_chunk: int = 400):
        self.max_words_per_chunk = max_words_per_chunk
        self._heading_re = re.compile(r"^(#{1,6})\s+(.*)")

    def parse_file(self, filepath: str) -> dict[str, Any]:
        """
        Read a Markdown file and return its metadata and semantic chunks.

        Returns:
            {
                "filepath":  str — original path string (for DB storage),
                "filename":  str — basename only,
                "file_hash": str — SHA-256 hex digest of raw file content,
                "title":     str | None — text of the first H1 heading,
                "chunks":    list[dict] — see _chunk_text() for chunk structure,
            }
        """
        path = Path(filepath)
        content = path.read_text(encoding="utf-8")
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        chunks, title = self._chunk_text(content)

        return {
            "filepath":  str(filepath),
            "filename":  path.name,
            "file_hash": file_hash,
            "title":     title,
            "chunks":    chunks,
        }

    # ───────────────────────────────────────────────────────────────────
    # Internal chunking logic
    # ───────────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> tuple[list[dict[str, Any]], str | None]:
        """
        Split document text into structured chunks.

        Returns (chunks, document_title) where document_title is the first H1
        heading found, or None if the document has no H1.
        """
        lines = text.split("\n")
        chunks: list[dict] = []
        chunk_index = 0

        in_code_block = False
        heading_stack: dict[int, str] = {}  # {level: heading_text}
        document_title: str | None = None

        current_buffer: list[str] = []
        current_type = "text"

        def heading_path() -> str:
            """Build breadcrumb string from heading stack, e.g. 'Root > Subtopic'."""
            path = [heading_stack[k] for k in sorted(heading_stack)]
            return " > ".join(path) if path else "Root"

        def flush(force_type: str | None = None) -> None:
            nonlocal chunk_index, current_type

            if not current_buffer:
                return

            content = "\n".join(current_buffer).strip()
            word_count = len(content.split())

            # Discard near-empty chunks (e.g., leftover blank lines between headings),
            # but always keep code blocks even if they are minimal.
            if word_count < self.MIN_CHUNK_WORDS and current_type != "code":
                current_buffer.clear()
                return

            chunks.append({
                "chunk_index":  chunk_index,
                "chunk_type":   force_type or current_type,
                "heading_path": heading_path(),
                "content":      content,
                # Word count as token approximation. Upgrade to tiktoken when needed.
                "token_count":  word_count,
            })
            chunk_index += 1
            current_buffer.clear()
            current_type = "text"

        for line in lines:
            # ── 1. Code block fences ──────────────────────────────────
            if line.strip().startswith("```"):
                if in_code_block:
                    # Closing fence: include it, emit the code block.
                    current_buffer.append(line)
                    in_code_block = False
                    flush(force_type="code")
                else:
                    # Opening fence: emit any preceding text, start code block.
                    flush()
                    in_code_block = True
                    current_buffer.append(line)
                    current_type = "code"
                continue

            if in_code_block:
                current_buffer.append(line)
                continue

            # ── 2. Headings ───────────────────────────────────────────
            m = self._heading_re.match(line)
            if m:
                flush()  # emit text accumulated before this heading

                level = len(m.group(1))
                heading_text = m.group(2).strip()

                if level == 1 and document_title is None:
                    document_title = heading_text

                # Remove this level and any deeper levels from the stack,
                # then record the new heading.
                for k in [k for k in heading_stack if k >= level]:
                    del heading_stack[k]
                heading_stack[level] = heading_text

                # Prepend the heading line to the next chunk so the stored
                # content is self-contained and readable without heading_path.
                current_buffer.append(line)
                continue

            # ── 3. Regular text ───────────────────────────────────────
            current_buffer.append(line)

            # Paragraph-boundary split: if a blank line tips us over the word
            # limit, flush. This keeps chunks semantically coherent while
            # respecting the size target.
            if line.strip() == "":
                if sum(len(l.split()) for l in current_buffer) >= self.max_words_per_chunk:
                    flush()

        # Guard against malformed documents with unclosed code fences.
        if in_code_block:
            current_type = "text"

        flush()  # emit whatever remains at EOF

        return chunks, document_title
