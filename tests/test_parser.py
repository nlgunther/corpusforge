"""
Tests for parsers/markdown_parser.py

MarkdownParser is pure logic with no external dependencies — ideal for
exhaustive unit testing. Every test calls _chunk_text() directly on a
string, or parse_file() on a tmp_path file, with no DB or model involved.

Coverage targets:
  - Normal document structure (title, sections, chunks)
  - Code blocks (fenced, type labeling, content preservation)
  - Edge cases (no H1, empty file, consecutive headings, unclosed fence)
  - Splitting behavior (large sections, paragraph boundaries)
  - Heading hierarchy in heading_path
  - Chunk content includes the heading line (self-contained chunks)
  - File hash consistency
"""

import hashlib
import pytest
from corpusforge.parsers.markdown_parser import MarkdownParser

# Use a small word limit for split tests so we don't need huge fixtures.
SMALL_LIMIT = 20


@pytest.fixture
def parser():
    """Default parser with standard word limit."""
    return MarkdownParser(max_words_per_chunk=400)


@pytest.fixture
def small_parser():
    """Parser with a small word limit to trigger paragraph splitting easily."""
    return MarkdownParser(max_words_per_chunk=SMALL_LIMIT)


# ═══════════════════════════════════════════════════════════════════════
# Basic structure
# ═══════════════════════════════════════════════════════════════════════

class TestBasicStructure:

    def test_title_extracted_from_h1(self, parser, simple_md):
        result = parser.parse_file(str(simple_md))
        assert result["title"] == "My Document"

    def test_filename_is_basename_only(self, parser, simple_md):
        result = parser.parse_file(str(simple_md))
        assert result["filename"] == "simple.md"
        assert "/" not in result["filename"]
        assert "\\" not in result["filename"]

    def test_filepath_preserved(self, parser, simple_md):
        result = parser.parse_file(str(simple_md))
        assert result["filepath"] == str(simple_md)

    def test_chunks_produced(self, parser, simple_md):
        result = parser.parse_file(str(simple_md))
        assert len(result["chunks"]) >= 1

    def test_chunk_index_sequential(self, parser, simple_md):
        chunks = parser.parse_file(str(simple_md))["chunks"]
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_chunk_has_required_keys(self, parser, simple_md):
        chunk = parser.parse_file(str(simple_md))["chunks"][0]
        assert "chunk_index"  in chunk
        assert "chunk_type"   in chunk
        assert "heading_path" in chunk
        assert "content"      in chunk
        assert "token_count"  in chunk

    def test_token_count_matches_content(self, parser, simple_md):
        """token_count is a word-count approximation: must match len(content.split())."""
        for chunk in parser.parse_file(str(simple_md))["chunks"]:
            assert chunk["token_count"] == len(chunk["content"].split())

    def test_default_chunk_type_is_text(self, parser, simple_md):
        for chunk in parser.parse_file(str(simple_md))["chunks"]:
            assert chunk["chunk_type"] == "text"


# ═══════════════════════════════════════════════════════════════════════
# Heading hierarchy
# ═══════════════════════════════════════════════════════════════════════

class TestHeadingHierarchy:

    def test_heading_path_reflects_nesting(self, parser):
        md = "# Root\n\n## Child\n\nSome content here with enough words.\n"
        chunks, _ = parser._chunk_text(md)
        # The chunk under ## Child should show the full hierarchy
        child_chunk = next(c for c in chunks if "Child" in c["heading_path"])
        assert "Root" in child_chunk["heading_path"]
        assert "Child" in child_chunk["heading_path"]

    def test_heading_path_resets_on_same_level(self, parser):
        md = "# Root\n\n## First\n\nContent for first.\n\n## Second\n\nContent for second.\n"
        chunks, _ = parser._chunk_text(md)
        second_chunk = next(c for c in chunks if "Second" in c["heading_path"])
        assert "First" not in second_chunk["heading_path"]
        assert "Second" in second_chunk["heading_path"]

    def test_heading_path_resets_deeper_levels_on_parent(self, parser):
        md = "# Root\n\n## Parent\n\n### Child\n\nDeep content here.\n\n## New Parent\n\nShallow content.\n"
        chunks, _ = parser._chunk_text(md)
        shallow = next(c for c in chunks if "New Parent" in c["heading_path"])
        assert "Child" not in shallow["heading_path"]

    def test_heading_line_included_in_chunk_content(self, parser):
        """Heading text must appear in chunk content for self-contained readability."""
        md = "# My Title\n\nSome introductory text with enough words here.\n"
        chunks, _ = parser._chunk_text(md)
        assert len(chunks) >= 1
        first_chunk_content = chunks[0]["content"]
        assert "My Title" in first_chunk_content

    def test_root_heading_path_when_no_heading(self, parser):
        md = "Just some plain text with no heading at all here.\n"
        chunks, _ = parser._chunk_text(md)
        assert chunks[0]["heading_path"] == "Root"


# ═══════════════════════════════════════════════════════════════════════
# Code blocks
# ═══════════════════════════════════════════════════════════════════════

class TestCodeBlocks:

    def test_code_block_gets_code_type(self, parser, code_block_md):
        chunks = parser.parse_file(str(code_block_md))["chunks"]
        code_chunks = [c for c in chunks if c["chunk_type"] == "code"]
        assert len(code_chunks) == 1

    def test_code_block_content_preserved(self, parser, code_block_md):
        chunks = parser.parse_file(str(code_block_md))["chunks"]
        code_chunk = next(c for c in chunks if c["chunk_type"] == "code")
        assert "def hello" in code_chunk["content"]
        assert "return" in code_chunk["content"]

    def test_code_block_fences_included(self, parser, code_block_md):
        """Opening and closing ``` lines must be in the chunk content."""
        chunks = parser.parse_file(str(code_block_md))["chunks"]
        code_chunk = next(c for c in chunks if c["chunk_type"] == "code")
        assert "```" in code_chunk["content"]

    def test_prose_before_code_is_text_type(self, parser, code_block_md):
        chunks = parser.parse_file(str(code_block_md))["chunks"]
        text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
        assert len(text_chunks) >= 1

    def test_prose_after_code_is_text_type(self, parser, code_block_md):
        chunks = parser.parse_file(str(code_block_md))["chunks"]
        # There should be text chunks both before and after the code block
        text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
        assert len(text_chunks) >= 2

    def test_unclosed_fence_does_not_crash(self, parser, unclosed_fence_md):
        """An unclosed code fence at EOF must not raise — treated as text."""
        result = parser.parse_file(str(unclosed_fence_md))
        assert isinstance(result["chunks"], list)


# ═══════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_no_h1_title_is_none(self, parser, no_h1_md):
        result = parser.parse_file(str(no_h1_md))
        assert result["title"] is None

    def test_empty_file_produces_no_chunks(self, parser, empty_md):
        result = parser.parse_file(str(empty_md))
        assert result["chunks"] == []
        assert result["title"] is None

    def test_consecutive_headings_no_junk_chunks(self, parser, consecutive_headings_md):
        """
        Consecutive headings with no content between them must not emit
        near-empty chunks (the MIN_CHUNK_WORDS guard).
        """
        chunks = parser.parse_file(str(consecutive_headings_md))["chunks"]
        # All chunks must have meaningful content (>= MIN_CHUNK_WORDS words)
        for chunk in chunks:
            if chunk["chunk_type"] != "code":
                assert chunk["token_count"] >= MarkdownParser.MIN_CHUNK_WORDS

    def test_minimum_word_guard(self, parser):
        """A heading with only 1-2 words of content must be discarded."""
        md = "# Title\n\nHi.\n\n## Real Section\n\nThis section has real content worth keeping.\n"
        chunks, _ = parser._chunk_text(md)
        # "Hi." is 1 word — should be dropped. Real section should survive.
        contents = [c["content"] for c in chunks]
        assert any("real content" in c for c in contents)
        # Verify no chunk has fewer than MIN_CHUNK_WORDS words (except code)
        for chunk in chunks:
            if chunk["chunk_type"] != "code":
                assert chunk["token_count"] >= MarkdownParser.MIN_CHUNK_WORDS

    def test_only_first_h1_becomes_title(self, parser):
        """If there are multiple H1 headings, only the first becomes the title."""
        md = "# First Title\n\nContent here.\n\n# Second Title\n\nMore content.\n"
        _, title = parser._chunk_text(md)
        assert title == "First Title"


# ═══════════════════════════════════════════════════════════════════════
# Large section splitting
# ═══════════════════════════════════════════════════════════════════════

class TestSplitting:

    def test_large_section_splits_into_multiple_chunks(self, small_parser, large_section_md):
        """
        A section with 500 words and a 20-word limit must produce multiple chunks.
        Splitting happens at blank-line paragraph boundaries.
        """
        chunks = small_parser.parse_file(str(large_section_md))["chunks"]
        assert len(chunks) > 1

    def test_split_chunks_are_all_text_type(self, small_parser, large_section_md):
        chunks = small_parser.parse_file(str(large_section_md))["chunks"]
        for chunk in chunks:
            assert chunk["chunk_type"] == "text"

    def test_split_chunks_have_sequential_indices(self, small_parser, large_section_md):
        chunks = small_parser.parse_file(str(large_section_md))["chunks"]
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_no_content_lost_in_split(self, small_parser):
        """
        Total word count across all chunks must equal the word count of the
        original content (modulo heading lines added to buffers).
        This is a loose check — we verify nothing is silently dropped.
        """
        # Build a doc with two clear paragraphs separated by a blank line
        para = "word " * 15  # 15 words per paragraph, limit is 20
        md = f"# Big\n\n{para}\n\n{para}\n\n{para}\n"
        chunks, _ = small_parser._chunk_text(md)
        total_words = sum(c["token_count"] for c in chunks)
        # 3 paragraphs × 15 words = 45, plus heading words — must be >= 45
        assert total_words >= 45


# ═══════════════════════════════════════════════════════════════════════
# File hash
# ═══════════════════════════════════════════════════════════════════════

class TestFileHash:

    def test_hash_is_sha256_hex(self, parser, simple_md):
        result = parser.parse_file(str(simple_md))
        h = result["file_hash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_content_same_hash(self, parser, tmp_path):
        content = "# Title\n\nSame content in both files.\n"
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text(content, encoding="utf-8")
        f2.write_text(content, encoding="utf-8")
        assert parser.parse_file(str(f1))["file_hash"] == \
               parser.parse_file(str(f2))["file_hash"]

    def test_different_content_different_hash(self, parser, tmp_path):
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("# Title\n\nContent A here.\n", encoding="utf-8")
        f2.write_text("# Title\n\nContent B here.\n", encoding="utf-8")
        assert parser.parse_file(str(f1))["file_hash"] != \
               parser.parse_file(str(f2))["file_hash"]

    def test_hash_matches_manual_sha256(self, parser, simple_md):
        content = simple_md.read_text(encoding="utf-8")
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert parser.parse_file(str(simple_md))["file_hash"] == expected
