"""
CorpusForge Compiler
=====================
Produces output documents from corpus data stored in SQLite.

All methods are local and deterministic — no LLM calls except compile_topic_llm,
which is the opt-in prose path retained for cases where narrative output is wanted.

Output modes:
  generate_topic_outline(topic_id)   — hierarchical checklist (Markdown)
  compile_linear_document(topic_id)  — chunks in reading order with XML tags
  export_tagged_document(file_id)    — single file with chunk tags
  compile_topic_llm(topic_id, ...)   — LLM-synthesized prose (requires API key)

All methods write to `output_dir` (default: current directory) and return a
CompileResult on success or None if the topic/file ID does not exist.

Usage:
    from corpusforge.compiler import LocalCompiler
    compiler = LocalCompiler()
    result = compiler.generate_topic_outline(topic_id=3)
    if result:
        print(f"Written {result.chunk_count} chunks to {result.output_path}")
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .db import CorpusDB


# ── Output descriptor ─────────────────────────────────────────────────

@dataclass
class CompileResult:
    """Returned by every LocalCompiler method on success."""
    output_path: str   # absolute or relative path of the written file
    chunk_count: int   # number of chunks included in the output


# ── Helpers ───────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    """
    Convert an arbitrary string to a filesystem-safe slug.

    Non-alphanumeric characters become underscores; leading/trailing
    underscores are stripped.

    Examples:
        "Cluster 3"       -> "Cluster_3"
        "Auth & Security" -> "Auth___Security"
        " leading "       -> "leading"
    """
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def _ensure_dir(path: str) -> Path:
    """Create directory if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Compiler ──────────────────────────────────────────────────────────

class LocalCompiler:
    """
    Produces output documents from the corpus database.

    Inject a CorpusDB instance for testing or to share a connection across
    operations. If not provided, a default instance is created.
    """

    def __init__(self, db: CorpusDB | None = None):
        self.db = db or CorpusDB()

    # ── Private helpers ───────────────────────────────────────────────

    def _get_topic(self, topic_id: int):
        """Return the topic row or None. Uses direct ID lookup via db.get_topic()."""
        return self.db.get_topic(topic_id)

    def _topic_filename_stem(self, topic_id: int, topic_name: str, prefix: str) -> str:
        """Build a deterministic output filename stem for a topic."""
        return f"{prefix}_{topic_id}_{_safe_filename(topic_name)}"

    # ── Public output methods ─────────────────────────────────────────

    def generate_topic_outline(
        self, topic_id: int, output_dir: str = "."
    ) -> CompileResult | None:
        """
        Generate a hierarchical checklist outline for a topic.

        Structure: one section per source file, one sub-section per heading
        path within that file, with chunks listed as Markdown checkboxes
        sorted by HDBSCAN similarity score (most relevant first).

        This is the primary local output format — no LLM required.

        Args:
            topic_id:   ID from cf topics.
            output_dir: directory to write into (created if absent).

        Returns CompileResult on success, None if topic_id not found.

        Example output filename: outline_3_Cluster_0.md
        """
        topic = self._get_topic(topic_id)
        if not topic:
            return None

        chunks = self.db.get_chunks_for_topic(topic_id)
        if not chunks:
            return None

        # Group chunks by file then by heading_path for the tree structure.
        # defaultdict(lambda: defaultdict(list)) gives us tree[file][heading] = [chunks]
        tree: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for c in chunks:
            tree[c["filename"]][c["heading_path"] or "Root"].append(c)

        out_path = _ensure_dir(output_dir) / f"{self._topic_filename_stem(topic_id, topic['name'], 'outline')}.md"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Topic {topic_id}: {topic['name']}\n")
            f.write(f"*{topic['description']}*\n\n---\n\n")

            for fname in sorted(tree):
                f.write(f"## {fname}\n\n")
                for hpath in sorted(tree[fname]):
                    f.write(f"### {hpath}\n")
                    # Sort within each heading group by similarity DESC.
                    for c in sorted(tree[fname][hpath], key=lambda x: x["similarity"], reverse=True):
                        preview = c["content"].replace("\n", " ")[:120].strip()
                        f.write(
                            f"- [ ] **[Chunk {c['id']}]** "
                            f"*(Relevance: {c['similarity']:.3f})* — {preview}...\n"
                        )
                    f.write("\n")

        return CompileResult(str(out_path), len(chunks))

    def compile_linear_document(
        self, topic_id: int, output_dir: str = "."
    ) -> CompileResult | None:
        """
        Reconstruct topic chunks in their original reading order with XML tags.

        Chunks are sorted by (file_id, chunk_index) to recover document order
        within each source file. File boundaries are marked with <file> tags;
        heading changes within a file with <header> tags; each chunk with a
        <chunk> tag carrying its ID and relevance score.

        This format is designed for human review of topic content in context,
        and as a stepping stone toward the full XML annotation pipeline.

        Args:
            topic_id:   ID from cf topics.
            output_dir: directory to write into (created if absent).

        Returns CompileResult on success, None if topic_id not found.

        Example output filename: linear_3_Cluster_0.md
        """
        topic = self._get_topic(topic_id)
        if not topic:
            return None

        chunks = self.db.get_chunks_for_topic(topic_id)
        if not chunks:
            return None

        # Original reading order: sort by source file, then by position in file.
        ordered = sorted(chunks, key=lambda x: (x["file_id"], x["chunk_index"]))

        out_path = _ensure_dir(output_dir) / f"{self._topic_filename_stem(topic_id, topic['name'], 'linear')}.md"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Topic {topic_id}: {topic['name']}\n")
            f.write(f"*{topic['description']}*\n\n---\n\n")

            current_file_id: int | None = None
            current_heading: str | None = None

            for c in ordered:
                # Emit a <file> boundary tag when the source file changes.
                if c["file_id"] != current_file_id:
                    if current_file_id is not None:
                        f.write("</file>\n\n")
                    current_file_id = c["file_id"]
                    current_heading = None
                    f.write(
                        f'<file id="{c["file_id"]}" name="{c["filename"]}">\n'
                        f"## Source: {c['filename']}\n\n"
                    )

                # Emit a <header> tag when the heading path changes within a file.
                hpath = c["heading_path"] or "Root"
                if hpath != current_heading:
                    current_heading = hpath
                    if hpath != "Root":
                        f.write(f'<header path="{hpath}">\n### {hpath}\n</header>\n\n')

                # Each chunk with its ID and HDBSCAN relevance score.
                relevance = dict(c).get("similarity", 0.0)
                f.write(
                    f'<chunk id="{c["id"]}" relevance="{relevance:.3f}">\n'
                    f"{c['content']}\n"
                    f"</chunk>\n\n"
                )

            if current_file_id is not None:
                f.write("</file>\n")

        return CompileResult(str(out_path), len(ordered))

    def export_tagged_document(
        self, file_id: int, output_dir: str = "exports"
    ) -> CompileResult | None:
        """
        Export a single source file with each chunk wrapped in <chunk> tags.

        This annotates the original document content with chunk IDs so the
        DB index can be traced back to the source text. The output is the
        foundation for the planned full XML annotation pipeline.

        Tags carry the chunk ID (for DB lookup) and will eventually carry
        topic and similarity attributes once the annotation pipeline matures.

        Args:
            file_id:    ID from cf files.
            output_dir: directory to write into (default: 'exports/').

        Returns CompileResult on success, None if file_id not found.

        Example output filename: exports/tagged_design.md
        """
        file_record = self.db.get_file(file_id)
        if not file_record:
            return None

        chunks = self.db.get_chunks_for_file(file_id)
        out_path = _ensure_dir(output_dir) / f"tagged_{file_record['filename']}"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f'<document file_id="{file_id}" filename="{file_record["filename"]}">\n\n')
            for c in chunks:
                f.write(
                    f'<chunk id="{c["id"]}" chunk_index="{c["chunk_index"]}" '
                    f'type="{c["chunk_type"]}">\n'
                    f"{c['content']}\n"
                    f"</chunk>\n\n"
                )
            f.write("</document>\n")

        return CompileResult(str(out_path), len(chunks))

    def compile_topic_llm(
        self,
        topic_id: int,
        summarizer,
        max_chunks: int = 12,
        output_dir: str = ".",
    ) -> CompileResult | None:
        """
        Synthesize topic chunks into a prose Markdown document via LLM.

        This is the opt-in prose path. For most use cases, prefer
        generate_topic_outline() or compile_linear_document() which are
        local, instant, and have no chunk-count limit.

        Args:
            topic_id:   ID from cf topics.
            summarizer: a CorpusSummarizer instance (injected so callers
                        control API key and model selection).
            max_chunks: cap on chunks sent to the LLM. Default 12 balances
                        coverage against context bloat and response latency.
                        Chunks are already sorted by relevance DESC.
            output_dir: directory to write into (created if absent).

        Returns CompileResult on success, None if topic_id not found.

        Example output filename: compiled_3_Cluster_0.md
        """
        topic = self._get_topic(topic_id)
        if not topic:
            return None

        chunks = self.db.get_chunks_for_topic(topic_id)
        if not chunks:
            return None

        # Take the top max_chunks by HDBSCAN relevance (already sorted DESC).
        selected = chunks[:max_chunks]
        chunk_texts = [c["content"] for c in selected]

        compiled_text = summarizer.compile_topic(
            topic_name=topic["name"],
            topic_description=topic["description"],
            chunk_texts=chunk_texts,
        )

        out_path = _ensure_dir(output_dir) / f"{self._topic_filename_stem(topic_id, topic['name'], 'compiled')}.md"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(compiled_text)

        # Register the output in the DB so combined_docs tracks what was generated.
        source_file_ids = list({c["file_id"] for c in selected})
        self.db.create_combined_doc(
            title=f"Compiled: {topic['name']}",
            topic_id=topic_id,
            output_path=str(out_path),
            source_file_ids=source_file_ids,
        )

        return CompileResult(str(out_path), len(selected))
