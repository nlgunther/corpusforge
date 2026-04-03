"""
CorpusForge Database Layer
===========================
Single source of truth for all SQLite operations.

Design decisions:
  - All timestamps are ISO 8601 UTC strings, managed by Python (not SQLite defaults).
    SQLite has no ON UPDATE trigger, so Python must pass timestamps explicitly.
    _now() is the single source of truth for timestamp format.
  - PRAGMA foreign_keys = ON is set on every connection in get_connection().
    Without this, ON DELETE CASCADE is silently ignored by SQLite.
  - sqlite3.Row is used as row_factory so callers get dict-like access (row['column'])
    rather than positional tuples. Makes code self-documenting and refactor-safe.
  - transaction() context manager: yields a single connection for multi-step
    atomic operations. All public write methods accept an optional `conn` parameter
    so they can participate in a caller-managed transaction without opening a new one.
    Single-step callers pass no conn and get autocommit behavior.
  - Embeddings are stored as BLOB via numpy.tobytes() / numpy.frombuffer().
    Both directions of BLOB serialization live here — embedder.py knows nothing
    about storage format. At corpus scale (~30 files, ~1000 chunks), in-memory
    similarity via get_all_embeddings() is simpler and faster than FAISS.

Atomic re-ingestion pattern (use transaction() for integrity):
    with db.transaction() as conn:
        file_id, action = db.upsert_file(..., conn=conn)
        if action == "updated":
            db.delete_chunks_for_file(file_id, conn=conn)
        db.insert_chunks_batch(file_id, chunks, conn=conn)
        # single commit on context exit
"""

import json
import sqlite3
import numpy as np
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# schema.sql lives alongside this file in the corpusforge package directory.
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Expected schema version — bump this when schema.sql changes.
_SCHEMA_VERSION = "1.1"


class CorpusDB:
    """
    Database wrapper for CorpusForge.

    Provides CRUD operations for all tables in the schema. All timestamp
    management and embedding serialization happen here — callers never
    need to generate timestamps or handle numpy BLOBs.

    Usage:
        db = CorpusDB()                          # uses default path
        db = CorpusDB("path/to/corpusforge.db")  # explicit path

        # Simple single-step call (auto-commit):
        file_id, action = db.upsert_file(...)

        # Multi-step atomic operation (caller-managed transaction):
        with db.transaction() as conn:
            file_id, action = db.upsert_file(..., conn=conn)
            if action == "updated":
                db.delete_chunks_for_file(file_id, conn=conn)
            db.insert_chunks_batch(file_id, chunks, conn=conn)
    """

    def __init__(self, db_path: str = "corpusforge.db"):
        self.db_path = db_path
        self._ensure_schema()

    # ───────────────────────────────────────────────────────────────────
    # Connection management
    # ───────────────────────────────────────────────────────────────────

    def _make_connection(self) -> sqlite3.Connection:
        """Open a configured connection."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL") # <--- Claude Issue 1a
        conn.execute("PRAGMA page_size = 4096")   # <--- Claude Issue 1a
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _auto_connect(self, conn: Optional[sqlite3.Connection] = None):
        """
        Internal context manager for methods that accept an optional conn.

        If conn is provided (caller is managing a transaction), yield it
        and do NOT commit or close — the caller owns the lifecycle.
        If conn is None, open a new connection, commit on success, always close.

        This is the mechanism that lets every public method work both as a
        standalone call (no conn) and as part of a larger transaction (with conn).
        """
        if conn is not None:
            yield conn  # caller manages commit and close
        else:
            c = self._make_connection()
            try:
                yield c
                c.commit()
            finally:
                c.close()

    @contextmanager
    def transaction(self):
        """
        Context manager for multi-step atomic operations.

        Yields a connection. On successful exit, commits. On exception, rolls back.
        Pass the yielded conn to any db method to enlist it in the transaction.

        Example:
            with db.transaction() as conn:
                file_id, action = db.upsert_file(..., conn=conn)
                if action == "updated":
                    db.delete_chunks_for_file(file_id, conn=conn)
                db.insert_chunks_batch(file_id, chunks, conn=conn)
        """
        conn = self._make_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self):
        """
        Initialize or validate the database schema.

        If the DB file doesn't exist, creates it from schema.sql and seeds
        the schema_version. If it exists, confirms schema_version matches
        _SCHEMA_VERSION — a mismatch means a migration is needed and we
        raise immediately rather than let cryptic SQL errors surface later.
        """
        db_exists = Path(self.db_path).exists()

        if not db_exists:
            if not _SCHEMA_PATH.exists():
                raise FileNotFoundError(
                    f"Schema file not found: {_SCHEMA_PATH}\n"
                    f"Expected alongside db.py in the corpusforge package directory."
                )
            with self.transaction() as conn:
                conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
                # Seed schema_version — done here because schema.sql can't
                # generate a Python UTC timestamp.
                conn.execute(
                    "INSERT INTO corpus_state (key, value, updated_at) VALUES (?, ?, ?)",
                    ("schema_version", _SCHEMA_VERSION, self._now()),
                )
            print(f"[CorpusForge] Initialized new database: {self.db_path}")
            return

        # DB file exists — verify schema version before any other operation.
        with self._auto_connect() as conn:
            row = conn.execute(
                "SELECT value FROM corpus_state WHERE key = 'schema_version'"
            ).fetchone()

        if row is None:
            raise RuntimeError(
                f"Database at '{self.db_path}' has no schema_version. "
                f"It may be corrupt or from an incompatible version. "
                f"Delete the file to reinitialize."
            )

        actual = row["value"]
        if actual != _SCHEMA_VERSION:
            raise RuntimeError(
                f"Schema version mismatch: DB has '{actual}', code expects '{_SCHEMA_VERSION}'. "
                f"A migration is required. See CHANGELOG for migration steps."
            )

    # ───────────────────────────────────────────────────────────────────
    # Timestamp helper — single source of truth
    # ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        """ISO 8601 UTC timestamp string. Used for all created_at / updated_at fields."""
        return datetime.now(timezone.utc).isoformat()

    # ───────────────────────────────────────────────────────────────────
    # Embedding serialization — both directions live here, not in embedder.py
    # ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _emb_to_blob(embedding: np.ndarray) -> bytes:
        """Serialize a NumPy float32 array to bytes for BLOB storage."""
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def blob_to_embedding(blob: bytes) -> np.ndarray:
        """
        Deserialize a BLOB from SQLite back to a NumPy float32 array.
        Returns a copy to prevent read-only buffer mutation errors downstream.
        """
        return np.frombuffer(blob, dtype=np.float32).copy()

    # ═══════════════════════════════════════════════════════════════════
    # EMBEDDING MODEL GUARD
    # ═══════════════════════════════════════════════════════════════════

    def assert_embedding_model(self, model_name: str) -> None:
        """
        Ensure the embedding model is consistent with the database.

        On first use: locks in the model name in corpus_state.
        On subsequent uses: raises RuntimeError if the model has changed,
        since mixing embedding spaces produces meaningless similarity scores.

        Call this once at the start of any ingest or search operation.
        """
        saved = self.get_state("embedding_model")
        if saved is None:
            self.set_state("embedding_model", model_name)
        elif saved != model_name:
            raise RuntimeError(
                f"Embedding model mismatch: DB was built with '{saved}', "
                f"current model is '{model_name}'. "
                f"To switch models, delete the DB and re-ingest the corpus."
            )

    # ═══════════════════════════════════════════════════════════════════
    # FILES
    # ═══════════════════════════════════════════════════════════════════

    def upsert_file(
        self,
        filepath: str,
        filename: str,
        fmt: str,
        file_hash: str,
        title: Optional[str] = None,
        user_summary: Optional[str] = None,
        corpus_role: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> tuple[int, str]:
        """
        Insert a new file or update if hash changed.

        Returns:
            (file_id, action) where action is 'inserted', 'updated', or 'unchanged'.

        On 'updated': the old file row is updated in place. The caller must call
        delete_chunks_for_file() before inserting new chunks. This is not automatic
        because the caller may want to diff old vs. new chunks before deleting.

        Pass conn to participate in a caller-managed transaction.
        """
        now = self._now()
        with self._auto_connect(conn) as c:
            row = c.execute(
                "SELECT id, file_hash FROM files WHERE filepath = ?", (filepath,)
            ).fetchone()

            if row is None:
                cursor = c.execute(
                    """INSERT INTO files
                       (filepath, filename, format, file_hash, title,
                        user_summary, corpus_role, ingested_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (filepath, filename, fmt, file_hash, title,
                     user_summary, corpus_role, now, now),
                )
                return cursor.lastrowid, "inserted"

            if row["file_hash"] == file_hash:
                return row["id"], "unchanged"

            # Hash changed — update metadata, clear stale auto_summary.
            c.execute(
                """UPDATE files
                   SET file_hash = ?, title = ?, auto_summary = NULL, updated_at = ?
                   WHERE id = ?""",
                (file_hash, title, now, row["id"]),
            )
            return row["id"], "updated"

    def get_file(self, file_id: int) -> Optional[sqlite3.Row]:
        """Get a single file by ID."""
        with self._auto_connect() as conn:
            return conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()

    def get_file_by_path(self, filepath: str) -> Optional[sqlite3.Row]:
        """Get a single file by filepath."""
        with self._auto_connect() as conn:
            return conn.execute(
                "SELECT * FROM files WHERE filepath = ?", (filepath,)
            ).fetchone()

    def get_all_files(self) -> list[sqlite3.Row]:
        """List all ingested files, newest first."""
        with self._auto_connect() as conn:
            return conn.execute(
                "SELECT * FROM files ORDER BY updated_at DESC"
            ).fetchall()

    def update_file_summary(
        self, file_id: int, auto_summary: str,
        conn: Optional[sqlite3.Connection] = None
    ) -> None:
        """Set the LLM-generated summary for a file."""
        with self._auto_connect(conn) as c:
            c.execute(
                "UPDATE files SET auto_summary = ?, updated_at = ? WHERE id = ?",
                (auto_summary, self._now(), file_id),
            )

    def update_file_metadata(self, file_id: int, **fields) -> None:
        """
        Update arbitrary file metadata fields.

        Valid fields: title, user_summary, auto_summary, corpus_role.
        Automatically sets updated_at. Rejects unknown field names to
        prevent SQL injection via dynamic column names.
        """
        ALLOWED = {"title", "user_summary", "auto_summary", "corpus_role"}
        invalid = set(fields.keys()) - ALLOWED
        if invalid:
            raise ValueError(f"Unknown file fields: {invalid}. Allowed: {ALLOWED}")

        fields["updated_at"] = self._now()
        set_clause = ", ".join(f"{k} = ?" for k in fields)

        with self._auto_connect() as conn:
            conn.execute(
                f"UPDATE files SET {set_clause} WHERE id = ?",
                [*fields.values(), file_id],
            )

    def update_topic_metadata(self, topic_id: int, **fields) -> None:
        """
        Update arbitrary topic metadata fields.

        Valid fields: name, description.
        Automatically sets updated_at. Rejects unknown field names.
        """
        ALLOWED = {"name", "description"}
        invalid = set(fields.keys()) - ALLOWED
        if invalid:
            raise ValueError(f"Unknown topic fields: {invalid}. Allowed: {ALLOWED}")

        fields["updated_at"] = self._now()
        set_clause = ", ".join(f"{k} = ?" for k in fields)

        with self._auto_connect() as conn:
            conn.execute(
                f"UPDATE topics SET {set_clause} WHERE id = ?",
                [*fields.values(), topic_id],
            )
    
    def delete_file(self, file_id: int) -> bool:
        """
        Delete a file and all its chunks (via CASCADE).
        Returns True if a row was actually deleted.
        """
        with self._auto_connect() as conn:
            cursor = conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            return cursor.rowcount > 0

    # ═══════════════════════════════════════════════════════════════════
    # CHUNKS
    # ═══════════════════════════════════════════════════════════════════

    def insert_chunks_batch(
        self, file_id: int, chunks: list[dict],
        conn: Optional[sqlite3.Connection] = None,
    ) -> list[int]:
        """
        Insert multiple chunks in a single transaction.

        Each dict in `chunks` must have: chunk_index, content, token_count.
        Optional keys: chunk_type (default 'text'), heading_path, embedding.

        Embedding values are np.ndarray; serialization to BLOB happens here.
        Returns list of chunk IDs in insertion order.

        Pass conn to participate in a caller-managed transaction.
        """
        chunk_ids = []
        with self._auto_connect(conn) as c:
            for chunk in chunks:
                emb = chunk.get("embedding")
                emb_blob = self._emb_to_blob(emb) if emb is not None else None

                cursor = c.execute(
                    """INSERT INTO chunks
                       (file_id, chunk_index, chunk_type, heading_path,
                        content, token_count, embedding)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        file_id,
                        chunk["chunk_index"],
                        chunk.get("chunk_type", "text"),
                        chunk.get("heading_path"),
                        chunk["content"],
                        chunk["token_count"],
                        emb_blob,
                    ),
                )
                chunk_ids.append(cursor.lastrowid)
        return chunk_ids

    def get_chunks_for_file(self, file_id: int) -> list[sqlite3.Row]:
        """Get all chunks for a file, ordered by chunk_index."""
        with self._auto_connect() as conn:
            return conn.execute(
                "SELECT * FROM chunks WHERE file_id = ? ORDER BY chunk_index",
                (file_id,),
            ).fetchall()

    def get_chunk(self, chunk_id: int) -> Optional[sqlite3.Row]:
        """Get a single chunk by ID."""
        with self._auto_connect() as conn:
            return conn.execute(
                "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()

    def get_all_embeddings(self) -> list[tuple[int, np.ndarray]]:
        """
        Load all chunk embeddings into memory for similarity computation.

        Returns list of (chunk_id, embedding_array) tuples.
        Skips chunks with NULL embeddings.
        At corpus scale (~1000 chunks @ 384 dims), full in-memory load is ~1.5MB.
        """
        with self._auto_connect() as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
            ).fetchall()

        return [
            (row["id"], self.blob_to_embedding(row["embedding"]))
            for row in rows
        ]

    def delete_chunks_for_file(
        self, file_id: int,
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        """
        Delete all chunks for a file (for re-ingestion).

        CASCADE automatically removes orphaned chunk_topics and cross_refs.
        Returns count of deleted chunks.

        Pass conn to participate in a caller-managed transaction.
        """
        with self._auto_connect(conn) as c:
            cursor = c.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            return cursor.rowcount

    def update_chunk_summary(self, chunk_id: int, summary: str) -> None:
        """Set the lazily-generated summary for a chunk."""
        with self._auto_connect() as conn:
            conn.execute(
                "UPDATE chunks SET summary = ? WHERE id = ?", (summary, chunk_id)
            )

    def update_chunk_embedding(self, chunk_id: int, embedding: np.ndarray) -> None:
        """Update the embedding for a chunk (e.g., after model change)."""
        with self._auto_connect() as conn:
            conn.execute(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                (self._emb_to_blob(embedding), chunk_id),
            )

    # ═══════════════════════════════════════════════════════════════════
    # TOPICS
    # ═══════════════════════════════════════════════════════════════════

    def create_topic(
        self,
        name: str,
        description: Optional[str] = None,
        parent_topic_id: Optional[int] = None,
    ) -> int:
        """Create a topic. Returns the topic ID."""
        now = self._now()
        with self._auto_connect() as conn:
            cursor = conn.execute(
                """INSERT INTO topics (name, description, parent_topic_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, description, parent_topic_id, now, now),
            )
            return cursor.lastrowid

    def get_topic(self, topic_id: int) -> Optional[sqlite3.Row]:
        """Get a single topic by ID. Returns None if not found.

        Prefer this over get_all_topics() + dict lookup when you only need
        one topic — avoids loading the entire topics table into memory.
        """
        with self._auto_connect() as conn:
            return conn.execute(
                "SELECT * FROM topics WHERE id = ?", (topic_id,)
            ).fetchone()

    def get_all_topics(self, roots_only: bool = False) -> list[sqlite3.Row]:
        """
        List topics. If roots_only=True, returns only broad (Pass 1) topics
        where parent_topic_id IS NULL.
        """
        query = "SELECT * FROM topics"
        if roots_only:
            query += " WHERE parent_topic_id IS NULL"
        query += " ORDER BY name"

        with self._auto_connect() as conn:
            return conn.execute(query).fetchall()

    def get_subtopics(self, parent_id: int) -> list[sqlite3.Row]:
        """Get child topics of a given parent."""
        with self._auto_connect() as conn:
            return conn.execute(
                "SELECT * FROM topics WHERE parent_topic_id = ? ORDER BY name",
                (parent_id,),
            ).fetchall()

    def update_topic(
        self, topic_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Update topic name and/or description."""
        fields = {}
        if name is not None:
            fields["name"] = name
        if description is not None:
            fields["description"] = description
        if not fields:
            return

        fields["updated_at"] = self._now()
        set_clause = ", ".join(f"{k} = ?" for k in fields)

        with self._auto_connect() as conn:
            conn.execute(
                f"UPDATE topics SET {set_clause} WHERE id = ?",
                [*fields.values(), topic_id],
            )

    def delete_all_topics(self) -> int:
        """
        Clear all topics and chunk_topics for full re-clustering.
        CASCADE handles chunk_topics cleanup.
        Returns count of deleted topics.
        """
        with self._auto_connect() as conn:
            cursor = conn.execute("DELETE FROM topics")
            return cursor.rowcount

    # ═══════════════════════════════════════════════════════════════════
    # CHUNK ↔ TOPIC MAPPING
    # ═══════════════════════════════════════════════════════════════════

    def assign_chunk_to_topic(self, chunk_id: int, topic_id: int, similarity: float) -> None:
        """Assign a chunk to a topic with its similarity score."""
        with self._auto_connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO chunk_topics (chunk_id, topic_id, similarity)
                   VALUES (?, ?, ?)""",
                (chunk_id, topic_id, similarity),
            )

    def assign_chunks_to_topic_batch(
        self, assignments: list[tuple[int, int, float]]
    ) -> None:
        """
        Batch assign chunks to topics.
        Each tuple: (chunk_id, topic_id, similarity).
        """
        with self._auto_connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO chunk_topics (chunk_id, topic_id, similarity)
                   VALUES (?, ?, ?)""",
                assignments,
            )

    def get_chunks_for_topic(self, topic_id: int) -> list[sqlite3.Row]:
        """Get all chunks assigned to a topic, with file info, ranked by similarity."""
        with self._auto_connect() as conn:
            return conn.execute(
                """SELECT c.*, ct.similarity, f.filepath, f.filename, f.title AS file_title
                   FROM chunks c
                   JOIN chunk_topics ct ON ct.chunk_id = c.id
                   JOIN files f ON f.id = c.file_id
                   WHERE ct.topic_id = ?
                   ORDER BY ct.similarity DESC""",
                (topic_id,),
            ).fetchall()

    def get_files_for_topic(self, topic_id: int) -> list[sqlite3.Row]:
        """Get files associated with a topic via the file_topics view."""
        with self._auto_connect() as conn:
            return conn.execute(
                """SELECT f.*, ft.avg_similarity, ft.chunk_count, ft.relevance
                   FROM file_topics ft
                   JOIN files f ON f.id = ft.file_id
                   WHERE ft.topic_id = ?
                   ORDER BY ft.relevance DESC""",
                (topic_id,),
            ).fetchall()

    def get_topics_for_file(self, file_id: int) -> list[sqlite3.Row]:
        """Get topics a file belongs to via the file_topics view."""
        with self._auto_connect() as conn:
            return conn.execute(
                """SELECT t.*, ft.avg_similarity, ft.chunk_count, ft.relevance
                   FROM file_topics ft
                   JOIN topics t ON t.id = ft.topic_id
                   WHERE ft.file_id = ?
                   ORDER BY ft.relevance DESC""",
                (file_id,),
            ).fetchall()

    # ═══════════════════════════════════════════════════════════════════
    # CROSS-REFERENCES (Phase 4: Rationalization)
    # ═══════════════════════════════════════════════════════════════════

    def insert_cross_ref(
        self,
        chunk_a_id: int,
        chunk_b_id: int,
        similarity: float,
        relationship: Optional[str] = None,
    ) -> int:
        """
        Record a detected relationship between two chunks.

        Enforces canonical ordering (a < b) to prevent duplicate pairs.
        Returns the cross_ref ID.
        """
        if chunk_a_id > chunk_b_id:
            chunk_a_id, chunk_b_id = chunk_b_id, chunk_a_id

        with self._auto_connect() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO cross_refs
                   (chunk_a_id, chunk_b_id, similarity, relationship, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (chunk_a_id, chunk_b_id, similarity, relationship, self._now()),
            )
            return cursor.lastrowid

    def insert_cross_refs_batch(
        self, refs: list[tuple[int, int, float, Optional[str]]]
    ) -> None:
        """
        Batch insert cross-references.
        Each tuple: (chunk_a_id, chunk_b_id, similarity, relationship).
        Canonical ordering is enforced automatically.
        """
        now = self._now()
        rows = []
        for a, b, sim, rel in refs:
            if a > b:
                a, b = b, a
            rows.append((a, b, sim, rel, now))

        with self._auto_connect() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO cross_refs
                   (chunk_a_id, chunk_b_id, similarity, relationship, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                rows,
            )

    def get_pending_cross_refs(self) -> list[sqlite3.Row]:
        """Get all unresolved cross-references for review, ranked by similarity."""
        with self._auto_connect() as conn:
            return conn.execute(
                """SELECT cr.*,
                          ca.content AS content_a, ca.heading_path AS heading_a,
                          fa.filename AS file_a,
                          cb.content AS content_b, cb.heading_path AS heading_b,
                          fb.filename AS file_b
                   FROM cross_refs cr
                   JOIN chunks ca ON ca.id = cr.chunk_a_id
                   JOIN chunks cb ON cb.id = cr.chunk_b_id
                   JOIN files fa ON fa.id = ca.file_id
                   JOIN files fb ON fb.id = cb.file_id
                   WHERE cr.status = 'pending'
                   ORDER BY cr.similarity DESC"""
            ).fetchall()

    def resolve_cross_ref(
        self,
        cross_ref_id: int,
        status: str,
        resolved_by: str,
        resolved_text: Optional[str] = None,
    ) -> None:
        """
        Resolve a cross-reference and automatically clean up redundant overlaps.
        """
        with self._auto_connect() as conn:
            # 1. Fetch the chunk IDs involved in this specific resolution
            row = conn.execute(
                "SELECT chunk_a_id, chunk_b_id FROM cross_refs WHERE id = ?", 
                (cross_ref_id,)
            ).fetchone()
            
            if not row:
                return

            # 2. Update the target cross-reference as requested
            conn.execute(
                """UPDATE cross_refs
                   SET status = ?, resolved_by = ?, resolved_text = ?, resolved_at = ?
                   WHERE id = ?""",
                (status, resolved_by, resolved_text, self._now(), cross_ref_id),
            )

            # 3. AUTO-CLEANUP: If we accepted a merge/dedup, all other PENDING 
            # overlaps involving either chunk are now mathematically redundant.
            if status == 'accepted':
                conn.execute(
                    """UPDATE cross_refs 
                       SET status = 'redundant', resolved_at = ? 
                       WHERE status = 'pending' 
                       AND id != ? 
                       AND (chunk_a_id IN (?, ?) OR chunk_b_id IN (?, ?))""",
                    (self._now(), cross_ref_id, row['chunk_a_id'], row['chunk_b_id'], 
                     row['chunk_a_id'], row['chunk_b_id'])
                )

    # ═══════════════════════════════════════════════════════════════════
    # FEEDBACK
    # ═══════════════════════════════════════════════════════════════════

    def log_feedback(
        self,
        context_type: str,
        decision: str,
        source: str,
        context_id: Optional[int] = None,
        detail: Optional[dict] = None,
    ) -> int:
        """
        Record a decision for future learning.

        Args:
            context_type: 'topic_assignment', 'topic_name', 'merge_proposal', etc.
            decision:     'accept', 'reject', 'modify'
            source:       'hil', 'llm', 'auto'
            context_id:   FK to the relevant table row
            detail:       arbitrary dict serialized as JSON

        Returns the feedback ID.
        """
        with self._auto_connect() as conn:
            cursor = conn.execute(
                """INSERT INTO feedback (context_type, context_id, source, decision, detail, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    context_type,
                    context_id,
                    source,
                    decision,
                    json.dumps(detail) if detail else None,
                    self._now(),
                ),
            )
            return cursor.lastrowid

    def get_feedback(
        self,
        context_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> list[sqlite3.Row]:
        """Query feedback log with optional filters."""
        query = "SELECT * FROM feedback WHERE 1=1"
        params = []
        if context_type:
            query += " AND context_type = ?"
            params.append(context_type)
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY created_at DESC"

        with self._auto_connect() as conn:
            return conn.execute(query, params).fetchall()

    # ═══════════════════════════════════════════════════════════════════
    # CORPUS STATE
    # ═══════════════════════════════════════════════════════════════════

    def get_state(self, key: str) -> Optional[str]:
        """Get a corpus state value by key. Returns None if not set."""
        with self._auto_connect() as conn:
            row = conn.execute(
                "SELECT value FROM corpus_state WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        """Set a corpus state value (insert or update)."""
        now = self._now()
        with self._auto_connect() as conn:
            conn.execute(
                """INSERT INTO corpus_state (key, value, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?""",
                (key, value, now, value, now),
            )

    # ═══════════════════════════════════════════════════════════════════
    # COMBINED DOCUMENTS (Phase 3)
    # ═══════════════════════════════════════════════════════════════════

    def create_combined_doc(
        self,
        title: str,
        topic_id: Optional[int] = None,
        output_path: Optional[str] = None,
        file_hash: Optional[str] = None,
        source_file_ids: Optional[list[int]] = None,
    ) -> int:
        """
        Create a combined document record with its ordered source file list.

        Args:
            source_file_ids: ordered list of file IDs contributing to this doc.
                             List order becomes inclusion_order.

        Returns the combined_doc ID.
        """
        now = self._now()
        with self._auto_connect() as conn:
            cursor = conn.execute(
                """INSERT INTO combined_docs (topic_id, title, output_path, file_hash, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (topic_id, title, output_path, file_hash, now, now),
            )
            doc_id = cursor.lastrowid

            if source_file_ids:
                conn.executemany(
                    """INSERT INTO combined_doc_sources (combined_doc_id, file_id, inclusion_order)
                       VALUES (?, ?, ?)""",
                    [(doc_id, fid, i) for i, fid in enumerate(source_file_ids)],
                )
            return doc_id

    def get_combined_docs(self) -> list[sqlite3.Row]:
        """List all combined documents, newest first."""
        with self._auto_connect() as conn:
            return conn.execute(
                """SELECT cd.*, t.name AS topic_name
                   FROM combined_docs cd
                   LEFT JOIN topics t ON t.id = cd.topic_id
                   ORDER BY cd.updated_at DESC"""
            ).fetchall()

    # ═══════════════════════════════════════════════════════════════════
    # CORPUS STATISTICS (for cf status)
    # ═══════════════════════════════════════════════════════════════════

    def get_corpus_stats(self) -> dict:
        """
        Compute live corpus statistics for the status display.

        Returns dict with file_count, chunk_count, topic_count,
        embedded_chunk_count, pending_cross_refs, total_feedback,
        and corpus_summary (from state table, may be None).
        """
        with self._auto_connect() as conn:
            # Run all count queries on the same connection for consistency.
            stats = {}
            for key, query in [
                ("file_count",           "SELECT COUNT(*) FROM files"),
                ("chunk_count",          "SELECT COUNT(*) FROM chunks"),
                ("topic_count",          "SELECT COUNT(*) FROM topics"),
                ("embedded_chunk_count", "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"),
                ("pending_cross_refs",   "SELECT COUNT(*) FROM cross_refs WHERE status = 'pending'"),
                ("total_feedback",       "SELECT COUNT(*) FROM feedback"),
            ]:
                stats[key] = conn.execute(query).fetchone()[0]

            row = conn.execute(
                "SELECT value FROM corpus_state WHERE key = 'corpus_summary'"
            ).fetchone()
            stats["corpus_summary"] = row["value"] if row else None

        return stats
