"""
CorpusForge Topic Engine
=========================
Groups semantic chunks into coherent topics using HDBSCAN clustering.

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise):
  - Density-based: does not force every chunk into a cluster. Outliers are labeled
    -1 ("noise") and are silently skipped — this is expected and correct behaviour.
  - Parameter-free: we do not guess the number of topics upfront; HDBSCAN finds
    dense regions in the vector space automatically.
  - all-MiniLM-L6-v2 vectors are L2-normalized, so Euclidean distance in the
    embedding space is mathematically equivalent to cosine distance. We use
    'euclidean' as the metric because sklearn's HDBSCAN optimises it better.

Workflow:
    engine = TopicEngine(min_cluster_size=3)
    n_topics = engine.cluster_corpus()   # destructive re-cluster
    engine.name_topics()                 # optional LLM naming pass (requires GEMINI_API_KEY)
"""

import numpy as np
from sklearn.cluster import HDBSCAN

from .db import CorpusDB


class TopicEngine:
    """
    Orchestrates HDBSCAN clustering and (optionally) LLM-based topic naming.

    The CorpusDB instance is injected so callers can share a single DB
    object across operations and tests can supply an isolated fixture DB.

    Usage:
        engine = TopicEngine(min_cluster_size=3)
        n = engine.cluster_corpus()   # returns number of topics created
        engine.name_topics()          # fills in human-readable names via LLM
    """

    def __init__(self, min_cluster_size: int = 3, db: CorpusDB | None = None):
        # min_cluster_size=3: a topic needs at least 3 related chunks to form.
        # Increase for larger corpora where you want broader, more stable topics.
        self.min_cluster_size = min_cluster_size
        self.db = db or CorpusDB()

    def cluster_corpus(self) -> int:
        """
        Pull all chunk embeddings, run HDBSCAN, write results to the DB.

        This is a DESTRUCTIVE operation — all existing topics and chunk_topic
        assignments are cleared before the new cluster labels are written.
        The vector space is treated as the authoritative state; the DB reflects
        the most recent clustering run.

        Records 'last_full_recluster' timestamp in corpus_state so cf status
        can show when clustering was last run.

        Returns the number of distinct topics created (noise chunks excluded).
        Returns 0 if there are no embeddings to cluster.
        """
        all_embeddings = self.db.get_all_embeddings()
        if not all_embeddings:
            return 0

        # Unpack [(chunk_id, vector), ...] into parallel arrays for sklearn.
        chunk_ids = np.array([item[0] for item in all_embeddings])
        vectors   = np.array([item[1] for item in all_embeddings])

        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",              # equivalent to cosine for L2-normalised vectors
            cluster_selection_method="eom",  # Excess of Mass: finds stable, broad clusters
            copy=True,                       # explicit: suppress sklearn FutureWarning re default change
        )
        clusterer.fit(vectors)

        labels        = clusterer.labels_        # int array; -1 = noise
        probabilities = clusterer.probabilities_  # float array; 0.0-1.0 cluster membership

        # Full re-cluster: wipe existing topics before writing new ones.
        # This keeps the DB consistent with the current vector space.
        self.db.delete_all_topics()

        topics_created = 0
        for label in sorted(set(labels)):  # sorted for deterministic topic IDs
            if label == -1:
                continue  # noise chunk — not assigned to any topic

            topic_id = self.db.create_topic(
                name=f"Cluster {label}",
                description="Pending LLM analysis...",
            )
            topics_created += 1

            # Find all chunks belonging to this cluster and batch-assign them.
            # HDBSCAN probability is used as the similarity score: it represents
            # how firmly a chunk belongs to its cluster (1.0 = core member).
            mask = (labels == label)
            assignments = [
                (int(cid), topic_id, float(prob))
                for cid, prob in zip(chunk_ids[mask], probabilities[mask])
            ]
            self.db.assign_chunks_to_topic_batch(assignments)

        # Record when clustering last ran so cf status can surface this.
        self.db.set_state("last_full_recluster", self.db._now())

        return topics_created

    def name_topics(self) -> None:
        """
        Pass a sample of each topic's top chunks to the LLM and update the
        topic name and description in the database.

        Makes one Gemini API call per topic, sequentially. For a corpus with
        N topics expect N round-trips — intentionally sequential to avoid
        rate-limit issues on free-tier API keys.

        Requires GEMINI_API_KEY. If the LLM is unavailable, each topic is
        silently left with its 'Cluster N / Pending LLM analysis...' defaults.

        Uses 3 sample chunks per topic — enough for the LLM to infer the
        common theme; more chunks add noise rather than signal for naming.
        """
        # Lazy relative import: avoids loading google-genai on every CLI invocation.
        from .summarizer import CorpusSummarizer
        summarizer = CorpusSummarizer()

        for topic in self.db.get_all_topics():
            chunks = self.db.get_chunks_for_topic(topic["id"])
            if not chunks:
                continue

            # Top-3 chunks by HDBSCAN probability (already sorted DESC by the query).
            sample_texts = [c["content"] for c in chunks[:3]]
            topic_info = summarizer.name_topic(sample_texts)

            self.db.update_topic(
                topic_id=topic["id"],
                name=topic_info.get("name", "Unnamed Topic"),
                description=topic_info.get("description", ""),
            )
