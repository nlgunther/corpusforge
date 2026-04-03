"""
CorpusForge Rationalizer
===========================
Phase 4: Analyzes topics to mathematically identify duplicate or overlapping chunks.

This module acts as the automated editor for the corpus. By computing an N x N
similarity matrix for all chunks within a given topic, it surfaces pairs of 
chunks that share high semantic overlap. 

These identified pairs are meant to be inserted into the `cross_refs` database 
table as 'pending' proposals, which can later be resolved (merged or rejected) 
by either a Human-in-the-Loop (HIL) or an LLM.
"""

import numpy as np
from typing import List, Tuple
from .db import CorpusDB

class TopicRationalizer:
    """
    Computes semantic overlaps within a specific topic using vector math.
    
    The database instance is injected so callers can share a single DB
    connection across operations, maintaining transaction integrity.
    """

    def __init__(self, db: CorpusDB | None = None):
        self.db = db or CorpusDB()

    def find_topic_overlaps(
        self, 
        topic_id: int, 
        threshold: float = 0.85
    ) -> List[Tuple[int, int, float]]:
        """
        Computes the cosine similarity between all chunks assigned to a topic.
        
        Args:
            topic_id: The ID of the topic to analyze.
            threshold: The minimum cosine similarity score (0.0 to 1.0) required 
                       to flag two chunks as an overlap. Default is 0.85.
                       
        Returns:
            A list of tuples representing overlapping pairs: 
            [(chunk_a_id, chunk_b_id, similarity_score), ...]
            The list is sorted in descending order of similarity.
        """
        
        # -------------------------------------------------------------------
        # 1. Data Extraction
        # -------------------------------------------------------------------
        # Fetch all chunks associated with this topic.
        # We need at least two chunks to perform a comparison.
        chunks = self.db.get_chunks_for_topic(topic_id)
        if len(chunks) < 2:
            return []

        # We must extract the actual numpy arrays from the database BLOBs.
        # get_chunks_for_topic returns sqlite3.Row objects without embeddings,
        # so we fetch the raw embedding data via get_chunk().
        chunk_ids = [c["id"] for c in chunks]
        embeddings = []
        valid_ids = []

        for cid in chunk_ids:
            chunk_data = self.db.get_chunk(cid)
            
            # Guard against missing embeddings (e.g., if a chunk failed to embed during ingest)
            if chunk_data and chunk_data["embedding"] is not None:
                vec = self.db.blob_to_embedding(chunk_data["embedding"])
                embeddings.append(vec)
                valid_ids.append(cid)

        # Final safety check after filtering out missing embeddings
        if len(embeddings) < 2:
            return []

        # -------------------------------------------------------------------
        # 2. Vectorized Math (Similarity Matrix)
        # -------------------------------------------------------------------
        # Stack the individual 384-dimensional vectors into a single 2D NumPy array.
        # Shape: (N, 384) where N is the number of valid chunks.
        matrix = np.vstack(embeddings)
        
        # Compute the N x N self-similarity matrix.
        # MATHEMATICAL NOTE: The 'all-MiniLM-L6-v2' model outputs L2-normalized vectors.
        # Because the magnitude of every vector is exactly 1.0, the dot product 
        # (A dot B) is mathematically identical to Cosine Similarity.
        # This allows us to use a highly optimized matrix multiplication (dot) 
        # instead of calculating norms iteratively.
        sim_matrix = np.dot(matrix, matrix.T)

        # -------------------------------------------------------------------
        # 3. Overlap Extraction
        # -------------------------------------------------------------------
        results = []
        num_chunks = len(valid_ids)
        
        # Iterate over the strict upper triangle of the similarity matrix.
        # We start j at (i + 1) to completely ignore:
        #   - The main diagonal (i == j): A chunk's similarity to itself (always 1.0)
        #   - The lower triangle (i > j): Redundant inverse comparisons (e.g., B vs A)
        for i in range(num_chunks):
            for j in range(i + 1, num_chunks):
                
                # Extract the scalar similarity score
                score = float(sim_matrix[i, j])
                
                # If the score meets or exceeds our strict overlap threshold, flag it
                if score >= threshold:
                    # Enforce a canonical ordering (smaller ID first) to ensure 
                    # uniqueness when inserting into the cross_refs table later.
                    chunk_a = valid_ids[i]
                    chunk_b = valid_ids[j]
                    if chunk_a > chunk_b:
                        chunk_a, chunk_b = chunk_b, chunk_a
                        
                    results.append((chunk_a, chunk_b, score))

        # Sort the final results so the most egregious overlaps (highest similarity) 
        # are at the top of the list for immediate review.
        return sorted(results, key=lambda x: x[2], reverse=True)
    
    def auto_resolve_exact_matches(self) -> int:
        """
        Automatically resolves pending cross-references that are exact duplicates 
        (similarity >= 0.999). Prefers to keep the chunk with the more specific heading path.
        
        Returns:
            The number of cross-references automatically resolved.
        """
        pending = self.db.get_pending_cross_refs()
        resolved_count = 0
        
        for ref in pending:
            # Catch floating point variations (0.999 is functionally identical in 384d space)
            if ref["similarity"] >= 0.999:
                h_a = ref["heading_a"] or "Root"
                h_b = ref["heading_b"] or "Root"
                
                # GLOBAL RULE: Keep the chunk with the most specific heading.
                # If Chunk A is just "Root" but Chunk B has a real heading, Chunk B wins.
                # Otherwise, Chunk A wins by default.
                if h_a == "Root" and h_b != "Root":
                    best_text = ref["content_b"]
                else:
                    best_text = ref["content_a"]
                    
                # Mark as 'accepted' and tag it 'auto' so we know the AI didn't do this
                self.db.resolve_cross_ref(
                    cross_ref_id=ref["id"], 
                    status="accepted", 
                    resolved_by="auto", 
                    resolved_text=best_text
                )
                resolved_count += 1
                
        return resolved_count
    
    def auto_resolve_subsets(self) -> int:
        """
        Automatically resolves pending cross-references where one chunk's text 
        is a complete substring of the other. 
        Preserves the larger, encompassing chunk.
        
        Returns:
            The number of cross-references automatically resolved.
        """
        pending = self.db.get_pending_cross_refs()
        resolved_count = 0
        
        for ref in pending:
            text_a = ref["content_a"].strip()
            text_b = ref["content_b"].strip()
            
            # Skip if they are exactly identical (handled by exact match resolver)
            if text_a == text_b:
                continue
                
            # If A is entirely contained within B, B is the superset.
            if text_a in text_b:
                self.db.resolve_cross_ref(
                    cross_ref_id=ref["id"], 
                    status="accepted", 
                    resolved_by="auto_subset", 
                    resolved_text=ref["content_b"]
                )
                resolved_count += 1
                
            # If B is entirely contained within A, A is the superset.
            elif text_b in text_a:
                self.db.resolve_cross_ref(
                    cross_ref_id=ref["id"], 
                    status="accepted", 
                    resolved_by="auto_subset", 
                    resolved_text=ref["content_a"]
                )
                resolved_count += 1
                
        return resolved_count