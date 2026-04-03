import pytest
import numpy as np
from corpusforge.rationalizer import TopicRationalizer

def _make_vec(val_x: float, val_y: float) -> np.ndarray:
    """
    Creates a normalized 384-dimensional vector for predictable math.
    By mapping our test values to the first two dimensions and padding 
    the rest with zeros, we can easily calculate cosine similarity by hand.
    """
    vec = np.zeros(384, dtype=np.float32)
    vec[0] = val_x
    vec[1] = val_y
    
    # L2 Normalization (just like sentence-transformers)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

class TestTopicRationalizer:
    
    def test_find_topic_overlaps_matrix_math(self, memory_db):
        """Tests that the cosine similarity matrix correctly surfaces overlaps."""
        
        # 1. Setup Data
        topic_id = memory_db.create_topic("Math Test Topic")
        file_id, _ = memory_db.upsert_file("test.md", "test.md", "md", "hash123")
        
        # 2. Create predictable vectors
        v_a = _make_vec(1.0, 0.0)         # Baseline
        v_b = _make_vec(1.0, 0.0)         # Identical to A (Sim: 1.0)
        v_c = _make_vec(0.0, 1.0)         # Orthogonal to A (Sim: 0.0)
        v_d = _make_vec(0.88, 0.4749)     # Cosine Similarity to A is exactly 0.88
        
        # 3. Insert Chunks
        chunks = [
            {"chunk_index": 0, "content": "Chunk A", "token_count": 1, "embedding": v_a},
            {"chunk_index": 1, "content": "Chunk B", "token_count": 1, "embedding": v_b},
            {"chunk_index": 2, "content": "Chunk C", "token_count": 1, "embedding": v_c},
            {"chunk_index": 3, "content": "Chunk D", "token_count": 1, "embedding": v_d},
        ]
        chunk_ids = memory_db.insert_chunks_batch(file_id, chunks)
        a_id, b_id, c_id, d_id = chunk_ids
        
        # 4. Assign to Topic (Similarity score here is HDBSCAN membership, not overlap)
        memory_db.assign_chunks_to_topic_batch([(cid, topic_id, 1.0) for cid in chunk_ids])
        
        # 5. Run the Rationalizer (Default threshold is 0.85)
        rat = TopicRationalizer(memory_db)
        overlaps = rat.find_topic_overlaps(topic_id, threshold=0.85)
        
        # We expect 3 overlapping pairs:
        # A & B (1.0), A & D (0.88), and B & D (0.88, since B is identical to A)
        assert len(overlaps) == 3
        
        # 6. Verify Results are correctly sorted and calculated
        # Highest similarity (A & B) should be first
        assert overlaps[0][0] == a_id
        assert overlaps[0][1] == b_id
        assert pytest.approx(overlaps[0][2], 0.01) == 1.0
        
        # C should not appear ANYWHERE in the results because it is orthogonal
        for pair in overlaps:
            assert pair[0] != c_id
            assert pair[1] != c_id

    def test_threshold_filtering(self, memory_db):
        """Tests that adjusting the threshold filters out lower similarities."""
        topic_id = memory_db.create_topic("Threshold Test")
        file_id, _ = memory_db.upsert_file("test2.md", "test2.md", "md", "hash123")
        
        v_a = _make_vec(1.0, 0.0)
        v_d = _make_vec(0.88, 0.4749) # Similarity 0.88
        
        chunk_ids = memory_db.insert_chunks_batch(file_id, [
            {"chunk_index": 0, "content": "A", "token_count": 1, "embedding": v_a},
            {"chunk_index": 1, "content": "D", "token_count": 1, "embedding": v_d},
        ])
        memory_db.assign_chunks_to_topic_batch([(cid, topic_id, 1.0) for cid in chunk_ids])
        
        rat = TopicRationalizer(memory_db)
        
        # Threshold 0.85 should catch it
        assert len(rat.find_topic_overlaps(topic_id, threshold=0.85)) == 1
        
        # Strict threshold 0.95 should ignore it
        assert len(rat.find_topic_overlaps(topic_id, threshold=0.95)) == 0

    def test_empty_or_single_chunk_topic_returns_empty(self, memory_db):
        """Tests early exit logic when there isn't enough data to compare."""
        rat = TopicRationalizer(memory_db)
        topic_id = memory_db.create_topic("Empty")
        
        assert rat.find_topic_overlaps(topic_id) == []