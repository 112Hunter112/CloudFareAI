"""
Tests for embedding generation and vector operations
"""
import pytest
import numpy as np
from fridge_rag import create_embedding, faiss_search, retrieve_chunks
from sentence_transformers import SentenceTransformer


class TestEmbeddings:
    """Test suite for embedding generation"""

    def test_create_embedding_shape(self, sample_chunks):
        """Test that embeddings have correct shape"""
        embeddings, model = create_embedding(sample_chunks)

        assert embeddings.shape[0] == len(sample_chunks)
        assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension

    def test_create_embedding_type(self, sample_chunks):
        """Test that embeddings are numpy arrays"""
        embeddings, model = create_embedding(sample_chunks)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32

    def test_embedding_model_type(self, sample_chunks):
        """Test that correct model is returned"""
        embeddings, model = create_embedding(sample_chunks)

        assert isinstance(model, SentenceTransformer)

    def test_embeddings_are_normalized(self, sample_chunks):
        """Test that embeddings are reasonable values"""
        embeddings, model = create_embedding(sample_chunks)

        # Embeddings should not be all zeros
        assert not np.all(embeddings == 0)

        # Check for reasonable value ranges (typically -1 to 1 range)
        assert np.all(np.abs(embeddings) < 10)

    def test_same_input_produces_same_embedding(self):
        """Test deterministic embedding generation"""
        chunks = ["Test sentence for embedding"]

        embeddings1, _ = create_embedding(chunks)
        embeddings2, _ = create_embedding(chunks)

        np.testing.assert_array_almost_equal(embeddings1, embeddings2, decimal=5)


class TestFAISSSearch:
    """Test suite for FAISS index operations"""

    def test_faiss_index_creation(self, mock_embeddings):
        """Test FAISS index is created successfully"""
        index = faiss_search(mock_embeddings)

        assert index is not None
        assert index.ntotal == len(mock_embeddings)

    def test_faiss_index_dimension(self, mock_embeddings):
        """Test FAISS index has correct dimension"""
        index = faiss_search(mock_embeddings)

        assert index.d == 384

    def test_retrieve_chunks_returns_results(self, sample_chunks, sample_query):
        """Test that retrieve_chunks returns expected number of results"""
        embeddings, model = create_embedding(sample_chunks)
        index = faiss_search(embeddings)

        results = retrieve_chunks(sample_query, index, sample_chunks, model, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_retrieve_chunks_relevance(self, sample_chunks, sample_query):
        """Test that retrieved chunks are relevant"""
        embeddings, model = create_embedding(sample_chunks)
        index = faiss_search(embeddings)

        results = retrieve_chunks(sample_query, index, sample_chunks, model, top_k=1)

        # The query about LED lights should retrieve the LED chunk
        assert "LED" in results[0] or "light" in results[0].lower()

    def test_retrieve_chunks_top_k_parameter(self, sample_chunks, sample_query):
        """Test that top_k parameter works correctly"""
        embeddings, model = create_embedding(sample_chunks)
        index = faiss_search(embeddings)

        for k in [1, 2, 3]:
            results = retrieve_chunks(sample_query, index, sample_chunks, model, top_k=k)
            assert len(results) == k

    def test_empty_query_handling(self, sample_chunks):
        """Test handling of empty query"""
        embeddings, model = create_embedding(sample_chunks)
        index = faiss_search(embeddings)

        # Should still return results even with empty query
        results = retrieve_chunks("", index, sample_chunks, model, top_k=1)
        assert len(results) == 1
