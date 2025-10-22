"""
Performance and benchmark tests
"""
import pytest
import time
from fridge_rag import create_embedding, faiss_search, retrieve_chunks


class TestPerformance:
    """Performance benchmarks for RAG operations"""

    @pytest.mark.performance
    def test_embedding_generation_performance(self, sample_chunks):
        """Test embedding generation completes in reasonable time"""
        start = time.time()
        embeddings, model = create_embedding(sample_chunks)
        duration = time.time() - start

        # Should complete in under 5 seconds for small chunks
        assert duration < 5.0, f"Embedding took {duration:.2f}s, expected < 5s"

    @pytest.mark.performance
    def test_faiss_index_creation_performance(self, mock_embeddings):
        """Test FAISS index creation speed"""
        start = time.time()
        index = faiss_search(mock_embeddings)
        duration = time.time() - start

        # Should be very fast for small datasets
        assert duration < 1.0, f"Index creation took {duration:.2f}s, expected < 1s"

    @pytest.mark.performance
    def test_retrieval_performance(self, sample_chunks):
        """Test that retrieval is fast"""
        embeddings, model = create_embedding(sample_chunks)
        index = faiss_search(embeddings)

        query = "test query"

        start = time.time()
        results = retrieve_chunks(query, index, sample_chunks, model, top_k=3)
        duration = time.time() - start

        # Retrieval should be very fast
        assert duration < 1.0, f"Retrieval took {duration:.2f}s, expected < 1s"

    @pytest.mark.performance
    @pytest.mark.parametrize("num_chunks", [10, 50, 100])
    def test_scalability_with_chunk_count(self, num_chunks):
        """Test performance scales with number of chunks"""
        chunks = [f"Sample text chunk number {i}" for i in range(num_chunks)]

        start = time.time()
        embeddings, model = create_embedding(chunks)
        embedding_time = time.time() - start

        start = time.time()
        index = faiss_search(embeddings)
        index_time = time.time() - start

        start = time.time()
        retrieve_chunks("test", index, chunks, model, top_k=3)
        retrieval_time = time.time() - start

        print(f"\nChunks: {num_chunks}, Embedding: {embedding_time:.2f}s, "
              f"Index: {index_time:.2f}s, Retrieval: {retrieval_time:.2f}s")

        # Basic scaling check - should complete in reasonable time
        assert embedding_time < 30.0
        assert index_time < 2.0
        assert retrieval_time < 2.0


class TestMemoryUsage:
    """Memory usage tests"""

    @pytest.mark.memory
    def test_embeddings_memory_footprint(self, sample_chunks):
        """Test that embeddings don't use excessive memory"""
        import numpy as np

        embeddings, model = create_embedding(sample_chunks)

        # Calculate memory usage in MB
        memory_mb = embeddings.nbytes / (1024 * 1024)

        print(f"\nEmbeddings memory usage: {memory_mb:.2f} MB")

        # Should be reasonable for small datasets
        assert memory_mb < 10.0, f"Embeddings use {memory_mb:.2f}MB, expected < 10MB"

    @pytest.mark.memory
    def test_faiss_index_memory(self, mock_embeddings):
        """Test FAISS index memory usage"""
        index = faiss_search(mock_embeddings)

        # Just verify index was created successfully
        assert index.ntotal > 0
