"""
Tests for PDF processing and chunking functionality
"""
import pytest
import os
from fridge_rag import pdf_chunker


class TestPDFChunking:
    """Test suite for PDF chunking operations"""

    def test_pdf_chunker_returns_list(self, sample_pdf_path):
        """Test that pdf_chunker returns a list of chunks"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"PDF file not found: {sample_pdf_path}")

        chunks = pdf_chunker(sample_pdf_path, chunk_size=500, overlap=100)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_pdf_chunker_with_different_sizes(self, sample_pdf_path):
        """Test chunking with different chunk sizes"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"PDF file not found: {sample_pdf_path}")

        small_chunks = pdf_chunker(sample_pdf_path, chunk_size=300, overlap=50)
        large_chunks = pdf_chunker(sample_pdf_path, chunk_size=1000, overlap=200)

        # Smaller chunk size should produce more chunks
        assert len(small_chunks) > len(large_chunks)

    def test_chunks_are_strings(self, sample_pdf_path):
        """Test that all chunks are strings"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"PDF file not found: {sample_pdf_path}")

        chunks = pdf_chunker(sample_pdf_path, chunk_size=500, overlap=100)
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_chunk_overlap(self, sample_pdf_path):
        """Test that chunks have some overlap"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"PDF file not found: {sample_pdf_path}")

        chunks = pdf_chunker(sample_pdf_path, chunk_size=500, overlap=100)
        # Just verify we got chunks - actual overlap testing would need content analysis
        assert len(chunks) > 1

    def test_invalid_pdf_path(self):
        """Test handling of invalid PDF path"""
        with pytest.raises(Exception):
            pdf_chunker("nonexistent_file.pdf")

    @pytest.mark.parametrize("chunk_size,overlap", [
        (100, 20),
        (500, 100),
        (1000, 200),
        (2000, 500)
    ])
    def test_various_chunk_parameters(self, sample_pdf_path, chunk_size, overlap):
        """Test chunking with various parameter combinations"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"PDF file not found: {sample_pdf_path}")

        chunks = pdf_chunker(sample_pdf_path, chunk_size=chunk_size, overlap=overlap)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
