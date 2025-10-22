"""
Integration tests for the complete RAG pipeline
"""
import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


class TestEndToEndPipeline:
    """Integration tests for the complete pipeline"""

    @pytest.mark.integration
    def test_chunks_file_generation(self, sample_pdf_path, temp_dir):
        """Test that chunks.json is generated correctly"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"PDF file not found: {sample_pdf_path}")

        from fridge_rag import pdf_chunker

        chunks = pdf_chunker(sample_pdf_path, 500, 100)

        output_file = os.path.join(temp_dir, "chunks.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        assert os.path.exists(output_file)

        # Verify file content
        with open(output_file, "r", encoding="utf-8") as f:
            loaded_chunks = json.load(f)

        assert len(loaded_chunks) == len(chunks)
        assert loaded_chunks == chunks

    @pytest.mark.integration
    def test_retrieved_chunks_format(self, sample_query, sample_chunks, temp_dir):
        """Test retrieved_chunks.json has correct format"""
        from fridge_rag import create_embedding, faiss_search, retrieve_chunks

        embeddings, model = create_embedding(sample_chunks)
        index = faiss_search(embeddings)
        top_chunks = retrieve_chunks(sample_query, index, sample_chunks, model, top_k=3)

        output_file = os.path.join(temp_dir, "retrieved_chunks.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "query": sample_query,
                "chunks": top_chunks
            }, f, ensure_ascii=False, indent=2)

        # Verify format
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "query" in data
        assert "chunks" in data
        assert isinstance(data["chunks"], list)
        assert len(data["chunks"]) == 3

    @patch('step2_ollama_answer.ollama.chat')
    def test_answer_file_generation(self, mock_chat, sample_retrieved_data, temp_dir):
        """Test that answer.txt is generated correctly"""
        from step2_ollama_answer import ask_ollama

        mock_chat.return_value = {
            "message": {
                "content": "To replace the LED lights, first unplug the refrigerator."
            }
        }

        query = sample_retrieved_data["query"]
        chunks = sample_retrieved_data["chunks"]

        answer = ask_ollama(query, chunks)

        output_file = os.path.join(temp_dir, "answer.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n\n")
            f.write(f"Answer: {answer}\n")

        assert os.path.exists(output_file)

        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Query:" in content
        assert "Answer:" in content
        assert query in content

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_without_ollama(self, sample_pdf_path, temp_dir):
        """Test the complete pipeline except Ollama call"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"PDF file not found: {sample_pdf_path}")

        from fridge_rag import pdf_chunker, create_embedding, faiss_search, retrieve_chunks

        # Step 1: Process PDF
        chunks = pdf_chunker(sample_pdf_path, 1000, 500)
        assert len(chunks) > 0

        # Step 2: Create embeddings
        embeddings, model = create_embedding(chunks)
        assert embeddings.shape[0] == len(chunks)

        # Step 3: Create FAISS index
        index = faiss_search(embeddings)
        assert index.ntotal == len(chunks)

        # Step 4: Retrieve chunks
        query = "How do I replace the LED lights?"
        top_chunks = retrieve_chunks(query, index, chunks, model, top_k=3)
        assert len(top_chunks) == 3

        # Verify retrieved chunks are strings
        for chunk in top_chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0


class TestDataPersistence:
    """Test data persistence and file I/O"""

    def test_json_encoding_utf8(self, sample_chunks, temp_dir):
        """Test that JSON files handle UTF-8 correctly"""
        output_file = os.path.join(temp_dir, "test_chunks.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sample_chunks, f, ensure_ascii=False, indent=2)

        with open(output_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded == sample_chunks

    def test_embeddings_save_load(self, mock_embeddings, temp_dir):
        """Test saving and loading embeddings"""
        import numpy as np

        output_file = os.path.join(temp_dir, "embeddings.npy")

        np.save(output_file, mock_embeddings)
        loaded = np.load(output_file)

        np.testing.assert_array_equal(mock_embeddings, loaded)

    def test_faiss_index_save_load(self, mock_embeddings, temp_dir):
        """Test saving and loading FAISS index"""
        import faiss
        from fridge_rag import faiss_search

        index = faiss_search(mock_embeddings)

        output_file = os.path.join(temp_dir, "faiss_index.bin")
        faiss.write_index(index, output_file)

        loaded_index = faiss.read_index(output_file)

        assert loaded_index.ntotal == index.ntotal
        assert loaded_index.d == index.d
