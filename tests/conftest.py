"""
Pytest configuration and shared fixtures
"""
import pytest
import os
import json
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def sample_pdf_path():
    """Returns path to the test PDF"""
    return "fridge-owners-manual.pdf"


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing"""
    return [
        "To replace the LED lights in your refrigerator, first unplug the unit.",
        "The water filter should be replaced every 6 months.",
        "Set the temperature to 37°F for optimal food preservation.",
        "Clean the condenser coils every 6 months to maintain efficiency.",
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return "How do I replace the LED lights?"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_embeddings(sample_chunks):
    """Generate mock embeddings for testing"""
    # Create random embeddings with correct dimensions (384 for all-MiniLM-L6-v2)
    np.random.seed(42)
    return np.random.rand(len(sample_chunks), 384).astype('float32')


@pytest.fixture
def sample_retrieved_data(sample_query, sample_chunks):
    """Sample retrieved chunks data structure"""
    return {
        "query": sample_query,
        "chunks": sample_chunks[:3]
    }


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response"""
    return {
        "message": {
            "content": "To replace the LED lights, first unplug the refrigerator for safety."
        }
    }
