"""
Tests for Ollama LLM integration
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from step2_ollama_answer import ask_ollama


class TestOllamaIntegration:
    """Test suite for Ollama answer generation"""

    @patch('step2_ollama_answer.ollama.chat')
    def test_ask_ollama_returns_string(self, mock_chat, sample_query, sample_chunks):
        """Test that ask_ollama returns a string response"""
        mock_chat.return_value = {
            "message": {
                "content": "To replace the LED lights, first unplug the unit."
            }
        }

        result = ask_ollama(sample_query, sample_chunks)

        assert isinstance(result, str)
        assert len(result) > 0

    @patch('step2_ollama_answer.ollama.chat')
    def test_ask_ollama_with_custom_model(self, mock_chat, sample_query, sample_chunks):
        """Test asking with a custom model name"""
        mock_chat.return_value = {
            "message": {
                "content": "Test response"
            }
        }

        result = ask_ollama(sample_query, sample_chunks, model_name="custom-model")

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs['model'] == "custom-model"

    @patch('step2_ollama_answer.ollama.chat')
    def test_ask_ollama_prompt_structure(self, mock_chat, sample_query, sample_chunks):
        """Test that the prompt is structured correctly"""
        mock_chat.return_value = {
            "message": {
                "content": "Test response"
            }
        }

        ask_ollama(sample_query, sample_chunks)

        call_kwargs = mock_chat.call_args[1]
        messages = call_kwargs['messages']

        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert 'Context:' in messages[0]['content']
        assert 'Question:' in messages[0]['content']
        assert sample_query in messages[0]['content']

    @patch('step2_ollama_answer.ollama.chat')
    def test_ask_ollama_includes_all_chunks(self, mock_chat, sample_query, sample_chunks):
        """Test that all context chunks are included in prompt"""
        mock_chat.return_value = {
            "message": {
                "content": "Test response"
            }
        }

        ask_ollama(sample_query, sample_chunks)

        call_kwargs = mock_chat.call_args[1]
        prompt = call_kwargs['messages'][0]['content']

        for chunk in sample_chunks:
            assert chunk in prompt

    @patch('step2_ollama_answer.ollama.chat')
    def test_ask_ollama_handles_empty_chunks(self, mock_chat, sample_query):
        """Test handling of empty chunks list"""
        mock_chat.return_value = {
            "message": {
                "content": "No context available"
            }
        }

        result = ask_ollama(sample_query, [])

        assert isinstance(result, str)

    @patch('step2_ollama_answer.ollama.chat')
    def test_ask_ollama_error_handling(self, mock_chat, sample_query, sample_chunks):
        """Test error handling when Ollama fails"""
        mock_chat.side_effect = Exception("Ollama connection failed")

        with pytest.raises(Exception) as exc_info:
            ask_ollama(sample_query, sample_chunks)

        assert "Ollama connection failed" in str(exc_info.value)
