import pytest
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_imports():
    """Test if all necessary imports are successful."""
    try:
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv
        import os
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_api_key_exists():
    """Test if the OpenAI API key is present in the environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key is not None, "OPENAI_API_KEY is not set in environment variables."

def test_llm_initialization():
    """Test if the LLM object is created without errors."""
    api_key = os.environ.get("OPENAI_API_KEY")
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.2, max_tokens=2048)
    except Exception as e:
        pytest.fail(f"LLM initialization failed: {e}")

def test_llm_invoke():
    """Test if the LLM invocation works without errors."""
    api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.2, max_tokens=2048)

    try:
        response = llm.invoke("if active respond with active")
        assert response is not None, "LLM invocation returned None."
        assert "active" in response, f"Unexpected response: {response}"
    except Exception as e:
        pytest.fail(f"LLM invocation failed: {e}")
