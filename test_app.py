# Automated test suite for FastAPI RAG endpoints
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app import app

client = TestClient(app)


# Health Check Tests
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["backend"] == "groqcloud"
    assert "retriever_initialized" in data


# Input Validation Tests
def test_empty_question_validation():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_whitespace_question_validation():
    response = client.post("/ask", json={"question": "     "})
    assert response.status_code == 422


def test_oversized_question_validation():
    long_question = "a" * 1005
    response = client.post("/ask", json={"question": long_question})
    assert response.status_code == 422


# Pipeline Execution and Error Handling Tests
@patch("support_rag.run_rag_pipeline")
def test_successful_ask_endpoint(mock_rag):
    mock_rag.return_value = {
        "answer": "Drink at least 8 to 10 glasses of water daily.",
        "sources": [{"source": "wellness.txt", "page": None}],
        "latency_ms": 245.50
    }

    response = client.post("/ask", json={"question": "What does the wellness note say about hydration?"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "hydration" in data["question"].lower()
    assert len(data["sources"]) > 0
    assert data["latency_ms"] == 245.50
    assert "8 to 10 glasses" in data["answer"]


@patch("support_rag.run_rag_pipeline")
def test_backend_failure_handling(mock_rag):
    mock_rag.side_effect = Exception("Groq connection timeout or invalid auth")

    response = client.post("/ask", json={"question": "What is the turnaround time?"})
    assert response.status_code == 500
    data = response.json()
    assert "Failed to generate answer from the backend LLM service" in data["detail"]
    assert "timeout" not in data["detail"]
