import pytest
import json
from flask import Flask
from main import app
from unittest.mock import patch

@pytest.fixture
def client():
    """
    Fixture to set up the Flask test client.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@patch("openai.ChatCompletion.create")  # Mock the OpenAI API call
def test_query_knowledge_graph(mock_openai, client):
    """
    Test the /query-knowledge-graph endpoint, mocking external API calls.
    """
    # Mock response from OpenAI API
    mock_openai.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "class": "Claim_789",
                        "relationships": {
                            "medicalNecessity": "Medical necessity not demonstrated"
                        },
                        "resolutions": {
                            "medicalNecessity": "Submit medical records demonstrating necessity"
                        },
                    })
                }
            }
        ]
    }

    # Send a POST request to the /query-knowledge-graph endpoint
    response = client.post(
        "/query-knowledge-graph",
        json={"query": "Why was the claim denied for medical necessity?"},
    )

    # Check the response status code
    assert response.status_code == 200

    # Check the response JSON
    assert response.json == {
        "results": [
            {
                "class": "Claim_789",
                "relationship": "medicalNecessity",
                "target": "Medical necessity not demonstrated",
                "resolution": "Submit medical records demonstrating necessity",
            }
        ]
    }
