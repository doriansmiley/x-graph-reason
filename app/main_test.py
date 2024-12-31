import requests
import subprocess
import time
import os
import signal
import pytest

# Constants
BASE_URL = "http://localhost:5000"
API_PORT = 5000

# Sample data for testing
CORPUS_PAYLOAD = {
    "corpus": [
        "Claim denied due to adjustment required for exceeding allowed amount. Resolution: Verify allowable amounts with the payor.",
        "Claim denied because preauthorization was not obtained. Resolution: Submit preauthorization forms retroactively."
    ]
}

QUERY_PAYLOAD = {
    "query": "Why was my claim denied?"
}


@pytest.fixture(scope="module")
def launch_api():
    """
    Fixture to launch the API if it is not already running.
    Ensures the Flask application is started and cleaned up after tests.
    """
    # Check if the API is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            # API is already running
            yield
            return
    except requests.ConnectionError:
        pass

    # API is not running, launch it
    process = subprocess.Popen(["python", "app/main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)  # Wait for the API to start

    yield  # Provide the fixture

    # Terminate the process after the tests
    os.kill(process.pid, signal.SIGTERM)
    process.wait()


def test_index_endpoint(launch_api):
    """
    Test the index endpoint to ensure API health.
    """
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.json() == {"message": "Knowledge Graph API is running."}


def test_update_corpus_endpoint(launch_api):
    """
    Test the `/update-corpus` endpoint with a valid payload.
    """
    response = requests.post(f"{BASE_URL}/update-corpus", json=CORPUS_PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {"message": "Corpus successfully processed and knowledge graph updated."}


def test_query_knowledge_graph_endpoint(launch_api):
    """
    Test the `/query-knowledge-graph` endpoint with a valid query.
    """
    response = requests.post(f"{BASE_URL}/query-knowledge-graph", json=QUERY_PAYLOAD)
    assert response.status_code == 200
    # Validate the structure of the response (the actual content may vary based on your implementation)
    assert "results" in response.json()
    assert isinstance(response.json()["results"], list)
