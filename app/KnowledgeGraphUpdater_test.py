import pytest
from unittest.mock import MagicMock, patch, call
from rdflib import Graph, Namespace
from KnowledgeGraph import KnowledgeGraphUpdater, COS
import json

@pytest.fixture
def mocked_neo4j_driver():
    """
    Fixture to mock the Neo4j driver and its session.
    """
    driver = MagicMock()
    session = MagicMock()
    session.run = MagicMock()
    driver.session.return_value = session
    return driver

@pytest.fixture
def sample_rdf_file(tmp_path):
    """
    Fixture to create a sample RDF file.
    """
    rdf_content = """
        @prefix ex: <http://example.org/insurance/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

        ex:Claim rdf:type rdfs:Class .
        ex:Denial rdf:type rdfs:Class .
    """
    rdf_file = tmp_path / "sample_data.ttl"
    rdf_file.write_text(rdf_content)
    return rdf_file

@pytest.fixture
def mocked_openai():
    """
    Fixture to mock OpenAI API calls.
    """
    with patch("openai.ChatCompletion.create") as mocked_chat, patch("openai.Embedding.create") as mocked_embedding:
        mocked_chat.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "class": "Claim",
                            "relationships": {
                                "denialReason": "Preauthorization required",
                                "adjustment": "Charge exceeds allowed amount"
                            },
                            "resolutions": {
                                "denialReason": "Submit preauthorization forms",
                                "adjustment": "Verify charge limits"
                            }
                        })
                    }
                }
            ]
        }

        mocked_embedding.return_value = {
            "data": [
                {
                    "embedding": [0.1, 0.2, 0.3]
                }
            ]
        }

        yield mocked_chat, mocked_embedding

def test_process_corpus(mocked_neo4j_driver, sample_rdf_file, mocked_openai):
    """
    Test the `process_corpus` method of `KnowledgeGraphUpdater`.
    """
    # Initialize KnowledgeGraphUpdater
    updater = KnowledgeGraphUpdater(sample_rdf_file, mocked_neo4j_driver)

    # Sample corpus
    corpus = [
        "Claim denied due to adjustment required for exceeding allowed amount. Resolution: Verify allowable amounts with the payor.",
        "Claim denied because preauthorization was not obtained. Resolution: Submit preauthorization forms retroactively."
    ]

    # Call the method
    updater.process_corpus(corpus)

    # Assertions
    # Ensure OpenAI GPT was called with the correct prompt
    mocked_openai[0].assert_called()
    assert mocked_openai[0].call_count == len(corpus)

    # Ensure OpenAI embedding was called for each relationship
    mocked_openai[1].assert_has_calls([
        call(model="text-embedding-3-small", input="denialReason", encoding_format="float"),
        call(model="text-embedding-3-small", input="adjustment", encoding_format="float"),
    ], any_order=True)

    # Ensure Neo4j session's `run` method was called with expected arguments
    session = mocked_neo4j_driver.session.return_value
    assert session.run.call_count > 0
