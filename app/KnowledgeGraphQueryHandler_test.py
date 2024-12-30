import pytest
from unittest.mock import MagicMock, patch, call
from QueryEngine import KnowledgeGraphQueryHandler
import textwrap


@pytest.fixture
def mocked_ecl():
    """
    Fixture to mock the External Continual Learner (ECL).
    """
    ecl = MagicMock()
    ecl.get_top_k_classes.return_value = ["Class1", "Class2", "Class3"]
    return ecl


@pytest.fixture
def mocked_neo4j_driver():
    """
    Fixture to mock the Neo4j driver and its session.
    """
    driver = MagicMock()
    session = MagicMock()

    # Mock the behavior of `session.run` to return an iterable of dictionaries
    session.run.return_value = [
        {
            "class": "Class1",
            "relationship": "denialReason",
            "target": "Preauthorization required",
            "resolution": "Submit preauthorization forms",
        },
        {
            "class": "Class2",
            "relationship": "adjustment",
            "target": "Charge exceeds allowed amount",
            "resolution": "Verify charge limits",
        },
    ]

    # Ensure the mocked driver properly handles the `with` statement
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__ = MagicMock()

    return driver

@pytest.fixture
def mocked_openai():
    """
    Fixture to mock OpenAI API calls.
    """
    with patch("openai.Embedding.create") as mocked_embedding:
        mocked_embedding.return_value = {
            "data": [
                {
                    "embedding": [0.1, 0.2, 0.3]
                }
            ]
        }
        yield mocked_embedding


def test_query_knowledge_graph(mocked_ecl, mocked_neo4j_driver, mocked_openai):
    """
    Test the `query_knowledge_graph` method of `KnowledgeGraphQueryHandler`.
    """
    # Initialize the query handler with mocked dependencies
    query_handler = KnowledgeGraphQueryHandler(mocked_ecl, mocked_neo4j_driver)

    # Test input
    query_text = "Why was my claim denied?"

    # Call the method
    result = query_handler.query_knowledge_graph(query_text)

    # Assertions
    # Verify OpenAI embedding API was called
    mocked_openai.assert_called_once_with(
        model="text-embedding-3-small",
        input=query_text,
        encoding_format="float"
    )

    # Verify ECL was queried with the generated embedding
    mocked_ecl.get_top_k_classes.assert_called_once_with([0.1, 0.2, 0.3], k=3)

    # Verify Neo4j session's `run` method was called with the expected query and classes
    session = mocked_neo4j_driver.session.return_value.__enter__.return_value
    session.run.assert_called_once_with(
        textwrap.dedent(
            """
            MATCH (c:Entity)-[r]->(t:Entity)
            OPTIONAL MATCH (r)-[:RESOLVES]->(res:Entity)
            WHERE c.name IN $classes
            RETURN c.name AS class, type(r) AS relationship, t.name AS target, res.name AS resolution
            """
        ),
        classes=["Class1", "Class2", "Class3"]
    )

    # Verify the result
    expected_result = [
        {
            "class": "Class1",
            "relationship": "denialReason",
            "target": "Preauthorization required",
            "resolution": "Submit preauthorization forms"
        },
        {
            "class": "Class2",
            "relationship": "adjustment",
            "target": "Charge exceeds allowed amount",
            "resolution": "Verify charge limits"
        }
    ]
    assert result == expected_result

    # Verify the session was closed once
    session.close.assert_called_once()
