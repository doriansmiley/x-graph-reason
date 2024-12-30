import pytest
from unittest.mock import MagicMock, call
from rdflib import Graph, Namespace, Literal
from KnowledgeGraph import Neo4jKnowledgeGraph
import textwrap

# Define the namespace
COS = Namespace("http://example.org/insurance/")


@pytest.fixture
def mocked_driver():
    """
    Fixture to mock the Neo4j driver and its session.
    """
    driver = MagicMock()
    session = MagicMock()
    session.run = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__ = MagicMock()
    return driver


@pytest.fixture
def rdf_graph():
    """
    Fixture to create a sample RDF graph.
    """
    graph = Graph()
    graph.add((COS["Claim123"], COS["denialReason"], Literal("Medical necessity not demonstrated")))
    graph.add((COS["Claim123"], COS["adjustment"], Literal("Charge exceeds allowed amount")))
    return graph


def test_add_rdf_to_graph(mocked_driver, rdf_graph):
    """
    Test the `add_rdf_to_graph` method to ensure RDF triples are added to the Neo4j graph.
    """
    # Initialize the Neo4jKnowledgeGraph with the mocked driver
    neo4j_kg = Neo4jKnowledgeGraph(mocked_driver)

    # Call the method with the sample RDF graph
    neo4j_kg.add_rdf_to_graph(rdf_graph)

    # Verify that the session's `run` method was called with the expected arguments for each triple
    session = mocked_driver.session.return_value
    expected_calls = [
        call(
            textwrap.dedent(
                """
                MERGE (s:Entity {name: $subj})
                MERGE (p:Property {name: $pred})
                MERGE (o:Entity {name: $obj})
                MERGE (s)-[:RELATION {property: p.name}]->(o)
                """
            ),
            subj=str(COS["Claim123"]),
            pred=str(COS["denialReason"]),
            obj="Medical necessity not demonstrated",
        ),
        call(
            textwrap.dedent(
                """
                MERGE (s:Entity {name: $subj})
                MERGE (p:Property {name: $pred})
                MERGE (o:Entity {name: $obj})
                MERGE (s)-[:RELATION {property: p.name}]->(o)
                """
            ),
            subj=str(COS["Claim123"]),
            pred=str(COS["adjustment"]),
            obj="Charge exceeds allowed amount",
        ),
    ]

    session.run.assert_has_calls(expected_calls, any_order=True)

    # Verify that the session was opened and closed once
    mocked_driver.session.assert_called_once()
    session.close.assert_called_once()
