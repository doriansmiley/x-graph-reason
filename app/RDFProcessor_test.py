import pytest
from rdflib import Graph, RDF, RDFS, Namespace, Literal
from KnowledgeGraph import RDFProcessor

# Define the namespace
COS = Namespace("http://example.org/insurance/")


@pytest.fixture
def rdf_processor():
    """
    Fixture to initialize the RDFProcessor instance.
    """
    return RDFProcessor()


def test_ensure_rdf_class(rdf_processor):
    """
    Test the `ensure_rdf_class` method to ensure RDF classes are added correctly.
    """
    class_name = "TestClass"
    rdf_processor.ensure_rdf_class(class_name)

    # Verify that the class was added to the RDF graph
    class_node = COS[class_name]
    assert (class_node, RDF.type, RDFS.Class) in rdf_processor.graph, "Class node was not added to the graph."


def test_ensure_rdf_relationships(rdf_processor):
    """
    Test the `ensure_rdf_relationships` method to ensure RDF relationships are added correctly.
    """
    relationships = {
        "denialReason": "Preauthorization required",
        "adjustment": "Charge exceeds allowed amount"
    }
    updated_relationships = rdf_processor.ensure_rdf_relationships(relationships)

    # Verify that the relationships were added to the RDF graph
    for rel, target in relationships.items():
        rel_node = COS[rel]
        target_node = COS[target.replace(" ", "_")]

        assert (rel_node, RDF.type, RDF.Property) in rdf_processor.graph, f"Relationship {rel} was not added as a property."
        assert (target_node, RDF.type, RDFS.Class) in rdf_processor.graph, f"Target node {target} was not added as a class."
        assert updated_relationships[rel] == target, f"Relationship {rel} was not updated correctly."


def test_update_rdf_with_relationships(rdf_processor):
    """
    Test the `update_rdf_with_relationships` method to ensure RDF relationships are added to the graph.
    """
    class_name = "Claim"
    relationships = {
        "denialReason": "Preauthorization required",
        "adjustment": "Charge exceeds allowed amount"
    }
    rdf_processor.update_rdf_with_relationships(class_name, relationships)

    # Verify that the relationships were added to the RDF graph
    class_node = COS[class_name]
    for rel, target in relationships.items():
        rel_node = COS[rel]
        target_node = COS[target.replace(" ", "_")]

        assert (class_node, rel_node, target_node) in rdf_processor.graph, f"Triple ({class_name}, {rel}, {target}) was not added to the graph."


def test_update_rdf_with_resolutions(rdf_processor):
    """
    Test the `update_rdf_with_resolutions` method to ensure RDF resolutions are added to the graph.
    """
    relationships = {
        "denialReason": "Preauthorization required",
        "adjustment": "Charge exceeds allowed amount"
    }
    resolutions = {
        "denialReason": "Submit preauthorization forms retroactively",
        "adjustment": "Verify allowable amounts with the payor"
    }
    rdf_processor.update_rdf_with_resolutions(relationships, resolutions)

    # Verify that the resolutions were added to the RDF graph
    for rel, resolution in resolutions.items():
        resolution_node = COS[resolution.replace(" ", "_")]
        rel_node = COS[rel]

        assert (resolution_node, RDF.type, RDFS.Class) in rdf_processor.graph, f"Resolution {resolution} was not added as a class."
        assert (rel_node, COS["resolves"], resolution_node) in rdf_processor.graph, f"Resolution link for {rel} was not added to the graph."
