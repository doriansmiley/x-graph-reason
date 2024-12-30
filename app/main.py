from KnowledgeGraph import KnowledgeGraphUpdater
from neo4j import GraphDatabase

# Neo4j connection details
# TODO use .env
neo4j_driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))

# Path to the RDF base file
rdf_file = "base_data.ttl"

# Initialize the KnowledgeGraphUpdater
kg_updater = KnowledgeGraphUpdater(rdf_file, neo4j_driver)

corpus = [
    "Claim denied due to adjustment required for exceeding allowed amount. Resolution: Verify allowable amounts with the payor.",
    "Claim denied because preauthorization was not obtained. Resolution: Submit preauthorization forms retroactively.",
    "Claim denied as medical necessity was not demonstrated. Resolution: Submit medical records demonstrating necessity."
]

# Process the corpus to dynamically update the RDF graph, Neo4j graph, and ECL
kg_updater.process_corpus(corpus)

"""
After processing the corpus, the RDF graph (self.rdf_processor.graph) will include:

@prefix ex: <http://example.org/insurance/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# Classes
ex:Claim rdf:type rdfs:Class .
ex:Denial rdf:type rdfs:Class .

# Relationships
ex:adjustment rdf:type rdf:Property .
ex:preauthorization rdf:type rdf:Property .
ex:medicalNecessity rdf:type rdf:Property .
ex:resolves rdf:type rdf:Property .

# Instances
ex:Claim123 rdf:type ex:Claim ;
    ex:adjustment "Exceeding allowed amount" ;
    ex:resolves "Verify allowable amounts with the payor" .

ex:Claim456 rdf:type ex:Claim ;
    ex:preauthorization "Preauthorization not obtained" ;
    ex:resolves "Submit preauthorization forms retroactively" .

ex:Claim789 rdf:type ex:Claim ;
    ex:medicalNecessity "Medical necessity not demonstrated" ;
    ex:resolves "Submit medical records demonstrating necessity" .

"""

