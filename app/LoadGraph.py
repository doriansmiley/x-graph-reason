from rdflib_neo4j import Neo4jStoreConfig, Neo4jStore, HANDLE_VOCAB_URI_STRATEGY
from rdflib import Graph
import os
from openai import OpenAI
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from KnowledgeGraph import KnowledgeGraphUpdater
from QueryEngine import KnowledgeGraphQueryHandler
from neo4j import GraphDatabase
from rdflib import Namespace
import faulthandler

faulthandler.enable()

# Load environment variables from .env file
load_dotenv()

# Neo4j connection details from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
AURA_INSTANCEID = os.getenv("AURA_INSTANCEID", "default-instance-id")
AURA_INSTANCENAME = os.getenv("AURA_INSTANCENAME", "default-instance-name")

auth_data = {'uri': NEO4J_URI,
             'database': "neo4j",
             'user': NEO4J_USERNAME,
             'pwd': NEO4J_PASSWORD}

prefixes = {
    'rdf': Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#'),
    'rdfs': Namespace('http://www.w3.org/2000/01/rdf-schema#'),
    'xsd': Namespace('http://www.w3.org/2001/XMLSchema#'),
    'ex': Namespace('http://example.org/insurance#'),
}

# Define your custom mappings & store config
config = Neo4jStoreConfig(auth_data=auth_data,
                          custom_prefixes=prefixes,
                          handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE,
                          batching=True)

# Load the RDF file content into memory
with open("app/base_data.ttl", "rb") as file:
    rdf_file_bytes = file.read()

# Create the RDF Graph, parse & ingest the data to Neo4j, and close the store(If the field batching is set to True in the Neo4jStoreConfig, remember to close the store to prevent the loss of any uncommitted records.)
neo4j_aura = Graph(store=Neo4jStore(config=config))
# Calling the parse method will implictly open the store
neo4j_aura.parse(rdf_file_bytes, format="ttl")
neo4j_aura.close(True)