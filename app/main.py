import os
from openai import OpenAI
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from KnowledgeGraph import KnowledgeGraphUpdater
from QueryEngine import KnowledgeGraphQueryHandler
from neo4j import GraphDatabase
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

# Initialize OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Neo4j driver
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Load the RDF file content into memory
with open("app/base_data.ttl", "rb") as file:
    rdf_file_bytes = file.read()

# Initialize the KnowledgeGraphUpdater
kg_updater = KnowledgeGraphUpdater(rdf_file_bytes, neo4j_driver, client)

# load the RDF data
kg_updater.load_rdf_data()

# Initialize the KnowledgeGraphQueryHandler
query_handler = KnowledgeGraphQueryHandler(kg_updater.ecl, neo4j_driver, client)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    """
    Default route to check API health.
    """
    return jsonify({"message": "Knowledge Graph API is running."})

@app.route("/update-corpus", methods=["POST"])
def update_corpus():
    """
    API endpoint to update the knowledge graph with a new corpus.
    Expected input: JSON with a "corpus" key containing a list of text strings.
    """
    try:
        data = request.get_json()
        corpus = data.get("corpus", [])
        if not corpus:
            return jsonify({"error": "No corpus provided"}), 400

        # Process the corpus to update the knowledge graph
        kg_updater.process_corpus(corpus)
        return jsonify({"message": "Corpus successfully processed and knowledge graph updated."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query-knowledge-graph", methods=["POST"])
def query_knowledge_graph():
    """
    API endpoint to query the knowledge graph.
    Expected input: JSON with a "query" key containing the user query.
    """
    try:
        data = request.get_json()
        query_text = data.get("query", "")
        if not query_text:
            return jsonify({"error": "No query provided"}), 400

        # Execute the query on the knowledge graph
        results = query_handler.query_knowledge_graph(query_text)
        return jsonify({"results": results}), 200
    except Exception as e:
        print(f"{e}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    print(f"Connecting to Neo4j Aura Instance: {AURA_INSTANCENAME} (ID: {AURA_INSTANCEID})")
    app.run(debug=True, host="0.0.0.0", port=5000)
