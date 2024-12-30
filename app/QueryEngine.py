import os
import openai
from dotenv import load_dotenv
from neo4j import GraphDatabase
import textwrap

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Set up OpenAI client
openai.api_key = OPENAI_API_KEY


class KnowledgeGraphQueryHandler:
    def __init__(self, ecl, kg_driver):
        """
        Initialize the query handler with the External Continual Learner (ECL)
        and Neo4j database driver.

        :param ecl: External Continual Learner (ECL) for class embeddings.
        :param kg_driver: Neo4j database driver.
        """
        self.ecl = ecl
        self.driver = kg_driver

    def query_knowledge_graph(self, query_text):
        """
        Query the knowledge graph using the input query text.
        Retrieves top-k classes from ECL based on input embeddings,
        then queries the Neo4j knowledge graph for relevant data.

        :param query_text: Input text for querying the knowledge graph.
        :return: Query results from the Neo4j knowledge graph.
        """
        input_embedding = self.generate_embedding(query_text)
        top_k_classes = self.ecl.get_top_k_classes(input_embedding, k=3)
        return self.execute_neo4j_query(top_k_classes)

    @staticmethod
    def generate_embedding(text):
        """
        Generate an embedding for the input text using OpenAI's embeddings API.

        :param text: Input text string.
        :return: Embedding vector as a list of floats.
        """
        try:
            response = openai.Embedding.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=text,
                encoding_format="float",
            )
            # Extract the embedding from the response
            embedding = response["data"][0]["embedding"]
            return embedding
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {e}")

    def execute_neo4j_query(self, classes):
        """
        Query the Neo4j knowledge graph to fetch denial reasons and resolutions.

        :param classes: List of top-k class names (from ECL) to query.
        :return: List of query results with class, relationship, target, and resolution.
        """
        query = textwrap.dedent(
            """
            MATCH (c:Entity)-[r]->(t:Entity)
            OPTIONAL MATCH (r)-[:RESOLVES]->(res:Entity)
            WHERE c.name IN $classes
            RETURN c.name AS class, type(r) AS relationship, t.name AS target, res.name AS resolution
            """
        )

        try:
            with self.driver.session() as session:
                result = session.run(query, classes=classes)
                return [
                    {
                        "class": record["class"],
                        "relationship": record["relationship"],
                        "target": record["target"],
                        "resolution": record["resolution"] or "No resolution available"
                    }
                    for record in result
                ]
        finally:
            session.close()
