
import textwrap

class KnowledgeGraphQueryHandler:
    def __init__(self, ecl, kg_driver, client):
        """
        Initialize the query handler with the External Continual Learner (ECL)
        and Neo4j database driver.

        :param ecl: External Continual Learner (ECL) for class embeddings.
        :param kg_driver: Neo4j database driver.
        :param client: the openai client instance
        """
        self.ecl = ecl
        self.driver = kg_driver
        self.client = client

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

    def generate_embedding(self, text):
        """
        Generate an embedding for the input text using OpenAI's embeddings API.

        :param text: Input text string.
        :return: Embedding vector as a list of floats.
        """
        try:
            response = self.client.embeddings.create(model='text-embedding-3-small',
            input=text,
            encoding_format="float")
            # Extract the embedding from the response
            embedding = response.data[0].embedding
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
            MATCH (c:Entity)-[r:`http://example.org/insurance/resolves`]->(res:Entity)
            MATCH (class)
            WHERE class.name IN $classes
            RETURN c.name AS class, 
                type(r) AS relationship, 
                res.name AS resolution,
                res.name AS target
            """
        )

        print(f"the query is:\n {query}")
        print(f"the classes are:\n {classes}")

        try:
            with self.driver.session() as session:
                result = session.run(query, classes=classes)
                
                cleaned_results = []
                for record in result:
                    cleaned_results.append({
                        "class": self._format_uri(record["class"]),
                        "relationship": record["relationship"],
                        "target": self._format_uri(record["target"]),
                        "resolution": record["resolution"] or "No resolution available"
                    })

                return cleaned_results
        finally:
            session.close()

    def _format_uri(self, uri):
        """
        Format URIs to make them more readable.
        Removes the namespace and returns the human-readable portion.
        """
        if uri and isinstance(uri, str):
            return uri.split("/")[-1].replace("_", " ")
        return uri
