class KnowledgeGraphQueryHandler:
    def __init__(self, ecl, kg_driver):
        self.ecl = ecl
        self.driver = kg_driver

    def query_knowledge_graph(self, query_text):
        input_embedding = self.generate_embedding(query_text)
        top_k_classes = self.ecl.get_top_k_classes(input_embedding, k=3)
        return self.execute_neo4j_query(top_k_classes)

    @staticmethod
    def generate_embedding(text):
        return np.mean([embedding_model.encode(word) for word in text.split()], axis=0)

    def execute_neo4j_query(self, classes):
        """
        Query the Neo4j knowledge graph to fetch denial reasons and resolutions.
        """
        query = """
        MATCH (c:Entity)-[r]->(t:Entity)
        OPTIONAL MATCH (r)-[:RESOLVES]->(res:Entity)
        WHERE c.name IN $classes
        RETURN c.name AS class, type(r) AS relationship, t.name AS target, res.name AS resolution
        """
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
