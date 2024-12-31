import json
import numpy as np
from rdflib import Graph, Namespace, RDF, RDFS, Literal
from sklearn.metrics import pairwise_distances
import textwrap
from scipy.spatial.distance import mahalanobis

# Set up OpenAI client

# Namespace for RDF
COS = Namespace("http://example.org/insurance/")

# External Continual Learner (ECL)
class ExternalContinualLearner:
    def __init__(self, client):
        self.class_means = {}  # Mean embeddings for each class
        self.shared_covariance = None  # Shared covariance matrix
        self.num_classes = 0
        self.epsilon = 1e-6  # Small value for regularization
        self.client = client

    def update_class(self, class_name, tag_embeddings):
        mean_embedding = np.mean(tag_embeddings, axis=0)
        self.class_means[class_name] = mean_embedding

        # Handle cases where tag_embeddings has less than 2 rows
        if tag_embeddings.shape[0] < 2:
            class_covariance = np.eye(tag_embeddings.shape[1]) * self.epsilon
        else:
            deviations = tag_embeddings - mean_embedding
            class_covariance = np.atleast_2d(np.cov(deviations, rowvar=False))

        if self.shared_covariance is None:
            self.shared_covariance = np.eye(tag_embeddings.shape[1]) * self.epsilon
        else:
            self.shared_covariance = (
                self.num_classes * self.shared_covariance + class_covariance
            ) / (self.num_classes + 1)

        self.num_classes += 1

        # Regularize the covariance matrix
        self.shared_covariance += self.epsilon * np.eye(self.shared_covariance.shape[0])

    def get_top_k_classes(self, input_embedding, k=3):
        if not self.class_means:
            raise ValueError("No classes have been added to the learner.")
        
        # Ensure input_embedding is a NumPy array
        input_embedding = np.array(input_embedding)


        embedding_size = input_embedding.shape[0]
        if self.shared_covariance.shape != (embedding_size, embedding_size):
            raise ValueError(f"Shared covariance matrix shape mismatch: expected ({embedding_size}, {embedding_size}), got {self.shared_covariance.shape}")

        VI = np.linalg.inv(self.shared_covariance)
        distances = {
            class_name: mahalanobis(input_embedding, mean, VI)
            for class_name, mean in self.class_means.items()
        }

        return sorted(distances, key=distances.get)[:k]


# RDF Processor
class RDFProcessor:
    def __init__(self):
        self.graph = Graph()

    def ensure_rdf_class(self, class_name):
        class_node = COS[class_name]
        if (class_node, RDF.type, RDFS.Class) not in self.graph:
            self.graph.add((class_node, RDF.type, RDFS.Class))
        return class_name

    def ensure_rdf_relationships(self, relationships):
        updated_relationships = {}
        for rel, target in relationships.items():
            rel_node = COS[rel]
            target_node = COS[target.replace(" ", "_")]

            if (rel_node, RDF.type, RDF.Property) not in self.graph:
                self.graph.add((rel_node, RDF.type, RDF.Property))
                self.graph.add((rel_node, RDFS.label, Literal(rel)))
            if (target_node, RDF.type, RDFS.Class) not in self.graph:
                self.graph.add((target_node, RDF.type, RDFS.Class))

            updated_relationships[rel] = target

        return updated_relationships

    def update_rdf_with_relationships(self, class_name, relationships):
        class_node = COS[class_name]
        self.graph.add((class_node, RDF.type, RDFS.Class))
        for rel, target in relationships.items():
            rel_node = COS[rel]
            target_node = COS[target.replace(" ", "_")]
            self.graph.add((class_node, rel_node, target_node))

    def update_rdf_with_resolutions(self, relationships, resolutions):
        """
        Update RDF graph with resolutions for relationships (denial reasons).
        """
        for rel, resolution in resolutions.items():
            resolution_node = COS[resolution.replace(" ", "_")]
            rel_node = COS[rel]

            # Add resolution as a node and link it to the relationship
            self.graph.add((resolution_node, RDF.type, RDFS.Class))
            self.graph.add((rel_node, COS["resolves"], resolution_node))

# Neo4j Knowledge Graph Integration
class Neo4jKnowledgeGraph:
    def __init__(self, driver):
        self.driver = driver

    def add_rdf_to_graph(self, rdf_graph):
        """
        Adds RDF triples to the Neo4j knowledge graph.

        :param rdf_graph: The RDF graph containing triples to add to Neo4j.
        """
        # Define a set of valid relationship types based on the TTL file
        valid_relationships = {
            "http://example.org/insurance/denialReason",
            "http://example.org/insurance/denialCode",
            "http://example.org/insurance/adjustment",
            "http://example.org/insurance/preauthorization",
            "http://example.org/insurance/medicalNecessity",
            "http://example.org/insurance/resolves"
        }

        session = self.driver.session()  # Open a single session
        try:
            for subj, pred, obj in rdf_graph:
                # Filter out invalid predicates
                if str(pred) not in valid_relationships:
                    print(f"Skipping invalid relationship: {subj} -[{pred}]-> {obj}")
                    query = textwrap.dedent("""
                            MERGE (s:Entity {name: $subj})
                            MERGE (o:Entity {name: $obj})
                        """)
                else:
                     print(f"Adding relationship: {subj} -[{pred}]-> {obj}")
                     query = textwrap.dedent(f"""
                            MERGE (s:Entity {{name: $subj}})
                            MERGE (o:Entity {{name: $obj}})
                            MERGE (s)-[:`{pred}`]->(o)
                        """)

                # Run the query with parameters for `subj` and `obj`
                session.run(
                    query,
                    subj=str(subj),
                    obj=str(obj)
                )
        finally:
            session.close()



class KnowledgeGraphUpdater:
    def __init__(self, rdf_file, neo4j_driver, client):
        """
        Initialize KnowledgeGraphUpdater with RDFProcessor, ECL, and Neo4jKnowledgeGraph.
        Load the supplied RDF file to populate the RDF graph, Neo4j graph, and ECL.

        :param rdf_file: Path to the RDF file to load initial data.
        :param neo4j_driver: Neo4j driver for connecting to the database.
        """
        # Initialize dependencies
        self.client = client
        self.rdf_processor = RDFProcessor()
        self.ecl = ExternalContinualLearner(client)
        self.kg = Neo4jKnowledgeGraph(neo4j_driver)

        # Load RDF data
        self.load_rdf_data(rdf_file)

    def load_rdf_data(self, rdf_file):
        """
        Load RDF data from the provided file into the RDF graph, Neo4j graph, and ECL.

        :param rdf_file: Path to the RDF file.
        """
        # Parse the RDF file
        self.rdf_processor.graph.parse(rdf_file, format="turtle")

        # Populate Neo4j from RDF
        self.kg.add_rdf_to_graph(self.rdf_processor.graph)

        # Initialize ECL from RDF graph
        self.initialize_ecl_from_rdf()

    def initialize_ecl_from_rdf(self):
        """
        Initialize the External Continual Learner (ECL) from the RDF graph.
        """
        for class_node in self.rdf_processor.graph.subjects(RDF.type, RDFS.Class):
            # Find all relationships for the class
            relationships = {
                str(pred): str(obj)
                for pred, obj in self.rdf_processor.graph.predicate_objects(subject=class_node)
                if pred != RDF.type  # Ignore the type predicate
            }

            # Generate embeddings for relationships
            relationship_embeddings = [
                self.generate_embedding(rel) for rel in relationships.keys()
            ]

            # Skip if no relationships are found
            if not relationship_embeddings:
                print(f"Skipping class {class_node} due to no relationships.")
                continue

            # Update the ECL with the class and its relationship embeddings
            self.ecl.update_class(str(class_node), np.array(relationship_embeddings))

    def process_corpus(self, corpus):
        """
        Process a corpus to dynamically update the RDF graph, Neo4j graph, and ECL.

        :param corpus: List of text strings to process.
        """
        for text in corpus:
            # Extract class name, relationships, and resolutions
            class_name, relationships, resolutions = self.extract_data_from_text_with_resolutions(text)

            # Update RDF
            self.rdf_processor.update_rdf_with_relationships(class_name, relationships)
            self.rdf_processor.update_rdf_with_resolutions(relationships, resolutions)

            # Update ECL with relationship embeddings
            relationship_embeddings = [self.generate_embedding(rel) for rel in relationships.keys()]
            self.ecl.update_class(class_name, np.array(relationship_embeddings))

        # Push RDF data to Neo4j
        self.kg.add_rdf_to_graph(self.rdf_processor.graph)

    def extract_data_from_text_with_resolutions(self, text):
        """
        Use OpenAI GPT to extract class, relationships, and resolutions from text.

        :param text: Input text to process.
        :return: Extracted class name, relationships, and resolutions.
        """
        prompt = f"""
        Extract entities from the following text. Identify:
        1. A main class (e.g., 'Claim').
        2. Relationships (tags) with their target entities (e.g., 'denialReason': 'Preauthorization required').
        3. Suggested resolutions for denial codes or reasons.

        Text: {text}

        Format the response as JSON:
        {{
            "class": "ClassName",
            "relationships": {{
                "relationship1": "TargetEntity1",
                "relationship2": "TargetEntity2"
            }},
            "resolutions": {{
                "relationship1": "Resolution1",
                "relationship2": "Resolution2"
            }}
        }}
        """
        try:
            response = self.client.chat.completions.create(model="gpt-4",
            messages=[{"role": "user", "content": prompt}])
            extracted_data = json.loads(response.choices[0].message.content)

            class_name = self.rdf_processor.ensure_rdf_class(extracted_data["class"])
            relationships = self.rdf_processor.ensure_rdf_relationships(extracted_data["relationships"])
            resolutions = extracted_data["resolutions"]

            return class_name, relationships, resolutions
        except Exception as e:
            raise RuntimeError(f"Error during OpenAI GPT extraction: {e}")

    def generate_embedding(self, text):
        """
        Generate an embedding for the input text using OpenAI's embeddings API.

        :param text: Input text string.
        :return: Embedding vector as a list of floats.
        """
        try:
            response = self.client.embeddings.create(model="text-embedding-3-small",
            input=text,
            encoding_format="float")
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {e}")