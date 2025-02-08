import sqlite3
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDFS, XSD

# ---------------------
# SET UP THE DATABASE
# ---------------------
conn = sqlite3.connect("graph.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE,        -- unique key: for resources, the URI; for literals, a composite string
    uri TEXT,               -- not null for resource/BNode nodes
    literal_value TEXT,     -- not null for literal nodes
    datatype TEXT,
    language TEXT,
    label TEXT              -- if available (e.g. from rdfs:label)
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL,
    predicate TEXT NOT NULL,
    object_id INTEGER NOT NULL,
    FOREIGN KEY(subject_id) REFERENCES nodes(id),
    FOREIGN KEY(object_id) REFERENCES nodes(id)
);
""")

cursor.execute("DELETE FROM edges;")
cursor.execute("DELETE FROM nodes;")

conn.commit()

# ---------------------
# HELPER: NODE CACHE & INSERTION
# ---------------------
# A cache to avoid inserting the same node twice.
node_cache = {}

def get_node_key(node):
    """Return a unique key string for a node."""
    if isinstance(node, URIRef):
        return str(node)
    elif isinstance(node, BNode):
        return "BNode:" + str(node)
    elif isinstance(node, Literal):
        # Create a composite key including datatype and language.
        dt = str(node.datatype) if node.datatype else ""
        lang = node.language if node.language else ""
        return f"Literal:{str(node)}:{dt}:{lang}"
    else:
        raise ValueError("Unknown node type: " + str(node))

def get_node_values(node):
    """Return a tuple (uri, literal_value, datatype, language) for insertion."""
    if isinstance(node, URIRef):
        return (str(node), None, None, None)
    elif isinstance(node, BNode):
        # For blank nodes, we may treat them similar to URIs.
        return (str(node), None, None, None)
    elif isinstance(node, Literal):
        return (None, str(node), str(node.datatype) if node.datatype else None, node.language if node.language else None)
    else:
        raise ValueError("Unknown node type: " + str(node))

def get_node_id(node):
    """Return the id of a node from the database, inserting it if necessary."""
    key = get_node_key(node)
    if key in node_cache:
        return node_cache[key]
    
    uri, literal_value, datatype, language = get_node_values(node)
    
    # Try to see if the node is already in the DB.
    cursor.execute("SELECT id FROM nodes WHERE key = ?", (key,))
    result = cursor.fetchone()
    if result:
        node_id = result[0]
    else:
        cursor.execute(
            "INSERT INTO nodes (key, uri, literal_value, datatype, language) VALUES (?, ?, ?, ?, ?)",
            (key, uri, literal_value, datatype, language)
        )
        conn.commit()
        node_id = cursor.lastrowid
    node_cache[key] = node_id
    return node_id

# ---------------------
# LOAD THE RDF GRAPH
# ---------------------
with open("app/base_data.ttl", "rb") as file:
    rdf_file_bytes = file.read()

g = Graph()
g.parse(data=rdf_file_bytes, format="turtle")

# ---------------------
# PROCESS TRIPLES INTO THE DATABASE
# ---------------------
for s, p, o in g:
    # Get node IDs for subject and object.
    subject_id = get_node_id(s)
    object_id = get_node_id(o)
    predicate = str(p)
    
    # Insert the edge (i.e. triple).
    cursor.execute(
        "INSERT INTO edges (subject_id, predicate, object_id) VALUES (?, ?, ?)",
        (subject_id, predicate, object_id)
    )
    conn.commit()
    
    # OPTIONAL: If the predicate is rdfs:label, update the subject's label column.
    if predicate == str(RDFS.label):
        # Here we assume that the object is a literal label.
        cursor.execute(
            "UPDATE nodes SET label = ? WHERE id = ?",
            (str(o), subject_id)
        )
        conn.commit()

print("Graph has been successfully inserted into the SQL database.")
