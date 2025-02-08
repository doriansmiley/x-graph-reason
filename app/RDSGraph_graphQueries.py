import sqlite3
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS
import gc
import json

# --- Build the RDF Graph from the database ---

# Create an empty RDF graph.
g = Graph()

# Connect to your SQLite database.
conn = sqlite3.connect("graph.db")
cursor = conn.cursor()

# Load nodes for building the graph (we only need minimal info for this purpose).
cursor.execute("SELECT id, key, uri, literal_value, datatype, language, label FROM nodes")
temp_nodes_dict = {}
for row in cursor.fetchall():
    node_id, key, uri, literal_value, datatype, language, label = row
    temp_nodes_dict[node_id] = {
        'key': key,
        'uri': uri,
        'literal_value': literal_value,
        'datatype': datatype,
        'language': language,
        'label': label
    }

# Function to create an rdflib node.
def make_node(node_info):
    if node_info['literal_value'] is not None:
        return Literal(node_info['literal_value'])
    else:
        return URIRef(node_info['key'])

# A helper function to convert a SQL row into a dictionary.
def row_to_dict(row):
    return {
        "id": row[0],
        "key": row[1],
        "uri": row[2],
        "literal_value": row[3],
        "datatype": row[4],
        "language": row[5],
        "label": row[6]
    }

# Load edges and add triples.
cursor.execute("SELECT subject_id, predicate, object_id FROM edges")
for subject_id, predicate, object_id in cursor.fetchall():
    subj = make_node(temp_nodes_dict[subject_id])
    obj = make_node(temp_nodes_dict[object_id])
    pred = URIRef(predicate)
    g.add((subj, pred, obj))

conn.close()
del temp_nodes_dict
gc.collect()

# --- Reopen a DB connection for batch queries during output ---
conn_print = sqlite3.connect("graph.db")
cursor_print = conn_print.cursor()

# Your SPARQL query (this is just one example)
query = """
PREFIX ex: <http://example.org/insurance#>
SELECT ?claim ?denial ?resolution ?denialReason
WHERE {
    ?claim a ex:Claim .
    ?denial a ex:Denial ;
            ex:denialReason ?denialReason ;
            ex:resolves ?resolution .
    FILTER(CONTAINS(str(?denialReason), "Preauthorization"))
}
"""

# For each result, batch query the DB for the corresponding nodes.
for row in g.query(query):
    claim_uri = str(row.claim)
    denial_uri = str(row.denial)
    resolution_uri = str(row.resolution)
    denial_reason = str(row.denialReason)
    
    # Batch query: fetch all nodes with keys equal to claim_uri, denial_uri, or resolution_uri.
    cursor_print.execute(
        "SELECT id, key, uri, literal_value, datatype, language, label FROM nodes WHERE key IN (?,?,?)",
        (claim_uri, denial_uri, resolution_uri)
    )
    batch_results = cursor_print.fetchall()
    
    # Build a simple mapping from key to full node row.
    mapping = { r[1]: r for r in batch_results }
    
    # Create a JSON object with the full details.
    result_obj = {
        "claim": row_to_dict(mapping.get(claim_uri)) if mapping.get(claim_uri) else None,
        "denial": row_to_dict(mapping.get(denial_uri)) if mapping.get(denial_uri) else None,
        "resolution": row_to_dict(mapping.get(resolution_uri)) if mapping.get(resolution_uri) else None,
        "denial_reason": denial_reason
    }
    
    # Print the JSON object.
    print(json.dumps(result_obj, indent=2))

conn_print.close()
