import sqlite3

# Connect to the existing database file.
conn = sqlite3.connect("graph.db")
cursor = conn.cursor()

# ---------------------
# Query all nodes.
cursor.execute("SELECT * FROM nodes;")
nodes = cursor.fetchall()
print("Nodes:")
for node in nodes:
    print(node)

# Query all edges.
cursor.execute("SELECT * FROM edges;")
edges = cursor.fetchall()
print("\nEdges:")
for edge in edges:
    print(edge)

# ---------------------
# Query 1: Find all claim instances that have a resolution.
#
# This query assumes that claim instances have keys starting with:
# "http://example.org/insurance#Claim"
# and denial instances with keys starting with:
# "http://example.org/insurance#Denial".
# It joins on the numeric suffix of the Claim/Denial and then uses the
# 'http://example.org/insurance#resolves' predicate to find the resolution.
query_claims_resolution = """
SELECT DISTINCT
    c.key AS ClaimURI,
    r.key AS ResolutionURI
FROM nodes c
JOIN nodes d
  ON SUBSTR(c.key, LENGTH('http://example.org/insurance#Claim')+1) =
     SUBSTR(d.key, LENGTH('http://example.org/insurance#Denial')+1)
JOIN edges ed
  ON d.id = ed.subject_id
JOIN nodes r
  ON ed.object_id = r.id
WHERE c.key LIKE 'http://example.org/insurance#Claim%'
  AND d.key LIKE 'http://example.org/insurance#Denial%'
  AND ed.predicate = 'http://example.org/insurance#resolves';
"""
cursor.execute(query_claims_resolution)
claims_with_resolution = cursor.fetchall()
print("\nClaims with a resolution:")
for row in claims_with_resolution:
    print(row)

# ---------------------
# Query 2: Find all nodes with a relationship to the Claim entity.
#
# This query assumes the Claim entity is defined by the key
# 'http://example.org/insurance#Claim'. It finds all nodes that are connected
# to that node as either the subject or the object of an edge.
query_nodes_claim_relationship = """
SELECT DISTINCT n.*
FROM nodes n
WHERE n.id IN (
    SELECT subject_id FROM edges 
      WHERE object_id = (SELECT id FROM nodes WHERE key = 'http://example.org/insurance#Claim')
    UNION
    SELECT object_id FROM edges 
      WHERE subject_id = (SELECT id FROM nodes WHERE key = 'http://example.org/insurance#Claim')
);
"""
cursor.execute(query_nodes_claim_relationship)
nodes_related_to_claim = cursor.fetchall()
print("\nNodes related to the Claim entity:")
for row in nodes_related_to_claim:
    print(row)

conn.close()
