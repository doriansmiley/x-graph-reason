import sqlite3
import networkx as nx
import matplotlib.pyplot as plt

# Connect to the SQLite database.
conn = sqlite3.connect("graph.db")
cursor = conn.cursor()

# Query all nodes including literal_value.
cursor.execute("SELECT id, key, label, literal_value FROM nodes;")
nodes_data = cursor.fetchall()

# Build a dictionary of node info.
node_info = {}
for node_id, key, label, literal_value in nodes_data:
    node_info[node_id] = {'key': key, 'label': label, 'literal_value': literal_value}

# Query all edges: subject_id, predicate, object_id.
cursor.execute("SELECT subject_id, predicate, object_id FROM edges;")
edges_data = cursor.fetchall()
conn.close()

# Create a directed graph.
G = nx.DiGraph()

# Only add nodes that are resources (i.e., not literal nodes).
for node_id, info in node_info.items():
    if info['literal_value'] is None:
        # Use the node label if available; otherwise, use a short version of the key.
        label = info['label'] if info['label'] is not None else info['key'].split('#')[-1]
        G.add_node(node_id, label=label)

# Define a set of predicates that we want to exclude from the visualization.
# These are typically used to define properties (labels, domains, ranges, types).
excluded_predicates = {
    'http://www.w3.org/2000/01/rdf-schema#label',
    # 'http://www.w3.org/2000/01/rdf-schema#domain',
    'http://www.w3.org/2000/01/rdf-schema#range',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
}

# Add only the edges that connect resource nodes and whose predicate is not excluded.
for subj, predicate, obj in edges_data:
    if subj in G.nodes and obj in G.nodes:
        if predicate not in excluded_predicates:
            G.add_edge(subj, obj, predicate=predicate)

# Use Graphviz's "dot" layout for a hierarchical structure.
# Note: You'll need to have pygraphviz installed (pip install pygraphviz)
try:
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
except ImportError:
    print("PyGraphviz not found, falling back to spring_layout")
    pos = nx.spring_layout(G, k=0.5, iterations=50)

# Compute positions using a spring layout.
# pos = nx.spring_layout(G, k=0.5, iterations=50)

# Draw nodes with labels.
node_labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

# Draw edges.
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10)

# Draw edge labels (showing only the last part of the predicate URI for clarity).
edge_labels = {}
for u, v, data in G.edges(data=True):
    predicate = data.get('predicate', '')
    short_pred = predicate.split('#')[-1] if '#' in predicate else predicate
    edge_labels[(u, v)] = short_pred

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title("Filtered RDF Graph")
plt.axis('off')
plt.show()
