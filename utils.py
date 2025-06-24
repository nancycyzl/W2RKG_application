import json, io, networkx as nx, pandas as pd, streamlit as st
# from streamlit_agraph import agraph, Node, Edge, Config
import numpy as np

def load_kg_file(kg_file):
    if kg_file:
        try:
            with open(kg_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading KG JSON: {e}")
    return []

def load_profiles_file(prof_file):
    if prof_file:
        try:
            return pd.read_csv(prof_file)
        except Exception as e:
            st.error(f"Error loading company profile CSV: {e}")
    return pd.DataFrame()

def build_W2R_graph(kg_triples):
    G = nx.MultiDiGraph()

    # add waste-to-resource graphs
    for entry in kg_triples:
        # Each entry: {waste, transforming_process, transformed_resource, reference}
        waste = entry.get('waste')
        resource = entry.get('transformed_resource')
        process = entry.get('transforming_process', '')
        reference = entry.get('reference', '')
        if waste and resource:
            G.add_node(waste, type='waste')
            G.add_node(resource, type='resource')
            G.add_edge(
                waste, resource, 
                process=process, reference=reference
            )
    return G

# def nx_to_agraph(Gsub, company_color="#3162C2"):
#     nodes, edges = [], []
#     for n in Gsub.nodes():
#         nodes.append(Node(id=n, label=n, size=20, shape="dot", color=company_color))

#     # for u, v, k, d in Gsub.edges(data=True, keys=True):
#     #     eid = f"{u}-{v}-{k}"
#     #     edges.append(Edge(id=eid, source=u, target=v, color="#7F8C8D", data=d | {"rel": k}))

#     for u, v, k, d in Gsub.edges(data=True, keys=True):
#         edge_id = f"{u}-{v}-{k}"
#         # add a hidden centre node
#         mid = f"mid_{edge_id}"
#         edge_props = {**d, "rel": k, "source": u, "target": v}  # ← inherit here
#         nodes.append(Node(id=mid, size=5, color="#d9d9d9", label="", data=edge_props))
#         # connect u─mid and mid─v instead of u─v
#         edges.append(Edge(id=edge_id+"_1", source=u,  target=mid, color='#666666',
#                         data=d | {"rel": k}))
#         edges.append(Edge(id=edge_id+"_2", source=mid, target=v, color='#666666',
#                         data=d | {"rel": k}))

#     return nodes, edges

def nx_to_agraph(Gsub, highlight_node=None):
    """
    Convert NetworkX graph to format compatible with st-link-analysis
    Returns nodes and edges data for st-link-analysis component
    """
    nodes_data = []
    edges_data = []
    
    # Convert nodes
    for n in Gsub.nodes():
        node_label = "query_company" if n == highlight_node else "company"
        node_data = {
            "id": n,
            "label": node_label
        }
        # Add node attributes if they exist
        if Gsub.nodes[n]:
            node_data['business'] = Gsub.nodes[n].get('business', 'n.a.')
            node_data['waste'] = Gsub.nodes[n].get('waste', 'n.a.')
            node_data['resource'] = Gsub.nodes[n].get('resource', 'n.a.')
        nodes_data.append({"data": node_data})
    
    # Convert edges - direct connections without mid_nodes
    for u, v, k, d in Gsub.edges(data=True, keys=True):
        edge_id = f"{u}-{v}-{k}"
        edge_data = {
            "id": edge_id,
            "label": "collaboration",
            "source": u,
            "target": v,
        }
        # add edge attributes
        edge_data['waste'] = d.get('waste', 'n.a.')
        edge_data['resource'] = d.get('resource', 'n.a.')
        edge_data['process'] = d.get('process', 'n.a.')
        edge_data['reference'] = d.get('reference', 'n.a.')
        edges_data.append({"data": edge_data})
    
    return nodes_data, edges_data

def save_list_to_text(alist, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in alist:
            f.write(f"{item}\n")

def read_list_from_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]






