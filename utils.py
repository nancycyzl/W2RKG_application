import json, io, networkx as nx, pandas as pd, streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

def load_kg_file(kg_file):
    if kg_file:
        try:
            return json.load(io.StringIO(kg_file.getvalue().decode()))
        except Exception as e:
            st.error(f"Error loading KG JSON: {e}")
    return []

def load_profiles_file(prof_file):
    if prof_file:
        try:
            return pd.read_json(prof_file)
        except Exception as e:
            st.error(f"Error loading company profile JSON: {e}")
    return pd.DataFrame()

def build_graph(kg_triples):
    G = nx.MultiDiGraph()
    for entry in kg_triples:
        # Each entry: {waste, transforming_process, transformed_resource, reference}
        waste = entry.get('waste')
        resource = entry.get('transformed_resource')
        process = entry.get('transforming_process', 'process')
        reference = entry.get('reference', '')
        if waste and resource:
            G.add_node(waste, type='waste')
            G.add_node(resource, type='resource')
            G.add_edge(
                waste, resource, 
                process=process, reference=reference
            )
    return G

def nx_to_agraph(Gsub, company_color="#3162C2", waste_color="#E67E22", resource_color="#27AE60"):
    nodes, edges = [], []
    for n in Gsub.nodes():
        ntype = Gsub.nodes[n].get("type", "company").lower()
        if ntype == "waste":
            color = waste_color
        elif ntype == "resource":
            color = resource_color
        else:
            color = company_color
        nodes.append(Node(id=n, label=n, size=20, shape="dot", color=color))
    for u, v, k, d in Gsub.edges(keys=True, data=True):
        edges.append(Edge(source=u, target=v, label=k, color="#7F8C8D", data=d | {"rel": k}))
    return nodes, edges

def save_list_to_text(alist, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in alist:
            f.write(f"{item}\n")

def read_list_from_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]







