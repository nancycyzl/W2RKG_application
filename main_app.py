# ──────────────────────────────────────────────
# Waste-to-Resource KG - quick demo UI
# • Streamlit + streamlit-agraph (interactive graph)
# • NetworkX for matching / filtering logic
# install:  pip install streamlit networkx streamlit-agraph pandas
# run:      streamlit run w2r_demo.py
# ──────────────────────────────────────────────
import json, io, networkx as nx, pandas as pd, streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
# Import tab rendering functions from separate files for clarity
from partner_finder_tab import render_partner_finder_tab
from network_planning_tab import render_network_planning_tab

from utils import load_kg_file, load_profiles_file, build_graph, create_material_embeddings

st.markdown(
    """
    <style>
    /* target every file-uploader */
    div[data-testid="stFileUploader"] > section {
        /* removes the whole dropzone panel (keeps the button) */
        display: none;
    }
    /* shrink the label margin */
    div[data-testid="stFileUploader"] > label {
        margin-bottom: 0.25rem;
        font-size: 0.9rem;
    }
    /* optional: shrink the button itself */
    div[data-testid="stFileUploader"] button {
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.set_page_config(page_title="W2RKG Demo", layout="wide")
st.title("⚡ Waste-to-Resource Matcher — demo")

col_up1, col_up2, _ = st.columns([1, 1, 3])  # Add a third, wider column for spacing
with col_up1:
    kg_file = st.file_uploader("📤 Upload W2RKG JSON", type="json", key="kg_main")
with col_up2:
    prof_file = st.file_uploader("📤 Upload company profile JSON", type="json", key="profile_main")

kg_triples = load_kg_file(kg_file)
profiles = load_profiles_file(prof_file)
G = build_graph(kg_triples)

G_waste_embeddings = create_material_embeddings(G.nodes)
G_resource_embeddings = create_material_embeddings(G.nodes)
profile_waste_embeddings = create_material_embeddings(profiles['waste'])
profile_resource_embeddings = create_material_embeddings(profiles['resource'])

tab1, tab2 = st.tabs(["🧭 Opportunity identification", "🗺️ Network planning"])

with tab1:
    render_partner_finder_tab(G, kg_file, prof_file)
with tab2:
    render_network_planning_tab(G, kg_file, prof_file)
