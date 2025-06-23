# ──────────────────────────────────────────────
# Waste-to-Resource KG - quick demo UI
# • Streamlit + streamlit-agraph (interactive graph)
# • NetworkX for matching / filtering logic
# install:  pip install streamlit networkx streamlit-agraph pandas
# run:      streamlit run w2r_demo.py
# ──────────────────────────────────────────────
import json, io, networkx as nx, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from streamlit_agraph import agraph, Node, Edge, Config
# Import tab rendering functions from separate files for clarity
from partner_finder_tab import render_partner_finder_tab
from network_planning_tab import render_network_planning_tab
from sentence_transformers import SentenceTransformer

from utils import load_kg_file, load_profiles_file, build_W2R_graph
from matching import obtain_W2RKG_embeddings, obtain_profile_embeddings

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


st.set_page_config(page_title="W2RKG Demonstration", layout="wide")
st.title("⚡ Waste-to-Resource Knowledge Graph — Demonstration")



# intialize kg_file and prof_file
kg_file_default = 'data_utils/fused_triples_aggregated.json'
prof_file_default = 'data_utils/Maestri_case1.csv'

col_up1, col_up2, _ = st.columns([1, 1, 3])  # Add a third, wider column for spacing
with col_up1:
    kg_file_uploaded = st.file_uploader("📤 Upload W2RKG JSON", type="json", key="kg_main")
with col_up2:
    prof_file_uploaded = st.file_uploader("📤 Upload company profile CSV", type="csv", key="profile_main")

# Use uploaded file if available, otherwise use default
kg_file = kg_file_uploaded if kg_file_uploaded is not None else kg_file_default
prof_file = prof_file_uploaded if prof_file_uploaded is not None else prof_file_default

kg_triples = load_kg_file(kg_file)     # a list of dicts
profiles_df = load_profiles_file(prof_file)    # a dataframe

model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings = obtain_W2RKG_embeddings(kg_triples, model)
P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings = obtain_profile_embeddings(profiles_df, model)

G = build_W2R_graph(kg_triples)    # a networkx graph for W2RKG
st.write(f"W2RKG has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

tab1, tab2 = st.tabs(["🧭 Opportunity identification", "🗺️ Network planning"])

with tab1:
    # render_partner_finder_tab(G,
    #                           G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
    #                           P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings)
    st.write("🚧 Partner finder functionality coming soon!")

with tab2:
    render_network_planning_tab(G, profiles_df,
                                G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                                P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings)
