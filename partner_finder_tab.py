import streamlit as st
import networkx as nx
from utils import nx_to_agraph
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle

from matching import build_partner_linkages

def render_partner_finder_tab(G, profiles_dict, model,
                              G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                              P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings):
    c_left, c_right = st.columns([1,3])
    with c_left:
        st.subheader("🔍 Query")
        with st.form("partner_finder_form"):
            query_waste  = st.text_input("Waste generated (use ; to separate)", key="query_waste")
            query_resource  = st.text_input("Resource demanded (use ; to separate)", key="query_resource")
            query_threshold = st.slider("Similarity threshold", min_value=0.5, max_value=1.0, value=0.8, step=0.01, key="query_threshold")
            update = st.form_submit_button("Update")

    query_waste_val = st.session_state.get('query_waste', '')
    query_resource_val = st.session_state.get('query_resource', '')
    query_threshold_val = st.session_state.get('query_threshold', 0.8)

    query_company_id = "Query company"

    J, num_collaborations = build_partner_linkages(G, profiles_dict, query_company_id, query_waste_val, query_resource_val, query_threshold_val, model,
                                                    G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                                                    P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings)
    
    with c_right:
        if not query_waste_val and not query_resource_val:
            st.info("Please input a waste material and/or resource material to find potential partners.")
        if model is None:
            st.info("Please select an embedding model to proceed.")
        else:
            st.subheader("🎯 Potential partners")
            st.write(f"Displaying {num_collaborations} collaborations, including {J.number_of_nodes()} companies.")
            if len(G_waste_list) == 0 or len(P_waste_list) == 0:
                st.info("Upload W2RKG JSON and company profile CSV to enable full graph matching.")
            else:
                nodes_data, edges_data = nx_to_agraph(J, highlight_node=query_company_id)
                elements = {
                    "nodes": nodes_data,
                    "edges": edges_data
                }
                node_style = [NodeStyle(label="company", color="#004d99", caption="id"),           # blue
                            NodeStyle(label="query_company", color="#cc6600", caption="id")]     # orange
                edge_style = [EdgeStyle(label="collaboration", color="#999999", directed=True)]    # grey

                result = st_link_analysis(
                    layout="cola",       # cose, random, grid, circle, concentric, breadthfirst, fcose, cola
                    elements=elements,
                    node_styles=node_style,
                    edge_styles=edge_style,
                    height=530
                )
