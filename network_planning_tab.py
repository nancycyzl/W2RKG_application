import streamlit as st
from utils import nx_to_agraph
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle

from matching import build_IS_network

def render_network_planning_tab(G, profiles_dict,
                                G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                                P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings):
    c2_left, c2_right = st.columns([1,3])
    with c2_left:
        st.subheader("🔧 Settings")
        with st.form("network_planning_form"):
            planner_threshold = st.slider("Similarity threshold", min_value=0.5, max_value=1.0, value=0.8, step=0.01, key="planner_threshold")
            update2 = st.form_submit_button("Update")
    planner_threshold_val = st.session_state.get('planner_threshold', 0.8)
    
    J, num_collaborations = build_IS_network(G, profiles_dict, planner_threshold_val,
                           G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                           P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings)
    
    with c2_right:
        st.subheader("🌐 Collaboration network")
        st.write(f"Displaying {num_collaborations} collaborations, including {J.number_of_nodes()} companies.")
        if len(G_waste_list) == 0 or len(P_waste_list) == 0:
            st.info("Upload W2RKG JSON and company profile CSV to enable full network view.")
        else:
            nodes_data, edges_data = nx_to_agraph(J)
            elements = {
                "nodes": nodes_data,
                "edges": edges_data
            }
            node_style = [NodeStyle(label="company", color="#004d99", caption="id")]
            edge_style = [EdgeStyle(label="collaboration", color="#999999", directed=True)]
            
            # Use st-link-analysis instead of streamlit-agraph
            result2 = st_link_analysis(
                layout="cola",       # cose, random, grid, circle, concentric, breadthfirst, fcose, cola
                elements=elements,
                node_styles=node_style,
                edge_styles=edge_style,
                height=530
            )
