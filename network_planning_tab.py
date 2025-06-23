import streamlit as st
from streamlit_agraph import agraph, Config
from utils import nx_to_agraph

from matching import build_W2R_Comp_network

def render_network_planning_tab(G, profiles_df,
                                G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                                P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings):
    c2_left, c2_mid, c2_right = st.columns([1,2,1])
    with c2_left:
        st.subheader("🔧 Filters")
        with st.form("network_planning_form"):
            planner_threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.01, key="planner_threshold")
            update2 = st.form_submit_button("Update")
    # Only update graph if button pressed
    if update2:
        st.session_state['planner_threshold'] = st.session_state['planner_threshold']
    planner_threshold_val = st.session_state.get('planner_threshold', 0.8)
    
    J = build_W2R_Comp_network(G, profiles_df, planner_threshold_val,
                           G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                           P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings)
    
    with c2_mid:
        st.subheader("🌐 Collaboration map")
        if len(G_waste_list) == 0 or len(P_waste_list) == 0:
            st.info("Upload W2RKG JSON and company profile CSV to enable full network view.")
        else:
            n2, e2 = nx_to_agraph(J)
            cfg2 = Config(width=700, height=550, directed=True, physics=True, selection=True)
            result2 = agraph(nodes=n2, edges=e2, config=cfg2)
            
    with c2_right:
        st.subheader("ℹ️ Details")
        if 'result2' in locals() and result2:
            if result2.get("nodes"):
                nid = result2["nodes"][0]
                st.markdown(f"**Selected node:** `{nid}`")
                st.json(J.nodes[nid])
            elif result2.get("edges"):
                ed = result2["edges"][0]
                st.markdown("**Selected link:**")
                st.json(ed["data"])
        else:
            st.write("Click a node or edge to view its data.") 