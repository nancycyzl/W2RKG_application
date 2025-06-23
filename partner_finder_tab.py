import streamlit as st
from streamlit_agraph import agraph, Config
import networkx as nx
from utils import nx_to_agraph

def render_partner_finder_tab(G, G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                              P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings):
    c_left, c_mid, c_right = st.columns([1,2,1])
    with c_left:
        st.subheader("🔍 Query")
        with st.form("partner_finder_form"):
            query_waste  = st.text_input("Waste generated (keyword)", key="query_waste")
            query_resource  = st.text_input("Resource demanded (keyword)", key="query_resource")
            query_threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.01, key="query_threshold")
            st.caption("Match by embeddings")
            update = st.form_submit_button("Update")
    # Only update graph if button pressed
    if update:
        st.session_state['pf_query_waste'] = st.session_state['query_waste']
        st.session_state['pf_query_resource'] = st.session_state['query_resource']
        st.session_state['pf_query_threshold'] = st.session_state['query_threshold']
    query_waste_val = st.session_state.get('pf_query_waste', '')
    query_resource_val = st.session_state.get('pf_query_resource', '')
    query_threshold_val = st.session_state.get('pf_query_threshold', 0.8)

    # create subgraph (test)
    H = G.copy()
    
    with c_mid:
        st.subheader("📈 Interactive graph")
        if len(G_waste_list) == 0 or len(P_waste_list) == 0:
            st.info("Upload W2RKG JSON and company profile CSV to enable full graph matching.")
        elif not H:
            st.info("Type a keyword to see matches.")
        else:
            nodes, edges = nx_to_agraph(H)
            cfg = Config(width=700, height=550, directed=True, physics=True, hierarchical=False, selection=True)
            result = agraph(nodes=nodes, edges=edges, config=cfg)
    with c_right:
        st.subheader("ℹ️ Details")
        if 'result' in locals() and result:
            if result.get("nodes"):
                nid = result["nodes"][0]
                st.markdown(f"**Selected node:** `{nid}`")
                st.json(G.nodes[nid])
            elif result.get("edges"):
                ed = result["edges"][0]
                st.markdown("**Selected link:**")
                st.json(ed["data"])
        else:
            st.write("Click a node or edge to view its data.") 