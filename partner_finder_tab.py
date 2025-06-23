import streamlit as st
from streamlit_agraph import agraph, Config
import networkx as nx
from utils import nx_to_agraph

def render_partner_finder_tab(G, kg_file, prof_file):
    c_left, c_mid, c_right = st.columns([1,2,1])
    with c_left:
        st.subheader("🔍 Query")
        with st.form("partner_finder_form"):
            waste_in  = st.text_input("Waste generated (keyword)", key="waste_in")
            res_need  = st.text_input("Resource demanded (keyword)", key="res_need")
            st.caption("Exact match—for demo purposes")
            update = st.form_submit_button("Update")
    # Only update graph if button pressed
    if update:
        st.session_state['pf_waste_in'] = st.session_state['waste_in']
        st.session_state['pf_res_need'] = st.session_state['res_need']
    waste_in_val = st.session_state.get('pf_waste_in', '')
    res_need_val = st.session_state.get('pf_res_need', '')

    sel_nodes = set()
    if waste_in_val:
        sel_nodes |= {n for n in G.nodes if waste_in_val.lower() == n.lower()}
    if res_need_val:
        sel_nodes |= {n for n in G.nodes if res_need_val.lower() == n.lower()}
    if sel_nodes:
        nbrs = set(sel_nodes)
        for n in sel_nodes:
            nbrs |= set(G.successors(n)) | set(G.predecessors(n))
        H = G.subgraph(nbrs).copy()
    else:
        H = nx.MultiDiGraph()
    
    with c_mid:
        st.subheader("📈 Interactive graph")
        if not kg_file or not prof_file:
            st.info("Upload both files to enable full graph matching.")
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