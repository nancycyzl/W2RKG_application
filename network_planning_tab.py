import streamlit as st
from streamlit_agraph import agraph, Config
from utils import nx_to_agraph

def render_network_planning_tab(G, kg_file, prof_file):
    c2_left, c2_mid, c2_right = st.columns([1,2,1])
    with c2_left:
        st.subheader("🔧 Filters")
        with st.form("network_planning_form"):
            waste_kw = st.text_input("Filter waste keyword", key="waste_kw")
            res_kw = st.text_input("Filter resource keyword", key="res_kw")
            min_cap = st.number_input("Min. capacity (if in edge data)", min_value=0.0, step=1.0, value=0.0, key="min_cap")
            update2 = st.form_submit_button("Update")
    # Only update graph if button pressed
    if update2:
        st.session_state['np_waste_kw'] = st.session_state['waste_kw']
        st.session_state['np_res_kw'] = st.session_state['res_kw']
        st.session_state['np_min_cap'] = st.session_state['min_cap']
    waste_kw_val = st.session_state.get('np_waste_kw', '')
    res_kw_val = st.session_state.get('np_res_kw', '')
    min_cap_val = st.session_state.get('np_min_cap', 0.0)
    J = G.copy()
    if waste_kw_val:
        J.remove_nodes_from([n for n in J if waste_kw_val.lower() not in n.lower()])
    if res_kw_val:
        J.remove_nodes_from([n for n in J if res_kw_val.lower() not in n.lower()])
    if min_cap_val > 0:
        J.remove_edges_from(
            [(u,v,k) for u,v,k,d in J.edges(keys=True, data=True)
                      if float(d.get("capacity", 0)) < min_cap_val]
        )
    
    with c2_mid:
        st.subheader("🌐 Collaboration map")
        if not kg_file or not prof_file:
            st.info("Upload both files to enable full network view.")
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