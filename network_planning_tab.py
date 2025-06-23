import streamlit as st
from streamlit_agraph import agraph, Config
from utils import nx_to_agraph

from matching import build_IS_network

def render_network_planning_tab(G, profiles_df,
                                G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                                P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings):
    c2_left, c2_mid, c2_right = st.columns([1,2,1])
    with c2_left:
        st.subheader("🔧 Filters")
        with st.form("network_planning_form"):
            planner_threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.01, key="planner_threshold")
            update2 = st.form_submit_button("Update")
    planner_threshold_val = st.session_state.get('planner_threshold', 0.8)
    
    J, num_collaborations = build_IS_network(G, profiles_df, planner_threshold_val,
                           G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                           P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings)
    
    with c2_mid:
        st.subheader("🌐 Collaboration map")
        st.write(f"Displaying {num_collaborations} collaborations, including {J.number_of_nodes()} companies.")
        if len(G_waste_list) == 0 or len(P_waste_list) == 0:
            st.info("Upload W2RKG JSON and company profile CSV to enable full network view.")
        else:
            n2, e2 = nx_to_agraph(J)
            cfg2 = Config(width=700, height=550, directed=True, physics=True, selection=True)
            result2 = agraph(nodes=n2, edges=e2, config=cfg2)

    # ---------- fast look-up helpers ----------
    company_node_lookup = {node.id: node for node in n2 if not node.id.startswith("mid_")}
    mid_node_lookup = {node.id: node for node in n2 if node.id.startswith("mid_")}
            
    with c2_right:
        st.subheader("ℹ️ Details")
        if result2:
            print(f"result2: {result2}")
            # single click ------------------
            if isinstance(result2, str):
                if result2 in company_node_lookup:
                    nid = result2
                    props = J.nodes[nid]
                    st.write(f"**Selected node:** `{nid}`")
                    st.json(props)
                elif result2 in mid_node_lookup:
                    st.write("**Transformation process:**")
                    st.json(mid_node_lookup[result2].data)
                else:
                    st.write(f"Selected item is {result2}.")
                    st.write(f"company node lookup: {company_node_lookup}")
                    st.write(f"mid node lookup: {mid_node_lookup}")

        else:
            st.write("Click a node or edge to view its data.") 