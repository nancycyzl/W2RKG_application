import json
import io
import os
import networkx as nx
import pandas as pd
from pyvis.network import Network
import matplotlib.pyplot as plt
import argparse
import pyvis


from matching import obtain_W2RKG_embeddings, obtain_profile_embeddings, build_W2R_Comp_network, filter_colloaration_links
from utils import load_kg_file, load_profiles_file, build_W2R_graph


def create_network(kg_file, prof_file, threshold=0.8):
    
    # the basic W2R graph
    kg_triples = load_kg_file(kg_file)
    G = build_W2R_graph(kg_triples)    # a networkx graph for W2RKG
    print(f"W2R graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


    # add company
    profiles_df = load_profiles_file(prof_file)    # a dataframe

    # model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    model = "NA"

    G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings = obtain_W2RKG_embeddings(kg_triples, model)
    P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings = obtain_profile_embeddings(profiles_df, model)

    G = build_W2R_Comp_network(G, profiles_df, threshold,
                            G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                            P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings)

    print(f"W2R graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    return G

def visualize_network(G, save_folder):
    # Find company nodes
    company_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'company']

    # Find all wastes/resources connected to companies
    connected_nodes = set(company_nodes)
    for company in company_nodes:
        for neighbor in G.neighbors(company):
            connected_nodes.add(neighbor)
        for neighbor in G.predecessors(company):
            connected_nodes.add(neighbor)

    # Also find waste-resource edges
    for waste in [n for n in connected_nodes if G.nodes[n].get('type') == 'waste']:
        for neighbor in G.neighbors(waste):
            if G.nodes[neighbor].get('type') == 'resource' and neighbor in connected_nodes:
                connected_nodes.add(waste)
                connected_nodes.add(neighbor)

    # Induce subgraph
    H = G.subgraph(connected_nodes).copy()
    # Remove isolated waste/resource nodes (not connected to any company)
    to_remove = [n for n in H.nodes if H.nodes[n].get('type') in ('waste', 'resource') and H.degree(n) == 0]
    H.remove_nodes_from(to_remove)
    # Draw
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(H)
    # Draw nodes by type
    node_colors = []
    for n in H.nodes():
        ntype = H.nodes[n].get('type', 'company')
        if ntype == 'company':
            node_colors.append('#3162C2')
        elif ntype == 'waste':
            node_colors.append('#E67E22')
        elif ntype == 'resource':
            node_colors.append('#27AE60')
        else:
            node_colors.append('#888888')
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(H, pos, font_size=8)
    nx.draw_networkx_edges(H, pos, arrows=True, arrowstyle='-|>', min_source_margin=10, min_target_margin=10, edge_color='grey')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'company_network.png'))
    plt.close()

def visualize_network_collaboration(G, save_folder):
    # create IS network
    H, num_collaborations = filter_colloaration_links(G)
    print(f"Total collaborations: {num_collaborations}")
    print(f"Total nodes: {H.number_of_nodes()}")
    print(f"Total edges: {H.number_of_edges()}")

    # Draw the collaboration network
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(H, seed=42)
    nx.draw_networkx_nodes(H, pos, node_color='#3162C2', node_size=700)
    # Node labels: only company name
    nx.draw_networkx_labels(H, pos, font_size=10)
    # Draw edges without labels
    nx.draw_networkx_edges(H, pos, arrows=True, arrowstyle='-|>', edge_color='grey', width=2)
    plt.title('Company Collaboration Network')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'company_network_collaboration.png'))
    plt.close()

def visualize_network_html(G, save_folder):
    """Create an interactive HTML visualization of the collaboration network"""

    # create IS network
    H, num_collaborations = filter_colloaration_links(G)
    
    # Create interactive network
    net = Network(height='750px', width='100%', bgcolor='#ffffff')
    
    # Add nodes
    for node, data in H.nodes(data=True):
        # Create title with node information
        title = f"Company: {node}"
        if 'waste' in data:
            title += f"\nWaste: {', '.join(map(str, data['waste']))}"
        if 'resource' in data:
            title += f"\nResource: {', '.join(map(str, data['resource']))}"
        
        net.add_node(node, label=node, title=title, 
                    color='#3162C2', size=25)
    
    # Add edges
    for u, v, data in H.edges(data=True):
        # Create title with edge information
        title = f"Collaboration: {u} → {v}"
        if 'waste' in data:
            title += f"\nWaste: {data['waste']}"
        if 'resource' in data:
            title += f"\nResource: {data['resource']}"
        if 'process' in data:
            title += f"\nProcess: {data['process']}"
        if 'reference' in data:
            title += f"\nReference: {data['reference']}"
        
        net.add_edge(u, v, title=title, color='#666666', arrows='to')
    
    # Set physics options for better layout
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "forceAtlas2Based",
        "timestep": 0.35
      }
    }
    """)
    
    # Save as HTML file
    html_path = os.path.join(save_folder, 'company_network_interactive.html')
    net.save_graph(html_path)
    print(f"Interactive visualization saved to: {html_path}")

def save_exchanges(G, save_folder):
    # For each company, find all (company -> waste -> resource -> company) paths
    all_exchanges = []

    for company1 in [n for n, d in G.nodes(data=True) if d.get('type') == 'company']:
        # Out edges from company1 to waste
        for waste in G.successors(company1):
            if G.nodes[waste].get('type') != 'waste':
                continue
            # Out edges from waste to resource
            for resource in G.successors(waste):
                if G.nodes[resource].get('type') != 'resource':
                    continue
                # Out edges from resource to company2
                for company2 in G.successors(resource):
                    if G.nodes[company2].get('type') != 'company':
                        continue
                    # Print the exchange path
                    all_exchanges.append([company1, company2, waste, resource])

    all_exchanges_df = pd.DataFrame(all_exchanges, columns=['donor', 'receiver', 'waste', 'resource'])
    all_exchanges_df.to_csv(os.path.join(save_folder, 'company_network_exchanges.csv'), index=False)

    return all_exchanges_df

def compare_with_maestri(all_exchanges_df, prof_file, save_folder):
    prof_df = pd.read_csv(prof_file)
    # Process company names in prof_df to match format in all_exchanges_df
    prof_df['donor_name'] = prof_df.apply(lambda row: f"ND - {row['donor_business']}" if row['donor_name'] == "ND" else row['donor_name'], axis=1)
    prof_df['receiver_name'] = prof_df.apply(lambda row: f"ND - {row['receiver_business']}" if row['receiver_name'] == "ND" else row['receiver_name'], axis=1)
    
    # get all donors and receivers
    all_donors = all_exchanges_df['donor'].unique().tolist() + prof_df['donor_name'].unique().tolist()
    all_receivers = all_exchanges_df['receiver'].unique().tolist() + prof_df['receiver_name'].unique().tolist()
    all_donors = list(set(all_donors))
    all_receivers = list(set(all_receivers))
    all_donors.sort()
    all_receivers.sort()

    comparison_data = []

    for donor in all_donors:
        for receiver in all_receivers:
            companies = f"{donor} -> {receiver}"
            # maestri exchanges
            maestri_exchanges_df = prof_df[(prof_df['donor_name'] == donor) & (prof_df['receiver_name'] == receiver)]
            maestri_num_exchanges = maestri_exchanges_df.shape[0]
            maestri_exchanges_details = '\n'.join([f"{row['waste']} -> {row['resource']}" for _, row in maestri_exchanges_df.iterrows()])

            # our exchanges
            our_exchanges_df = all_exchanges_df[(all_exchanges_df['donor'] == donor) & (all_exchanges_df['receiver'] == receiver)]
            our_num_exchanges = our_exchanges_df.shape[0]
            our_exchanges_details = '\n'.join([f"{row['waste']} -> {row['resource']}" for _, row in our_exchanges_df.iterrows()])

            if maestri_num_exchanges > 0 or our_num_exchanges > 0:
                comparison_data.append([companies, maestri_num_exchanges, our_num_exchanges, maestri_exchanges_details, our_exchanges_details])

    comparison_df = pd.DataFrame(comparison_data, columns=['companies', 'maestri_num_exchanges', 'our_num_exchanges', 'maestri_exchanges_details', 'our_exchanges_details'])
    comparison_df.to_csv(os.path.join(save_folder, 'comparison_with_maestri.csv'), index=False)



def main(args):    
    G = create_network(args.kg_file, args.prof_file, threshold=args.threshold)    # W2RKG + company nodes
    all_exchanges_df = save_exchanges(G, args.save_folder)
    visualize_network(G, args.save_folder)    # connected waste/resource nodes + company nodes
    visualize_network_collaboration(G, args.save_folder)    # collaboration network (all company nodes)
    visualize_network_html(G, args.save_folder)    # interactive HTML visualization of the collaboration network

    compare_with_maestri(all_exchanges_df, args.prof_file, args.save_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_file', type=str, default='data_utils/fused_triples_aggregated.json')
    parser.add_argument('--prof_file', type=str, default='data_utils/Maestri_case1.csv')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--save_folder', type=str, default='case_study1')
    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    main(args)
