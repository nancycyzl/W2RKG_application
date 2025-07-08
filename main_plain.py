import json
import io
import os
import networkx as nx
import pandas as pd
from pyvis.network import Network
import matplotlib.pyplot as plt
import argparse
import pyvis
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from matching import obtain_W2RKG_embeddings, obtain_profile_embeddings, build_W2R_Comp_network, filter_colloaration_links
from utils import load_kg_file, load_profiles_file, build_W2R_graph


def create_network(kg_file, prof_file, threshold=0.8, return_embeddings=False):
    
    # the basic W2R graph
    kg_triples = load_kg_file(kg_file)
    G = build_W2R_graph(kg_triples)    # a networkx graph for W2RKG
    print(f"W2R graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


    # add company
    profiles_df = load_profiles_file(prof_file)    # a dataframe

    # model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    model = "NA"

    G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings = obtain_W2RKG_embeddings(kg_triples, model)
    P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings = obtain_profile_embeddings(profiles_df, prof_file, model)

    G = build_W2R_Comp_network(G, profiles_df, threshold,
                            G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                            P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings)

    print(f"W2R graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


    graph_embeddings = [G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings]
    profile_embeddings = [P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings]

    if return_embeddings:
        return G, graph_embeddings, profile_embeddings
    else:
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

def match_G_material_to_P(provider_wastes, receiver_resources, waste, resource, graph_embeddings, profile_embeddings, threshold=0.8):
    # G waste embedding & P waste embedding list -> matched P waste list
    G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings = graph_embeddings
    P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings = profile_embeddings

    G_waste_embed = G_waste_embeddings[G_waste_list.index(waste)]
    P_waste_embeds = [P_waste_embeddings[P_waste_list.index(waste)] for waste in provider_wastes]
    G_waste_sims = cosine_similarity(G_waste_embed.reshape(1, -1), np.array(P_waste_embeds))[0]
    matched_prof_wastes = [provider_wastes[i] for i, sim in enumerate(G_waste_sims) if sim >= threshold]
    
    # G resource embedding -> P resource embedding list
    G_resource_embed = G_resource_embeddings[G_resource_list.index(resource)]
    P_resource_embeds = [P_resource_embeddings[P_resource_list.index(resource)] for resource in receiver_resources]
    G_resource_sims = cosine_similarity(G_resource_embed.reshape(1, -1), np.array(P_resource_embeds))[0]
    matched_prof_resources = [receiver_resources[i] for i, sim in enumerate(G_resource_sims) if sim >= threshold]

    return matched_prof_wastes, matched_prof_resources


def save_exchanges(G, graph_embeddings, profile_embeddings, threshold=0.8, save_folder=None):
    # For each company, find all (company -> waste -> resource -> company) paths
    all_exchanges = []
    G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings = graph_embeddings
    P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings = profile_embeddings

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
                    # get the exchange path
                    # all_exchanges.append([company1, company2, waste, resource])    # company -> graph waste node -> graph resource node -> company

                    # match the graph materials to the profile materials
                    provider_wastes = G.nodes[company1].get('waste', [])
                    receiver_resources = G.nodes[company2].get('resource', [])

                    matched_prof_wastes, matched_prof_resources = match_G_material_to_P(provider_wastes, receiver_resources, waste, resource, graph_embeddings, profile_embeddings, threshold)

                    all_exchanges.append([company1, company2, ";".join(matched_prof_wastes), ";".join(matched_prof_resources), waste, resource])


    all_exchanges_df = pd.DataFrame(all_exchanges, columns=['donor', 'receiver', 'P_wastes', 'P_resources', 'G_waste', 'G_resource'])

    if save_folder is not None:
        all_exchanges_df.to_csv(os.path.join(save_folder, 'company_network_exchanges.csv'), index=False)

    return all_exchanges_df

def compare_with_maestri(all_exchanges_df, case_file, save_folder):
    prof_df = pd.read_csv(case_file)
    # Process company names in prof_df to match format in all_exchanges_df
    prof_df['donor_name'] = prof_df.apply(lambda row: f"ND - {row['donor_business']}" if row['donor_name'] == "ND" else row['donor_name'], axis=1)
    prof_df['receiver_name'] = prof_df.apply(lambda row: f"ND - {row['receiver_business']}" if row['receiver_name'] == "ND" else row['receiver_name'], axis=1)

    # get all donors and receivers
    all_donors = all_exchanges_df['donor'].unique().tolist() + prof_df['donor_name'].unique().tolist()
    all_receivers = all_exchanges_df['receiver'].unique().tolist() + prof_df['receiver_name'].unique().tolist()
    all_companies = list(set(all_donors + all_receivers))
    all_companies.sort()

    # get donors' wastes and receivers' resources
    waste_dict = {}
    resource_dict = {}
    for company in all_companies:
        waste_dict[company] = []
        resource_dict[company] = []
        # get its wastes
        donor_df = prof_df[prof_df['donor_name'] == company]
        waste_dict[company] = donor_df['waste'].unique().tolist()
        # get its resources
        receiver_df = prof_df[prof_df['receiver_name'] == company]
        resource_dict[company] = receiver_df['resource'].unique().tolist()
    

    # save comparison table
    comparison_data = []

    for donor in all_companies:
        for receiver in all_companies:

            # wastes and resources
            wastes = '\n'.join(waste_dict[donor])
            resources = '\n'.join(resource_dict[receiver])

            # maestri exchanges
            maestri_exchanges_df = prof_df[(prof_df['donor_name'] == donor) & (prof_df['receiver_name'] == receiver)]
            maestri_num_exchanges = maestri_exchanges_df.shape[0]
            maestri_exchanges_details = '\n'.join([f"{row['waste']} -> {row['resource']}" for _, row in maestri_exchanges_df.iterrows()])

            # our exchanges
            our_exchanges_df = all_exchanges_df[(all_exchanges_df['donor'] == donor) & (all_exchanges_df['receiver'] == receiver)]
            our_num_exchanges = our_exchanges_df.shape[0]
            our_exchanges_details = '\n'.join([f"{row['G_waste']} -> {row['G_resource']}" for _, row in our_exchanges_df.iterrows()])
            
            # final identified exchanges
            final_exchanges = []
            for _, row in our_exchanges_df.iterrows():
                matched_P_wastes = row['P_wastes'].split(';')
                matched_P_resources = row['P_resources'].split(';')
                for matched_P_waste in matched_P_wastes:
                    for matched_P_resource in matched_P_resources:
                        final_exchanges.append(f"{matched_P_waste} -> {matched_P_resource}")
            final_exchanges = list(set(final_exchanges))
            final_exchanges_details = '\n'.join(final_exchanges)
            final_num_exchanges = len(final_exchanges)

            if maestri_num_exchanges > 0 or our_num_exchanges > 0:
                comparison_data.append([donor, receiver, wastes, resources, maestri_num_exchanges, our_num_exchanges, final_num_exchanges, maestri_exchanges_details, our_exchanges_details, final_exchanges_details])

    comparison_df = pd.DataFrame(comparison_data, columns=['donor', 'receiver', 'donor_wastes', 'receiver_resources', 'maestri_num_exchanges', 'our_num_exchanges', 'final_num_exchanges', 'maestri_exchanges_details', 'our_exchanges_details', 'final_exchanges_details'])
    comparison_df.to_csv(os.path.join(save_folder, 'comparison_with_maestri.csv'), index=False)
    print(f"Comparison table saved to: {os.path.join(save_folder, 'comparison_with_maestri.csv')}")

    # check how many maestri collaborations are in our identified ones
    # Count overlapping collaborations (where both Maestri and our approach identified exchanges)
    maestri_num_collaborations = len(comparison_df[comparison_df['maestri_num_exchanges'] >= 1])
    our_num_collaborations = len(comparison_df[comparison_df['our_num_exchanges'] >= 1])
    final_num_collaborations = len(comparison_df[comparison_df['final_num_exchanges'] >= 1])
    overlapping_collaborations = len(comparison_df[(comparison_df['maestri_num_exchanges'] >= 1) & (comparison_df['final_num_exchanges'] >= 1)])
    print(f"Number of collaborations identified by Maestri: {maestri_num_collaborations}")
    print(f"Number of collaborations identified by our approach (before matching): {our_num_collaborations}")
    print(f"Number of collaborations identified by our approach (after matching): {final_num_collaborations}")
    print(f"\nNumber of overlapping collaborations between Maestri and our approach: {overlapping_collaborations}")
    

    # --- Create networks for visualization ---
    # 1. Maestri: MultiDiGraph (multiple edges)
    G_maestri_multi = nx.MultiDiGraph()
    for _, row in prof_df.iterrows():
        G_maestri_multi.add_edge(row['donor_name'], row['receiver_name'])
    # 2. Maestri: DiGraph (single edge)
    G_maestri_single = nx.DiGraph()
    for _, row in prof_df.iterrows():
        G_maestri_single.add_edge(row['donor_name'], row['receiver_name'])

    # 3. Ours: MultiDiGraph (multiple edges)
    G_ours_multi = nx.MultiDiGraph()
    for _, row in comparison_df.iterrows():
        for _ in range(int(row["final_num_exchanges"])):
            G_ours_multi.add_edge(row['donor'], row['receiver'])
    # 4. Ours: DiGraph (single edge)
    G_ours_single = nx.DiGraph()
    for _, row in comparison_df.iterrows():
        for _ in range(int(row["final_num_exchanges"])):
            G_ours_single.add_edge(row['donor'], row['receiver'])

    # Add all companies as nodes to each graph (to show isolated nodes)
    for company in all_companies:
        G_maestri_multi.add_node(company)
        G_maestri_single.add_node(company)
        G_ours_multi.add_node(company)
        G_ours_single.add_node(company)

    # Fix node positions for all graphs using the union of all nodes
    # Use sorted list to ensure consistent node order across runs
    all_nodes = sorted(set(G_maestri_multi.nodes()) | set(G_maestri_single.nodes()) | set(G_ours_multi.nodes()) | set(G_ours_single.nodes()))
    G_union = nx.DiGraph()
    G_union.add_nodes_from(all_nodes)
    pos = nx.spring_layout(G_union, seed=15)

    # Plotting helper
    def plot_graph(G, pos, title, filename):
        # Print graph statistics
        print(f"\n{title}:")
        print(f"Number of nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos, node_color='#3162C2', node_size=700)
        nx.draw_networkx_labels(G, pos, font_size=9)
        # Draw all edges, including multiple edges for MultiDiGraph
        if isinstance(G, nx.MultiDiGraph):
            # Group edges by (u, v)
            from collections import defaultdict
            edge_dict = defaultdict(list)
            for u, v, k in G.edges(keys=True):
                edge_dict[(u, v)].append(k)
            # Draw each edge with a different curvature
            for (u, v), keys in edge_dict.items():
                n = len(keys)
                if n == 1:
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v, keys[0])], arrows=True, arrowstyle='-|>', edge_color='grey', width=2)
                else:
                    # Spread curvatures between -0.5 and 0.5
                    for idx, k in enumerate(keys):
                        rad = 0.5 * (2 * idx / (n - 1) - 1) if n > 1 else 0.0
                        nx.draw_networkx_edges(
                            G, pos, edgelist=[(u, v, k)],
                            arrows=True, arrowstyle='-|>', edge_color='grey', width=2,
                            connectionstyle=f'arc3,rad={rad}'
                        )
        else:
            nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', edge_color='grey', width=2)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, filename))
        plt.close()


    plot_graph(G_maestri_multi, pos, 'Maestri (MultiDiGraph)', f'comparison_maestri_multiedge.png')
    plot_graph(G_maestri_single, pos, 'Maestri (DiGraph)', f'comparison_maestri_singleedge.png')
    plot_graph(G_ours_multi, pos, 'Ours (MultiDiGraph)', f'comparison_ours_multiedge.png')
    plot_graph(G_ours_single, pos, 'Ours (DiGraph)', f'comparison_ours_singleedge.png')
    print(f"Network plots saved.")

def compare_multiple_thresholds(args, graph_embeddings, profile_embeddings):
    thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    num_exchanges_list = []
    for threshold in thresholds:
        print(f"Processing threshold: {threshold}...")
        G = create_network(args.kg_file, args.prof_file, threshold=threshold)    # W2RKG + company nodes
        all_exchanges_df = save_exchanges(G, graph_embeddings, profile_embeddings, threshold=threshold, save_folder=None)
        num_exchanges_list.append(all_exchanges_df.shape[0])
        print(f"Number of exchanges: {all_exchanges_df.shape[0]}")
    
    # create table and save to csv
    threshold_data = pd.DataFrame({
        'similarity_threshold': thresholds,
        'number_of_exchanges': num_exchanges_list
    })
    threshold_data.to_csv(os.path.join(args.save_folder, 'num_exchanges_vs_threshold.csv'), index=False)

    # plot and save
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, num_exchanges_list, 
            marker='o', linewidth=2, markersize=8)
    plt.xlabel('Similarity threshold')
    plt.ylabel('Number of exchanges')
    plt.title('Number of exchanges vs Similarity threshold')
    plt.grid(True)
    plt.savefig(os.path.join(args.save_folder, 'num_exchanges_vs_threshold.png'))
    plt.close()
        


def main(args):
    if args.test_one_case:    
        G, graph_embeddings, profile_embeddings = create_network(args.kg_file, args.prof_file, threshold=args.threshold, return_embeddings=True)    # W2RKG + company nodes
        all_exchanges_df = save_exchanges(G, graph_embeddings, profile_embeddings, threshold=args.threshold, save_folder=args.save_folder)

    if args.save_visualization:
        if not args.test_one_case:
            raise ValueError("Need to enable test_one_case for visualization.")
        visualize_network(G, args.save_folder)    # connected waste/resource nodes + company nodes
        visualize_network_collaboration(G, args.save_folder)    # collaboration network (all company nodes)
        visualize_network_html(G, args.save_folder)    # interactive HTML visualization of the collaboration network

    if args.compare_with_maestri:
        if not args.test_one_case:
            raise ValueError("Need to enable test_one_case for comparison with Maestri.")
        compare_with_maestri(all_exchanges_df, args.case_file, args.save_folder)

    if args.compare_multiple_thresholds:
        compare_multiple_thresholds(args, graph_embeddings, profile_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_file', type=str, default='data_utils/fused_triples_aggregated.json')
    parser.add_argument('--prof_file', type=str, default='data_utils/Maestri_profiles_case1.json')
    parser.add_argument('--case_file', type=str, default='data_utils/Maestri_case1.csv')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--save_folder', type=str, default='case_study1')

    parser.add_argument('--test_one_case', action='store_true')
    parser.add_argument('--save_visualization', action='store_true')
    parser.add_argument('--compare_with_maestri', action='store_true')
    parser.add_argument('--compare_multiple_thresholds', action='store_true')
    args = parser.parse_args()

    save_folder = os.path.join(args.save_folder, f'threshold_{args.threshold}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    args.save_folder = save_folder

    main(args)
