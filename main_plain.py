import json
import io
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from matching import obtain_W2RKG_embeddings, obtain_profile_embeddings, build_W2R_Comp_network, filter_colloaration_links
from utils import load_kg_file, load_profiles_file, build_W2R_graph


def create_network(kg_file, prof_file, threshold=0.8):
    
    # the basic W2R graph
    kg_triples = load_kg_file(kg_file)
    G = build_W2R_graph(kg_triples)    # a networkx graph for W2RKG
    print(f"W2R graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


    # add company
    profiles_df = load_profiles_file(prof_file)    # a dataframe

    model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

    G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings = obtain_W2RKG_embeddings(kg_triples, model)
    P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings = obtain_profile_embeddings(profiles_df, model)

    G = build_W2R_Comp_network(G, profiles_df, threshold,
                            G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                            P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings)

    print(f"W2R graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    return G

def visualize_network(G):
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
    pos = nx.spring_layout(H, seed=42)
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
    plt.savefig('company_network.png')
    plt.close()

def visualize_network_collaboration(G):
    # create IS network
    H, num_collaborations = filter_colloaration_links(G)
    print(f"Total collaborations: {num_collaborations}")
    print(f"Total nodes: {H.number_of_nodes()}")
    print(f"Total edges: {H.number_of_edges()}")

    # Draw the collaboration network
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(H, seed=42)
    nx.draw_networkx_nodes(H, pos, node_color='#3162C2', node_size=700)
    # Node labels: company name + waste/resource
    node_labels = {}
    for n, d in H.nodes(data=True):
        label = f"{n}\n"
        if 'waste' in d:
            label += f"waste: {', '.join(map(str, d['waste']))}\n"
        if 'resource' in d:
            label += f"resource: {', '.join(map(str, d['resource']))}"
        node_labels[n] = label.strip()
    nx.draw_networkx_labels(H, pos, labels=node_labels, font_size=7)
    # Edge labels: process + reference
    edge_labels = {}
    for u, v, d in H.edges(data=True):
        label = ''
        if d.get('process'):
            label += f"process: {d['process']}\n"
        if d.get('reference'):
            label += f"ref: {d['reference']}"
        edge_labels[(u, v)] = label.strip()
    nx.draw_networkx_edges(H, pos, arrows=True, arrowstyle='-|>', edge_color='green', width=2)
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=7)
    plt.title('Company Collaboration Network')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('company_network_collaboration.png')
    plt.close()

def save_exchanges(G):
    # For each company, find all (company -> waste -> resource -> company) paths
    all_exchanges = []
    print("---------------Exchanges---------------")
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
                    print(f"[{company1}] --generates--> [{waste}] --transformed_into--> [{resource}] --needed_by--> [{company2}]")
    print("---------------------------------------")
    
    all_exchanges_df = pd.DataFrame(all_exchanges, columns=['donor', 'receiver', 'waste', 'resource'])
    all_exchanges_df.to_csv('company_network_exchanges.csv', index=False)

if __name__ == "__main__":
    kg_file = 'data_utils/fused_triples_aggregated.json'
    prof_file = 'data_utils/Maestri_case1.csv'
    G = create_network(kg_file, prof_file, threshold=0.8)
    save_exchanges(G)
    visualize_network(G)
    visualize_network_collaboration(G)

