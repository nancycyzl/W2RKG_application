import json
import io
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from matching import obtain_W2RKG_embeddings, obtain_profile_embeddings, build_W2R_Comp_network
from utils import load_kg_file, load_profiles_file, build_W2R_graph


def create_network():
    
    # the basic W2R graph
    kg_file_default = 'data_utils/fused_triples_aggregated.json'
    kg_triples = load_kg_file(kg_file_default)
    G = build_W2R_graph(kg_triples)    # a networkx graph for W2RKG
    print(f"W2R graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


    # add company
    prof_file_default = 'data_utils/Maestri_case1.csv'
    profiles_df = load_profiles_file(prof_file_default)    # a dataframe

    model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

    G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings = obtain_W2RKG_embeddings(kg_triples, model)
    P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings = obtain_profile_embeddings(profiles_df, model)

    G = build_W2R_Comp_network(G, profiles_df, 0.9,
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

def print_exchanges(G):
    # For each company, find all (company -> waste -> resource -> company) paths
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
                    print(f"[{company1}] --generates--> [{waste}] --transformed_into--> [{resource}] --needed_by--> [{company2}]")
    print("---------------------------------------")

if __name__ == "__main__":
    G = create_network()
    print_exchanges(G)
    visualize_network(G)

