'''
Functionality:
1. match a user-query to Maestri using W2RKG
2. match among Maestri using W2RKG

Steps:
1. embed W2RKG waste and resource
2. embed user-query (a single waste / resource or Maestri wastes / resources)
3. match the user-query and W2RKG
'''


from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from utils import save_list_to_text, read_list_from_text, remove_isolated_nodes, convert_str_to_list


def embed_profile_and_save(profiles_dict, model, case_id):
    # obtain waste / resource names and create embeddings
    waste_list = []
    resource_list = []
    for company in profiles_dict:
        waste_list.extend(profiles_dict[company]['waste generation'])
        resource_list.extend(profiles_dict[company]['resource demand'])

    # remove duplicates
    waste_list = list(set(waste_list))
    resource_list = list(set(resource_list))

    if model is None:
        model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

    waste_embeddings = model.encode(waste_list, convert_to_tensor=True)
    waste_embeddings_np = waste_embeddings.cpu().numpy()

    resource_embeddings = model.encode(resource_list, convert_to_tensor=True)
    resource_embeddings_np = resource_embeddings.cpu().numpy()

    case_id_placeholder = f"_case{case_id}" if case_id is not None else ""

    # save list and embeddings
    save_list_to_text(waste_list, f'app_data/maestri_waste_list{case_id_placeholder}.txt')
    save_list_to_text(resource_list, f'app_data/maestri_resource_list{case_id_placeholder}.txt')
    np.save(f'app_data/maestri_waste_embeddings{case_id_placeholder}.npy', waste_embeddings_np)
    np.save(f'app_data/maestri_resource_embeddings{case_id_placeholder}.npy', resource_embeddings_np)

    print(f"Created profile embeddings for case {case_id} and saved to app_data/.")

    return waste_list, resource_list, waste_embeddings_np, resource_embeddings_np


def embed_w2rkg_and_save(w2rkg_dict, model):
    waste_list = []
    resource_list = []
    print(f"In embed_w2rkg_and_save, w2rkg_dict has {len(w2rkg_dict)} entries.")
    for w2r in w2rkg_dict:
        waste_list.append(w2r['waste'])
        resource_list.append(w2r['transformed_resource'])

    # remove duplicates
    waste_list = list(set(waste_list))
    resource_list = list(set(resource_list))

    if model is None:
        model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    
    # create embeddings
    waste_embeddings = model.encode(waste_list, convert_to_tensor=True, batch_size=2)
    waste_embeddings_np = waste_embeddings.cpu().numpy()

    resource_embeddings = model.encode(resource_list, convert_to_tensor=True, batch_size=2)
    resource_embeddings_np = resource_embeddings.cpu().numpy()

    # save list and embeddings
    save_list_to_text(waste_list, 'app_data/w2rkg_waste_list.txt')
    save_list_to_text(resource_list, 'app_data/w2rkg_resource_list.txt')
    np.save('app_data/w2rkg_waste_embeddings.npy', waste_embeddings_np)
    np.save('app_data/w2rkg_resource_embeddings.npy', resource_embeddings_np) 

    print(f"Created W2RKG embeddings and saved to app_data/.")

    return waste_list, resource_list, waste_embeddings_np, resource_embeddings_np


def obtain_profile_embeddings(profiles_dict, prof_file, model):
    # obtain case_id
    case_id = prof_file.basename().split('.')[0].split('case')[1]
    case_id_placeholder = f"_case{case_id}" if case_id is not None else ""

    try:
        waste_list = read_list_from_text(f'app_data/maestri_waste_list{case_id_placeholder}.txt')
        resource_list = read_list_from_text(f'app_data/maestri_resource_list{case_id_placeholder}.txt')
        waste_embeddings = np.load(f'app_data/maestri_waste_embeddings{case_id_placeholder}.npy')
        resource_embeddings = np.load(f'app_data/maestri_resource_embeddings{case_id_placeholder}.npy')
        print("Loaded profile embeddings successfully")
    except Exception as e:
        print(f"Error loading profile embeddings: {e}")
        print("Creating profile embeddings from scratch...")
        waste_list, resource_list, waste_embeddings, resource_embeddings = embed_profile_and_save(profiles_dict, model, case_id)
     
    return waste_list, resource_list, waste_embeddings, resource_embeddings


def obtain_W2RKG_embeddings(kg_triples, model):
    try:
        waste_list = read_list_from_text('app_data/w2rkg_waste_list.txt')
        resource_list = read_list_from_text('app_data/w2rkg_resource_list.txt')
        waste_embeddings = np.load('app_data/w2rkg_waste_embeddings.npy')
        resource_embeddings = np.load('app_data/w2rkg_resource_embeddings.npy')
        print("Loaded W2RKG embeddings successfully")
    except Exception as e:
        print(f"Error loading W2RKG embeddings: {e}")
        print("Creating W2RKG embeddings from scratch")
        waste_list, resource_list, waste_embeddings, resource_embeddings = embed_w2rkg_and_save(kg_triples, model)

    return waste_list, resource_list, waste_embeddings, resource_embeddings


def match_material_to_G(material_embedding, G_material_list, G_material_embeddings, similarity_threshold):
    '''
    1. get the embedding of the material (from profile)
    2. calculate cosine similarity between the material and the W2RKG
    3. return the matched material list (similarity score > threshold)
    '''
    similarity_scores = cosine_similarity(material_embedding.reshape(1, -1), G_material_embeddings)
    matched_indices = [i for i in range(len(G_material_list)) if similarity_scores[0][i] >= similarity_threshold]
    
    return matched_indices


def build_W2R_Comp_network(G, profiles_dict, similarity_threshold,
                           G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                           P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings):
    '''
    Input: W2RKG + company profiles
    Output: W2RKG + company nodes connected
    '''
    # Add company nodes and their connections to wastes/resources

    for company in profiles_dict:
        waste_list = profiles_dict[company]['waste generation']
        resource_list = profiles_dict[company]['resource demand']

        # check if company node exists
        if G.has_node(company):
            # If 'waste' already exists, append; else, create a new list
            if 'waste' in G.nodes[company]:
                if isinstance(G.nodes[company]['waste'], list):
                    G.nodes[company]['waste'].extend(waste_list)
                else:
                    G.nodes[company]['waste'] = [G.nodes[company]['waste'], waste_list]
            else:
                G.nodes[company]['waste'] = waste_list

            # If 'resource' already exists, append; else, create a new list
            if 'resource' in G.nodes[company]:
                if isinstance(G.nodes[company]['resource'], list):
                    G.nodes[company]['resource'].extend(resource_list)
                else:
                    G.nodes[company]['resource'] = [G.nodes[company]['resource'], resource_list]
            else:
                G.nodes[company]['resource'] = resource_list

        else:
            G.add_node(company, type='company', waste=waste_list, resource=resource_list)

        # match company's waste
        for waste in waste_list:
            waste_embedding = P_waste_embeddings[P_waste_list.index(waste)]
            matched_G_waste_ids = match_material_to_G(waste_embedding, G_waste_list, G_waste_embeddings, similarity_threshold)
            for matched_G_waste_id in matched_G_waste_ids:
                G.add_edge(company, G_waste_list[matched_G_waste_id], type='generates')
        
        # match company's resource
        for resource in resource_list:
            resource_embedding = P_resource_embeddings[P_resource_list.index(resource)]
            matched_G_resource_ids = match_material_to_G(resource_embedding, G_resource_list, G_resource_embeddings, similarity_threshold)
            for matched_G_resource_id in matched_G_resource_ids:
                G.add_edge(G_resource_list[matched_G_resource_id], company, type='supplies')

    return G

def filter_colloaration_links(G, center_company=None):
    '''
    Input: W2RKG + company nodes connected
    Output: IS network

    If center_company is provided, only include collaborations that involve this company.

    Each company node in H will have its 'waste' and 'resource' properties (if any) from G, accumulating all relevant values.
    Each edge in H will have 'process' and 'reference' properties from the corresponding W2RKG path in G.  
    '''
    H = nx.MultiDiGraph()
    num_collaborations = 0
    # Find all company->waste->resource->company paths
    for company1 in [n for n, d in G.nodes(data=True) if d.get('type') == 'company']:
        for waste in G.successors(company1):
            if G.nodes[waste].get('type') != 'waste':
                continue
            for resource in G.successors(waste):
                if G.nodes[resource].get('type') != 'resource':
                    continue
                for company2 in G.successors(resource):
                    if G.nodes[company2].get('type') != 'company':
                        continue

                    company_1_business = G.nodes[company1].get('business', 'n.a.')
                    company_2_business = G.nodes[company2].get('business', 'n.a.')
                    # Get process/reference from the first matching edge (handles MultiDiGraph and DiGraph)
                    process = None
                    reference = None
                    if G.is_multigraph():
                        for key, edge_data in G[waste][resource].items():
                            process = edge_data.get('process', None)
                            reference = edge_data.get('reference', None)
                            break
                    else:
                        edge_data = G[waste][resource]
                        process = edge_data.get('process', None)
                        reference = edge_data.get('reference', None)
                    # Add/append waste to company1
                    if H.has_node(company1):
                        wastes = H.nodes[company1].get('waste', [])
                        if waste not in wastes:
                            wastes.append(waste)
                        H.nodes[company1]['waste'] = wastes
                    else:
                        H.add_node(company1, type='company', waste=[waste], business=company_1_business)
                    # Add/append resource to company2
                    if H.has_node(company2):
                        resources = H.nodes[company2].get('resource', [])
                        if resource not in resources:
                            resources.append(resource)
                        H.nodes[company2]['resource'] = resources
                    else:
                        H.add_node(company2, type='company', resource=[resource], business=company_2_business)
                    if center_company is None or company1 == center_company or company2 == center_company:
                        H.add_edge(company1, company2, waste=waste, resource=resource, process=process, reference=reference)
                    #num_collaborations += 1

    H = remove_isolated_nodes(H)
    num_collaborations = H.number_of_edges()

    return H, num_collaborations


def build_IS_network(G, profiles_dict, similarity_threshold,
                     G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                     P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings):
    '''
    Input: W2RKG network + profiles
    Output: IS network
    
    Steps:
    1. build W2RKG + company nodes connected
    2. find all company->waste->resource->company paths to create the IS network
    '''
    G = G.copy()

    # obtain W2RKG + company nodes connected
    G = build_W2R_Comp_network(G, profiles_dict, similarity_threshold,
                               G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                               P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings)

    # create IS network
    H, num_collaborations = filter_colloaration_links(G)

    return H, num_collaborations


def obtain_query_embedding(query, model):
    # first check if query exists in query_lookup
    lookup_file = f'app_data/query_lookup.txt'
    embedding_file = f'app_data/query_embedding_lookup.npy'
    if os.path.exists(lookup_file) and os.path.exists(embedding_file):
        query_lookup = read_list_from_text(lookup_file)
        embedding_lookup = np.load(embedding_file)

        if query in query_lookup:
            # load embedding
            query_embedding = embedding_lookup[query_lookup.index(query)]
            print(f"Loaded query embedding for {query} from query_lookup.txt")
            return query_embedding
    else:
        # initialize
        query_lookup = []
        embedding_lookup = np.empty((0 , 0))

    # create embedding
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().numpy()    # shape (1024)
    query_embedding_np = query_embedding_np[np.newaxis, :] # shape (1, 1024)
    
    # update query lookup and embedding lookup
    query_lookup.append(query)
    if embedding_lookup.size == 0:
        embedding_lookup = query_embedding_np
    else:
        embedding_lookup = np.vstack([embedding_lookup, query_embedding_np])

    # save query lookup and embedding lookup
    if len(query_lookup) == embedding_lookup.shape[0]:
        save_list_to_text(query_lookup, lookup_file)
        np.save(embedding_file, embedding_lookup)

    return query_embedding_np


def match_query_company_to_G(G, company_id, waste_query, resource_query, similarity_threshold, model,
                     G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings):
    '''
    Input: W2RKG network + company nodes connected
    Output: W2RKG + company nodes connected + query company node
    '''
    waste_query_list = convert_str_to_list(waste_query)
    resource_query_list = convert_str_to_list(resource_query)

    G.add_node(company_id, type='company', waste=waste_query_list, resource=resource_query_list)
    for one_waste_query in waste_query_list:
        waste_query_embedding = obtain_query_embedding(one_waste_query, model)
        matched_G_waste_ids = match_material_to_G(waste_query_embedding, G_waste_list, G_waste_embeddings, similarity_threshold)
        for matched_G_waste_id in matched_G_waste_ids:
            G.add_edge(company_id, G_waste_list[matched_G_waste_id], type='generates')

    for one_resource_query in resource_query_list:
        resource_query_embedding = obtain_query_embedding(one_resource_query, model)
        matched_G_resource_ids = match_material_to_G(resource_query_embedding, G_resource_list, G_resource_embeddings, similarity_threshold)
        for matched_G_resource_id in matched_G_resource_ids:
            G.add_edge(G_resource_list[matched_G_resource_id], company_id, type='supplies')

    return G


def build_partner_linkages(G, profiles_dict, company_id, waste_query, resource_query, similarity_threshold, model,
                           G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                           P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings):
    '''
    Input: W2RKG network + profiles
    Output: query company + collaborators
    '''
    G = G.copy()

    # obtain W2RKG + company nodes connected
    G = build_W2R_Comp_network(G, profiles_dict, similarity_threshold,
                               G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                               P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings)
    
    # add query company node
    G = match_query_company_to_G(G, company_id, waste_query, resource_query, similarity_threshold, model,
                                 G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings)
    
    # filter collaborations
    H, num_collaborations = filter_colloaration_links(G, center_company=company_id)

    return H, num_collaborations

    