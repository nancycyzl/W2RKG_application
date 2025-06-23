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
from sklearn.metrics.pairwise import cosine_similarity

from utils import save_list_to_text, read_list_from_text


def embed_profile_and_save(profiles_df, model):
    # obtain case_id
    case_id_all = profiles_df['case_id'].unique().tolist()
    case_id = case_id_all[0] if len(case_id_all) == 1 else None

    # obtain waste / resource names and create embeddings
    waste_list = profiles_df['waste'].unique().tolist()
    resource_list = profiles_df['resource'].unique().tolist()

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
    
    # create embeddings
    waste_embeddings = model.encode(waste_list, convert_to_tensor=True)
    waste_embeddings_np = waste_embeddings.cpu().numpy()

    resource_embeddings = model.encode(resource_list, convert_to_tensor=True)
    resource_embeddings_np = resource_embeddings.cpu().numpy()

    # save list and embeddings
    save_list_to_text(waste_list, 'app_data/w2rkg_waste_list.txt')
    save_list_to_text(resource_list, 'app_data/w2rkg_resource_list.txt')
    np.save('app_data/w2rkg_waste_embeddings.npy', waste_embeddings_np)
    np.save('app_data/w2rkg_resource_embeddings.npy', resource_embeddings_np) 

    print(f"Created W2RKG embeddings and saved to app_data/.")

    return waste_list, resource_list, waste_embeddings_np, resource_embeddings_np


def obtain_profile_embeddings(profiles_df, model):
    # obtain case_id
    case_id_all = profiles_df['case_id'].unique().tolist()
    case_id = case_id_all[0] if len(case_id_all) == 1 else None
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
        waste_list, resource_list, waste_embeddings, resource_embeddings = embed_profile_and_save(profiles_df, model)
     
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


def match_material_to_G(material, P_material_list, P_material_embeddings, G_material_list, G_material_embeddings, similarity_threshold):
    '''
    1. get the embedding of the material (from profile)
    2. calculate cosine similarity between the material and the W2RKG
    3. return the matched material list (similarity score > threshold)
    '''
    material_embedding = P_material_embeddings[P_material_list.index(material)]
    similarity_scores = cosine_similarity(material_embedding.reshape(1, -1), G_material_embeddings)
    matched_indices = [i for i in range(len(G_material_list)) if similarity_scores[0][i] >= similarity_threshold]
    
    return matched_indices


def build_W2R_Comp_network(G, profiles_df, similarity_threshold,
                           G_waste_list, G_resource_list, G_waste_embeddings, G_resource_embeddings,
                           P_waste_list, P_resource_list, P_waste_embeddings, P_resource_embeddings):
    # Add company nodes and their connections to wastes/resources
    for _, row in profiles_df.iterrows():
        donor_name = row['donor_name']
        donor_business = row['donor_business'] 
        receiver_name = row['receiver_name']
        receiver_business = row['receiver_business']
        waste = row['waste']
        resource = row['resource']

        if donor_name == "ND":
            donor_name = f"ND - {donor_business}"
        if receiver_name == "ND":
            receiver_name = f"ND - {receiver_business}"

        # add company node and its properties
        if G.has_node(donor_name):
            # If 'waste' already exists, append; else, create a new list
            if 'waste' in G.nodes[donor_name]:
                if isinstance(G.nodes[donor_name]['waste'], list):
                    G.nodes[donor_name]['waste'].append(waste)
                else:
                    G.nodes[donor_name]['waste'] = [G.nodes[donor_name]['waste'], waste]
            else:
                G.nodes[donor_name]['waste'] = [waste]
        else:
            G.add_node(donor_name, type='company', business=donor_business, waste=[waste])

        if G.has_node(receiver_name):
            # If 'resource' already exists, append; else, create a new list
            if 'resource' in G.nodes[receiver_name]:
                if isinstance(G.nodes[receiver_name]['resource'], list):
                    G.nodes[receiver_name]['resource'].append(resource)
                else:
                    G.nodes[receiver_name]['resource'] = [G.nodes[receiver_name]['resource'], resource]
            else:
                G.nodes[receiver_name]['resource'] = [resource]
        else:
            G.add_node(receiver_name, type='company', business=receiver_business, resource=[resource])


        # match company's waste
        matched_G_waste_ids = match_material_to_G(waste, P_waste_list, P_waste_embeddings, G_waste_list, G_waste_embeddings, similarity_threshold)
        for matched_G_waste_id in matched_G_waste_ids:
            G.add_edge(donor_name, G_waste_list[matched_G_waste_id], type='generates')

        # match company's resource
        matched_G_resource_ids = match_material_to_G(resource, P_resource_list, P_resource_embeddings, G_resource_list, G_resource_embeddings, similarity_threshold)
        for matched_G_resource_id in matched_G_resource_ids:
            G.add_edge(G_resource_list[matched_G_resource_id], receiver_name, type='supplies')

    return G

