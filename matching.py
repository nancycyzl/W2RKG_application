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

from utils import save_list_to_text, read_list_from_text


def read_maestri_data(file_path):

    # Read the Excel file with header=[0, 1] to handle multi-level headers
    maestri_df = pd.read_excel(file_path, header=[0, 1, 2])

    # Extract two columns: waste description and final use (transformed resource) and rename them
    maestri_df.columns = ['_'.join(col).strip() for col in maestri_df.columns.values]
    w2r_df = maestri_df[['EXCHANGE IDENTIFICATION_Unnamed: 0_level_1_Exchange Identifier ("Case Identifier, Source identifier, Progressive number")',
                            'INVOLVED COMPANIES_Donor_Company name', 'INVOLVED COMPANIES_Donor_Main business',
                            'INVOLVED COMPANIES_Receiver_Company name', 'INVOLVED COMPANIES_Receiver_Main business',
                            'EXCHANGE DESCRIPTION_Exchange Input_Waste description', 
                            'EXCHANGE DESCRIPTION_Exchange details_Final use of the waste by the receiver company']].dropna()

    w2r_df.columns = ['exchange_id', 'donor_name', 'donor_business', 'receiver_name', 'receiver_business', 'waste', 'resource']
    
    # obtain the case_id from exchange_id (case_id, source_id, progressive_number)
    w2r_df['case_id'] = w2r_df['exchange_id'].str.split(',').str[0].astype(int)
    
    # Move case_id to the first column
    cols = ['case_id'] + [col for col in w2r_df.columns if col != 'case_id']
    w2r_df = w2r_df[cols]

    return w2r_df

def read_w2rkg_data(file_path):
    w2rkg_dict = json.load(open(file_path, 'r', encoding='utf-8'))
    return w2rkg_dict     # list of dicts


def create_embeddings_maestri(w2r_df, model, case_id=None):
    # filter case if provided
    if case_id is not None:
        w2r_df = w2r_df[w2r_df['case_id'] == case_id]

    waste_list = w2r_df['waste'].unique().tolist()
    resource_list = w2r_df['resource'].unique().tolist()

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

    return waste_list, resource_list, waste_embeddings_np, resource_embeddings_np


def create_embeddings_w2rkg(w2r_df, model):
    waste_list = []
    resource_list = []
    for w2r in w2r_df:
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

    return waste_list, resource_list, waste_embeddings_np, resource_embeddings_np


def embed_maestri_and_save(model, case_id=None):
    maestri_df = read_maestri_data('data_utils/Maestri.xlsx')
    print("Embedding Maestri...")
    maestri_waste_list, maestri_resource_list, maestri_waste_embeddings, maestri_resource_embeddings = create_embeddings_maestri(maestri_df, model, case_id=case_id)
    print(f"Total waste: {len(maestri_waste_list)}, total resource: {len(maestri_resource_list)}")

    return maestri_waste_list, maestri_resource_list, maestri_waste_embeddings, maestri_resource_embeddings


def embed_w2rkg_and_save(model):
    w2rkg_dict = read_w2rkg_data('data_utils/fused_triples_aggregated.json')
    print("Embedding W2RKG...")
    w2rkg_waste_list, w2rkg_resource_list, w2rkg_waste_embeddings, w2rkg_resource_embeddings = create_embeddings_w2rkg(w2rkg_dict, model)
    print(f"Total waste: {len(w2rkg_waste_list)}, total resource: {len(w2rkg_resource_list)}")

    return w2rkg_waste_list, w2rkg_resource_list, w2rkg_waste_embeddings, w2rkg_resource_embeddings


def load_list_and_embeddings(data_type, model):
    try:
        waste_list = read_list_from_text(f'app_data/{data_type}_waste_list.txt')
        resource_list = read_list_from_text(f'app_data/{data_type}_resource_list.txt')
        waste_embeddings = np.load(f'app_data/{data_type}_waste_embeddings.npy')
        resource_embeddings = np.load(f'app_data/{data_type}_resource_embeddings.npy')
        print(f"Loaded {data_type} list and embeddings successfully")

    except Exception as e:
        print(f"Error loading {data_type} list and embeddings: {e}")
        print(f"File not found for {data_type}, create from scratch")
        if data_type == 'maestri':
            waste_list, resource_list, waste_embeddings, resource_embeddings = embed_maestri_and_save(model, case_id=None)
        elif data_type == 'w2rkg':
            waste_list, resource_list, waste_embeddings, resource_embeddings = embed_w2rkg_and_save(model)

    return waste_list, resource_list, waste_embeddings, resource_embeddings



def main():
    model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

    # load name list and embeddings
    maestri_waste_list, maestri_resource_list, maestri_waste_embeddings, maestri_resource_embeddings = load_list_and_embeddings('maestri', model)
    w2rkg_waste_list, w2rkg_resource_list, w2rkg_waste_embeddings, w2rkg_resource_embeddings = load_list_and_embeddings('w2rkg', model)



if __name__ == '__main__':
    main()