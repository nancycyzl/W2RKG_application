'''
1. Obtain the exchanges only in Maestri, only in our approach, and in both
2. For those only in our approach, check whether they appear in other Maestri cases + ISDATA (use script to faciliate checking)
   Saved results in exchanges_only_in_our_analysis.xlsx
3. Calculate statistics: how many exist in other Maestri cases + ISDATA (unique waste-resource pairs)
4. for those only in our approach and do not exist in other cases, check how many are correct
   4.1 Manaully check and save results in exchanges_only_in_our_analysis_label_0_analysis.xlsx
   4.2 Use script to check the results
'''

import pandas as pd
import argparse
import os

from data_utils.create_maestri_csv import read_maestri_data
from rapidfuzz import fuzz
import re

def parse_exchanges(comparison_df):
    '''
    Exchanges is list of string
    Return: a list of [[waste, resource], [waste, resource], ...] (one element is exchanges for one pair of companies)
    '''
    all_maestri_exchanges = []
    all_our_exchanges = []

    for _, row in comparison_df.iterrows():
        maestri_exchanges = row["maestri_exchanges_details"]   # a string of {waste} -> {resource} split by \n
        our_exchanges = row["final_exchanges_details"]   # a string of {waste} -> {resource} split by \n

        maestri_exchanges_one_pair_of_company = []
        if not pd.isna(maestri_exchanges):
            maestri_exchanges = maestri_exchanges.split("\n")
            for exchange in maestri_exchanges:
                exchange_W2R = exchange.split(" -> ")
                maestri_exchanges_one_pair_of_company.append(exchange_W2R)
        all_maestri_exchanges.append(maestri_exchanges_one_pair_of_company)

        our_exchanges_one_pair_of_company = []
        if not pd.isna(our_exchanges):
            our_exchanges = our_exchanges.split("\n")
            for exchange in our_exchanges:
                exchange_W2R = exchange.split(" -> ")
                our_exchanges_one_pair_of_company.append(exchange_W2R)
        all_our_exchanges.append(our_exchanges_one_pair_of_company)

    return all_maestri_exchanges, all_our_exchanges


def save_exchanges_three_types(maestri_exchange_list, our_exchange_list, save_folder):
    exchanges_only_in_maestri = []
    exchanges_only_in_our = []
    exchanges_in_both = []

    num_collaboration_maestri = 0
    num_collaboration_our = 0
    num_collaboration_both = 0

    for maestri_exchanges, our_exchanges in zip(maestri_exchange_list, our_exchange_list):
        if len(maestri_exchanges) > 0:
            num_collaboration_maestri += 1
        if len(our_exchanges) > 0:
            num_collaboration_our += 1
        if len(maestri_exchanges) > 0 and len(our_exchanges) > 0:
            num_collaboration_both += 1

        maestri_set = set(tuple(x) for x in maestri_exchanges)
        our_set = set(tuple(x) for x in our_exchanges)
        exchanges_only_in_maestri.extend(list(maestri_set - our_set))
        exchanges_only_in_our.extend(list(our_set - maestri_set))
        exchanges_in_both.extend(list(maestri_set & our_set))

    # Save to files
    pd.DataFrame(exchanges_only_in_maestri, columns=["waste", "resource"]).to_csv(f"{save_folder}/exchanges_only_in_maestri.csv", index=False)
    pd.DataFrame(exchanges_only_in_our, columns=["waste", "resource"]).to_csv(f"{save_folder}/exchanges_only_in_our.csv", index=False)
    pd.DataFrame(exchanges_in_both, columns=["waste", "resource"]).to_csv(f"{save_folder}/exchanges_in_both.csv", index=False)

    print(f"Number of exchanges only in Maestri: {len(exchanges_only_in_maestri)}")
    print(f"Number of exchanges only in our approach: {len(exchanges_only_in_our)}")
    print(f"Number of exchanges in both: {len(exchanges_in_both)}")
    
    print(f"Number of collaborations in Maestri: {num_collaboration_maestri}")
    print(f"Number of collaborations in our approach: {num_collaboration_our}")
    print(f"Number of collaborations in both: {num_collaboration_both}")


def obtain_exchanges_in_Maestri(maestri_filepath, exclude_case=None):
    maestri_df = read_maestri_data(maestri_filepath)
    if exclude_case is not None:
        maestri_df = maestri_df[maestri_df["case_id"] != exclude_case]

    exchanges_in_Maestri = []
    for _, row in maestri_df.iterrows():
        exchange_id = row["exchange_id"]
        waste = row["waste"]
        resource = row["resource"]
        exchanges_in_Maestri.append([exchange_id, waste, resource])
    
    return exchanges_in_Maestri

def obtain_exchanges_in_ISDATA(isdata_filepath):
    isdata_df = pd.read_excel(isdata_filepath, header=[0,1])
    isdata_df.columns = ['_'.join(col).strip() for col in isdata_df.columns.values]

    isdata_w2r_df = isdata_df[["Exchange Identifier_(xxxx.x)", "By-product/Waste (CN)_Common name(s)", "Secondary Materials (CN)_Common Name 1, Common name 2"]]
    isdata_w2r_df.columns = ["exchange_id", "waste", "resource"]

    exchanges_in_ISDATA = []
    for _, row in isdata_w2r_df.iterrows():
        if not pd.isna(row["waste"]) and not pd.isna(row["resource"]):
            exchange_id = row["exchange_id"]
            waste = row["waste"]
            resource = row["resource"]
            exchanges_in_ISDATA.append([exchange_id, waste, resource])
    
    return exchanges_in_ISDATA

def preprocess_material_name(s):
    # Lowercase, remove punctuation, and normalize whitespace
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_material_similarity(str1, str2):
    # If str1 is empty, return 1
    if not str1:
        return 1.0
    # If str1 is in str2 or str2 is in str1, return 1
    if str1 in str2 or str2 in str1:
        return 1.0
    s1 = preprocess_material_name(str1)
    s2 = preprocess_material_name(str2)
    return fuzz.token_set_ratio(s1, s2) / 100.0

def main(args):
    if args.stage == "1":
        ##### 1. Obtain the exchanges only in Maestri, only in our approach, and in both
        comparison_df = pd.read_csv(os.path.join(args.base_folder, "comparison_with_maestri.csv"))
        maestri_exchange_list, our_exchange_list = parse_exchanges(comparison_df)

        print("Number of exchanges in Maestri: ", comparison_df["maestri_num_exchanges"].sum())
        print("Number of exchanges in our approach: ", comparison_df["final_num_exchanges"].sum())

        save_exchanges_three_types(maestri_exchange_list, our_exchange_list, save_folder=args.base_folder)

    elif args.stage == "2":
        ##### 2. For those only in our approach, check whether they appear in other Maestri cases + ISDATA
        maestri_other_exchanges = obtain_exchanges_in_Maestri("data_utils/Maestri.xlsx", exclude_case=1)
        isdata_exchanges = obtain_exchanges_in_ISDATA("data_utils/ISDATA.xlsx")
        total_ref_exchanges = maestri_other_exchanges + isdata_exchanges
        print("Number of exchanges in Maestri: ", len(maestri_other_exchanges))
        print("Number of exchanges in ISDATA: ", len(isdata_exchanges))
        print("Number of total reference exchanges: ", len(total_ref_exchanges))

        contune_query = True
        while contune_query:
            waste = input("Enter the waste (q to quit): ")
            resource = input("Enter the resource (q to quit): ")
            if waste == "q" or resource == "q":
                break
            
            # filter references
            for exchange_id, ref_waste, ref_resource in total_ref_exchanges:
                if (compute_material_similarity(waste, ref_waste) > args.material_similarity_threshold and
                    compute_material_similarity(resource, ref_resource) > args.material_similarity_threshold):
                    print(f"Exchange ID {exchange_id}: {ref_waste} -> {ref_resource}")

    elif args.stage == "3":
        ##### 3. Calculate statistics: how many exist in other Maestri cases + ISDATA (unique waste-resource pairs)
        exchanges_only_in_our_df = pd.read_excel(os.path.join(args.base_folder, "exchanges_only_in_our_analysis.xlsx"))

        # Find rows with same waste-resource combination but different labels
        duplicates = exchanges_only_in_our_df.groupby(['waste', 'resource'])['exist in other cases'].nunique()
        inconsistent_pairs = duplicates[duplicates > 1].index
        
        if len(inconsistent_pairs) > 0:
            print("\nFound inconsistent labels for these waste-resource pairs:")
            for waste, resource in inconsistent_pairs:
                rows = exchanges_only_in_our_df[
                    (exchanges_only_in_our_df['waste'] == waste) & 
                    (exchanges_only_in_our_df['resource'] == resource)
                ]
                print(f"\nWaste: {waste}")
                print(f"Resource: {resource}") 
                print("Labels:", rows['exist in other cases'].tolist())
        else:
            print("\nAll labels are consistent for all waste-resource pairs")

        # Count unique waste-resource combinations
        unique_combinations = exchanges_only_in_our_df[['waste', 'resource']].drop_duplicates()
        print(f"Number of unique waste-resource combinations: {len(unique_combinations)}")
        print(f"Number of total waste-resource pairs: {len(exchanges_only_in_our_df)}")

        # Count unique waste-resource combinations with label = 1
        unique_combinations_with_label_1 = exchanges_only_in_our_df[exchanges_only_in_our_df['exist in other cases'] == 1][['waste', 'resource']].drop_duplicates()
        print(f"Number of unique waste-resource combinations with label = 1: {len(unique_combinations_with_label_1)}")
        print(f"Number of waste-resource pairs with label = 1: {len(exchanges_only_in_our_df[exchanges_only_in_our_df['exist in other cases'] == 1])}")

        # Count unique waste-resource combinations with label = 0
        unique_combinations_with_label_0 = exchanges_only_in_our_df[exchanges_only_in_our_df['exist in other cases'] == 0][['waste', 'resource']].drop_duplicates()
        print(f"Number of unique waste-resource combinations with label = 0: {len(unique_combinations_with_label_0)}")
        print(f"Number of waste-resource pairs with label = 0: {len(exchanges_only_in_our_df[exchanges_only_in_our_df['exist in other cases'] == 0])}")

        # Count unique waste-resource combinations with label = /
        unique_combinations_with_label_unknown = exchanges_only_in_our_df[exchanges_only_in_our_df['exist in other cases'] == "/"][['waste', 'resource']].drop_duplicates()
        print(f"Number of unique waste-resource combinations with label = /: {len(unique_combinations_with_label_unknown)}")
        label_unknown = exchanges_only_in_our_df[exchanges_only_in_our_df['exist in other cases'] == '/']
        print(f"Number of waste-resource pairs with label = /: {len(label_unknown)}")

        # save the unique waste-resource pairs with label = 0
        unique_combinations_with_label_0.to_csv(os.path.join(args.base_folder, "exchanges_only_in_our_analysis_label_0.csv"), index=False)

    elif args.stage == "4":
        ##### 4. for those only in our approach and do not exist in other cases, check how many are correct
        df = pd.read_excel(os.path.join(args.base_folder, "exchanges_only_in_our_analysis_label_0_analysis.xlsx"))
        print("Number of correct waste-to-resource pairs: ", df[df["correct"] == 1].shape[0])
        print("Number of incorrect waste-to-resource pairs: ", df[df["correct"] == 0].shape[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, default="case_study1/threshold_0.75")
    parser.add_argument("--stage", type=str, default="4", choices=["1", "2", "3", "4"])
    parser.add_argument("--material_similarity_threshold", type=float, default=0.7)
    args = parser.parse_args()
    main(args)





