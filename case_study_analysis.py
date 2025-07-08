import pandas as pd
import argparse
import os

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


def main(args):
    comparison_df = pd.read_csv(os.path.join(args.base_folder, "comparison_with_maestri.csv"))
    maestri_exchange_list, our_exchange_list = parse_exchanges(comparison_df)

    print("Number of exchanges in Maestri: ", comparison_df["maestri_num_exchanges"].sum())
    print("Number of exchanges in our approach: ", comparison_df["final_num_exchanges"].sum())

    save_exchanges_three_types(maestri_exchange_list, our_exchange_list, save_folder=args.base_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, default="case_study1/threshold_0.75")
    args = parser.parse_args()
    main(args)





