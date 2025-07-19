'''
For every exchange in company_network_exchanges.csv
Check:
1. whether its in other casees (in other cases)
2. whether its W2R is corrct (novel)'''

import pandas as pd
import tqdm
import os

base_folder = 'case_study1/threshold_0.75'

all_exchanges_df = pd.read_csv(os.path.join(base_folder, 'company_network_exchanges.csv'))
exchanges_in_both = pd.read_csv(os.path.join(base_folder, 'exchanges_in_both.csv'))
exchanges_only_ours = pd.read_excel(os.path.join(base_folder, "exchanges_only_in_our_analysis.xlsx"))
exchanges_correct_analysis = pd.read_excel(os.path.join(base_folder, "exchanges_only_in_our_analysis_label_0_analysis.xlsx"))

# print(all_exchanges_df.head(5))
# print(exchanges_in_both.head(5))
# print(exchanges_only_ours.head(5))
# print(exchanges_correct_analysis.head(5))

def check_in_this_case(waste, resource):
    for index, row in exchanges_in_both.iterrows():
        if row['waste'] == waste and row['resource'] == resource:
            return True
    return False


def check_in_other_cases(waste, resource):
    for index, row in exchanges_only_ours.iterrows():
        if row['waste'] == waste and row['resource'] == resource:
            in_other_cases = row["exist in other cases"]
            comment = row["comment"]
            if str(in_other_cases) == "1":
                return True, comment
    return False, None

def check_correctness(waste, resource):
    for index, row in exchanges_correct_analysis.iterrows():
        if row['waste'] == waste and row['resource'] == resource:
            is_correct = row["correct"]
            comment = row["comment"]
            if str(is_correct) == "1":
                return True, comment
    return False, None

all_results = []
for index, row in tqdm.tqdm(all_exchanges_df.iterrows(), total=len(all_exchanges_df)):
    donor_name = row['donor']
    receiver_name = row['receiver']
    P_wastes = row['P_wastes'].split(';')
    P_resources = row['P_resources'].split(';')

    for waste in P_wastes:
        for resource in P_resources:
            # # 1. check whether in case study
            # if check_in_this_case(waste, resource):
            #     all_results.append([donor_name, receiver_name, waste, resource, "in this case", ""])
            #     continue

            # 2. check whether in other cases
            in_other_cases, comment = check_in_other_cases(waste, resource)
            if in_other_cases:
                all_results.append([donor_name, receiver_name, waste, resource, "in other cases", comment])
                continue

            # 3. check whether is novel
            is_novel, comment = check_correctness(waste, resource)
            if is_novel:
                all_results.append([donor_name, receiver_name, waste, resource, "novel", comment])
                continue

# save to excel
all_results_df = pd.DataFrame(all_results, columns=['donor', 'receiver', 'waste', 'resource', 'label', 'comment'])
all_results_df.drop_duplicates(inplace=True)
all_results_df.to_excel(os.path.join(base_folder, 'supplement_table.xlsx'), index=False)
print("Length of all_results_df: ", len(all_results_df))

                

    


