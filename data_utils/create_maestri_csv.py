import pandas as pd

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

    # if resource = "Raw material", then use "Raw material for {business}"
    # Replace "Raw material" with "Raw material for {business}"
    w2r_df.loc[w2r_df['resource'] == 'Raw material', 'resource'] = w2r_df.loc[w2r_df['resource'] == 'Raw material'].apply(
        lambda row: f"Raw material for {row['receiver_business']}", axis=1)
    
    # obtain the case_id from exchange_id (case_id, source_id, progressive_number)
    w2r_df['case_id'] = w2r_df['exchange_id'].str.split(',').str[0].astype(int)
    
    # Move case_id to the first column
    cols = ['case_id'] + [col for col in w2r_df.columns if col != 'case_id']
    w2r_df = w2r_df[cols]

    return w2r_df



def main():

    maestri_df = read_maestri_data('data_utils/Maestri.xlsx')
    print(maestri_df.head())
    maestri_df.to_csv('data_utils/Maestri_all.csv', index=False)

    # Create separate CSV files for each case
    unique_cases = maestri_df['case_id'].unique()
    for case in unique_cases:
        case_df = maestri_df[maestri_df['case_id'] == case]
        if case_df.shape[0] >= 10:
            print(f"Case {case} has {case_df.shape[0]} rows")
            output_file = f'data_utils/Maestri_case{case}.csv'
            case_df.to_csv(output_file, index=False)
            print(f"Saved case {case} data to {output_file}")


if __name__ == '__main__':
    main()