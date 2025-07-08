import pandas as pd

df = pd.read_csv('case_study1/threshold_0.9/company_network_exchanges.csv')

print(df.columns)

print(df["P_resources"])