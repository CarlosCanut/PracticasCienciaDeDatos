import pandas as pd

data = pd.read_csv("2020_LoL_esports_match_data_from_OraclesElixir_20201201.csv")
"""print(data)
print("")
for x in data.columns:
    print(x)"""
    
print(pd.unique(data['league']))

secondary_leagues = ['KeSPA', 'LPL', 'LEC', 'LCS.A', 'LCS', 'LCK', 'Riot']

new_data = data[ data['league'].isin(secondary_leagues)]

print(pd.unique(new_data['league']))


