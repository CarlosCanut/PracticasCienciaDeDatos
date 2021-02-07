import pandas as pd
import pprint

worlds_data = pd.read_excel("./Worlds_2020_Main_Event.xlsx")
pprint.pprint(worlds_data.columns)

# agrupa los campeones de cada jugador
worlds_data["champ"] = worlds_data.groupby(['player_name'])['champ'].transform(lambda x: ','.join(x))
worlds_data = worlds_data.drop_duplicates()

# agrupa los matchups de cada jugador
worlds_data["matchup"] = worlds_data.groupby(['player_name'])['matchup'].transform(lambda x: ','.join(x))
worlds_data = worlds_data.drop_duplicates()

# obtiene medias de todas las stats y genera el df final
new_data = worlds_data.groupby(['player_name','team','role','champ','matchup']).agg({'kda':['mean'],'kp':['mean'],'G%':['mean'],'DMG%':['mean'],'CSD@10':['mean'],'GD@10':['mean'],'XPD@10':['mean'],'game_time':['mean'],'Result':['mean']}).reset_index()

print(new_data)
new_data.to_excel("players_stats.xlsx")



"""
players = { "player_name": pd.unique(worlds_data["player_name"]),
            "team": pd.unique(worlds_data[worlds_data["player_name"] == ])}

players = worlds_data.groupby(["player_name"]).mean()
new_df = pd.DataFrame(players)
"""


"""
# numeric data means
means = worlds_data.groupby(['player_name','team','role','champ','matchup']).agg({'kda':['mean'],'kp':['mean'],'G%':['mean'],'DMG%':['mean'],'CSD@10':['mean'],'GD@10':['mean'],'XPD@10':['mean'],'game_time':['mean'],'Result':['mean']}).reset_index()
print(means)"""