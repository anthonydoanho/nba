import json
import pandas as pd

from nba_api.stats.endpoints import draftcombinespotshooting 
from nba_api.stats.endpoints import draftcombinenonstationaryshooting 
from nba_api.stats.endpoints import draftcombinestats 
from nba_api.stats.endpoints import draftcombineplayeranthro 
from nba_api.stats.endpoints import leagueleaders

with open("src/inputs.json", "r") as f:
    inputs = json.load(f)

draft = inputs['draftYear']
years = inputs['seasons']

players = pd.DataFrame()
for year in years:
	stats = leagueleaders.LeagueLeaders(
		season=year,
		season_type_all_star='Regular Season',
		stat_category_abbreviation='MIN'
		,per_mode48='Totals'
	).get_data_frames()[0]
	players = pd.concat((players, stats)) 
playersSum = players.groupby(by='PLAYER_ID', as_index=False).sum()
	
measurements = draftcombinestats.DraftCombineStats(season_all_time=draft).get_data_frames()[0]
spotShooting = draftcombinespotshooting.DraftCombineSpotShooting(season_year=draft).get_data_frames()[0]
nonStationaryShooting = draftcombinenonstationaryshooting.DraftCombineNonStationaryShooting(season_year=draft).get_data_frames()[0]

spotShootingTrunc = spotShooting[['PLAYER_ID'] + inputs['spotShootingCols']]
spotShootingTrunc = spotShootingTrunc[spotShootingTrunc[inputs['spotShootingCols']].any(axis=1)].sort_values(by='PLAYER_ID', ascending=False)

nonStationaryShootingTrunc = nonStationaryShooting[['PLAYER_ID'] + inputs['nonStationaryShootingCols']]
nonStationaryShootingTrunc = nonStationaryShootingTrunc[nonStationaryShootingTrunc[inputs['nonStationaryShootingCols']].any(axis=1)].sort_values(by='PLAYER_ID', ascending=False)

draftPlayers = measurements[['PLAYER_ID'] + inputs['measurementCols']].sort_values(by='PLAYER_ID', ascending=False)

df = pd.merge(nonStationaryShootingTrunc, spotShootingTrunc, on='PLAYER_ID', how='outer')
df = pd.merge(draftPlayers, df, on='PLAYER_ID', how='inner')

df = pd.merge(df, playersSum[['PLAYER_ID', inputs['target']]], on='PLAYER_ID', how='inner').sort_values(by='MIN', ascending=False)

import pdb; pdb.set_trace()
