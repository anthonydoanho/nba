import json
import pandas as pd

from nba_api.stats.endpoints import draftcombinespotshooting 
from nba_api.stats.endpoints import draftcombinenonstationaryshooting 
from nba_api.stats.endpoints import draftcombinestats 
from nba_api.stats.endpoints import draftcombineplayeranthro 
from nba_api.stats.endpoints import leagueleaders

class NBAPrep:
	def __init__(self, jsonFile):
		with open(jsonFile, "r") as f:
			inputs = json.load(f)

		self.target = inputs['target']
		self.draft = inputs['draftYear']
		self.years = inputs['seasons']
		self.measurementCols = inputs['measurementCols']
		self.spotShootingCols = inputs['spotShootingCols']
		self.nonStationaryShootingCols = inputs['nonStationaryShootingCols']
	
	def players(self):
		players = pd.DataFrame()
		for year in self.years:
			stats = leagueleaders.LeagueLeaders(
				season=year,
				season_type_all_star='Regular Season',
				stat_category_abbreviation='MIN'
				,per_mode48='Totals'
			).get_data_frames()[0]
			players = pd.concat((players, stats)) 
		playersSum = players.groupby(by='PLAYER_ID', as_index=False).sum()
		
		return playersSum

	def measurements(self):
		measurements = draftcombinestats.DraftCombineStats(season_all_time = self.draft).get_data_frames()[0]
		draftPlayers = measurements[['PLAYER_ID'] + self.measurementCols].sort_values(by='PLAYER_ID', ascending=False)
		
		return draftPlayers
	
	def spotShooting(self):
		spotShooting = draftcombinespotshooting.DraftCombineSpotShooting(season_year = self.draft).get_data_frames()[0]
		spotShootingTrunc = spotShooting[['PLAYER_ID'] + self.spotShootingCols]
		spotShootingTrunc = spotShootingTrunc[spotShootingTrunc[self.spotShootingCols].any(axis=1)].sort_values(by='PLAYER_ID', ascending=False)

		return spotShootingTrunc

	def nonStationaryShooting(self):
		nonStationaryShooting = draftcombinenonstationaryshooting.DraftCombineNonStationaryShooting(season_year = self.draft).get_data_frames()[0]
		nonStationaryShootingTrunc = nonStationaryShooting[['PLAYER_ID'] + self.nonStationaryShootingCols]
		nonStationaryShootingTrunc = nonStationaryShootingTrunc[nonStationaryShootingTrunc[self.nonStationaryShootingCols].any(axis=1)].sort_values(by='PLAYER_ID', ascending=False)

		return nonStationaryShootingTrunc

	def merging(self, playersSum, draftPlayers, spotShootingTrunc, nonStationaryShootingTrunc):
		df = pd.merge(nonStationaryShootingTrunc, spotShootingTrunc, on='PLAYER_ID', how='outer')
		df = pd.merge(draftPlayers, df, on='PLAYER_ID', how='inner')
		df = pd.merge(df, playersSum[['PLAYER_ID', self.target]], on='PLAYER_ID', how='inner').sort_values(by= self.target, ascending=False)

		return df

if __name__ == '__main__':
	draft = NBAPrep('src/inputs.json')
	players = draft.players()
	measurements = draft.measurements()
	spotShooting = draft.spotShooting()
	nonStationaryShooting = draft.nonStationaryShooting()
	df = draft.merging(players, measurements, spotShooting, nonStationaryShooting)

	import pdb; pdb.set_trace()
