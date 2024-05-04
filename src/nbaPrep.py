import json
import pandas as pd
from sklearn.linear_model import LinearRegression
import time

from nba_api.stats.endpoints import draftcombinespotshooting 
from nba_api.stats.endpoints import draftcombinenonstationaryshooting 
from nba_api.stats.endpoints import draftcombinestats 
from nba_api.stats.endpoints import draftcombineplayeranthro 
from nba_api.stats.endpoints import leagueleaders

class NBAPrep:
	def __init__(self, target, draftYear, seasons, measurementCols, spotShootingCols, nonStationaryShootingCols):
		self.target = target
		self.draft = draftYear
		self.years = seasons
		self.measurementCols = measurementCols
		self.spotShootingCols = spotShootingCols
		self.nonStationaryShootingCols = nonStationaryShootingCols
	
	def players(self):
		playersSum = pd.DataFrame()
		for i, draftClass in enumerate(self.years):
			print('Pulling data for the following years: ' + str(draftClass))
			players = pd.DataFrame()
			for year in draftClass:
				time.sleep(0.6)
				stats = leagueleaders.LeagueLeaders(
					season = year,
					season_type_all_star='Regular Season',
					stat_category_abbreviation='MIN'
					,per_mode48='Totals'
				).get_data_frames()[0]
				players = pd.concat((players, stats)) 
			players = players.groupby(by='PLAYER_ID', as_index=False).sum()
			players['DRAFT_CLASS'] = self.draft[i]
			playersSum = pd.concat((playersSum, players))
		
		return playersSum

	def combine(self):
		measurements = pd.DataFrame()
		spotShooting = pd.DataFrame()
		nonStationaryShooting = pd.DataFrame()
		for year in self.draft: 
			print('Pulling data for draft: ' + str(year))
			m = draftcombinestats.DraftCombineStats(season_all_time = year).get_data_frames()[0]
			m['DRAFT_CLASS'] = year
			time.sleep(0.6)
			s = draftcombinespotshooting.DraftCombineSpotShooting(season_year = year).get_data_frames()[0]
			time.sleep(0.6)
			n = draftcombinenonstationaryshooting.DraftCombineNonStationaryShooting(season_year = self.draft).get_data_frames()[0]
			time.sleep(0.6)
			measurements = pd.concat((measurements, m))
			spotShooting = pd.concat((spotShooting, s))
			nonStationaryShooting = pd.concat((nonStationaryShooting, n))

		measurements = measurements[measurements['WEIGHT'] != ''] # weight is not blank
		measurements['WEIGHT'] = measurements['WEIGHT'].astype(float)
		draftPlayers = measurements[['PLAYER_ID', 'DRAFT_CLASS'] + self.measurementCols].sort_values(by='PLAYER_ID', ascending=False)
		
		spotShootingTrunc = spotShooting[['PLAYER_ID'] + self.spotShootingCols]
		spotShootingTrunc = spotShootingTrunc[spotShootingTrunc[self.spotShootingCols].any(axis=1)].sort_values(by='PLAYER_ID', ascending=False)

		nonStationaryShootingTrunc = nonStationaryShooting[['PLAYER_ID'] + self.nonStationaryShootingCols]
		nonStationaryShootingTrunc = nonStationaryShootingTrunc[nonStationaryShootingTrunc[self.nonStationaryShootingCols].any(axis=1)].sort_values(by='PLAYER_ID', ascending=False)
		
		return draftPlayers, spotShootingTrunc, nonStationaryShootingTrunc

	def merging(self, playersSum, draftPlayers, spotShootingTrunc, nonStationaryShootingTrunc):
		df = pd.merge(nonStationaryShootingTrunc, spotShootingTrunc, on='PLAYER_ID', how='outer')
		df = pd.merge(draftPlayers, df, on='PLAYER_ID', how='outer')
		df.loc[df['PLAYER_ID']==2006, 'PLAYER_ID'] = 1626204 # Correcting Larry Nance's PLAYER_ID
		df = pd.merge(df, playersSum[['PLAYER_ID', 'DRAFT_CLASS', self.target]], on=['PLAYER_ID', 'DRAFT_CLASS'], how='left').sort_values(by= self.target, ascending=False)

		df['MIN'] = df['MIN'].fillna(0)

		df = df.dropna(how='all', axis=1)

		return df

if __name__ == '__main__':
	with open('src/inputs.json', "r") as f:
		inputs = json.load(f)

	draft = NBAPrep(inputs['target'], inputs['draftYear'], inputs['seasons'], inputs['measurementCols'], inputs['spotShootingCols'], inputs['nonStationaryShootingCols'])
	players = draft.players()
	measurements, spotShooting, nonStationaryShooting = draft.combine()
	df = draft.merging(players, measurements, spotShooting, nonStationaryShooting)
	
	y_train = df['MIN']
	X_train = df.drop(['PLAYER_ID', 'MIN', 'FIRST_NAME', 'LAST_NAME'], axis=1)
	
	draft = NBAPrep(inputs['target'], inputs['draftYearTest'], inputs['seasonsTest'], inputs['measurementCols'], inputs['spotShootingCols'], inputs['nonStationaryShootingCols'])
	players = draft.players()
	measurements, spotShooting, nonStationaryShooting = draft.combine()
	df = draft.merging(players, measurements, spotShooting, nonStationaryShooting)
	
	y_test = df['MIN']
	X_test = df.drop(['PLAYER_ID', 'MIN', 'FIRST_NAME', 'LAST_NAME'], axis=1)
	
	model = LinearRegression(alpha=1e-2).fit(X_train, y_train)
	model.score(X_test, y_test)
