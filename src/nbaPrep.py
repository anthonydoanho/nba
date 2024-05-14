import json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import time

from nba_api.stats.endpoints import draftcombinespotshooting 
from nba_api.stats.endpoints import draftcombinenonstationaryshooting 
from nba_api.stats.endpoints import draftcombinestats 
from nba_api.stats.endpoints import draftcombineplayeranthro 
from nba_api.stats.endpoints import leagueleaders

class NBAPrep:
	def __init__(self, target, draftYear, seasons, measurementCols, spotShootingCols, nonStationaryShootingCols, dropCols, testTrainSplit):
		self.target = target
		self.draft = draftYear
		self.years = seasons
		self.measurementCols = measurementCols
		self.spotShootingCols = spotShootingCols
		self.nonStationaryShootingCols = nonStationaryShootingCols
		self.dropCols = dropCols
		self.testTrainSplit = testTrainSplit
	
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

	def drop(self, df, dropCols):
		# drop cols are selected from previous feature importance analyses
		df = df.drop(dropCols, axis=1)

		return df

	def splits(self, df, target, testTrainSplit):
		y = df[target]
		X = df.drop(target, axis=1)
		X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=testTrainSplit, random_state=42)

		return X_train, X_test, y_train, y_test

	def featureImportance(self, X_train, y_train):

		xgb_regressor = xgb.XGBRegressor(random_state=42)
		xgb_regressor.fit(X_train, y_train)

		xgb_feature_importances = xgb_regressor.feature_importances_
		xgb_feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': xgb_feature_importances})
		xgb_feature_importance_df = xgb_feature_importance_df.sort_values(by='Importance', ascending=False)
		print("XGBoost Feature Importance:")
		print(xgb_feature_importance_df)

		#lgb_regressor = lgb.LGBMRegressor(random_state=42)
		#lgb_regressor.fit(X_train, y_train)

		#lgb_feature_importances = lgb_regressor.feature_importances_
		#lgb_feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': lgb_feature_importances})
		#lgb_feature_importance_df = lgb_feature_importance_df.sort_values(by='Importance', ascending=False)
		#print("\nLightGBM Feature Importance:")
		#print(lgb_feature_importance_df)
		#import pdb; pdb.set_trace()

if __name__ == '__main__':
	with open('src/inputs.json', "r") as f:
		inputs = json.load(f)

	draft = NBAPrep(inputs['target'], inputs['draftYear'], inputs['seasons'], inputs['measurementCols']
		, inputs['spotShootingCols'], inputs['nonStationaryShootingCols'], inputs['dropCols'], inputs['testTrainSplit'])
	measurements, spotShooting, nonStationaryShooting = draft.combine()
	players = draft.players()
	df = draft.merging(players, measurements, spotShooting, nonStationaryShooting)
	df = draft.drop(df, draft.dropCols)
	
	X_train, y_train, X_test, y_test = draft.splits(df, draft.target, draft.testTrainSplit)
	import pdb; pdb.set_trace()
	draft.featureImportance(X_train, y_train)

	draft = NBAPrep(inputs['target'], inputs['draftYearTest'], inputs['seasonsTest'], inputs['measurementCols']
		, inputs['spotShootingCols'], inputs['nonStationaryShootingCols'], inputs['dropCols'])
	players = draft.players()
	measurements, spotShooting, nonStationaryShooting = draft.combine()
	df = draft.merging(players, measurements, spotShooting, nonStationaryShooting)
	df = draft.drop(df, draft.dropCols)

	y_test = df['MIN']
	X_test = df.drop(['PLAYER_ID', 'MIN', 'FIRST_NAME', 'LAST_NAME'], axis=1)
	
	model = LinearRegression(alpha=1e-2).fit(X_train, y_train)
	model.score(X_test, y_test)
