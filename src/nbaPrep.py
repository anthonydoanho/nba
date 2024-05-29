import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import time

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
		self.dropCols = inputs['dropCols']
		self.testTrainSplit = inputs['testTrainSplit']
		self.xgbParams = inputs['xgbParamsGridSearch'] 
	
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
			print('Pulling data for ' + str(year) + ' draft' )
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
		#corrMatrix = X.corr()
		#axis_corr = sns.heatmap(
		#corrMatrix,
		#vmin=-1, vmax=1, center=0,
		#cmap=sns.diverging_palette(50, 500, n=500),
		#square=True
		#)

		#plt.show()
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testTrainSplit, random_state=42)

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

	def train(self, X_train, y_train):
		model = xgb.XGBRegressor()
		reg_cv = GridSearchCV(model, {
			'colsample_bytree':self.xgbParams['params']['colsample_bytree'], 
			'max_depth':self.xgbParams['params']['max_depth'],
			'min_child_weight':self.xgbParams['params']['min_child_weight'], 
			'n_estimators':self.xgbParams['params']['n_estimators'],
			})
		reg_cv.fit(X_train, y_train)
		import pdb; pdb.set_trace()

if __name__ == '__main__':
	jsonFile = 'src/inputs.json'
	draft = NBAPrep(jsonFile)
	
	measurements, spotShooting, nonStationaryShooting = draft.combine()
	players = draft.players()
	df = draft.merging(players, measurements, spotShooting, nonStationaryShooting)
	df = draft.drop(df, draft.dropCols)
	
	X_train, X_test, y_train, y_test = draft.splits(df, draft.target, draft.testTrainSplit)
	# draft.featureImportance(X_train, y_train)

	draft.train(X_train, y_train)
