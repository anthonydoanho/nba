import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV

import pandasCleanup as pc

from nba_api.stats.endpoints import draftcombinespotshooting 
from nba_api.stats.endpoints import draftcombinenonstationaryshooting 
from nba_api.stats.endpoints import draftcombinestats 
from nba_api.stats.endpoints import draftcombineplayeranthro 
from nba_api.stats.endpoints import leagueleaders

class NBAPrep:
	def __init__(self, jsonFile):
		with open(jsonFile, "r") as f:
			inputs = json.load(f)

		self.draft = inputs['draftYear']
		self.dropCols = inputs['dropCols']
		self.joinCols = inputs['joinCols']
		self.measurementCols = inputs['measurementCols']
		self.measurementColsFix = inputs['measurementColsFix']
		self.nonStationaryShootingCols = inputs['nonStationaryShootingCols']
		self.spotShootingCols = inputs['spotShootingCols']
		self.target = inputs['target']
		self.testTrainSplit = inputs['testTrainSplit']
		self.xgbParams = inputs['xgbParamsGridSearch'] 
		self.years = inputs['seasons']
	
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
			players = players.groupby(by=['PLAYER_ID', 'PLAYER'], as_index=False).sum()
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
			s['DRAFT_CLASS'] = year
			time.sleep(0.6)
			n = draftcombinenonstationaryshooting.DraftCombineNonStationaryShooting(season_year = year).get_data_frames()[0]
			n['DRAFT_CLASS'] = year
			# time.sleep(0.6)
			measurements = pd.concat((measurements, m))
			spotShooting = pd.concat((spotShooting, s))
			nonStationaryShooting = pd.concat((nonStationaryShooting, n))

		draftPlayers = measurements[self.joinCols + self.measurementCols].sort_values(by=self.joinCols, ascending=False)
		
		spotShootingTrunc = spotShooting[self.joinCols + self.spotShootingCols]
		spotShootingTrunc = spotShootingTrunc[spotShootingTrunc[self.spotShootingCols].any(axis=1)].sort_values(by=self.joinCols, ascending=False)

		nonStationaryShootingTrunc = nonStationaryShooting[self.joinCols + self.nonStationaryShootingCols]
		nonStationaryShootingTrunc = nonStationaryShootingTrunc[nonStationaryShootingTrunc[self.nonStationaryShootingCols].any(axis=1)].sort_values(by=self.joinCols, ascending=False)
		
		return draftPlayers, spotShootingTrunc, nonStationaryShootingTrunc

	def merging(self, playersSum, draftPlayers, spotShootingTrunc, nonStationaryShootingTrunc):
		df = pd.merge(nonStationaryShootingTrunc, spotShootingTrunc, on=self.joinCols, how='outer')
		df = pd.merge(draftPlayers, df, on=self.joinCols, how='outer')
		df.loc[df['PLAYER_ID']==2006, 'PLAYER_ID'] = 1626204 # Correcting Larry Nance's PLAYER_ID
		df = pd.merge(df, playersSum[self.joinCols + [self.target]], on=self.joinCols, how='left').sort_values(by= self.target, ascending=False)

		df['MIN'] = df['MIN'].fillna(0)

		df = df.dropna(how='all', axis=1)

		return df

	def drop(self, df, dropCols):
		# drop cols are selected from previous feature importance analyses
		df = df.drop(dropCols, axis=1)

		return df

	def positions(self, df):
		# split players based on 'HEIGHT_W_SHOES'
		
		df_1 = df[df['HEIGHT_W_SHOES'] < 78]
		df_2 = df[(df['HEIGHT_W_SHOES'] >= 78) & (df['HEIGHT_W_SHOES'] < 82)]
		df_3 = df[df['HEIGHT_W_SHOES'] >= 82]

		dfList = [df_1, df_2, df_3]

		return dfList

	def splits(self, dfList, target, testTrainSplit):
		X_trainList, X_testList = [], []
		y_trainList, y_testList = [], []
		for df in dfList:
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
			X_trainList.append(X_train)
			X_testList.append(X_test)
			y_trainList.append(y_train)
			y_testList.append(y_test)

		return X_trainList, X_testList, y_trainList, y_testList

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

	def train(self, dfPlayers, X_trainList, y_trainList, X_testList, y_testList):
		model = xgb.XGBRegressor()
		for i, position  in enumerate(['Guards', 'Forwards', 'Centers']):
			print('Performing grid search on ' + position)
			reg_cv = GridSearchCV(model, {
				'eta':self.xgbParams['params' + position]['eta'],
				'colsample_bytree':self.xgbParams['params' + position]['colsample_bytree'], 
				'gamma':self.xgbParams['params' + position]['gamma'],
				'max_depth':self.xgbParams['params' + position]['max_depth'],
				'min_child_weight':self.xgbParams['params' + position]['min_child_weight'], 
				'n_estimators':self.xgbParams['params' + position]['n_estimators'],
				'nthread':self.xgbParams['params' + position]['nthread'],
				'objective':self.xgbParams['params' + position]['objective'],
				'reg_alpha':self.xgbParams['params' + position]['reg_alpha'],
				'reg_lambda':self.xgbParams['params' + position]['reg_lambda'],
				'scale_pos_weight':self.xgbParams['params' + position]['scale_pos_weight'],
				'subsample':self.xgbParams['params' + position]['subsample'],
				'seed':self.xgbParams['params' + position]['seed'],
				})
			print("Training gridsearch")
			reg_cv.fit(X_trainList[i], y_trainList[i])
			print(reg_cv.best_params_)
			model = xgb.XGBRegressor(**reg_cv.best_params_)
			model.fit(X_trainList[i], y_trainList[i])
			predictions = model.predict(X_testList[i])
			print("Predicting " + position)
			mse = mean_squared_error(y_testList[i], predictions)
			mae = mean_absolute_error(y_testList[i], predictions)
			print('mse: ' + str(mse) + ', mae: ' + str(mae))
			test = pd.DataFrame({'y_test':y_testList[i], 'predictions':predictions})
			merged = dfPlayers.merge(test, how='inner', left_index=True, right_index=True)
			merged['absDiff'] = abs(merged['y_test'] - merged['predictions'])
			merged = merged.sort_values(by='predictions', ascending=False)

if __name__ == '__main__':
	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_rows', None)
	jsonFile = 'src/inputs.json'
	draft = NBAPrep(jsonFile)
	
	measurements, spotShooting, nonStationaryShooting = draft.combine()

	y = pc.pandasCleanup(measurements, draft.measurementColsFix)
	measurements = y.main()
	
	players = draft.players()
	df = draft.merging(players, measurements, spotShooting, nonStationaryShooting)
	dfPlayers = df[['PLAYER_ID', 'FIRST_NAME', 'LAST_NAME']]
	df = draft.drop(df, draft.dropCols)
	
	dfList = draft.positions(df)

	X_trainList, X_testList, y_trainList, y_testList = draft.splits(dfList, draft.target, draft.testTrainSplit)
	# draft.featureImportance(X_train, y_train)

	draft.train(dfPlayers, X_trainList, y_trainList, X_testList, y_testList)
