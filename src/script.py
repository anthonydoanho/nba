import argparse
import pandas as pd

from nbaPrep import NBAPrep

import pandasCleanup as pc
from mlPrep import MLPrep

def main(args):
	jsonFile = args.inputFile
	draft = NBAPrep(jsonFile)
	ml_prep = MLPrep(jsonFile)
	
	if args.saveFile: 
		measurements, spotShooting, nonStationaryShooting = draft.combine()

		y = pc.pandasCleanup(measurements, draft.measurementColsFix)
		measurements = y.main()
		
		players = draft.players()
		df = draft.merging(players, measurements, spotShooting, nonStationaryShooting)
		dfPlayers = df[draft.dfPlayers]
		df = draft.dropColumns(df, draft.dropCols)
		pc.saveFile.to_csv(df, args.saveFile)
	
	if args.loadFile:
		df = pd.read_csv(args.loadFile)	
	
	dfList = draft.positions(df)

	#featureCorrelation 
	if args.correlation:
		#MLPrep.featureCorrelation(df.loc[:, df.columns != 'MIN'])
		ml_prep.featureCorrelation(df.loc[:, df.columns != 'MIN'])
	# featureCorrelation = 
	X_trainList, X_evalList, X_testList, y_trainList, y_evalSplit, y_testList = MLPrep.splits(dfList, draft.target, draft.trainSplit, draft.evalSplit, draft.testSplit)
	# draft.featureImportance(X_train, y_train)

	draft.train(dfPlayers, X_trainList, y_trainList, X_testList, y_testList)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a model to predict NBA success from combine statistics',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-c', '--correlation', action='store_true', help='View feature correlation matrix')
	parser.add_argument('-i', '--inputFile', help='Path to input JSON file', required=True)
	
	source_group = parser.add_mutually_exclusive_group(required=True)
	source_group.add_argument('-l', '--loadFile', help='Path to a local datafram csv to load')
	source_group.add_argument('-s', '--saveFile', help='Fetch data from NBA API and save to this path')

	args = parser.parse_args()
	
	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_rows', None)

	main(args)
