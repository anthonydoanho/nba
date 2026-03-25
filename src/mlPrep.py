import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplcursors
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV

import pandasCleanup as pc

class MLPrep:
	def __init__(self, jsonFile):
		with open(jsonFile, "r") as f:
			inputs = json.load(f)

		self.trainSplit = inputs['trainSplit']
		self.testSplit = inputs['testSplit']

	def splits(self, dfList, target, trainSplit, evalSplit, testSplit):
		trainSplit = trainSplit / (trainSplit + evalSplit + testSplit)
		evalSplit = evalSplit / (trainSplit + evalSplit + testSplit)
		testSplit = testSplit / (trainSplit + evalSplit + testSplit)

		testSplitRelative = testSplit / (testSplit + evalSplit)

		X_trainList, X_evalList, X_testList = [], [], []
		y_trainList, y_evalList, y_testList = [], [], []
		for df in dfList:
			y = df[target]
			X = df.drop(target, axis=1)
			self.featureCorrelation(X)

			X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - trainSplit), random_state=42)
			X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=(testSplitRelative), random_state=42)
			X_trainList.append(X_train)
			X_evalList.append(X_eval)
			X_testList.append(X_test)
			y_trainList.append(y_train)
			y_evalList.append(y_eval)
			y_testList.append(y_test)

		import pdb; pdb.set_trace()

		return X_trainList, X_evalList, X_testList, y_trainList, y_evalList, y_testList

	def featureCorrelation(self, X):
		corrMatrix = X.corr()
		labels = corrMatrix.columns.tolist()
		n = len(labels)
		
		'''
		fig, ax = plt.subplots(figsize=(14,7))
		im = ax.imshow(corrMatrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
		plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

		ax.set_xticks(range(n))
		ax.set_yticks(range(n))
		ax.set_xticklabels(labels, rotation=90, fontsize=7)
		ax.set_yticklabels(labels, fontsize=7)
		
		ax.set_title("Combine Feature Correlation Matrix", fontsize=13, pad=12)
		fig.tight_layout()

		cursor = mplcursors.cursor(im, hover=True)

		@cursor.connect("add")
		def on_add(sel):
			j, i = int(round(sel.target[0])), int(round(sel.target[1]))
			sel.annotation.set_text(
				f"{labels[i]} vs {labels[j]}\nr = {corrMatrix[i, j]:.3f}"
			)
			sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

		def on_resize(event):
			fig.tight_layout()
			fig.canvas.draw_idle()

		fig.canvas.mpl_connect('resize_event', on_resize)
		'''
		sns.set_theme(font_scale=.8)
		axis_corr = sns.heatmap(
		corrMatrix,
		vmin=-1, vmax=1, center=0,
		xticklabels=1,
		cmap='RdYlGn'
		#square=True
		)
		#plt.xticks(rotation=45, ha='right')
		plt.tight_layout(pad=.25)

		plt.show()
		import pdb; pdb.set_trace()
