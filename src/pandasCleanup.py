import numpy as np
import pandas as pd

class pandasCleanup:
	def __init__(self, df):
		self.df = df

	def cleanBlank(self, col):
		# replaces '' with None
		self.df[col] = self.df[col].replace('', None)

	def consistentNull(self):
		# makes null values consistent with column dtypes
		for col in self.df.columns:
			if self.df[col].dtype == 'object':
				self.df[col] = self.df[col].replace(np.nan, None)
			# elif self.df[col].dtype in ['int64', 'float64']:
			# 	self.df[col] = self.df[col].replace(None, np.nan)

	def newType(self, col, newType):
		self.df[col] = self.df[col].astype(newType)
	
	def main(self):
		self.cleanBlank('WEIGHT')
		self.newType('WEIGHT', 'float64')
		self.consistentNull()
		return self.df

