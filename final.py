import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from scipy.sparse import hstack

ClassifiertrainFileFeature = pd.read_csv('convert2.csv')
ClassifiertrainFileFeature = ClassifiertrainFileFeature.drop(ClassifiertrainFileFeature.columns[0], axis = 1)
ClassifiertrainFileFeature = ClassifiertrainFileFeature.drop(ClassifiertrainFileFeature.columns[1], axis = 1)
ClassifiertrainFileTruth = pd.read_csv('convert2.csv', usecols = [1])

RegressortrainFileFeature = pd.read_csv('revenue2.csv')
RegressortrainFileFeature = RegressortrainFileFeature.drop(RegressortrainFileFeature.columns[0], axis = 1)
RegressortrainFileFeature = RegressortrainFileFeature.drop(RegressortrainFileFeature.columns[1], axis = 1)
RegressortrainFileTruth = pd.read_csv('revenue2.csv', usecols = [1])

testFileFeature = pd.read_csv('df_holdout_scholarjet.csv')
testFileFeature = testFileFeature.drop(testFileFeature.columns[0], axis = 1)
testFileCuid = pd.read_csv('df_holdout_scholarjet.csv', usecols=[0])

ClassifiertrainFileFeature = ClassifiertrainFileFeature.fillna(-100)
ClassifiertrainFileTruth = ClassifiertrainFileTruth.fillna(-100)
RegressortrainFileFeature = RegressortrainFileFeature.fillna(-100)
RegressortrainFileTruth = RegressortrainFileTruth.fillna(-100)
testFileFeature = testFileFeature.fillna(-100)


lassoThing = linear_model.Lasso(max_iter=2000)
lassoThing = lassoThing.fit(RegressortrainFileFeature,RegressortrainFileTruth)

gbc = GradientBoostingClassifier()
gbc = gbc.fit(ClassifiertrainFileFeature,ClassifiertrainFileTruth)

lassoResult = lassoThing.predict(testFileFeature)
gbcResult = gbc.predict(testFileFeature)

outregress = np.asarray(lassoResult)
outclass = np.asarray(gbcResult)

