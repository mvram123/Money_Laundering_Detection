import numpy as np
import math
from numpy import genfromtxt

import pandas
import pickle
import time
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("Reading Filtered Data 3 ...")

dataframeX = pandas.read_csv('data_processed/filtered_data_3.csv')
col_names = list(dataframeX.columns.values)
dataMat = dataframeX.to_numpy()

print("Read Dataset")
print("Creating X and Y Variables...")

X = dataMat[:,1:-2]
Y = dataMat[:,-1]

print(f"Shape of X: {X.shape} and Shape of Y: {Y.shape}")

print("Instiantiating Model...")
model = RandomForestClassifier(random_state=42)
print("Fitting Data...")
model.fit(X, Y.astype(int))
print("Checking Feature Importances...")
feature_imp = model.feature_importances_

sorted_feature_vals = np.sort(feature_imp)
sorted_feature_indexes = np.argsort(feature_imp)

print("Significant Features in decreasing order of importance: ")
print("Storing Feature Importances in reports...")
fea_imp = [[col_names[i+2], feature_imp[i]] for i in reversed(sorted_feature_indexes)]
features = pandas.DataFrame(fea_imp, columns=["features", "importance_score"])
features.to_csv("reports/feature_importances.csv", index=False)
print("Storing Feature Importances Done")

