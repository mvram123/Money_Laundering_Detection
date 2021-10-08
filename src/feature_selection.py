import numpy as np
import math
from numpy import genfromtxt
import pandas
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(filename='logs/model_development.txt',
					filemode='a',
					format='%(asctime)s %(message)s',
					datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("----------")
logging.warning("FEATURE SELECTION STAGE")

logging.warning("Reading Filtered Data 3 ...")

dataframeX = pandas.read_csv('data_processed/filtered_data_3.csv')
col_names = list(dataframeX.columns.values)
dataMat = dataframeX.to_numpy()

logging.warning("Read Filtered Data 3")

logging.warning("Creating X and Y Variables...")

X = dataMat[:,1:-2]
Y = dataMat[:,-1]

logging.warning(f"Shape of X: {X.shape} and Shape of Y: {Y.shape}")

logging.warning("Instiantiating Random Forest Model...")

model = RandomForestClassifier(random_state=42)

logging.warning("Fitting Data...")

model.fit(X, Y.astype(int))

logging.warning("Checking Feature Importances...")

feature_imp = model.feature_importances_

sorted_feature_vals = np.sort(feature_imp)
sorted_feature_indexes = np.argsort(feature_imp)

logging.warning("Significant Features in decreasing order of importance: ")

logging.warning("Storing Feature Importances in reports...")

fea_imp = [[col_names[i+2], feature_imp[i]] for i in reversed(sorted_feature_indexes)]
features = pandas.DataFrame(fea_imp, columns=["features", "importance_score"])
features.to_csv("reports/feature_importances.csv", index=False)

logging.warning("Storing Feature Importances Done")

