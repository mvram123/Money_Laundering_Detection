import time
import json
import pandas
import numpy as np
import csv
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def getClusterPredictions(data, true_k):
	model = KMeans(n_clusters=true_k)
	model.fit(data)
	prediction = model.predict(data)

	return prediction

def getBestCluster(X,_min=2,_max=10):
	selected_cluster = 0
	previous_sil_coeff = 0.001 #some random small number not 0
	sc_vals = []
	for n_cluster in range(_min, _max):
	    kmeans = KMeans(n_clusters=n_cluster).fit(X)
	    label = kmeans.labels_

	    sil_coeff = silhouette_score(X, label, metric='euclidean', sample_size=1000)
	    sc_vals.append(sil_coeff)
	    # print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

	    percent_change = (sil_coeff-previous_sil_coeff)*100/previous_sil_coeff

	    # return when below a threshold of 1%
	    if percent_change<1:
	    	selected_cluster = n_cluster-1

	    previous_sil_coeff = sil_coeff

	return selected_cluster or _max, sc_vals

data_path = "data_processed/filtered_data_3.csv"
print("Reading Filtered 3 Dataset...")

X_dataframe = pandas.read_csv(data_path)
X = X_dataframe.to_numpy()

print("Read dataset")
col_names = list(X_dataframe.columns.values)

X_trimmed_features = np.zeros((X.shape[0],1))
print("Importing Feature Importances...")
feature_path = "reports/feature_importances.csv"
features = pandas.read_csv(feature_path)
print("Selecting Top 13 Features for CLustering...")
top_13 = features.iloc[:13, 0].tolist()
print("Top 13 Features stored in List")

for feature in top_13:
	X_trimmed_features = np.concatenate((X_trimmed_features,np.expand_dims(X_dataframe[feature],axis=1)),axis=1)
X_trimmed_features = X_trimmed_features[:,1:]
print("Choosing Best Number Of Clusters...")
min_value = 2
max_value = 10
true_k, sc_vals = getBestCluster(X_trimmed_features,_min=min_value,_max=max_value)
true_k = 5
print("Storing Silhoutte Scores... ")
sil_score = [[i, sc_vals[i-min_value]] for i in range(min_value, max_value)]
sil = pandas.DataFrame(sil_score, columns=["no_of_clusters", "silhoutte_score"])
sil.to_csv("reports/silhoutte_scores.csv", index=False)
print("Storing Silhoutte Scores Done")
print("Creating Clusters with Best No Of Clusters...")

prediction = getClusterPredictions(X_trimmed_features, true_k)
seg_dict = {}
for i in range(X.shape[0]):
	seg_dict[X[i,0]] = prediction[i]

print("Inputing Filtered Data 2 Dataset...")

filter_1_path = "data_processed/filtered_data_2.csv"
X_dataframe_pri = pandas.read_csv(filter_1_path)
X_pri = X_dataframe_pri.to_numpy()
col_names = list(X_dataframe_pri.columns.values)

print("Read Filtered 2 Data")
print("Creating Final Dataset with segments...")
X_with_segments = []
for i in range(X_pri.shape[0]):
		X_with_segments.append(np.concatenate(([[seg_dict[X_pri[i,3]]]],np.expand_dims(X_pri[i,:],axis=0)),axis=1)[0])

segmented_columns = ['segment','step','trans_type','amount','nameOrig','oldbalanceOrg',
                    'nameDest','oldbalanceDest','accountType','isFraud','isFlaggedFraud']

data_segmented = pandas.DataFrame(X_with_segments, columns = segmented_columns)
data_segmented = data_segmented.drop('isFlaggedFraud', axis=1)
data_segmented.to_csv('data_processed/final_data.csv', index=False)
print("Storing Final Dataset Done")