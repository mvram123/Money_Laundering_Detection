import pandas
import json
import numpy as np
import csv
import logging
from random import randint

data_path = "data_processed/filtered_data.csv"

logging.basicConfig(filename='logs/model_development.txt',
					filemode='a',
					format='%(asctime)s %(message)s',
					datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("----------")
logging.warning("DATA PREPROCESSING 2 STAGE")

logging.warning("Reading Preprocessed 1 dataset...")

X = pandas.read_csv(data_path)
X = X.to_numpy()

logging.warning("Read Preprocessed 1 dataset...")

csv_dataset_primary = []
step = 0
trans_type = 1
amount = 2
nameOrig = 3
oldbalanceOrg = 4
nameDest = 6
oldbalanceDest = 7
accountType = 8
isFraud = 9
isFlaggedFraud = 10

logging.warning("Changing Labels of Type Column ...")

transfer = ["WIRE_IN", "WIRE_OUT"]
for i in range(X.shape[0]):
	arr = []
	arr.append(X[i,step])
	if X[i,trans_type] =="PAYMENT":
		arr.append("CREDIT")
	elif X[i,trans_type] =="TRANSFER":
		arr.append(transfer[randint(0,1)])
	else:
		arr.append(X[i,trans_type])
	arr.append(X[i,amount])
	arr.append(X[i,nameOrig])
	arr.append(X[i,oldbalanceOrg])
	arr.append(X[i,nameDest])
	arr.append(X[i,oldbalanceDest])
	if X[i,trans_type] == "TRANSFER":
		arr.append("FOREIGN")
	else:
		arr.append("DOMESTIC")

	arr.append(X[i,isFraud])
	arr.append(X[i,isFlaggedFraud])

	csv_dataset_primary.append(arr)

logging.warning("Changing Labels Done")
logging.warning("Storing Data in Data_processed Folder...")


columns=['step','trans_type','amount','nameOrig','oldbalanceOrg',
        'nameDest','oldbalanceDest','accountType','isFraud','isFlaggedFraud']

data_primary = pandas.DataFrame(csv_dataset_primary, columns=columns)

data_primary.to_csv('data_processed/filtered_data_2.csv', index=False)

logging.warning("Storing Data Done")



