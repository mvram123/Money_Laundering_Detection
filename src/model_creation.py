import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from catboost import CatBoostClassifier

print("Reading Final Dataset...")
data_path = "data_processed/final_data.csv"
dataMat = pd.read_csv(data_path)
data = dataMat.to_numpy()
print("Read Dataset")

print("Checking Categorical Features...")
cat_feat = [i for i in dataMat.columns if dataMat[i].dtypes == 'O']

print("Checking Missing Values...")
a = dict(dataMat.isnull().sum())
b = [[i, a[i]] for i in a.keys()]
missing = pd.DataFrame(b, columns=['features', 'null_values_count'])
print("Storing Missing Values...")
missing.to_csv("reports/missing_values.csv", index=False)
print("Storing Missing Done")

print("Encoding Categorical Features...")
encoder = LabelEncoder()
for i in cat_feat:
  dataMat[i] = encoder.fit_transform(dataMat[i])
print("Features Encoding Done")

print("Creating X and y ...")
X = dataMat.iloc[:, :-1]
y = dataMat['isFraud']
print(f"Shape of X: {X.shape} and Shape of y: {y.shape}")

print("Splitting Dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Instantiating Model...")
model = CatBoostClassifier(random_state=42, class_weights={0:1, 1:12}, silent=True)
print("Fitting Model...")
model.fit(X_train, y_train)
y_pred_cat = model.predict(X_test)
print("Saving Model...")
model_path = "saved_models/model.pkl"
pickle.dump(model, open(model_path, 'wb'))

print("Saving Model Metrics...")
metric_file_path = "reports/performance.json"
with open(metric_file_path, "r") as f:
    data = json.load(f)

model_metric = {
    "time_stamp": datetime.now().strftime("%d-%m-%Y_%H:%M:%S"),
    "confusion_matrix": confusion_matrix(y_test, y_pred_cat).tolist(),
    "precision": precision_score(y_test, y_pred_cat),
    "recall": recall_score(y_test, y_pred_cat),
    "f1_score": f1_score(y_test, y_pred_cat)
}

data['model_metric'].append(model_metric)
with open(metric_file_path, "w") as f:
    json.dump(data, f, indent=4)

print("Model Metrics Stored")


