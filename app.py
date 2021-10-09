from flask import Flask, request, render_template
from flask import Response
from datetime import datetime
import pandas
import pickle
import json
import requests

headers = {"Authorization": "Bearer ya29.a0ARrdaM_YkI6oUm949UJteFylUpoLGG114jpBLlEiTSJZkfPSqwPaUcWmJKHRN9aPNBpOZoXdbCjC5BRezFaooZSVvfFqMyKkbmb_ZuuxrzkARnNJh06-Dm-xvq4FVlpmnoBm1IF2n4seBnhRjV79Si4XmNS7"}

model_path = 'saved_models/model.pkl'
model = pickle.load(open(model_path, 'rb'))

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predictRouteClient():

        if request.form is not None:
            path = request.form['filepath']
            data = pandas.read_csv(path)
            y_pred = model.predict(data)
            data['isFraud'] = y_pred
            output_path = "predictions/"+ "Output_"+ datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".csv"
            data.to_csv(output_path, index=False)
            file_name = "Output_"+ datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".csv"
            para = {
                    "name": file_name,
                    "parents":["1UGQyeMhA8YR0UFT5suftSckVvfXcBd-d"]

                    }
            files = {
                    'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
                    'file': open("./"+ output_path, "rb")
                    }
            r = requests.post(
                     "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                    headers=headers,
                     files=files
                    )
            print(r.text)
    
            return file_name
        else:
            print('Nothing Matched')
    

if __name__ == "__main__":
    app.run()