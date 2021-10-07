from flask import Flask, request, render_template
from flask import Response
from datetime import datetime
import pandas
import pickle


model_path = 'saved_models/model.pkl'
model = pickle.load(open(model_path, 'rb'))

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predictRouteClient():
    try:
        if request.form is not None:
            path = request.form['filepath']
            print(path)
            print("Reading Data...")
            data = pandas.read_csv(path)
            print("Predicting Outputs...")
            y_pred = model.predict(data)
            data['isFraud'] = y_pred
            print("Storing Outputs...")
            output_path = "predictions/"+ "Output_"+ datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".csv"
            data.to_csv(output_path, index=False)
            return output_path
        else:
            print('Nothing Matched')
    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)
    

if __name__ == "__main__":
    app.run(debug=True)