import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle


app = Flask(__name__)
model = pickle.load(open('RandomForest.pkl', 'rb'))


cols = ['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat',
       'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache',
       'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue ',
       'Gastrointestinal ', 'Abroad travel', 'Contact with COVID Patient',
       'Attended Large Gathering', 'Visited Public Exposed Places',
       'Family working in Public Exposed Places', 'Wearing Masks',
       'Sanitization from Market']

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    

    features = [x for x in request.form.values()]
    df = pd.DataFrame([features], columns=cols)
    prediction =model.predict(df)

    
    if prediction == 0:
        output = f'THERE IS A CHANCE YOU ARE NOT COVID POSITIVE.'
    else:
        output = f'THERE IS A CHANCE YOU ARE COVID POSITIVE. PLEASE CONSULT A DOCTOR.'


    return render_template('index.html', prediction=output)

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    df = pd.DataFrame([data.values()], columns=cols)

    prediction =model.predict(df)
    if prediction == 0:
        output = f'THERE IS A CHANCE YOU ARE NOT COVID POSITIVE.'
    else:
        output = f'THERE IS A CHANCE YOU ARE COVID POSITIVE. PLEASE CONSULT A DOCTOR.'

    return output

if __name__ == "__main__":
    app.run(debug=True)
    
    