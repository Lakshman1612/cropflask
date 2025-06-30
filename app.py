from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import sklearn
import os

print("Scikit-learn version:", sklearn.__version__)

dtr = pickle.load(open('rf.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        crop = request.form['Crop']
        crop_year = int(request.form['Crop_Year'])
        season = request.form['Season']
        state = request.form['State']
        area = float(request.form['Area'])
        rainfall = float(request.form['Annual_Rainfall'])
        fertilizer = float(request.form['Fertilizer'])
        pesticide = float(request.form['Pesticide'])

        input_df = pd.DataFrame([{
            'Crop': crop,
            'Crop_Year': crop_year,
            'Season': season,
            'State': state,
            'Area': area,
            'Annual_Rainfall': rainfall,
            'Fertilizer': fertilizer,
            'Pesticide': pesticide
        }])

        transformed_features = preprocessor.transform(input_df)
        prediction = dtr.predict(transformed_features)[0]
        return render_template('index.html', prediction=f"{round(prediction, 2)}kg/ha")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
