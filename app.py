import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    #Back end calculations - Predictions
    prediction = model.predict(final_features)

    if (prediction == 0):
        return "Prediction : Plant is Knema linifolia (Roxb) Warb."

    elif (prediction == 1):
        return "Prediction : Plant is  knema angustifolia (Roxb.) Warb."

    elif (prediction == 2):
        return "Prediction : Plant is knema eratica (Hook. f. & Thomson) J. Sinclair."

    elif (prediction == 3):
        return "Prediction : Plant is  Knema tenuinervia W.J.de Wilde subsp. Tenuinervia."

    else:
        return "Prediction : Plant is knema globularia (Lam) Warb."

    
if __name__ == "__main__":
    app.run(debug=True)
