import numpy as np
from collections.abc import Mapping
from flask import Flask, request, jsonify, render_template
import pickle

from markupsafe import escape
# flask app
app = Flask(__name__)
# loading model
model = pickle.load(open('project.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    final_features = [x for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)
    
    if prediction == 'Y':
       output='The Sample is delivered on TIME'
    else:
       output='The Sample is NOT delivered on TIME'
       
    return render_template('index.html', output=output )

if __name__ == "__main__":
    app.run(debug=True)
