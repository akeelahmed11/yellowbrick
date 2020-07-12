import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("regression_model")

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():

	int_features = [float(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
	prediction = model.predict(final_features)
	
	output = round(prediction[0], 2)

	return render_template('index.html', prediction_text='Strenth of concrete is: {} MPa'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
	app.run(debug=True)