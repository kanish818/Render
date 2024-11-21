# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'random_forest_model.pkl'  # Use the Random Forest model file name
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form and convert it to the correct format
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'Safe to drink' if prediction[0] == 1 else 'Not safe to drink'

        return render_template('index.html', prediction_text='Water Quality Prediction: {}'.format(output))
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
