from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("Model loaded successfully!")
except:
    print("Model not found. Please run model.py first to train the model.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    try:
        # Get data from request
        data = request.json
        
        # Create feature array
        features = np.array([[
            data['age'],
            1 if data['sex'] == 'female' else 0,
            data['bmi'],
            data['children'],
            1 if data['smoker'] == 'yes' else 0,
            {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}[data['region']]
        ]])
        
        # Scale features if needed (for KNN and SVR models)
        model_name = str(type(model).__name__)
        if 'KNeighbors' in model_name or 'SVR' in model_name:
            features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return jsonify({
            'prediction': round(prediction, 2),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)