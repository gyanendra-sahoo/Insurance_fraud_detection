
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model
with open('fraud_detection_model.pkl', 'rb') as f:
    model_components = pickle.load(f)

model = model_components['model']
scaler = model_components['scaler']
label_encoders = model_components['label_encoders']
feature_columns = model_components['feature_columns']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        
        # Preprocess data (same as training)
        # ... (add preprocessing steps here)
        
        X = df[feature_columns]
        X_scaled = scaler.transform(X)
        
        fraud_prob = model.predict_proba(X_scaled)[0][1]
        
        if fraud_prob < 0.3:
            risk = 'Low'
        elif fraud_prob < 0.7:
            risk = 'Medium'
        else:
            risk = 'High'
        
        return jsonify({
            'fraud_probability': float(fraud_prob),
            'risk_category': risk
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
        