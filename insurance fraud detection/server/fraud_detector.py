# Flask API for Fraud Detection
# Run this file to start the web service

from flask import Flask, request, jsonify, render_template_string, send_file
import pandas as pd
import numpy as np
import os
import io
import base64
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class FraudDetectionAPI:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        self.is_trained = False
    
    def preprocess_data(self, df):
        """Preprocess input data"""
        df_processed = df.copy()
        
        # Date parsing
        date_columns = ['Policy_Start_Date', 'Policy_Expiry_Date', 'Claims_Date', 'Accident_Date']
        for col in date_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
        
        # Handle missing values
        categorical_cols = ['authorities_contacted', 'Police_Report', 'Education', 'Occupation', 
                          'Vehicle_Color', 'Marital_Status', 'Gender']
        
        for col in categorical_cols:
            if col in df_processed.columns:
                if col in ['authorities_contacted', 'Police_Report']:
                    df_processed[col].fillna('No', inplace=True)
                else:
                    mode_val = df_processed[col].mode()
                    fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                    df_processed[col].fillna(fill_val, inplace=True)
        
        # Fill numerical missing values
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Create new features
        if 'Policy_Expiry_Date' in df_processed.columns and 'Policy_Start_Date' in df_processed.columns:
            df_processed['Policy_Duration'] = (df_processed['Policy_Expiry_Date'] - df_processed['Policy_Start_Date']).dt.days
        
        if 'Claims_Date' in df_processed.columns and 'Accident_Date' in df_processed.columns:
            df_processed['Claim_Lag_Days'] = (df_processed['Claims_Date'] - df_processed['Accident_Date']).dt.days
        
        if 'Accident_Hour' in df_processed.columns:
            df_processed['Is_Night_Accident'] = ((df_processed['Accident_Hour'] >= 22) | (df_processed['Accident_Hour'] <= 6)).astype(int)
        
        if 'Vehicle_Cost' in df_processed.columns:
            threshold = df_processed['Vehicle_Cost'].quantile(0.75)
            df_processed['High_Cost_Vehicle'] = (df_processed['Vehicle_Cost'] > threshold).astype(int)
        
        # Feature engineering
        if 'Accident_Date' in df_processed.columns:
            df_processed['Accident_Weekday'] = df_processed['Accident_Date'].dt.dayofweek
            df_processed['Weekend_Accident'] = (df_processed['Accident_Weekday'] >= 5).astype(int)
        
        if 'Claim_Amount' in df_processed.columns and 'Vehicle_Cost' in df_processed.columns:
            df_processed['Claim_Vehicle_Ratio'] = df_processed['Claim_Amount'] / (df_processed['Vehicle_Cost'] + 1)
            df_processed['High_Claim_Ratio'] = (df_processed['Claim_Vehicle_Ratio'] > 0.8).astype(int)
        
        if 'Age' in df_processed.columns:
            df_processed['High_Risk_Age'] = ((df_processed['Age'] < 25) | (df_processed['Age'] > 65)).astype(int)
        
        if 'Claim_Lag_Days' in df_processed.columns:
            df_processed['Late_Reporting'] = (df_processed['Claim_Lag_Days'] > 30).astype(int)
        
        # Encode categorical variables
        categorical_cols_to_encode = ['Education', 'Occupation', 'Vehicle_Color', 'Marital_Status', 
                                    'Gender', 'authorities_contacted', 'Police_Report']
        
        for col in categorical_cols_to_encode:
            if col in df_processed.columns:
                if col in self.label_encoders:
                    try:
                        df_processed[col + '_encoded'] = self.label_encoders[col].transform(df_processed[col].astype(str))
                    except ValueError:
                        le = self.label_encoders[col]
                        df_processed[col + '_encoded'] = df_processed[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else 0
                        )
                else:
                    le = LabelEncoder()
                    df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
        
        return df_processed
    
    def train_model(self, df):
        """Train a quick fraud detection model"""
        df_processed = self.preprocess_data(df)
        
        # Select numerical features
        numerical_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_remove = ['Claim_ID', 'Policy_Num', 'Vehicle_Registration', 'Check_Point', 'Fraud']
        numerical_features = [col for col in numerical_features if col not in cols_to_remove]
        
        # Create synthetic fraud labels if not present
        if 'Fraud' not in df_processed.columns:
            fraud_indicators = 0
            if 'High_Claim_Ratio' in df_processed.columns:
                fraud_indicators += df_processed['High_Claim_Ratio']
            if 'Late_Reporting' in df_processed.columns:
                fraud_indicators += df_processed['Late_Reporting']
            if 'Is_Night_Accident' in df_processed.columns:
                fraud_indicators += df_processed['Is_Night_Accident']
            
            df_processed['Fraud'] = (fraud_indicators >= 2).astype(int)
        
        X = df_processed[numerical_features]
        y = df_processed['Fraud']
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        self.feature_columns = numerical_features
        self.is_trained = True
    
    def predict_fraud(self, df):
        """Predict fraud for input dataframe"""
        if not self.is_trained:
            self.train_model(df)
        
        df_processed = self.preprocess_data(df)
        
        # Prepare features
        available_features = [col for col in self.feature_columns if col in df_processed.columns]
        X = df_processed[available_features]
        
        # Handle missing features
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        X = X[self.feature_columns]
        
        # Make predictions
        fraud_probabilities = self.model.predict_proba(X)[:, 1]
        
        # Create risk categories
        def categorize_risk(prob):
            if prob < 0.3:
                return 'Low'
            elif prob < 0.7:
                return 'Medium'
            else:
                return 'High'
        
        results = df.copy()
        results['Fraud_Probability'] = fraud_probabilities
        results['Risk_Category'] = [categorize_risk(prob) for prob in fraud_probabilities]
        
        return results

# Initialize the fraud detector
fraud_detector = FraudDetectionAPI()

@app.route('/')
def home():
    """Serve the main webpage"""
    # Read the HTML template from the artifact
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Auto Insurance Fraud Detection</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }
            .header { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 30px; text-align: center; }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
            .upload-section { padding: 40px; text-align: center; }
            .btn { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 15px 30px; border: none; border-radius: 25px; font-size: 1.1em; cursor: pointer; transition: all 0.3s ease; margin: 10px; }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(0,0,0,0.2); }
            .upload-box { border: 3px dashed #667eea; border-radius: 15px; padding: 40px; margin: 20px 0; background: #f8f9ff; transition: all 0.3s ease; }
            .results { padding: 20px; margin-top: 20px; background: #f8f9fa; border-radius: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛡️ Auto Insurance Fraud Detection</h1>
                <p>Upload your CSV file to detect potential fraud cases</p>
            </div>
            <div class="upload-section">
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="upload-box">
                        <input type="file" name="file" accept=".csv" required>
                        <p>Select your CSV file</p>
                    </div>
                    <button type="submit" class="btn">🔍 Analyze for Fraud</button>
                </form>
                <div id="results" class="results" style="display: none;"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const resultsDiv = document.getElementById('results');
                
                resultsDiv.innerHTML = '<p>Analyzing... Please wait.</p>';
                resultsDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.error) {
                        resultsDiv.innerHTML = `<p style="color: red;">Error: ${result.error}</p>`;
                    } else {
                        displayResults(result);
                    }
                } catch (error) {
                    resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            });
            
            function displayResults(data) {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `
                    <h3>📊 Analysis Results</h3>
                    <p><strong>Total Records:</strong> ${data.summary.total_records}</p>
                    <p><strong>High Risk:</strong> ${data.summary.high_risk} (${(data.summary.high_risk/data.summary.total_records*100).toFixed(1)}%)</p>
                    <p><strong>Medium Risk:</strong> ${data.summary.medium_risk} (${(data.summary.medium_risk/data.summary.total_records*100).toFixed(1)}%)</p>
                    <p><strong>Low Risk:</strong> ${data.summary.low_risk} (${(data.summary.low_risk/data.summary.total_records*100).toFixed(1)}%)</p>
                    <br>
                    <a href="/download" class="btn">💾 Download Full Results</a>
                `;
            }
        </script>
    </body>
    </html>
    '''
    return html_template

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and fraud prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'})
        
        # Read CSV file
        df = pd.read_csv(file)
        
        if df.empty:
            return jsonify({'error': 'CSV file is empty'})
        
        # Make predictions
        results = fraud_detector.predict_fraud(df)
        
        # Store results for download
        app.config['LAST_RESULTS'] = results
        
        # Calculate summary statistics
        total_records = len(results)
        high_risk = len(results[results['Risk_Category'] == 'High'])
        medium_risk = len(results[results['Risk_Category'] == 'Medium'])
        low_risk = len(results[results['Risk_Category'] == 'Low'])
        
        summary = {
            'total_records': total_records,
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk
        }
        
        # Return top high-risk cases
        high_risk_cases = results[results['Risk_Category'] == 'High'].sort_values('Fraud_Probability', ascending=False).head(10)
        
        return jsonify({
            'summary': summary,
            'high_risk_sample': high_risk_cases.to_dict('records'),
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download')
def download():
    """Download full results as CSV"""
    try:
        results = app.config.get('LAST_RESULTS')
        if results is None:
            return "No results available for download", 404
        
        # Create CSV in memory
        output = io.StringIO()
        results.to_csv(output, index=False)
        output.seek(0)
        
        # Convert to bytes
        csv_bytes = io.BytesIO()
        csv_bytes.write(output.getvalue().encode())
        csv_bytes.seek(0)
        
        return send_file(
            csv_bytes,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'fraud_analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    
    except Exception as e:
        return f"Error generating download: {str(e)}", 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        # Accept JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'})
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        results = fraud_detector.predict_fraud(df)
        
        result = results.iloc[0]
        
        return jsonify({
            'fraud_probability': float(result['Fraud_Probability']),
            'risk_category': result['Risk_Category'],
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': fraud_detector.is_trained,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Fraud Detection API...")
    print("📱 Web Interface: http://localhost:5000")
    print("🔌 API Endpoint: http://localhost:5000/api/predict")
    print("❤️  Health Check: http://localhost:5000/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)