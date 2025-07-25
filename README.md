🚗 Auto Insurance Fraud Detection using Machine Learning
This project aims to detect fraudulent auto insurance claims using both rule-based heuristics and machine learning models. It includes data preprocessing, feature engineering, rule-based fraud indicators, and predictive modeling with proper evaluation metrics and visualization.

🧾 Dataset Overview
Dataset: Auto Insurance Claims Dataset

Records: ~70,000 rows

Columns: 53 features (numerical, categorical, date-time)

Target Variable: Fraud_Suspected (1 = Fraud, 0 = Not Fraud)

🧠 Problem Statement
Insurance fraud leads to significant financial losses. This project helps build an automated system that flags potentially fraudulent claims using:

Domain-driven rules

Machine Learning models
🧮 Preprocessing & Feature Engineering
Converted date columns to datetime format

Extracted time intervals (e.g., days between policy start and accident)

Applied label encoding and one-hot encoding for categorical variables

Removed unnecessary features (like Claim_ID, Accident_Date, etc.)

🧾 Rules-Based Features
We engineered the following domain-specific binary rules (1 = suspicious, 0 = not suspicious):

Rule Feature	Description
Suspicious_Late_Claim	Accident reported >30 days after policy start
Suspicious_Multiple_Claims	Customer has more than 2 claims
Suspicious_Police_Not_Filed	No police report filed
Suspicious_High_Claim	Total claim amount > ₹100,000
Suspicious_Recent_Policy	Policy age < 60 days at time of claim
Suspicious_Policy_Claim_Same_Day	Claim made on the same day policy started
Suspicious_High_Vehicle_Cost	Vehicle cost > ₹1,000,000

These rules were combined with original features to enhance model learning.

🤖 Machine Learning Models
Models trained and evaluated:

Logistic Regression

Random Forest Classifier

XGBoost Classifier

Decision Tree

Support Vector Machine (SVM)

All models were trained with and without rule-based features for comparison.

📊 Model Evaluation
Each model was evaluated using:

Accuracy Score

Confusion Matrix

Precision, Recall, F1-Score (especially for minority class "fraud")

ROC AUC (optional)

Best performing model:
✅ Random Forest with rule-based features + encoding
📈 Accuracy: ~72%
📌 Fraud Recall: ~14–20% (still limited due to class imbalance)

📉 Visualization
Feature importance plot (for tree models)

Confusion matrix heatmaps

Distribution of fraud vs. non-fraud

Rule-based feature correlation

🔧 How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/Insurance_Fraud_ML.git
cd Insurance_Fraud_ML
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the notebook or script:

bash
Copy
Edit
jupyter notebook notebooks/EDA_and_Modeling.ipynb
📦 Dependencies
pandas

numpy

matplotlib

seaborn

scikit-learn

xgboost

imbalanced-learn (if using SMOTE or resampling)

jupyter (for notebook use)

✅ Future Improvements
Implement advanced sampling techniques like SMOTE to handle class imbalance

Use ensemble learning or stacking

Integrate with a Flask API or web dashboard for real-time scoring

Improve rule thresholds using data statistics

👤 Author
Gyanendra Sahoo (CSE Engineer)
Python | Embedded | Machine Learning
