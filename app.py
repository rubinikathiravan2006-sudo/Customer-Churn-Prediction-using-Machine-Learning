import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import gradio as gr

# ==============================
# LOAD DATASET
# ==============================
# Replace with your dataset path
DATASET_PATH = "customer_churn_dataset.csv"

try:
    df = pd.read_csv(DATASET_PATH)
except:
    # Sample dataset creation if dataset not found
    data = {
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0, 1, 0],
        'Tenure': [12, 5, 30, 2, 45],
        'MonthlyCharges': [70, 90, 60, 100, 50],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'Two year'],
        'InternetService': ['Fiber optic', 'DSL', 'DSL', 'Fiber optic', 'No'],
        'Churn': ['Yes', 'Yes', 'No', 'Yes', 'No']
    }
    df = pd.DataFrame(data)

# ==============================
# DATA PREPROCESSING
# ==============================
label_encoders = {}

for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# ==============================
# FEATURES AND TARGET
# ==============================
X = df.drop('Churn', axis=1)
y = df['Churn']

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# FEATURE SCALING
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# MODEL TRAINING
# ==============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==============================
# MODEL EVALUATION
# ==============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_churn(gender, senior, tenure, monthly, contract, internet):

    input_data = pd.DataFrame({
        'Gender': [gender],
        'SeniorCitizen': [senior],
        'Tenure': [tenure],
        'MonthlyCharges': [monthly],
        'Contract': [contract],
        'InternetService': [internet]
    })

    # Encode categorical values
    for column in input_data.columns:
        if column in label_encoders:
            le = label_encoders[column]
            input_data[column] = le.transform(input_data[column])

    # Scale data
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    confidence = np.max(probability) * 100

    if prediction == 1:
        result = "⚠ Customer is likely to CHURN"
    else:
        result = "✅ Customer is likely to STAY"

    output = f"""
# 📊 Prediction Result

## {result}

### Confidence Score: {confidence:.2f}%

### Model Accuracy: {accuracy * 100:.2f}%
"""

    return output

# ==============================
# GRADIO INTERFACE
# ==============================
interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(['Male', 'Female'], label='Gender'),
        gr.Radio([0, 1], label='Senior Citizen (0 = No, 1 = Yes)'),
        gr.Slider(0, 100, value=12, label='Tenure (Months)'),
        gr.Slider(0, 200, value=70, label='Monthly Charges'),
        gr.Dropdown(['Month-to-month', 'One year', 'Two year'], label='Contract Type'),
        gr.Dropdown(['DSL', 'Fiber optic', 'No'], label='Internet Service')
    ],
    outputs=gr.Markdown(),
    title='📈 Customer Churn Prediction System',
    description='AI-powered churn prediction system using Machine Learning and customer behavior analysis.',
    theme='soft'
)

# ==============================
# LAUNCH APP
# ==============================
interface.launch()
