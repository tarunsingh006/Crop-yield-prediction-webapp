from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load resources
model = joblib.load('crop_yield_model.pkl')
encoders = {col: joblib.load(f'{col}_encoder.pkl') for col in ['Crop', 'Season', 'State']}
scaler = joblib.load('scaler.pkl')

# Get feature order from model
FEATURE_ORDER = model.feature_names_
with open('feature_order.txt') as f:
    FEATURE_ORDER = f.read().splitlines()

@app.context_processor
def inject_supported_values():
    return {
        'supported_crops': list(encoders['Crop'].classes_),
        'supported_seasons': list(encoders['Season'].classes_),
        'supported_states': list(encoders['State'].classes_)
    }

@app.route('/')
def home():
    return render_template('index.html')

def prepare_input(input_data):
    """Convert and validate input data with proper ordering"""
    # Convert to DataFrame with correct feature order
    df = pd.DataFrame([input_data], columns=FEATURE_ORDER)
    
    # Encode categorical features
    for col in ['Crop', 'Season', 'State']:
        df[col] = encoders[col].transform(df[col])
    
    # Scale numerical features
    numeric_cols = ['Crop_Year', 'Area', 'Production', 
                   'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect and clean input data
        input_data = {
            'Crop': request.form['crop'].strip().title(),
            'Season': request.form['season'].strip().title(),
            'State': request.form['state'].strip().title(),
            'Crop_Year': float(request.form['year']),
            'Area': float(request.form['area']),
            'Production': float(request.form['production']),
            'Annual_Rainfall': float(request.form['rainfall']),
            'Fertilizer': float(request.form['fertilizer']),
            'Pesticide': float(request.form['pesticide'])
        }
        
        # Validate input features
        missing_features = set(FEATURE_ORDER) - set(input_data.keys())
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
            
        # Prepare input with correct feature order
        df = prepare_input(input_data)
        
        # Verify feature order match
        if list(df.columns) != FEATURE_ORDER:
            raise ValueError(f"Feature order mismatch. Expected: {FEATURE_ORDER}")
        
        # Predict
        prediction = model.predict(df)[0]
        return render_template(
            'result.html',
            success=True,
            prediction=round(prediction, 2),
            input_data=input_data
        )
        
    except Exception as e:
        return render_template(
            'result.html',
            success=False,
            error_message=str(e),
            input_data=input_data
        )

if __name__ == '__main__':
    app.run(debug=False)