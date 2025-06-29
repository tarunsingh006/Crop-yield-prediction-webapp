import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(df):
    """Clean and standardize data formats"""
    # Clean categorical columns
    categorical_cols = ['Crop', 'Season', 'State']
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.title()
    
    # Clean numeric columns
    numeric_cols = ['Crop_Year', 'Area', 'Production', 
                   'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna()

def train_model():
    # Load and clean data
    df = pd.read_csv('dataset.csv')
    df = clean_data(df)
    
    # Encode categorical features
    categorical_cols = ['Crop', 'Season', 'State']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        joblib.dump(le, f'{col}_encoder.pkl')
    
    # Scale numerical features
    numeric_cols = ['Crop_Year', 'Area', 'Production', 
                   'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    joblib.dump(scaler, 'scaler.pkl')
    
    # Prepare features and target
    X = df.drop('Yield', axis=1)
    y = df['Yield']
    
    # Ensure consistent feature order
    feature_order = X.columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=150,
        max_depth=5
    )
    model.fit(X_train, y_train)
    
    # Save model with feature order
    model.feature_names_ = feature_order
    joblib.dump(model, 'crop_yield_model.pkl')
    
    # Save feature order for verification
    with open('feature_order.txt', 'w') as f:
        f.write('\n'.join(feature_order))
    
    print("Training successful. Feature order:", feature_order)

if __name__ == '__main__':
    train_model()