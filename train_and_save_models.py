#!/usr/bin/env python3
"""
Train and save ML models for Pakistan House Price Prediction
This script trains the models and saves them along with preprocessing objects for the web app.
"""

import numpy as np
import pandas as pd
import warnings
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import XGBoost (skip if not available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost loaded successfully")
except (ImportError, Exception) as e:
    print(f"XGBoost not available: {e}")
    print("Skipping XGBoost model")
    XGBOOST_AVAILABLE = False

# Disable warnings
warnings.filterwarnings('ignore')

def create_models_directory():
    """Create directory for saved models"""
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models' directory")
    else:
        print("'models' directory already exists")

def load_and_preprocess_data():
    """Load and preprocess the house price data"""
    print("Loading data...")
    df = pd.read_csv('house_prices.csv')

    # Data cleaning
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    df = df.drop_duplicates().reset_index(drop=True)

    # Feature engineering
    df['area'] = df['Area_in_Marla'] * 272.25
    df.drop('Area_in_Marla', axis=1, inplace=True)

    # Re-arrange columns
    df = df[["property_type", "location", "city", "purpose", "baths", "bedrooms", "area", "price"]]
    df.columns = ["type", "location", "city", "purpose", "baths", "beds", "area", "price"]

    return df

def get_unique_values(df):
    """Get unique values for categorical features for the web app"""
    unique_values = {}
    unique_values['property_types'] = sorted(df['type'].unique())
    unique_values['locations'] = sorted(df['location'].unique())
    unique_values['cities'] = sorted(df['city'].unique())
    unique_values['purposes'] = sorted(df['purpose'].unique())

    # Get min/max values for numerical features
    unique_values['baths_range'] = (int(df['baths'].min()), int(df['baths'].max()))
    unique_values['beds_range'] = (int(df['beds'].min()), int(df['beds'].max()))
    unique_values['area_range'] = (float(df['area'].min()), float(df['area'].max()))

    return unique_values

def preprocess_data(df):
    """Preprocess data and return encoders/scalers for saving"""
    # Define categorical and numerical columns
    cat_cols = ["type", "location", "city", "purpose"]
    num_cols = ["area", "baths", "beds"]

    # Create and fit encoders/scalers
    encoders = {}
    scalers = {}

    # Label encoding for categorical features
    for column in cat_cols:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder

    # Standard scaling for numerical features
    for column in num_cols:
        scaler = StandardScaler()
        df[column] = scaler.fit_transform(df[[column]])
        scalers[column] = scaler

    return df, encoders, scalers

def train_models(X_train, y_train):
    """Train all available models"""
    models = {
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        # 'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),  # Commented out for faster training
        # 'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)  # Commented out for faster training
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

def save_models_and_preprocessing(trained_models, encoders, scalers, unique_values):
    """Save all models and preprocessing objects"""
    print("\nSaving models and preprocessing objects...")

    # Save trained models
    for name, model in trained_models.items():
        filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} model to {filename}")

    # Save encoders
    for feature, encoder in encoders.items():
        filename = f"models/{feature}_encoder.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(encoder, f)
        print(f"Saved {feature} encoder to {filename}")

    # Save scalers
    for feature, scaler in scalers.items():
        filename = f"models/{feature}_scaler.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Saved {feature} scaler to {filename}")

    # Save unique values for UI
    with open('models/unique_values.pkl', 'wb') as f:
        pickle.dump(unique_values, f)
    print("Saved unique values to models/unique_values.pkl")

def main():
    print("=== Training and Saving Models for Pakistan House Price Prediction ===\n")

    # Create models directory
    create_models_directory()

    # Load and preprocess data
    df = load_and_preprocess_data()
    print(f"Dataset shape: {df.shape}")

    # Get unique values for web app (before preprocessing)
    raw_df = load_and_preprocess_data()  # Reload to get raw values
    unique_values = get_unique_values(raw_df)

    # Preprocess data
    df, encoders, scalers = preprocess_data(df)

    # Split data
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Train models
    trained_models = train_models(X_train, y_train)

    # Save models and preprocessing objects
    save_models_and_preprocessing(trained_models, encoders, scalers, unique_values)

    # Quick evaluation
    print("\n=== Quick Model Evaluation ===")
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"{name}: R² = {r2:.4f}, RMSE = {rmse:,.0f}")

    print("\n=== All models and preprocessing objects saved successfully! ===")
    print("You can now run the web app with: streamlit run app.py")

if __name__ == "__main__":
    main()
