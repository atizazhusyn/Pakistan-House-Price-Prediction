#!/usr/bin/env python3
"""
Pakistan House Price Prediction Analysis
This script runs the complete ML analysis in the correct order.
"""

# Import required libraries
import numpy as np
import pandas as pd
import warnings
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

print("=== Pakistan House Price Prediction Analysis ===\n")

# Step 1: Load and explore data
print("1. Loading data...")
df = pd.read_csv('house_prices.csv')
print(f"Dataset shape: {df.shape}")
print("First few rows:")
print(df.head())
print()

# Step 2: Data cleaning
print("2. Data cleaning...")
df.drop(["Unnamed: 0"], axis=1, inplace=True)
print(f"After dropping column: {df.shape}")

# Check for null values
print("Null values per column:")
print(df.isna().sum())
print()

# Drop duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"After dropping duplicates: {df.shape}")
print()

# Step 3: Feature engineering
print("3. Feature engineering...")
df['area'] = df['Area_in_Marla'] * 272.25
df.drop('Area_in_Marla', axis=1, inplace=True)

# Re-arrange columns
df = df[["property_type", "location", "city", "purpose", "baths", "bedrooms", "area", "price"]]
df.columns = ["type", "location", "city", "purpose", "baths", "beds", "area", "price"]

print("Updated columns:", list(df.columns))
print("Sample data:")
print(df.head())
print()

# Step 4: Data preprocessing
print("4. Data preprocessing...")

# Define categorical and numerical columns
cat_cols = ["type", "location", "city", "purpose"]
num_cols = ["area", "baths", "beds"]

# Label encoding for categorical features
encoder = LabelEncoder()
for column in cat_cols:
    df[column] = encoder.fit_transform(df[column])

# Standard scaling for numerical features
scaler = StandardScaler()
for column in num_cols:
    df[column] = scaler.fit_transform(df[[column]])

print("After preprocessing:")
print(df.head())
print()

# Step 5: Split data
print("5. Splitting data...")
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
print()

# Step 6: Train and evaluate models
print("6. Training and evaluating models...")

models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'model': model
    }

    print(".2f")
    print(".2f")
    print(".2f")
    print(".4f")

# Step 7: Test with example
print("\n7. Testing with example...")

# Get a test sample (using index 211 like in the original notebook)
test_idx = 211
if test_idx < len(X_test):
    test_sample = X_test.iloc[test_idx].values.reshape(1, -1)

    print("Predictions for a test sample:")
    for name, result in results.items():
        pred = result['model'].predict(test_sample)[0]
        print(f"{name}: Rs. {pred:,.0f}")

    actual_price = y_test.iloc[test_idx]
    print(f"\nActual price: Rs. {actual_price:,.0f}")

    # Price range (13% margin like in original)
    margin = 0.13
    price_range_low = actual_price * (1 - margin)
    price_range_high = actual_price * (1 + margin)
    print(f"Price range: Rs. {price_range_low:,.0f} - Rs. {price_range_high:,.0f}")
else:
    print("Test index out of range, using first test sample instead")
    test_sample = X_test.iloc[0].values.reshape(1, -1)

    print("Predictions for first test sample:")
    for name, result in results.items():
        pred = result['model'].predict(test_sample)[0]
        print(f"{name}: Rs. {pred:,.0f}")

    actual_price = y_test.iloc[0]
    print(f"\nActual price: Rs. {actual_price:,.0f}")

print("\n=== Analysis Complete ===")
print("Note: In real estate, predicting a price range is often more practical than a single value due to market variability.")
