#!/usr/bin/env python3
"""
Pakistan House Price Prediction - Flask Web App
A simple Flask web application for predicting house prices in Pakistan.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load models and preprocessing objects
def load_models():
    """Load trained models and preprocessing objects"""
    models = {}
    encoders = {}
    scalers = {}

    try:
        # Load unique values
        with open('models/unique_values.pkl', 'rb') as f:
            unique_values = pickle.load(f)

        # Load encoders
        cat_cols = ["type", "location", "city", "purpose"]
        for col in cat_cols:
            with open(f'models/{col}_encoder.pkl', 'rb') as f:
                encoders[col] = pickle.load(f)

        # Load scalers
        num_cols = ["area", "baths", "beds"]
        for col in num_cols:
            with open(f'models/{col}_scaler.pkl', 'rb') as f:
                scalers[col] = pickle.load(f)

        # Load models
        model_files = [
            'decision_tree_model.pkl',
            'random_forest_model.pkl',
            'gradient_boosting_model.pkl',
            'xgboost_model.pkl'
        ]

        for model_file in model_files:
            model_path = f'models/{model_file}'
            if os.path.exists(model_path):
                model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)

        return models, encoders, scalers, unique_values

    except FileNotFoundError:
        return None, None, None, None

# Load models at startup
models, encoders, scalers, unique_values = load_models()

def preprocess_input(property_type, location, city, purpose, baths, beds, area):
    """Preprocess user input for prediction"""
    # Create input dataframe
    input_data = pd.DataFrame({
        'type': [property_type],
        'location': [location],
        'city': [city],
        'purpose': [purpose],
        'baths': [baths],
        'beds': [beds],
        'area': [area]
    })

    # Encode categorical features
    for col in ['type', 'location', 'city', 'purpose']:
        input_data[col] = encoders[col].transform(input_data[col])

    # Scale numerical features
    for col in ['area', 'baths', 'beds']:
        input_data[col] = scalers[col].transform(input_data[[col]])

    return input_data

def format_currency(amount):
    """Format amount as Pakistani Rupees"""
    return f"Rs. {amount:,.0f}"

@app.route('/')
def home():
    """Render the main page"""
    if models is None:
        return """
        <h1>Models not found</h1>
        <p>Please run <code>python train_and_save_models.py</code> first to train and save the models.</p>
        <a href="/train">Train Models</a>
        """
    return render_template('index.html', unique_values=unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        property_type = request.form.get('property_type')
        location = request.form.get('location')
        city = request.form.get('city')
        purpose = request.form.get('purpose')
        baths = int(request.form.get('baths'))
        beds = int(request.form.get('beds'))
        area = float(request.form.get('area'))

        # Preprocess input
        input_data = preprocess_input(property_type, location, city, purpose, baths, beds, area)

        # Get predictions from all models
        predictions = {}
        for model_name, model in models.items():
            prediction = model.predict(input_data)[0]
            predictions[model_name] = prediction

        # Calculate summary statistics
        prices = list(predictions.values())
        avg_price = np.mean(prices)
        min_price = min(prices)
        max_price = max(prices)

        # Realistic price range (13% margin like in original)
        margin = 0.13
        realistic_low = avg_price * (1 - margin)
        realistic_high = avg_price * (1 + margin)

        result = {
            'predictions': {name: format_currency(price) for name, price in predictions.items()},
            'summary': {
                'average': format_currency(avg_price),
                'range': f"{format_currency(min_price)} - {format_currency(max_price)}",
                'realistic_range': f"{format_currency(realistic_low)} - {format_currency(realistic_high)}"
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/train')
def train_page():
    """Page to trigger model training"""
    return """
    <h1>Train Models</h1>
    <p>Run this command in your terminal:</p>
    <code>python train_and_save_models.py</code>
    <br><br>
    <a href="/">Back to Home</a>
    """

@app.route('/api/models')
def get_models():
    """API endpoint to get available models"""
    return jsonify({
        'models': list(models.keys()),
        'unique_values': unique_values
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
