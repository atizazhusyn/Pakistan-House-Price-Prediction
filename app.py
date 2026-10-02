#!/usr/bin/env python3
"""
Pakistan House Price Prediction Web App
A Streamlit web application for predicting house prices in Pakistan using multiple ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Pakistan House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5em;
        margin-bottom: 30px;
    }
    .prediction-card {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 10px 0;
    }
    .price-display {
        font-size: 1.8em;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
    }
    .sidebar-info {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_preprocessing():
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

    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        st.error("Please run 'python train_and_save_models.py' first to train and save the models.")
        return None, None, None, None

def preprocess_input(property_type, location, city, purpose, baths, beds, area, encoders, scalers):
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

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Pakistan House Price Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### 📊 About This App")
        st.markdown("""
        This application predicts house prices in Pakistan using multiple machine learning models:

        - **Decision Tree Regressor**
        - **Random Forest Regressor**
        - **Gradient Boosting Regressor**
        - **XGBoost Regressor** (if available)

        The predictions are based on property features like type, location, city, number of bathrooms/bedrooms, and area.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 🔧 How to Use")
        st.markdown("""
        1. Fill in the property details in the form
        2. Click 'Predict Price' to get estimates
        3. View predictions from all available models
        4. Consider the price range for realistic expectations
        """)

    # Load models and preprocessing objects
    models, encoders, scalers, unique_values = load_models_and_preprocessing()

    if models is None:
        st.error("❌ Models not found! Please run the training script first.")
        st.code("python train_and_save_models.py")
        return

    st.success(f"✅ Loaded {len(models)} trained models successfully!")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🏡 Property Details")

        # Create form
        with st.form("prediction_form"):
            st.markdown("**Property Information**")

            # Property Type
            property_type = st.selectbox(
                "Property Type",
                unique_values['property_types'],
                help="Select the type of property (House, Flat, etc.)"
            )

            # Location
            location = st.selectbox(
                "Location/Area",
                unique_values['locations'],
                help="Select the specific location or area"
            )

            # City
            city = st.selectbox(
                "City",
                unique_values['cities'],
                help="Select the city"
            )

            # Purpose
            purpose = st.selectbox(
                "Purpose",
                unique_values['purposes'],
                help="Purpose of the property"
            )

            st.markdown("**Property Specifications**")

            # Numerical inputs
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                baths = st.slider(
                    "Bathrooms",
                    min_value=unique_values['baths_range'][0],
                    max_value=unique_values['baths_range'][1],
                    value=2,
                    help="Number of bathrooms"
                )

            with col_b:
                beds = st.slider(
                    "Bedrooms",
                    min_value=unique_values['beds_range'][0],
                    max_value=unique_values['beds_range'][1],
                    value=2,
                    help="Number of bedrooms"
                )

            with col_c:
                area = st.number_input(
                    "Area (sq ft)",
                    min_value=float(unique_values['area_range'][0]),
                    max_value=float(unique_values['area_range'][1]),
                    value=1000.0,
                    step=100.0,
                    help="Area in square feet"
                )

            # Submit button
            submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

    with col2:
        st.markdown("### 📈 Quick Stats")
        st.metric("Available Models", len(models))
        st.metric("Cities Covered", len(unique_values['cities']))
        st.metric("Locations Covered", len(unique_values['locations']))

        # Model performance preview
        st.markdown("### 🎯 Model Performance")
        st.info("Models trained on ~49K properties with R² scores ranging from 0.81-0.87")

    # Prediction logic
    if submitted:
        try:
            # Preprocess input
            input_data = preprocess_input(
                property_type, location, city, purpose,
                baths, beds, area, encoders, scalers
            )

            st.markdown("---")
            st.markdown("### 💰 Price Predictions")

            # Get predictions from all models
            predictions = {}
            for model_name, model in models.items():
                prediction = model.predict(input_data)[0]
                predictions[model_name] = prediction

            # Display predictions
            cols = st.columns(len(predictions))

            for i, (model_name, price) in enumerate(predictions.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4 style="text-align: center; margin-bottom: 10px;">{model_name}</h4>
                        <div class="price-display">{format_currency(price)}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Summary statistics
            prices = list(predictions.values())
            avg_price = np.mean(prices)
            min_price = min(prices)
            max_price = max(prices)

            st.markdown("### 📊 Prediction Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Average Price", format_currency(avg_price))

            with col2:
                st.metric("Price Range", f"{format_currency(min_price)} - {format_currency(max_price)}")

            with col3:
                price_range = max_price - min_price
                st.metric("Range Spread", format_currency(price_range))

            # Realistic price range (13% margin like in original)
            margin = 0.13
            realistic_low = avg_price * (1 - margin)
            realistic_high = avg_price * (1 + margin)

            st.success(f"💡 **Realistic Price Range:** {format_currency(realistic_low)} - {format_currency(realistic_high)}")

            st.info("""
            **Note:** House prices can vary significantly based on market conditions,
            property condition, and other factors. Consider this as a rough estimate
            and consult with local real estate experts for accurate valuations.
            """)

        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.error("Please check your input values and try again.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit | Pakistan House Price Prediction Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
