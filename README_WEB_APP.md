# 🏠 Pakistan House Price Prediction - Web App

A beautiful web-based interface for predicting house prices in Pakistan using multiple machine learning models.

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train and Save Models
```bash
python train_and_save_models.py
```
This will:
- Train all available ML models (Decision Tree, Random Forest, Gradient Boosting, XGBoost)
- Save models and preprocessing objects to the `models/` directory
- Create necessary files for the web app

### Step 3: Run the Web App
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📊 Features

### 🏡 Property Input Form
- **Property Type**: House, Flat, Penthouse, etc.
- **Location**: Specific area/location in Pakistan
- **City**: Major cities across Pakistan
- **Purpose**: For Sale/For Rent
- **Bathrooms**: Number of bathrooms (slider)
- **Bedrooms**: Number of bedrooms (slider)
- **Area**: Property area in square feet

### 🤖 Multiple ML Models
The app uses ensemble of models for robust predictions:
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor** (if available)

### 📈 Smart Price Estimation
- Individual predictions from each model
- Average price calculation
- Realistic price range (±13% margin)
- Performance metrics for each model

## 🗂️ Project Structure

```
Pakistan-House-Price-Prediction-main/
├── house_prices.csv              # Raw dataset
├── train_and_save_models.py       # Model training script
├── app.py                        # Streamlit web application
├── run_analysis.py               # Original analysis script
├── requirements.txt              # Python dependencies
├── models/                       # Saved models and preprocessing objects
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── xgboost_model.pkl
│   ├── *_encoder.pkl            # Label encoders
│   ├── *_scaler.pkl             # Standard scalers
│   └── unique_values.pkl        # UI configuration
└── README_WEB_APP.md            # This file
```

## 🎨 Web App Features

### Beautiful UI
- Modern, responsive design
- Mobile-friendly interface
- Intuitive form layout
- Real-time predictions

### Interactive Elements
- Dropdown menus for categorical inputs
- Sliders for numerical inputs
- Form validation
- Loading states

### Results Display
- Individual model predictions
- Visual price cards
- Summary statistics
- Realistic price ranges

## 🔧 Technical Details

### Data Preprocessing
- Label encoding for categorical features
- Standard scaling for numerical features
- Feature engineering (Marla to square feet conversion)

### Model Training
- 80/20 train-test split
- Random state for reproducibility
- Evaluation metrics: MAE, MSE, RMSE, R²

### Web App Architecture
- **Frontend**: Streamlit
- **Backend**: Scikit-learn models
- **Data Storage**: Pickle serialization
- **Styling**: Custom CSS

## 🚨 Troubleshooting

### Models Not Found Error
If you see "Models not found" error:
```bash
# Make sure you've run the training script
python train_and_save_models.py
```

### XGBoost Not Available
XGBoost may not work on some Windows systems due to OpenMP dependency. The app will gracefully skip XGBoost and use the other three models.

### Port Already in Use
If port 8501 is busy:
```bash
streamlit run app.py --server.port 8502
```

## 📝 Usage Tips

1. **Realistic Inputs**: Use realistic values for bathrooms, bedrooms, and area based on property type
2. **Price Ranges**: Consider the suggested price range rather than exact predictions
3. **Market Factors**: Remember that actual prices depend on many external factors
4. **Model Ensemble**: The average prediction from multiple models is often more reliable

## 🤝 Contributing

Feel free to improve the web app by:
- Adding more models
- Improving the UI design
- Adding more features (maps, charts, etc.)
- Optimizing performance

## 📄 License

This project is open source. Feel free to use and modify as needed.

---

**Happy house hunting! 🏠✨**
