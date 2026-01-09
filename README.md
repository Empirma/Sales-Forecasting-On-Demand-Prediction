# Sales Forecasting & Demand Prediction System

A comprehensive machine learning-powered desktop application for forecasting sales, analyzing product demand, visualizing data, and retraining predictive models. Built with PyQt6 for a modern, user-friendly interface that enables business users to make data-driven decisions.

## Overview

This graduation project provides an end-to-end solution for retail and e-commerce businesses to:
- Predict future sales based on marketing budgets
- Forecast product demand across different categories
- Visualize historical trends and patterns
- Retrain models with new data for continuous improvement
- Export predictions and analytics for business intelligence

## Key Features

### 1. **Sales Forecasting**
- Upload marketing budget data in CSV format
- Generate sales forecasts using Random Forest and SARIMAX models
- View comprehensive metrics:
  - Average daily sales
  - Total projected sales
  - Peak sales periods
- Interactive visualizations with confidence intervals

### 2. **Demand Product Analysis**
- Forecast demand for specific product categories
- Select products and forecast start dates
- Analyze demand patterns:
  - Predicted demand trends
  - Average daily demand
  - Total demand over period
  - Peak demand identification

### 3. **Data Visualization Dashboard**
- **General Graphs**: Bar plots, line plots, pie charts, box plots, histograms, scatter plots
- **Specialized Analyses**:
  - Top 20 best-selling products
  - Order size and value analysis
  - Seasonal revenue patterns by year
  - General seasonal trends

### 4. **Model Retraining**
- Upload new sales and marketing data
- Automatic data versioning and merging
- Feature engineering pipeline:
  - Data preprocessing
  - Time series feature extraction
  - Lag features and rolling statistics
- Train both sales and demand forecasting models
- MLflow experiment tracking

---

## Installation and Setup

### Prerequisites
- **Python**: 3.10 or higher
- **Operating System**: Windows
- **Dependencies**: See `requirments.txt`

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Graduation Project FInal 2"
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirments.txt
   ```

4. **Run the application**
   ```bash
   python app_qt.py
   ```

---

## Usage Guide

### Running the Desktop Application

```bash
python app_qt.py
```

### Using the Main Pages

#### **Forecasting Sales**
1. Navigate to "Forecasting Sales" from the sidebar
2. Upload marketing budget CSV file
3. Review data preview and date range
4. Click "Generate Forecast"
5. View predictions, metrics, and charts

#### **Demand Product Analysis**
1. Select "Demand Product" from sidebar
2. Choose product category from dropdown
3. Set forecast start date
4. Click "Generate Forecast"
5. Analyze demand predictions and visualizations

#### **Visualizations**
1. Open "Visualizations" page
2. Choose between General Graphs or Specified Analyses
3. Configure graph settings (columns, chart type)
4. Generate and explore visualizations

#### **Retraining Models**
1. Go to "Retraining Models" page
2. Upload new sales data CSV
3. Upload marketing data CSV (optional)
4. Click "Train Models"
5. Monitor training progress and metrics
6. Models saved automatically in `models/` directory

---

## Project Structure

```
Graduation Project FInal 2/
├── app_qt.py                       # Main PyQt6 desktop application
├── main.py                         # Data preprocessing pipeline script
├── build_exe.py                    # Build script for standalone executable
├── Main_Project.ipynb              # Jupyter notebook for exploration
├── Sales_Forecasting_App.spec      # PyInstaller configuration
├── requirments.txt                 # Python dependencies
├── README.md                       # This file
│
├── utils/                          # Utility modules
│   ├── models_functions.py         # ML model training and prediction
│   ├── preprocessing.py            # Data preprocessing pipeline
│   ├── EDA.py                      # Exploratory data analysis functions
│   └── __pycache__/
│
├── data/                           # Data files
│   ├── depi_v0.csv                 # Raw data
│   ├── depi_ungrouped.csv          # Cleaned ungrouped data
│   ├── depi_grouped.csv            # Grouped aggregated data
│   ├── depi_time_series.csv        # Time series featured data
│   ├── marketing_data.csv          # Marketing budget data
│   └── marketing_test.csv          # Test marketing data
│
├── models/                         # Saved ML models
│   └── mlruns/                     # MLflow experiment tracking
│
├── imgs/                           # Generated visualizations
│   ├── demand_models_comparison/
│   ├── sales_models_comparison/
│   ├── models_comparison/
│   └── random_forest_performance/
│
├── mlruns/                         # MLflow tracking files
├── catboost_info/                  # CatBoost training logs
├── build/                          # PyInstaller build artifacts
└── __pycache__/                    # Python cache files
```

---

## Technologies & Libraries

### **Core Technologies**
- **PyQt6**: Modern desktop GUI framework
- **Python 3.10+**: Programming language

### **Machine Learning & Data Science**
- **Scikit-learn**: Machine learning algorithms (Random Forest)
- **Statsmodels**: Time series analysis (SARIMAX)
- **MLflow**: Experiment tracking and model management
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### **Visualization**
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical visualizations

### **Additional Libraries**
- **Joblib**: Model serialization
- **Holidays**: Holiday calendars for feature engineering
- **PyInstaller**: Standalone executable creation

---

## Building Standalone Executable

To create a standalone `.exe` application for Windows:

```bash
python build_exe.py
```

The executable will be created as `Sales_Forecasting_App.exe` in the project root.

### Distribution
When distributing, include:
- `Sales_Forecasting_App.exe`
- `models/` folder
- `mlruns/` folder

These should be placed in the same directory for the app to function correctly.

---

## Data Pipeline

The data preprocessing pipeline (`main.py`) performs the following steps:

1. **Load Data**: Read raw CSV data
2. **Investigate Data**: Analyze structure and quality
3. **Handle Unneeded Columns**: Remove irrelevant features
4. **Handle Missing Values**: Clean missing columns and rows
5. **Group Data**: Aggregate by date for time series
6. **Feature Engineering**: Create derived features
7. **Time Series Features**: Add lag features, rolling statistics, date components
8. **Save Processed Data**: Export to CSV files

Run the pipeline:
```bash
python main.py
```

---

## Machine Learning Models

### **Sales Forecasting**
- **Random Forest Regressor**: Captures non-linear patterns
- **SARIMAX**: Seasonal ARIMA for time series patterns
- **Features**: Date components, lag features, rolling statistics, marketing budget

### **Demand Forecasting**
- **Random Forest Regressor**: Product-specific demand prediction
- **Features**: Date components, lag features, rolling statistics, product categories

### **Model Evaluation Metrics**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

---

## Future Enhancements

- [ ] Add support for multiple model algorithms (XGBoost, LightGBM, Neural Networks)
- [ ] Implement automated hyperparameter tuning
- [ ] Export forecasts to Excel/CSV with one click
- [ ] Add real-time data integration APIs
- [ ] Implement A/B testing for model comparison
- [ ] Add anomaly detection for unusual patterns
- [ ] Multi-language support
- [ ] Cloud deployment option (AWS, Azure, GCP)
- [ ] Mobile application companion
- [ ] Advanced analytics dashboard with KPIs

---
