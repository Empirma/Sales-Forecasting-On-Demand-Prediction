# Sales Forecasting & On-Demand Predictions

This is a **Streamlit** application designed for forecasting sales, analyzing product demand, visualizing data, and retraining predictive models. The app provides an intuitive interface for business users to make data-driven decisions using machine learning models.

## Features

### 1. **Forecasting Sales**
- Upload marketing budget data in CSV format.
- View data previews and date ranges.
- Generate sales forecasts with visualizations and metrics:
  - Average daily sales
  - Total sales
  - Peak sales

### 2. **Demand Product Analysis**
- Analyze demand for specific product categories.
- Select a product and start date to generate forecasts.
- Visualize predicted demand, average daily demand, total demand, and peak demand.

### 3. **Data Visualization Dashboards**
- Explore general graphs or conduct specified analyses:
  - General Graphs:
    - Bar plots, line plots, pie charts, box plots, histograms, and scatter plots.
  - Specified Analyses:
    - Top 20 best-selling products
    - Order size and value analysis
    - Seasonal revenue patterns by year
    - General seasonal patterns

### 4. **Retraining Models**
- Upload new data and marketing data files.
- Automatically handle existing data versions and merge new data.
- Preprocess and engineer features for time series forecasting.
- Retrain models for sales and demand forecasting.

---

## Installation and Setup

### Prerequisites
- Python 3.10+
- Streamlit
- Required Python libraries (see `requirements.txt`)

---

## Usage

### Navigation
Use the **Sidebar** to navigate between the different pages:
- `Forecasting Sales`
- `Demand Product`
- `Visualizations`
- `Retraining Models`

### File Uploads
- Ensure files uploaded are in CSV format and follow the expected schema.

### Data Visualization
- Select the desired analysis type and configure graph settings for visualization.

### Retraining Models
1. Upload new datasets for retraining.
2. Start the retraining process by clicking the "Train Models" button.

---

## File Structure

```
.
├── app.py                    # Main Streamlit application
├── utils/
│   ├── models_functions.py   # Model-related utilities
│   ├── preprocessing.py      # Data preprocessing utilities
│   ├── EDA.py                # Exploratory Data Analysis utilities
├── data/
│   ├── depi_grouped.csv      # Example grouped data
│   ├── depi_ungrouped.csv    # Example ungrouped data
│   ├── depi_time_series.csv  # Example time series data
│   ├── marketing_data.csv    # Example marketing data
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Technologies Used
- **Streamlit**: For the web app interface.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib**: For visualizations.
- **Scikit-learn**: Preprocessing and machine learning utilities.
- **Joblib**: Model persistence.
- **RandomForestRegressor**: For predictive modeling.

---

