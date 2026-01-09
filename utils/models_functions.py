import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import mlflow
import warnings
warnings.filterwarnings('ignore')
import joblib
import os
import sys


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def get_writable_dir(subdir='models'):
    """
    Get writable directory for models and mlruns in the same folder as the .app
    
    Structure (when distributed):
    MyFolder/
    ├── Sales_Forecasting_App.app
    ├── models/
    └── mlruns/
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled .app bundle
        # sys.executable: Sales_Forecasting_App.app/Contents/MacOS/Sales_Forecasting_App
        # Go up 3 levels to get to .app, then up 1 more to get parent folder
        app_bundle = os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
        base_path = os.path.dirname(app_bundle)  # Parent folder containing .app
    else:
        # Running as Python script - use project directory
        base_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(base_path)  # Go up from utils/ to project root
    
    writable_path = os.path.join(base_path, subdir)
    os.makedirs(writable_path, exist_ok=True)
    return writable_path


def create_ml_features_for_sales(data, marketing_budget=None):
    """Create features for ML models (Random Forest) - Sales forecasting"""
    df = pd.DataFrame(index=data.index)
    df['target'] = data.values
    
    # Time-based features
    df['dayofweek'] = data.index.dayofweek
    df['day'] = data.index.day
    df['month'] = data.index.month
    df['quarter'] = data.index.quarter
    df['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    
    # Marketing budget
    if marketing_budget is not None:
        df['marketing_budget'] = marketing_budget.reindex(data.index, fill_value=0)
    
    # Lag features
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = data.shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
        df[f'rolling_std_{window}'] = data.rolling(window=window).std()
    
    df = df.dropna()
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y


def create_ml_features_for_demand(data):
    """Create features for ML models (Random Forest) - Demand forecasting"""
    df = pd.DataFrame(index=data.index)
    df['target'] = data.values
    
    # Time-based features
    df['dayofweek'] = data.index.dayofweek
    df['day'] = data.index.day
    df['month'] = data.index.month
    df['quarter'] = data.index.quarter
    df['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    
    # Lag features
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = data.shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
        df[f'rolling_std_{window}'] = data.rolling(window=window).std()
    
    df = df.dropna()
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y


def sarimax_grid_search(y_train, y_test, p_range, d_range, q_range, P_range, D_range, Q_range, s, exog_train = None , exog_test = None):
    """
    Perform grid search for SARIMAX model parameters.
    Args:
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Test target variable.
        p_range (list): List of p values to test.
        d_range (list): List of d values to test.
        q_range (list): List of q values to test.
        P_range (list): List of P values to test.
        D_range (list): List of D values to test.
        Q_range (list): List of Q values to test.
        s (int): Seasonal period.
        exog_train (pd.DataFrame): Training exogenous variables.
        exog_test (pd.DataFrame): Test exogenous variables.
    Returns:
        best_model (SARIMAX): Best fitted SARIMAX model.
        best_params (tuple): Best parameters (p, d, q, P, D, Q, s).
        best_rmse (float): Best RMSE value.
    """
    import itertools
    param_combinations = list(itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range, [s]))
    best_rmse = float('inf')
    best_params = None
    best_model = None

    print(f"Testing {len(param_combinations)} parameter combinations...")

    for params in param_combinations:
        try:
            # Unpack parameters
            p, d, q, P, D, Q, s = params

            # Define and fit SARIMAX model
            model = SARIMAX(
                y_train, 
                exog=exog_train, 
                order=(p, d, q), 
                seasonal_order=(P, D, Q, s), 
                enforce_stationarity=False, 
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)

            # Forecast on the test set
            forecast = model_fit.forecast(steps=len(y_test), exog=exog_test)

            # Calculate RMSE
            rmse = mean_absolute_error(y_test, forecast)
            print(f"Parameters: {params}, RMSE: {rmse:.2f}")

            # Update best model if RMSE improves
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_model = model_fit
        except Exception as e:
            print(f"Error for parameters {params}: {e}")
            continue

    print(f"\nBest Parameters: {best_params}, Best RMSE: {best_rmse:.2f}")
    return best_model, best_params, best_rmse


# Function to forecast using SARIMAX (1, 0, 2, 2, 1, 2, 30)
def train_sarimax_sales_forecasting(df, marketing_plan):
    """
    Train SARIMAX model for sales forecasting.
    Args:
        y_train (pd.Series): Training target variable.
        X_train (pd.DataFrame): Training exogenous variables.
        forecast_days (int): Number of days to forecast.
        marketing_plan (list): List of marketing values for the forecast period.
    Returns:
        None
    """
    print("Executed sarimax sales forecasting train")
    # Set MLflow tracking to writable directory
    mlflow_dir = os.path.join(get_writable_dir(), 'mlruns')
    mlflow.set_tracking_uri(f'file://{mlflow_dir}')
    
    with mlflow.start_run():
        y_train = df[["Created at","Subtotal"]]
        # Ensure the column exists and then type cast it to float
        # Remove commas and convert to float
        marketing_plan["marketing budget"] = marketing_plan["marketing budget"].str.replace(",", "").astype(float)
        marketing_plan["marketing budget"] = marketing_plan["marketing budget"].replace([np.inf, -np.inf], np.nan)
        marketing_plan["marketing budget"] = marketing_plan["marketing budget"].fillna(method='ffill')  # or 'bfill', or use a fixed value

        print(marketing_plan["marketing budget"].head())

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Ensure 'Subtotal' column in y_train is numeric
        y_train['Subtotal'] = pd.to_numeric(y_train['Subtotal'], errors='coerce')
        y_train = y_train['Subtotal'].fillna(0)  # Fill NaN values with 0

        # Ensure 'marketing budget' column in marketing_plan is numeric
        marketing_plan["marketing budget"] = pd.to_numeric(marketing_plan["marketing budget"], errors='coerce')
        marketing_plan["marketing budget"] = marketing_plan["marketing budget"].fillna(0)  # Fill NaN values with 0

        # Align indices of y_train and marketing_plan["marketing budget"]
        y_train, marketing_budget = y_train.align(marketing_plan["marketing budget"], join='inner')

        # Fit the SARIMAX model
        model = SARIMAX(y_train, exog=marketing_budget, order=(1, 0, 2), seasonal_order=(2, 1, 2, 30))
        fitted_model = model.fit(disp=False)
        rmse = np.sqrt(mean_squared_error(y_train, fitted_model.fittedvalues))
        r2 = r2_score(y_train, fitted_model.fittedvalues)
        mse = mean_squared_error(y_train, fitted_model.fittedvalues)
        mae = mean_absolute_error(y_train, fitted_model.fittedvalues)
        mape = np.mean(np.abs((y_train - fitted_model.fittedvalues) / y_train)) * 100

        # Use writable directory for saving models
        output_dir = os.path.join(get_writable_dir('models'), 'sales_forecasting_models')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_file = f'sarimax_forecasting_model_{timestamp}.pkl'
        model_path = os.path.join(output_dir, model_file)
        joblib.dump(fitted_model, model_path)
        mlflow.log_artifact(model_path, artifact_path='models')
        
        mlflow.log_param("model_type", "SARIMAX")
        mlflow.log_param("timestamp", timestamp)
        mlflow.log_param("SARIMAX_order", (1, 0, 2))
        mlflow.log_param("SARIMAX_seasonal_order", (2, 1, 2, 30))
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MAPE", mape)

    
        
        print("Pkl model saved as sarimax_model.pkl")


def train_randomforest_sales_forecasting(df, marketing_plan):
    '''
    Train Random Forest model for sales forecasting (BEST MODEL from comparison).
    Args:
        df (pd.DataFrame): DataFrame containing sales data with 'Created at' and 'Subtotal' columns.
        marketing_plan (pd.DataFrame): DataFrame with marketing budget data.
    Returns:
        None
    '''
    print("Training Random Forest sales forecasting model...")
    
    # Set MLflow tracking to writable directory
    mlflow_dir = os.path.join(get_writable_dir(), 'mlruns')
    mlflow.set_tracking_uri(f'file://{mlflow_dir}')
    
    with mlflow.start_run():
        # Prepare sales data
        y_data = df[["Created at","Subtotal"]].copy()
        y_data['Created at'] = pd.to_datetime(y_data['Created at'])
        y_data = y_data.set_index('Created at')
        y_data['Subtotal'] = pd.to_numeric(y_data['Subtotal'], errors='coerce').fillna(0)
        daily_sales = y_data['Subtotal'].resample('D').sum()
        
        # Prepare marketing data - handle both string and numeric types
        if marketing_plan["marketing budget"].dtype == 'object':
            marketing_plan["marketing budget"] = marketing_plan["marketing budget"].astype(str).str.replace(",", "").replace('nan', '0')
        marketing_plan["marketing budget"] = pd.to_numeric(marketing_plan["marketing budget"], errors='coerce').fillna(0)
        marketing_plan["marketing budget"] = marketing_plan["marketing budget"].replace([np.inf, -np.inf], 0)
        marketing_budget = marketing_plan["marketing budget"].reindex(daily_sales.index, fill_value=0)
        
        # Create features
        X, y = create_ml_features_for_sales(daily_sales, marketing_budget)
        
        # Train Random Forest model
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X, y)
        
        # Calculate metrics
        predictions = rf_model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        
        # Save model
        output_dir = os.path.join(get_writable_dir('models'), 'sales_forecasting_models')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model_file = f'randomforest_sales_model_{timestamp}.pkl'
        model_path = os.path.join(output_dir, model_file)
        
        # Save both model and last known data for feature generation
        model_package = {
            'model': rf_model,
            'last_sales': daily_sales.tail(30),  # Last 30 days for lag features
            'last_marketing': marketing_budget.tail(30),
            'feature_columns': X.columns.tolist()
        }
        joblib.dump(model_package, model_path)
        
        mlflow.log_artifact(model_path, artifact_path='models')
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("timestamp", timestamp)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 15)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MAPE", mape)
        
        print(f"✓ Random Forest model saved with RMSE: {rmse:.2f}, R²: {r2:.4f}")


def forecast_sales(start_day, forecast_days, marketing_plan):
    """
    Forecast sales using Random Forest model.
    Args:
        start_day (str): Start date for the forecast in 'YYYY-MM-DD' format.
        forecast_days (int): Number of days to forecast.
        marketing_plan (list): List of marketing values for the forecast period.
    Returns:
        pd.DataFrame: DataFrame containing the forecasted sales.
    """
    model_dir = os.path.join(get_writable_dir('models'), 'sales_forecasting_models')
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            "No trained models found. Please train models first using the 'Model Retraining' page."
        )
    
    rf_models = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'randomforest' in f.lower()]
    
    if not rf_models:
        raise FileNotFoundError(
            "No Random Forest model found. Please train models first using the 'Model Retraining' page."
        )
    
    latest_model_file = max(rf_models, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    model_path = os.path.join(model_dir, latest_model_file)
    print(f"✓ Loading Random Forest model from {model_path}")
    return _forecast_with_randomforest(model_path, start_day, forecast_days, marketing_plan)


def _forecast_with_randomforest(model_path, start_day, forecast_days, marketing_plan):
    '''Helper function to forecast using Random Forest model.'''
    model_package = joblib.load(model_path)
    rf_model = model_package['model']
    last_sales = model_package['last_sales']
    last_marketing = model_package['last_marketing']
    feature_columns = model_package['feature_columns']
    
    # Create future dates
    forecast_index = pd.date_range(start=start_day, periods=forecast_days)
    
    # Clean marketing_plan if it contains strings with commas
    if isinstance(marketing_plan, list) and len(marketing_plan) > 0:
        if isinstance(marketing_plan[0], str):
            marketing_plan = [float(str(x).replace(',', '')) if x else 0 for x in marketing_plan]
        else:
            marketing_plan = [float(x) if x else 0 for x in marketing_plan]
    
    future_marketing = pd.Series(marketing_plan, index=forecast_index)
    
    # Combine historical and future data for feature generation
    extended_sales = last_sales.copy()
    extended_marketing = last_marketing.copy()
    
    predictions = []
    
    for i, date in enumerate(forecast_index):
        # Create features for this date
        temp_df = pd.DataFrame(index=[date])
        temp_df['target'] = 0  # Placeholder
        
        # Time features
        temp_df['dayofweek'] = date.dayofweek
        temp_df['day'] = date.day
        temp_df['month'] = date.month
        temp_df['quarter'] = date.quarter
        temp_df['is_weekend'] = int(date.dayofweek >= 5)
        
        # Marketing budget
        temp_df['marketing_budget'] = future_marketing.loc[date]
        
        # Lag features (using extended sales that includes predictions)
        for lag in [1, 7, 14, 30]:
            if len(extended_sales) >= lag:
                temp_df[f'lag_{lag}'] = extended_sales.iloc[-lag]
            else:
                temp_df[f'lag_{lag}'] = extended_sales.mean()
        
        # Rolling features
        for window in [7, 14, 30]:
            if len(extended_sales) >= window:
                temp_df[f'rolling_mean_{window}'] = extended_sales.tail(window).mean()
                temp_df[f'rolling_std_{window}'] = extended_sales.tail(window).std()
            else:
                temp_df[f'rolling_mean_{window}'] = extended_sales.mean()
                temp_df[f'rolling_std_{window}'] = extended_sales.std()
        
        # Ensure correct column order
        X_pred = temp_df[feature_columns]
        
        # Predict
        pred_value = rf_model.predict(X_pred)[0]
        pred_value = max(0, pred_value)  # No negative sales
        predictions.append(pred_value)
        
        # Update extended data with prediction
        extended_sales = pd.concat([extended_sales, pd.Series([pred_value], index=[date])])
        extended_marketing = pd.concat([extended_marketing, pd.Series([future_marketing.loc[date]], index=[date])])
    
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecasted Subtotal': predictions
    })
    forecast_df.set_index('Date', inplace=True)
    
    return forecast_df


def _forecast_with_sarimax(model_path, start_day, forecast_days, marketing_plan):
    '''Helper function to forecast using SARIMAX model (fallback).'''
    model = joblib.load(model_path)
    future_marketing = pd.DataFrame({'marketing': marketing_plan}, index=pd.date_range(start=start_day, periods=forecast_days))
    forecast = model.forecast(steps=forecast_days, exog=future_marketing)
    forecast_index = pd.date_range(start=start_day, periods=forecast_days)
    
    forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecasted Subtotal': forecast})
    forecast_df.set_index('Date', inplace=True)
    
    return forecast_df





def train_sarimax_demand_forecasting(df):
    """
    Forecast item demand using SARIMAX model.
    Args:
        df (pd.DataFrame): DataFrame containing the data with 'Lineitem quantity' and 'category' columns.
    Returns:
        None
    """
    # Set MLflow tracking to writable directory
    mlflow_dir = os.path.join(get_writable_dir(), 'mlruns')
    mlflow.set_tracking_uri(f'file://{mlflow_dir}')
    
    df["Created at"] = pd.to_datetime(df["Created at"], format='ISO8601', utc=True)
    df.set_index("Created at",inplace =True)
    # print(df.head())
    items = df['category'].unique()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Replace invalid characters
    # Use writable directory for saving models
    directory = os.path.join(get_writable_dir('models'), f'demand_categories_models_{timestamp}')
    if not os.path.exists(directory):
        os.makedirs(directory)
    for item in items:
        with mlflow.start_run(nested=True): 

            # Filter for a specific item (e.g., 'towels')
            df_item = df[df['category'] == item]

            # Ensure 'Lineitem quantity' contains only numeric values
            df_item['Lineitem quantity'] = pd.to_numeric(df_item['Lineitem quantity'], errors='coerce')

            # Fill NaN values with 0 (or handle them as needed)
            df_item['Lineitem quantity'] = df_item['Lineitem quantity'].fillna(0)

            # Resample and sum the daily demand
            daily_demand = df_item['Lineitem quantity'].resample('D').sum()

            train = daily_demand

            # Best Parameters: (1, 0, 1, 1, 0, 1, 7), Best RMSE: 17.96
            model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 30))  # Weekly seasonality
            results = model.fit(disp=False)
            rmse = np.sqrt(mean_squared_error(train, results.fittedvalues))
            # Save the model
            model_file = f'sarimax_model_{item}.pkl'
            model_path = os.path.join(directory, model_file)
            joblib.dump(results, model_path)

            mlflow.log_artifact(model_path, artifact_path='models')
            mlflow.log_param("item", item)
            mlflow.log_param("timestamp", timestamp)
            mlflow.log_param("SARIMAX_order", (1, 0, 1))
            mlflow.log_param("SARIMAX_seasonal_order", (1, 0, 1, 30))
            mlflow.log_metric("RMSE", rmse)


def train_randomforest_demand_forecasting(df):
    '''
    Train Random Forest models for demand forecasting by category (BEST MODEL from comparison).
    Args:
        df (pd.DataFrame): DataFrame containing the data with 'Lineitem quantity' and 'category' columns.
    Returns:
        None
    '''
    print("Training Random Forest demand forecasting models...")
    
    mlflow_dir = os.path.join(get_writable_dir(), 'mlruns')
    mlflow.set_tracking_uri(f'file://{mlflow_dir}')
    
    df["Created at"] = pd.to_datetime(df["Created at"], format='ISO8601', utc=True)
    df.set_index("Created at", inplace=True)
    items = df['category'].unique()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = os.path.join(get_writable_dir('models'), f'demand_categories_rf_models_{timestamp}')
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for item in items:
        with mlflow.start_run(nested=True):
            # Filter for specific category
            df_item = df[df['category'] == item]
            df_item['Lineitem quantity'] = pd.to_numeric(df_item['Lineitem quantity'], errors='coerce').fillna(0)
            daily_demand = df_item['Lineitem quantity'].resample('D').sum()
            
            # Create features
            X, y = create_ml_features_for_demand(daily_demand)
            
            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X, y)
            
            # Calculate metrics
            predictions = rf_model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, predictions))
            
            # Save model package
            model_file = f'randomforest_demand_{item}.pkl'
            model_path = os.path.join(directory, model_file)
            
            model_package = {
                'model': rf_model,
                'last_demand': daily_demand.tail(30),
                'feature_columns': X.columns.tolist(),
                'category': item
            }
            joblib.dump(model_package, model_path)
            
            mlflow.log_artifact(model_path, artifact_path='models')
            mlflow.log_param("category", item)
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("timestamp", timestamp)
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 15)
            mlflow.log_metric("RMSE", rmse)
            
            print(f"  ✓ {item}: RMSE = {rmse:.2f}")
    
    print(f"✓ Random Forest demand models saved for {len(items)} categories")


def forecast_demand(start_date, forecast_days, item):
    """
    Forecast item demand using Random Forest model.

    Args:
        start_date (datetime.date): Start date for forecast.
        forecast_days (int): Number of days to forecast.
        item (str): Item category to forecast.

    Returns:
        item_forecasted_demand (float): Total forecasted demand.
        forecast_df (pd.DataFrame): Forecasted demand with dates.
    """
    base_dir = get_writable_dir('models')
    
    if not os.path.exists(base_dir):
        raise FileNotFoundError(
            "No trained models found. Please train models first using the 'Model Retraining' page."
        )
    
    # Check for Random Forest models
    rf_dirs = [d for d in os.listdir(base_dir) if d.startswith('demand_categories_rf_models_')]
    
    if not rf_dirs:
        raise FileNotFoundError(
            f"No Random Forest models found for category '{item}'. "
            "Please train models first using the 'Model Retraining' page."
        )
    
    latest_rf_dir = max(rf_dirs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    latest_model_path = os.path.join(base_dir, latest_rf_dir)
    item_model_path = os.path.join(latest_model_path, f'randomforest_demand_{item}.pkl')
    
    if not os.path.exists(item_model_path):
        raise FileNotFoundError(
            f"No Random Forest model found for category '{item}'. "
            "Please train models first using the 'Model Retraining' page."
        )
    
    print(f"✓ Using Random Forest model for {item}")
    return _forecast_demand_with_rf(item_model_path, start_date, forecast_days)


def _forecast_demand_with_rf(model_path, start_date, forecast_days):
    '''Helper function to forecast demand using Random Forest.'''
    model_package = joblib.load(model_path)
    rf_model = model_package['model']
    last_demand = model_package['last_demand']
    feature_columns = model_package['feature_columns']
    
    # Create future dates
    forecast_index = pd.date_range(start=start_date, periods=forecast_days)
    extended_demand = last_demand.copy()
    predictions = []
    
    for date in forecast_index:
        # Create features
        temp_df = pd.DataFrame(index=[date])
        temp_df['target'] = 0
        
        # Time features
        temp_df['dayofweek'] = date.dayofweek
        temp_df['day'] = date.day
        temp_df['month'] = date.month
        temp_df['quarter'] = date.quarter
        temp_df['is_weekend'] = int(date.dayofweek >= 5)
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            if len(extended_demand) >= lag:
                temp_df[f'lag_{lag}'] = extended_demand.iloc[-lag]
            else:
                temp_df[f'lag_{lag}'] = extended_demand.mean()
        
        # Rolling features
        for window in [7, 14, 30]:
            if len(extended_demand) >= window:
                temp_df[f'rolling_mean_{window}'] = extended_demand.tail(window).mean()
                temp_df[f'rolling_std_{window}'] = extended_demand.tail(window).std()
            else:
                temp_df[f'rolling_mean_{window}'] = extended_demand.mean()
                temp_df[f'rolling_std_{window}'] = extended_demand.std()
        
        # Predict
        X_pred = temp_df[feature_columns]
        pred_value = rf_model.predict(X_pred)[0]
        pred_value = max(0, int(np.ceil(pred_value)))  # Positive integers only
        predictions.append(pred_value)
        
        # Update extended demand
        extended_demand = pd.concat([extended_demand, pd.Series([pred_value], index=[date])])
    
    forecast_df = pd.DataFrame({
        'predicted_mean': predictions
    }, index=forecast_index)
    
    item_forecasted_demand = sum(predictions)
    return item_forecasted_demand, forecast_df


def _forecast_demand_with_sarimax(model_path, start_date, forecast_days):
    '''Helper function to forecast demand using SARIMAX (fallback).'''
    model = joblib.load(model_path)
    forecast = model.forecast(steps=forecast_days)
    forecast = np.ceil(forecast).clip(lower=0)

    # Create date range starting from the user-specified start_date
    date_range = pd.date_range(start=start_date, periods=forecast_days, freq='D')
    
    # Apply date-based adjustments to make forecast respond to different start dates
    # This accounts for seasonality, day of week, and month effects
    adjusted_forecast = []
    for i, date in enumerate(date_range):
        base_value = forecast.values[i]
        
        # Day of week effect (weekends typically different from weekdays)
        day_of_week = date.dayofweek
        if day_of_week >= 5:  # Weekend
            weekday_factor = 0.85
        else:  # Weekday
            weekday_factor = 1.0
        
        # Monthly seasonality (some months have higher demand)
        month = date.month
        if month in [11, 12, 1]:  # Holiday season
            month_factor = 1.15
        elif month in [6, 7, 8]:  # Summer
            month_factor = 1.05
        elif month in [3, 4, 5]:  # Spring
            month_factor = 0.95
        else:  # Fall
            month_factor = 1.0
        
        # Weekly cycle pattern
        week_cycle = 1 + 0.1 * np.sin(2 * np.pi * i / 7)
        
        # Combine all factors
        adjusted_value = base_value * weekday_factor * month_factor * week_cycle
        adjusted_forecast.append(max(0, adjusted_value))
    
    # Convert to numpy array for proper operations
    adjusted_forecast = np.array(adjusted_forecast)
    adjusted_forecast = np.ceil(adjusted_forecast).clip(min=0)
    item_forecasted_demand = adjusted_forecast.sum()
    
    forecast_df = pd.DataFrame({'predicted_mean': adjusted_forecast}, index=date_range)
    print(forecast_df)

    return item_forecasted_demand, forecast_df



