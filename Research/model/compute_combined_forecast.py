import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d 
import warnings

def forecast_arima(train_data, product_id):
    product_data = train_data[train_data['product_id'] == product_id].copy()
    product_data['day'] = pd.to_datetime(product_data['day'])
    product_data.set_index('day', inplace=True)
    time_series = product_data['product_sold']

    # Ensure the index is a DateTimeIndex
    time_series.index = pd.date_range(start=time_series.index[0], periods=len(time_series))

    model = ARIMA(time_series, order=(5, 1, 0))

    # Suppress the convergence warnings FIX THIS!!!
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=Warning)
        
        arima_model = model.fit()

    # Forecast using the proper DateTime index
    forecast_index = pd.date_range(start=time_series.index[-1], periods=30)  # Forecasting next 30 days
    forecast = arima_model.forecast(steps=30, index=forecast_index)

    return forecast


def predict_demand(train_data, test_data, features, target):
    X_train = train_data[features]
    y_train = train_data[target]  
    X_test = test_data[features]
    y_test = test_data[target]

    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return predictions, mse


def compute_combined_forecast(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Example features and target for machine learning model
    ml_features = ['price', 'rating', 'total_rating', 'status', 'units_sold']
    ml_target = 'product_sold'

    # Example product ID to perform the forecast
    desired_product_id = 1

    # Get ARIMA forecast for the desired product
    arima_forecast = forecast_arima(train_data, desired_product_id)

    # Get ML-based demand predictions
    ml_predictions, mse = predict_demand(train_data, test_data, ml_features, ml_target)

    # Interpolating ARIMA forecast to match the length of ML predictions
    arima_interpolated = interp1d(np.linspace(0, 1, len(arima_forecast)), arima_forecast)
    arima_forecast_resized = arima_interpolated(np.linspace(0, 1, len(ml_predictions)))

    # Weighted combination of ARIMA and ML predictions
    combined_forecast = 0.7 * arima_forecast_resized + 0.3 * ml_predictions

    return combined_forecast, mse