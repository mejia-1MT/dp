import pandas as pd
import numpy as np
from itertools import groupby
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

    
# Function to adjust the sequence when sales jump by 100 but are rounded to nearest 1000
def adjust_sales_sequence(df):
    for pid in df['product_id'].unique():
        product_data = df[df['product_id'] == pid]
        sales = product_data['product_sold'].tolist()

        for i in range(1, len(sales)):
            if sales[i] - sales[i - 1] > 100:
                diff = sales[i] - sales[i - 1]
                num_days = i - sales.index(sales[i - 1]) - 1
                increase_per_day = diff / (num_days + 1)

                # Adjust sales for affected days
                for j in range(sales.index(sales[i - 1]) + 1, i):
                    sales[j] = round(sales[j - 1] + increase_per_day)

        # Update the DataFrame with adjusted sales
        df.loc[df['product_id'] == pid, 'product_sold'] = sales

    return df

# Function to predict customer amount based on adjusted data
def predict_units_sold(data):
    # Filter data where product_sold > 1000
    filtered_data = data[data['product_sold'] > 1000].copy()  # Ensure a copy is made

    # Calculate daily units sold ('sold_diff')
    sold_diff_values = filtered_data.groupby('product_id')['product_sold'].diff().fillna(0)
    filtered_data.loc[:, 'sold_diff'] = sold_diff_values

    # Features and target variable
    X = filtered_data[['day', 'price', 'rating', 'total_rating', 'status']]
    y = filtered_data['sold_diff']

    # Initialize the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    rf_model.fit(X, y)

    # Predict daily units sold for filtered data
    predictions = rf_model.predict(X)
    filtered_data.loc[:, 'units_sold'] = predictions

    # Merge predictions back to the original dataframe
    merged_data = data.merge(filtered_data[['product_id', 'day', 'units_sold']], on=['product_id', 'day'], how='left')

    # Update units_sold for product_sold <= 1000
    merged_data.loc[merged_data['product_sold'] <= 1000, 'units_sold'] = merged_data['customer_that_day']

    return merged_data
