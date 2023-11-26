import pandas as pd
import numpy as np 

def load_dataset(file_path):
    try:
        data = pd.read_excel(file_path)
        assert all(col in data.columns for col in ["product_id", "day", "product_sold", "price", "rating", "total_rating", "status"])
        return data
    except Exception as _:
      print("Error getting dataset")
      return None
    

def calculate_customer_value(data):
    # Sort data by 'product_id' and 'day' to ensure chronological order
    data.sort_values(by=['product_id', 'day'], inplace=True)

    # Calculate the change in 'product_sold' from the previous day
    data['customer_that_day'] = data.groupby('product_id')['product_sold'].diff().fillna(0)

    return data

