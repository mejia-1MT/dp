from data_prep import load_dataset, calculate_customer_value
from regression_analysis import  predict_units_sold
from dqn_agent import DQNAgent
from environment import PricingEnvironment
from train_dqn import train_dqn
import torch.optim as optim
import torch.nn as nn

def main():
    
    file_path = './Shopee-Product-20oz-September-2023-FINAL.xlsx'
    data = load_dataset(file_path)

    # Step 1: Data Preparation
    if data is not None:
        print("**************************")
        with_raw_customers = calculate_customer_value(data.copy())

        

        adjusted_df = predict_units_sold(with_raw_customers)
        print(adjusted_df.drop(columns=["customer_that_day"]).to_string(index=False))
        processed_products_data = adjusted_df.drop(columns=["customer_that_day"])

        initial_price = 1990.0  # float(input("Enter the initial price for the new product: "))

        # Define the price range percentage
        price_range_percentage = 0.2  # ±20% of the initial price

        # Calculate the upper and lower bounds of the price range
        price_lower_bound = initial_price - (initial_price * price_range_percentage)
        price_upper_bound = initial_price + (initial_price * price_range_percentage)

        # Initialize agent with a single continuous output for price
        input_dim = 7  # Your input dimension based on features
        output_dim = 1  # Single continuous output for price
        agent = DQNAgent(input_dim, output_dim, price_lower_bound, price_upper_bound)
        env = PricingEnvironment(processed_products_data, initial_price)  # Initialize with appropriate data

        # Call the training function
        train_dqn(agent, env)
        
    else:
        print("Failed to load the dataset.")
    
    
if __name__ == "__main__":
    main()