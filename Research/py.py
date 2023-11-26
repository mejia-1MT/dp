import pandas as pd
import numpy as np
import random

# Replace 'your_excel_file.xlsx' with the path to your Excel file
file_path = 'Shopee-Product-20oz-September-2023-FINAL.xlsx'


data = pd.read_excel(file_path, header=None)
# Define the number of products
num_products = 25
rows_per_product = 30

# Initialize an empty list to store DataFrames for each product
product_dfs = []

# Loop through the rows in steps of (rows_per_product + 2) to skip title and headers for each product
for i in range(0, len(data), rows_per_product + 2):
    # Create a DataFrame for the current product (skipping the first 2 rows as headers)
    product_data = data.iloc[i + 2:i + rows_per_product + 2]
    product_data.columns = ["day", "product_sold", "price", "rating", "total_rating", "status"]
    product_df = product_data.copy()
    
    # Append the DataFrame to the list of product DataFrames
    product_dfs.append(product_df)

product_df = pd.concat(product_dfs, ignore_index=True)
product_df['product_id'] = np.repeat(range(1, num_products + 1), rows_per_product)

# Add a new product to the dataset
new_product_data = pd.DataFrame({
    'day': [1] * rows_per_product,
    'product_sold': [0] * rows_per_product,
    'price': [120] * rows_per_product,  # Initial price for the new product
    'rating': [0] * rows_per_product,
    'total_rating': [0] * rows_per_product,
    'status': [1] * rows_per_product
})
new_product_data['product_id'] = num_products + 1  # Assign a unique product ID
product_df = pd.concat([product_df, new_product_data], ignore_index=True)

class PricingEnvironment:
    def __init__(self, data):
        self.data = data
        self.num_products = len(data['product_id'].unique())
        self.current_day = 0
        self.current_product = None
        self.done = False
        self.prices = []
        self.state = {}

        # Calculate and set the mean price during initialization
        self.mean_price = data['price'].mean()
        
        
    def _get_state(self):
        return self.data[self.data['product_id'] == self.current_product]

    def reset(self, product_id=None):
        self.current_day = 0
        self.done = False
        self.current_product = product_id or random.randint(1, self.num_products)
        self.prices = list(self._get_state()['price'])  # Convert prices to a list
        
        # Convert 'product_sold' column to a list when constructing self.state
        self.state = {
            'product_sold': list(self._get_state()['product_sold']),
            'price': list(self._get_state()['price']),
            'rating': list(self._get_state()['rating']),
            'total_rating': list(self._get_state()['total_rating']),
            'status': list(self._get_state()['status'])
        }
        
        return self.state
    
    def get_dynamic_price(self, state, new_product_data):
        scaled_product_sold = np.mean(state['product_sold']) / self.num_products
        scaled_price = (np.mean(state['price']) - new_product_data['price'].iloc[0]) / self.mean_price  # Use new product's initial price
        scaled_rating = np.mean(state['rating'])
        scaled_total_rating = np.mean(state['total_rating'])

        dynamic_price = new_product_data['price'].iloc[0] * (scaled_product_sold + scaled_price + scaled_rating + scaled_total_rating)

        return dynamic_price

    def step(self, actions):
        # Take a list of actions (new prices) and transition to the next state
        if self.done:
            raise ValueError("Episode is done, call reset() to start a new episode.")

        # Update prices of products based on actions
        self.prices = actions

        # Move to the next day
        self.current_day += 1

        # If 30 days for the current product are processed, move to the next product
        if self.current_day % 30 == 0:
            self.current_product += 1
            if self.current_product > self.num_products:
                self.done = True
                return None, 0, True, {}
            self.prices = list(self._get_state()['price'])  # Reset prices for the new product
        
        # Get the next state (products' data for the next day)
        self.state = self._get_state()
        
        # Ensure that both product_sold and prices are lists before using zip
        product_sold_list = list(self.state['product_sold'])
        prices_list = list(self.prices)
        revenue = sum(product_sold * price for product_sold, price in zip(product_sold_list, prices_list))

        return self.state, revenue, self.done, {}

   

# Calls
env = PricingEnvironment(product_df)
initial_state = env.reset()

import torch
import torch.nn as nn
import torch.optim as optim
import random


class DQNAgent(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNAgent, self).__init__()
        self.fc = nn.Linear(input_size, 150)  # Adjust the output size to match your input size
        self.out = nn.Linear(150, output_size)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.out(x)

class DQNAgentWrapper:
    def __init__(self, input_size, output_size, epsilon=0.1, lr=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = DQNAgent(input_size, output_size).to(self.device)
        self.target_model = DQNAgent(input_size, output_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.steps = 0 


    def preprocess_state(self, state):
        # Convert string values to numerical format and return the processed state
        processed_state = {
            'product_sold': [float(value) for value in state['product_sold']],  
            'price': [float(value) for value in state['price']],  
            'rating': [float(value) for value in state['rating']], 
            'total_rating': [float(value) for value in state['total_rating']],  
            'status': [1.0 if status == 'Available' else 0.0 for status in state['status']]
        }

        # Flatten the dictionary values into a list
        state_values = [value for values_list in processed_state.values() for value in values_list]

        return state_values
        
        
    def choose_prices(self, state):
        # Preprocess the state to convert string values to numerical format
        processed_state = self.preprocess_state(state)

        # Choose prices based on the preprocessed state using the trained DQN agent
        with torch.no_grad():
            state_tensor = torch.tensor(processed_state, dtype=torch.float32).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
            q_values = self.model(state_tensor)

            # Choose the top N prices based on q-values (N is the number of products)
            top_prices_indices = torch.argsort(q_values, descending=True).flatten()[:self.get_action_space()]

            chosen_prices = [float(top_price) for top_price in top_prices_indices]

        return chosen_prices
        

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        q_values = self.model(state)
        next_q_values = self.target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = reward + (1 - done) * self.gamma * next_q_values.max(dim=1)[0]

        loss = nn.MSELoss()(q_value, next_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % 1000 == 0:
            self.update_target_model()

            self.steps += 1


    def get_input_size(self):
        # Return the input size expected by the neural network
        return self.model.fc.in_features
    
    def get_action_space(self):
        # Return the number of products
        return self.model.out.out_features
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

        # Initialize the environment and RL agent (assuming you have already defined your RL agent)


env = PricingEnvironment(product_df)
input_size = 150  # Adjust according to the size of your state
output_size = num_products  # Number of possible actions (prices)
agent = DQNAgentWrapper(input_size, output_size)  # Call agent

# Simulation for 30 days
for day in range(1, 31):
     # Choose prices for the new product using the DQNAgentWrapper
    chosen_prices = agent.choose_prices(initial_state)
    
    # Update the environment with the chosen prices
    next_state, revenue, done, _ = env.step(chosen_prices)
    
    # Train the agent with the experience from the day
    processed_initial_state = agent.preprocess_state(initial_state)
    processed_next_state = agent.preprocess_state(next_state)
    agent.train(processed_initial_state, chosen_prices, revenue, processed_next_state, done)

    # Update initial_state for the next iteration
    initial_state = next_state

    # Get dynamic pricing calculated price
    dynamic_price = env.get_dynamic_price(initial_state, new_product_data)
    
    # Get the number of customers (assuming it's proportional to product_sold)
    num_customers = sum(initial_state['product_sold'])
    
    # Display the information for the current day
    print(f"Day{day}: Revenue={revenue:.2f}, Price={dynamic_price:.2f}, Customers={num_customers}")