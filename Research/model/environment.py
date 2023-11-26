import numpy as np
from compute_combined_forecast import compute_combined_forecast

class PricingEnvironment:
    def __init__(self, products_data, initial_price):
        self.products_data = products_data  # Data for existing products
        self.num_products = len(products_data)
        self.current_day = 0
        self.initial_price = initial_price

    def reset(self):
        self.current_day = 0
        return self._get_state()

    def _get_state(self):
        # Features of the products for the current day
        current_product_features = self.products_data.iloc[self.current_day, :]

        return current_product_features.values  # Return only the product features
    
    def step(self, prices):
        

        # Get demand for each product based on the given prices
        demands = self._calculate_demand(prices)

        # Calculate reward based on demands and prices
        rewards = self._calculate_reward(demands, prices)
        
        # Move to the next day
        self.current_day += 1
        
        # Check if the episode is done after 30 days
        done = self.current_day >= 30

        # Get the next state
        next_state = self._get_state() if not done else None

        return next_state, rewards, done, {}

    def _calculate_demand(self, prices):
        combined_forecast, _ = compute_combined_forecast(self.products_data)
        
        # Assume a simple demand estimation based on the forecast and prices for all products
        demands = np.maximum(100 - prices * combined_forecast, 10)
        
        return demands
    
    def _calculate_reward(self, demands, prices):
        costs = self.products_data.iloc[self.current_day, -self.num_products:]  # Costs for all products
        revenues = demands * prices  # Revenues from selling the products
        profits = revenues - (demands * costs)  # Profits from selling the products
        return profits
