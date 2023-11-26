import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import random        
import pandas as pd

# Define the DQN (Deep Q-Network)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    

# Define the ReplayBuffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Define the PricingEnvironment
class PricingEnvironment:
    def __init__(self, data, num_products):
        self.data = data
        self.num_products = num_products
        self.reset()

    def reset(self):
        self.current_product = 0  # Start with the first product
        self.current_day = 0
        self.done = False

    def step(self, action):
        new_product_price = action  # Assuming 'action' is the price set by the agent for the new product
        state = self.get_state(new_product_price)  # Update state based on the new product and market conditions

        # Fetch relevant data for the current day
        product_data = self.data[self.data["product_id"] == self.current_product + 1]
        if self.current_day < len(product_data):
            current_row = product_data.iloc[self.current_day]

            # Get the price and other relevant information for the current day
            current_price = current_row["price"]
            customers_modifier = self.calculate_customers_modifier(current_price, new_product_price)

            # Update the number of customers based on the pricing decision and market conditions
            self.current_customers = int(current_row["product_sold"] * customers_modifier)
            self.data.at[self.current_day, "product_sold"] = self.current_customers

            reward = self.calculate_reward(new_product_price, self.current_customers)  # Update the calculate_reward method accordingly
        else:
            reward = 0

        self.current_day += 1
        if self.current_day == len(self.data):
            self.done = True

        return state, reward, self.done

    def get_state(self, new_product_price):
        # For example, assuming the state is a vector of relevant features for pricing:
        product_data = self.data[self.data["product_id"] == self.current_product + 1]  # Assuming product_id starts from 1

        # Initialize the state with zeros
        state = np.zeros(state_size)

        if self.current_day < len(product_data):
            current_row = product_data.iloc[self.current_day]

            state[:6] = np.array([
                current_row["day"],
                current_row["product_sold"],
                current_row["price"],
                current_row["rating"],
                current_row["total_rating"],
                current_row["status"]
            ])

        # Ensure that the state has the correct shape for the neural network
        state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)

        return state_tensor

    def calculate_customers_modifier(self, current_price, new_price, base_customers=5):
        price_difference = new_price - current_price
        sensitivity = 10 # Higher sensitivity for more variability
        base_modifier = base_customers / self.num_products / 100
        sigmoid_value =  0.5 / (25 + np.exp(-sensitivity * price_difference / current_price))
        random_factor = random.uniform(0.8, 1.2)  # Adjust the range for variability
        customers_modifier = base_modifier + sigmoid_value * random_factor
        return customers_modifier



    def calculate_reward(self, action, customers):
        product_data = self.data[self.data["product_id"] == self.current_product + 1]

        if self.current_day < len(product_data):
            current_row = product_data.iloc[self.current_day]

            target_price = current_row["price"]
            deviation_penalty = -abs(action - target_price)
            reward = customers + deviation_penalty

            # Debugging print statements

        else:
            reward = 0

        return reward



# Hyperparameters
state_size = 7  # Change based on your dataset features
action_size = 1  # Assuming the agent sets the price (continuous action space)
learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update = 10
memory_capacity = 10000
batch_size = 64
num_episodes = 1000

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load your dataset
# Replace 'your_excel_file.xlsx' with the path to your Excel file
file_path = 'Shopee-Product-40oz-September-2023-FINAL.xlsx'
data = pd.read_excel(file_path)

# Ensure the dataset has the required columns: "product_id", "day", "product_sold", "price", "rating", "total_rating", "status"
assert all(col in data.columns for col in ["product_id", "day", "product_sold", "price", "rating", "total_rating", "status"])

# Print the entire DataFrame
#print(data.to_string(index=False))


# Initialize the DQN, target network, optimizer, and replay buffer
dqn = DQN(state_size, action_size)
target_network = DQN(state_size, action_size)
target_network.load_state_dict(dqn.state_dict())
target_network.eval()



  
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(memory_capacity)

# Training loop
epsilon = epsilon_start
#means is 1914.0


# Simulation loop
epsilon = epsilon_start
daily_prices = []
daily_customers = []
daily_revenues = []
max_price = 2500.0  # Example: setting a maximum price
#1916 for s40
#1461 for s20 
initial_price = float(input("\nEnter the initial price: "))  # Get the initial price from user input
price_fluctuation_range = 350.0  # Adjust this range as needed


for day in range(1, 31):  # 30-day simulation
    env = PricingEnvironment(data, num_products=25)

    total_reward = 0
    action = 0
    while not env.done:
        if random.random() < epsilon:
            # Random action for exploration within the defined range around the initial price
            action = np.random.uniform(max(0, initial_price - price_fluctuation_range),
                                        min(max_price, initial_price + price_fluctuation_range))
        else:
            state = env.get_state(action)
            with torch.no_grad():
                q_values = dqn(state)
                # Scale the output of the network to fit within the price range around the initial price
                action = max(0.0, min(q_values.item(), initial_price + price_fluctuation_range))
                action = min(max_price, max(action, initial_price - price_fluctuation_range))

        # Calculate the number of customers based on the modifier
        current_price = data.at[env.current_day, 'price']
        new_price = action
        customers_modifier = env.calculate_customers_modifier(current_price, new_price)
        env.current_customers = int(data.at[env.current_day, 'product_sold'] * customers_modifier)
        static_customer = int(data.at[0, 'product_sold'] * customers_modifier)
        # Other parts of your simulation loop remain unchanged...
        # ... (omitted for brevity)

        next_state, reward, done = env.step(action)
        total_reward += reward

        # Decay epsilon for exploration-exploitation trade-off
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Append to daily_prices, daily_customers, etc. as needed
        daily_prices.append(action)
        daily_customers.append(env.current_customers)
        daily_revenues.append(action * env.current_customers)

# Print the daily breakdown for the entire simulation
print("\nDynamic Pricing Simulation Breakdown:")
for day in range(1, 31):
    print(f"Day {day}, Price: {daily_prices[day-1]:.2f}, Customers: {daily_customers[day-1]}, Revenue: {daily_revenues[day-1]:.2f}")



import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 6))

# plt.subplot(2, 1, 1)
# plt.plot(range(1, 31), daily_prices[:30], marker='o', color='green')  # Slice daily_prices to match 30 days
# plt.title('Dynamic Pricing: Price Over 30 Days')
# plt.xlabel('Day')
# plt.ylabel('Price')
# plt.grid(True)

# plt.subplot(2, 1, 2)# Slice daily_customers to match 30 days
# plt.plot(range(1, 31), daily_revenues[:30], marker='o', color='red', label='Revenue')  # Slice static_daily_revenues to match 30 days
# plt.title('Dynamic Pricing: Customers and Revenue Over 30 Days')
# plt.xlabel('Day')
# plt.ylabel('Revenue')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()
static_daily_prices = [initial_price] * 30  # Create a list with the same initial price for 30 days

static_daily_customers = []
static_daily_revenues = []

# Simulation loop for static pricing
for day in range(30):  # 30-day simulation for static pricing
    env = PricingEnvironment(data, num_products=25)
    total_reward = 0

    while not env.done:
        current_price = data.at[env.current_day, 'price']
        customers_modifier = env.calculate_customers_modifier(current_price, initial_price)
        static_customer = daily_customers[env.current_day]  # Use dynamic pricing customers for static pricing
        env.current_customers = static_customer
        action = initial_price  # Static action (initial price)
        next_state, reward, done = env.step(action)
        total_reward += reward

    static_daily_customers.append(static_customer)
    static_daily_revenues.append(static_daily_prices[day] * daily_customers[day])  # Calculate static revenue  # Calculate static revenue


# Print the daily breakdown for the entire simulation (static pricing)
print("\nStatic Pricing Simulation Breakdown:")
for day in range(1, 31):
    print(f"Day {day}, Price: {static_daily_prices[day-1]:.2f} Customers: {daily_customers[day-1]}, Revenue: {static_daily_revenues[day-1]:.2f}")


# Calculate total revenue for dynamic pricing
total_dynamic_revenue = sum(daily_revenues[:30])

# Calculate total revenue for static pricing
total_static_revenue = sum(static_daily_revenues[:30])

print(f"\nTotal Revenue (Dynamic Pricing): ${total_dynamic_revenue:.2f}")
print(f"Total Revenue (Static Pricing): ${total_static_revenue:.2f}")


# Finding the day with the highest and lowest revenue for dynamic pricing
max_dynamic_revenue_day = daily_revenues.index(max(daily_revenues[:30])) + 1
min_dynamic_revenue_day = daily_revenues.index(min(daily_revenues[:30])) + 1

# Finding the day with the highest and lowest revenue for static pricing
max_static_revenue_day = static_daily_revenues.index(max(static_daily_revenues[:30])) + 1
min_static_revenue_day = static_daily_revenues.index(min(static_daily_revenues[:30])) + 1
# Print details of days with highest and lowest revenue for both dynamic and static pricing

print("\nDetails of Days with Highest and Lowest Revenue:")
print(f"Day with the Highest Revenue (Dynamic Pricing): {max_dynamic_revenue_day}")
print(f"Price: {daily_prices[max_dynamic_revenue_day - 1]:.2f}, Customers: {daily_customers[max_dynamic_revenue_day - 1]}, Revenue: {daily_revenues[max_dynamic_revenue_day - 1]:.2f}")
print(f"\nDay with the Highest Revenue (Static Pricing): {max_static_revenue_day}")
print(f"Price: {static_daily_prices[max_static_revenue_day - 1]:.2f}, Customers: {daily_customers[max_static_revenue_day - 1]}, Revenue: {static_daily_revenues[max_static_revenue_day - 1]:.2f}")
print(f"\nDay with the Lowest Revenue (Dynamic Pricing): {min_dynamic_revenue_day}")
print(f"Price: {daily_prices[min_dynamic_revenue_day - 1]:.2f}, Customers: {daily_customers[min_dynamic_revenue_day - 1]}, Revenue: {daily_revenues[min_dynamic_revenue_day - 1]:.2f}")
print(f"\nDay with the Lowest Revenue (Static Pricing): {min_static_revenue_day}")
print(f"Price: {static_daily_prices[min_static_revenue_day - 1]:.2f}, Customers: {daily_customers[min_static_revenue_day - 1]}, Revenue: {static_daily_revenues[min_static_revenue_day - 1]:.2f}")


plt.figure(figsize=(12, 8))

# Subplot 1 for Dynamic Pricing
plt.subplot(2, 2, 1)
plt.plot(range(1, 31), daily_prices[:30], marker='o', color='#0000FF')  # Slice daily_prices to match 30 days
plt.title('Dynamic Pricing: Price Over 30 Days')
plt.xlabel('Day')
plt.ylabel('Price')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(range(1, 31), daily_revenues[:30], marker='o', color='#87CEEB', label='Revenue')  # Slice static_daily_revenues to match 30 days
plt.title('Dynamic Pricing: Customers and Revenue Over 30 Days')
plt.xlabel('Day')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)

# Subplot 2 for Static Pricing
plt.subplot(2, 2, 3)
plt.plot(range(1, 31), static_daily_prices[:30], marker='o', color='#008000')  # Slice daily_prices to match 30 days
plt.title('Static Pricing: Price Over 30 Days')
plt.xlabel('Day')
plt.ylabel('Price')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(range(1, 31), static_daily_revenues[:30], marker='o', color='#90EE90', label='Revenue')  # Slice static_daily_revenues to match 30 days
plt.title('Static Pricing: Customers and Revenue Over 30 Days')
plt.xlabel('Day')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()