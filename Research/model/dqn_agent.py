import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from dqn_model import DQN


class DQNAgent:
    def __init__(self, input_dim, output_dim, price_lower_bound, price_upper_bound, memory_capacity=10000,
                 batch_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.dqn = DQN(input_dim, output_dim)
        self.target_dqn = DQN(input_dim, output_dim)
        self.memory_capacity = memory_capacity
        self.memory = []
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.price_lower_bound = price_lower_bound
        self.price_upper_bound = price_upper_bound

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Assuming 'state' is already of shape (input_dim,)
        # Assuming input_dim = 7, unsqueeze to add batch dimension

        if np.random.rand() <= self.epsilon:
            return np.random.uniform(self.price_lower_bound, self.price_upper_bound)
        else:
            with torch.no_grad():
                q_value = self.dqn(state_tensor)
                return q_value.item()
        
    def update_target_network(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

