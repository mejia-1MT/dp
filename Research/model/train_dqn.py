import torch
import torch.optim as optim
import torch.nn as nn
import random
from dqn_agent import DQNAgent
from environment import PricingEnvironment
from replay_buffer import ReplayBuffer, Transition
import numpy as np

def train_dqn(agent, env, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            # Convert the state to a tensor
            state_tensor = torch.FloatTensor(state)
            
            # Select an action based on the current state
            action = agent.select_action(state_tensor)
            
            # Perform the action and observe the next state and reward
            next_state, reward, done, _ = env.step(action)
            
            # Update the total reward for the episode
            total_reward += reward
            
            # Convert the next state to a tensor
            next_state_tensor = torch.FloatTensor(next_state) if next_state is not None else None
            
            # Store the experience in the agent's memory
            agent.memory.append((state_tensor, action, reward, next_state_tensor, done))
            
            # Move to the next state
            state = next_state
            
            # Perform an optimization step if the memory is sufficiently full
            if len(agent.memory) >= agent.memory_capacity:
                agent.optimize_model()
            
            if done:
                break

        # Update the target network periodically
        if episode % 10 == 0:
            agent.update_target_network()

        # Decay epsilon for exploration
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")