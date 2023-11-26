import numpy as np 
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def check_next_state_lengths(self):
        return [len(trans.next_state) for trans in self.memory if trans is not None]
    
    def preprocess_next_state(self):
        # Filter out None or empty next_state elements
        valid_next_states = [trans.next_state for trans in self.memory if trans is not None and trans.next_state]
        problematic_transitions = [trans for trans in self.memory if trans is not None and not trans.next_state]
        
        if problematic_transitions:
            print(f"Problematic Transitions: {problematic_transitions}")
        
        if not valid_next_states:
            return  # No valid next_state elements found
        
        max_length = max(len(next_state) for next_state in valid_next_states)
        for trans in self.memory:
            if trans is not None and trans.next_state:
                if len(trans.next_state) < max_length:
                    trans.next_state += [0] * (max_length - len(trans.next_state))
                # Alternatively, truncate longer sequences if needed
                # desired_length = 10  # For example, to ensure all sequences have length 10
                # trans.next_state = trans.next_state[:desired_length]

    def convert_to_numpy(self):
        # Filter out None or empty next_state elements
        valid_next_states = [trans.next_state for trans in self.memory if trans is not None and trans.next_state]
        if not valid_next_states:
            return np.array([])  # Return an empty array if no valid next_state elements found

        max_length = max(len(next_state) for next_state in valid_next_states)
        next_state_array = np.array(
            [next_state + [0] * (max_length - len(next_state)) for next_state in valid_next_states]
        )
        return next_state_array
    
    def get_state_batch(self):
        return [trans.state for trans in self.memory if trans is not None]

    def get_action_batch(self):
        return [trans.action for trans in self.memory if trans is not None]

    def get_reward_batch(self):
        return [trans.reward for trans in self.memory if trans is not None]