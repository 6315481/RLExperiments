import numpy as np
from utils import argmax

class GreedyEpsPolicyQ:
    def __init__(self, Q, action_spaces, eps=0.1):
        self.Q = Q
        self.eps = eps
        self.action_spaces = action_spaces
    
    def get_action(self, state):
        return np.random.choice([argmax(state, self.Q), np.random.choice(self.action_spaces)], p=[1-self.eps, self.eps])

class GreedyPolicyQ:
    def __init__(self, Q):
        self.Q = Q
    
    def get_action(self, state):
        return argmax(state, self.Q)
