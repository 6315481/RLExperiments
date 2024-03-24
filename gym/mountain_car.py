import numpy as np
import gym
from enum import IntEnum
from collections import namedtuple
from learner import execute_Q_learning, execute_sarsa_learning
from utils import evaluate_policy
from policies import GreedyPolicyQ

State = namedtuple("State", ["pos", "velocity"])
class Action(IntEnum):
    LEFT = 0
    STAY = 1
    RIGHT= 2

class MountainCarEnv:
    def __init__(self, num_tiles, render_mode="human"):
        num_tiles = num_tiles
        min_pos, max_pos = -1.2, 0.6
        min_velocity, max_velocity = -0.07, 0.07
        pos_spaces = np.linspace(min_pos, max_pos, num=num_tiles)
        velocity_spaces = np.linspace(min_velocity, max_velocity, num=num_tiles)
        
        self.state_spaces = []
        self.action_spaces = [Action.LEFT, Action.STAY, Action.RIGHT]
        for pos_min, pos_max in zip(pos_spaces[:-1], pos_spaces[1:]):
            for velocity_min, velocity_max in zip(velocity_spaces[:-1], velocity_spaces[1:]):
                self.state_spaces.append(State((pos_min, pos_max), (velocity_min, velocity_max)))
        
        self.env = gym.make("MountainCar-v0", render_mode=render_mode)
        self.eps = 0.00001

    def to_state(self, pos, velocity):
        for state in self.state_spaces:
            if (state.pos[0] - self.eps <= pos <= state.pos[1] + self.eps) and (state.velocity[0] - self.eps <= velocity <= state.velocity[1] + self.eps):
                return state

        raise ValueError(f"No matching state {pos}, {velocity}")

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(int(action))
        pos, velocity = observation[0], observation[1]
        return self.to_state(pos, velocity), reward, terminated, truncated, info

    def render(self):
        self.env.render()
    
    def reset(self):
        state, info = self.env.reset()
        return self.to_state(state[0], state[1]), info

    def close(self):
        self.env.close()

n_tiles = 16
env = MountainCarEnv(n_tiles, render_mode="rgb_array")

n_iter = 1000000
Q = execute_sarsa_learning(n_iter, env, alpha=1/16)

policy = GreedyPolicyQ(Q)
env = MountainCarEnv(n_tiles, render_mode="human")
evaluate_policy(env, 30, policy)
env.close()