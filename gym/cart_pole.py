import gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

state, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    pos, velocity, angle, angle_velocity = state[0], state[1], state[2], state[3]
    print(f"pos: {pos}, velocity: {velocity}, angle: {angle}, angle_velocity: {angle_velocity}")
    if terminated or truncated:
        state, info = env.reset()

env.close()