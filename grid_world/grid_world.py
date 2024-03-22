import numpy as np
from utils import visualize_maze_with_path

grid = [
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    [0 ,1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
]
height = len(grid)
width = len(grid[-1])
start = (0, 0)
goal = (height - 1, width - 1)
goal = (6, 4)
actions = ["left", "up", "down", "right"]

def get_reward(state, action):
    if state == goal:
        return 0
    
    ns = transition(state, action)
    if ns == goal:
        return 1
    else:
        return -1

def transition(state, action):    
    i, j = state
    if action == "left":
        ni, nj = (i, max(j - 1, 0))
    elif action == "up":
        ni, nj = (max(i - 1, 0), j)
    elif action == "right":
        ni, nj = (i, min(j + 1, width - 1))
    elif action == "down":
        ni, nj = (min(i + 1, height - 1), j)
    else:
        raise NotImplemented()
    
    if grid[ni][nj] == 1:
        return state
    else:
        return (ni, nj)

Q = {}
N = {}
for i in range(height):
    for j in range(width):
        state = (i, j)
        Q[state] = {}
        N[state] = {}
        for action in actions:
            Q[state][action] = 0
            N[state][action] = 0


def argmax(state, Q):
    return max(Q[state], key=Q[state].get)

def get_action(state, eps=0.1):
    return np.random.choice([argmax(state, Q), np.random.choice(actions)], p=[1-eps, eps])

def evaluate_current_policy(Q, is_visualize=False):
    state = start
    steps_taken = 0
    trajectory = [start]
    while state != goal and steps_taken < 10000:
        action = argmax(state, Q)
        state = transition(state, action)
        trajectory.append(state)
        steps_taken += 1    

    if steps_taken >= 10000:
        print("steps_taken >= 10000")
    else:
        print(f"steps_taken: {steps_taken}")

    if is_visualize:
        visualize_maze_with_path(grid, start, goal, trajectory)

n_episodes = 30000
gamma = 1
alpha = 0.01
for n in range(n_episodes):
    state = start
    while state != goal:
        action = get_action(state)
        next_state = transition(state, action)
        reward = get_reward(state, action)
        N[state][action] = N[state][action] + 1
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * max(Q[next_state].values()) - Q[state][action])
        state = next_state

    if n % 100 == 0:
        evaluate_current_policy(Q)

evaluate_current_policy(Q, is_visualize=True)
