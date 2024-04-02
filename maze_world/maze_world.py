import numpy as np
import random
from utils import visualize_maze_with_path
from maze_generator import generate_maze

width = 100
height = 100
num_islands = 500
maze = generate_maze(width, height, num_islands)
start = (0, 0)
goal = (height - 1, width - 1)
actions = ["left", "up", "down", "right"]
path_states = [(i, j)  for i in range(height) for j in range(width) if maze[i][j] == 0]

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
    
    if maze[ni][nj] == 1:
        return state
    else:
        return (ni, nj)

Q = {}
V = {}
N = {}
for i in range(height):
    for j in range(width):
        state = (i, j)
        Q[state] = {}
        N[state] = {}
        V[state] = 0
        for action in actions:
            Q[state][action] = 0
            N[state][action] = 0


def argmax(state, Q):
    return max(Q[state], key=Q[state].get)

def policy_greedy_eps_Q(state, eps=0.1):
    return np.random.choice([argmax(state, Q), np.random.choice(actions)], p=[1-eps, eps])

def policy_greedy_Q(state):
    return argmax(state, Q)

def policy_greedy_V(state):
    cur_max = -1000000
    cur_action = "left"
    for action in actions:
        next_state = transition(state, action)
        reward = get_reward(state, action)
        if reward + V[next_state] > cur_max:
            cur_action = action
            cur_max = reward + V[next_state]
    return cur_action

def evaluate_current_policy(policy, is_visualize=False):
    state = start
    steps_taken = 0
    trajectory = [start]
    while state != goal and steps_taken < 10000:
        action = policy(state)
        state = transition(state, action)
        trajectory.append(state)
        steps_taken += 1    

    if steps_taken >= 10000:
        print("steps_taken >= 10000")
    else:
        print(f"steps_taken: {steps_taken}")

    if is_visualize:
        visualize_maze_with_path(maze, start, goal, trajectory)

def execute_Q_learning(n_episodes, gamma=1, alpha=0.01):
    for n in range(n_episodes):
        state = start
        while state != goal:
            action = policy_greedy_eps_Q(state)
            next_state = transition(state, action)
            reward = get_reward(state, action)
            N[state][action] = N[state][action] + 1
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * max(Q[next_state].values()) - Q[state][action])
            state = next_state

        if n % 100 == 0:
            evaluate_current_policy(policy_greedy_Q)


def execute_value_iteration(n_iter, gamma=0.99):
    for n in range(n_iter):
        for state in path_states:
            V[state] = max([get_reward(state, action) + gamma * V[transition(state, action)] for action in actions])
        if n % 100 == 0:
            evaluate_current_policy(policy_greedy_V)

if __name__ == "__main__":
    n_episodes = 50000
#    execute_Q_learning(n_episodes)
    execute_value_iteration(1000)
    evaluate_current_policy(policy_greedy_V, is_visualize=True)

print(path_states)
#print(Q)