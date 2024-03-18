import numpy as np
import matplotlib.pyplot as plt

# 状態・行動の定義
num_B_arm = 170
B_actions = [f"arm{i}" for i in range(num_B_arm)]
A_actions = ["left", "right"]
states = ["A", "B", "C", "Terminate"]

# 遷移の定義
transitions = {
    ("A", "left"): "B",
    ("A", "right"): "C",
    ("C", "right"): "Terminate",
    ("Terminate", ""): "Terminate"
}
for action in B_actions:
    transitions[("B", action)] = "Terminate"

# N, Qの初期化
N = {}
N_d = {}
Q = {}
Q_d = {}
def init_Q_N():
    for state_action in transitions.keys():
        Q_d[state_action] = Q[state_action] = N_d[state_action] = N[state_action] = 0

def get_reward(state, action):
    if state == "B" and action.startswith("arm"):
        return np.random.normal(loc=-0.05, scale=1.0)
    else:
        return 0

def argmax(state, Q):
    cur_max = -10000
    cur_argmax = -1
    for key in Q.keys():
        if key[0] == state and cur_max < Q[key]:
            cur_max = Q[key]
            cur_argmax = key
    return cur_argmax[1]

def get_action(state, Q, eps=0.1):
    if state == "Terminate":
        return ""
    elif state == "A":
        return np.random.choice([np.random.choice(A_actions), argmax(state, Q)], p=[eps, 1-eps])
    elif state == "B":
        return np.random.choice([np.random.choice(B_actions), argmax(state, Q)], p=[eps, 1-eps])
    elif state == "C":
        return "right"
    else:
        raise NotImplementedError(f"state:{state}は存在しません")


def update_by_Q_learning():
    cur_state = "A"
    while cur_state != "Terminate":
        action = get_action(cur_state, Q)
        reward, next_state = get_reward(cur_state, action), transitions[(cur_state, action)]
        next_optimal_action = argmax(next_state, Q)
        N[(cur_state, action)] += 1
        alpha = 1/N[(cur_state, action)]
        Q[(cur_state, action)] = Q[(cur_state, action)] + alpha * (reward + Q[(next_state, next_optimal_action)] - Q[(cur_state, action)])
        cur_state = next_state
    
def update_by_DQ_learning():
    cur_state = "A"
    while cur_state != "Terminate":
        which_update = np.random.choice(["m", "d"])
        if which_update == "m":
            action = get_action(cur_state, Q)
            reward, next_state = get_reward(cur_state, action), transitions[(cur_state, action)]
            next_optimal_action = argmax(next_state, Q)
            N[(cur_state, action)] += 1
            alpha = 1/N[(cur_state, action)]
            Q[(cur_state, action)] = Q[(cur_state, action)] + alpha * (reward + Q_d[(next_state, next_optimal_action)] - Q[(cur_state, action)])                
        else:
            action = get_action(cur_state, Q_d)
            reward, next_state = get_reward(cur_state, action), transitions[(cur_state, action)]
            next_optimal_action = argmax(next_state, Q_d)
            N_d[(cur_state, action)] += 1
            alpha = 1/N_d[(cur_state, action)]
            Q_d[(cur_state, action)] = Q_d[(cur_state, action)] + alpha * (reward + Q[(next_state, next_optimal_action)] - Q_d[(cur_state, action)])                
        cur_state = next_state


def plot_history(history_max_Q, history_max_DQ):
    plt.figure(figsize=(12, 6))
    plt.plot(history_max_Q, label='Max Q Value (Q-learning)')
    plt.plot(history_max_DQ, label='Max Q Value (Double Q-learning)')
    plt.xlabel('Episode')
    plt.ylabel('Max Q Value in State A')
    plt.title('Max Q Value in State A Over Time')
    plt.legend()
    plt.show()

def execute_learning(n_episodes, update_func):
    history_max_Q = []
    init_Q_N()
    for n in range(n_episodes):
        update_func()

        action = argmax("A", Q)
        history_max_Q.append(Q[("A", action)])
        if n % 500 == 0:
            print(f"{n}: {Q[('A', action)]}")

    return history_max_Q

n_episodes = 100000
history_max_Q = execute_learning(n_episodes, update_by_Q_learning)
history_max_DQ = execute_learning(n_episodes, update_by_DQ_learning)

plot_history(history_max_Q, history_max_DQ)
