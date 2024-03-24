from policies import GreedyEpsPolicyQ

def execute_Q_learning(n_iter, env, gamma=1, alpha=0.01):
    Q = {}
    for state in env.state_spaces:
        Q[state] = {}
        for action in env.action_spaces:
            Q[state][action] = 100

    policy = GreedyEpsPolicyQ(Q, env.action_spaces)
    state, _ = env.reset()
    cur_reward = 0
    for n in range(n_iter):
        action = policy.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * max(Q[next_state].values()) - Q[state][action])
        cur_reward += reward
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()
            print(f"reward: {cur_reward} reset state {state}")
            cur_reward = 0

    return Q

def execute_sarsa_learning(n_iter, env, gamma=1, alpha=0.01):
    Q = {}
    for state in env.state_spaces:
        Q[state] = {}
        for action in env.action_spaces:
            Q[state][action] = 100

    policy = GreedyEpsPolicyQ(Q, env.action_spaces)
    state, _ = env.reset()
    cur_reward = 0
    action = policy.get_action(state)
    for n in range(n_iter):
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_action = policy.get_action(state)
        
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
        cur_reward += reward
        state, action = next_state, next_action

        if terminated or truncated:
            state, _ = env.reset()
            print(f"reward: {cur_reward} reset state {state}")
            cur_reward = 0
            action = policy.get_action(state)

    return Q