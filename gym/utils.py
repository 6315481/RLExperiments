

def argmax(state, Q):
    return max(Q[state], key=Q[state].get)

def evaluate_policy(env, n_episodes, policy):
    state, _ = env.reset()
    cur_steps = 0
    cur_episodes = 0
    cur_reward = 0
    while cur_episodes < n_episodes:
        action = policy.get_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        cur_steps += 1
        cur_reward += reward
        if terminated or truncated:
            print(f"Steps taken: {cur_steps}")
            print(f"Reward: {cur_reward}")
            state, _ = env.reset()
            cur_reward = cur_steps = 0
            cur_episodes += 1