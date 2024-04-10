Sure! Here's a simple example of reinforcement learning (Q-learning) applied to a basic game scenario, specifically the classic "CartPole" problem from OpenAI's Gym environment. In this game, the agent controls a cart trying to balance a pole upright.
<br>
You'll need to have the Gym library installed (pip install gym). Here's a Python script implementing Q-learning for this task:


```py
import gym
import numpy as np

# Environment setup
env = gym.make('CartPole-v1')

# Q-learning parameters
num_episodes = 1000
learning_rate = 0.1
discount_factor = 0.99

# Discretization parameters
num_buckets = (1, 1, 6, 3)  # Cart position, Cart velocity, Pole angle, Pole velocity
num_actions = env.action_space.n
q_table = np.zeros(num_buckets + (num_actions,))

# Discretization function
def discretize(observation):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], np.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -np.radians(50)]
    ratios = [(observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(observation))]
    discretized_obs = [int(round((num_buckets[i] - 1) * ratios[i])) for i in range(len(observation))]
    discretized_obs = [min(num_buckets[i] - 1, max(0, discretized_obs[i])) for i in range(len(observation))]
    return tuple(discretized_obs)

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    state = discretize(state)
    done = False
    
    while not done:
        if np.random.random() < 0.5:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values
        
        next_state, reward, done, _ = env.step(action)
        next_state = discretize(next_state)
        
        if not done or episode == num_episodes - 1:
            q_value = q_table[state][action]
            max_value = np.max(q_table[next_state])
            new_q_value = (1 - learning_rate) * q_value + learning_rate * (reward + discount_factor * max_value)
            q_table[state][action] = new_q_value

        state = next_state

env.close()
```
<br>
This code performs Q-learning with discretization of the state space. The state space for the CartPole environment is continuous, so we discretize it into a finite number of buckets to apply Q-learning.
<br>
This is just a basic example. Depending on the complexity of the game and its state space, you might need more advanced techniques like Deep Q-Networks (DQN) or other variants of reinforcement learning algorithms. Additionally, parameter tuning and exploration strategies are crucial for the success of reinforcement learning algorithms.
