import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from policynetwork import PolicyNetwork

# Initialize the environment
env = gym.make("LunarLander-v2")
num_episodes = 1000  # Number of episodes for training
learning_rate = 0.1  # Learning rate for policy updates
discount_factor = 0.99  # Discount factor for future rewards


# Initialize the policy network
policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

def choose_action(observation, epsilon=0.1):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore: choose a random action
    
    # Convert observation to a PyTorch tensor
    observation_tensor = torch.tensor(observation, dtype=torch.float32)
    
    # Get Q-values from the policy network
    with torch.no_grad():  # No gradient computation for action selection
        q_values = policy_net(observation_tensor)
    
    # Choose the action with the highest Q-value
    return torch.argmax(q_values).item()

def normalize_observation(observation):
    # Clip observations to a reasonable range
    return np.clip(observation, -1, 1)


# List to store total rewards for each episode
rewards = []

epsilon = 1.0  # Start with high exploration
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.999  # Decay rate per episode

for episode in range(num_episodes):
    observation, info = env.reset()
    total_reward = 0
    episode_over = False

    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Slower decay

    while not episode_over:
        # Choose an action using the policy
        action = choose_action(observation, epsilon)

        # Take the action in the environment
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_observation = normalize_observation(next_observation)

        reward = np.clip(reward, -1, 1)
        total_reward += reward

        # Update the policy network
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        next_observation_tensor = torch.tensor(next_observation, dtype=torch.float32)

        # Compute Q-values for the current state
        q_values = policy_net(observation_tensor)

        # Compute Q-values for the next state
        with torch.no_grad():  # No gradient computation for the next state
            next_q_values = policy_net(next_observation_tensor)

        # Compute the TD target
        td_target = reward + discount_factor * torch.max(next_q_values).item()

        # Compute the TD error for the chosen action
        td_error = td_target - q_values[action]

        # Compute the loss (squared TD error)
        loss = td_error ** 2

        # Backpropagate the loss and update the network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update observation and check if the episode is over
        observation = next_observation     
        episode_over = terminated or truncated

    # Save the total reward for this episode
    rewards.append(total_reward)

    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()

# Save the trained model
torch.save(policy_net.state_dict(), "trained_policy.pth")
print("Trained model saved as 'trained_policy.pth'")

# Plot the rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()