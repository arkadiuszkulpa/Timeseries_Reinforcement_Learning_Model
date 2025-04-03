import gymnasium as gym
import torch
from policynetwork import PolicyNetwork

def run_trained_model(policy_net, env_name="LunarLander-v2", num_episodes=5):
    """
    Run the trained model in human mode to visualize its performance.
    """
    # Create the environment in human mode for rendering
    env = gym.make(env_name, render_mode="human")

    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_over = False
        total_reward = 0

        print(f"Episode {episode + 1}:")

        while not episode_over:
            # Convert observation to a PyTorch tensor
            observation_tensor = torch.tensor(observation, dtype=torch.float32)

            # Get Q-values from the policy network
            with torch.no_grad():
                q_values = policy_net(observation_tensor)

            # Choose the action with the highest Q-value
            action = torch.argmax(q_values).item()

            # Take the action in the environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Update observation and check if the episode is over
            observation = next_observation
            episode_over = terminated or truncated

        print(f"Total Reward: {total_reward}")

    env.close()

# Initialize the policy network
env = gym.make("LunarLander-v2")  # Temporary environment to get dimensions
policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)

# Load the trained model
policy_net.load_state_dict(torch.load("trained_policy.pth"))
policy_net.eval()  # Set the model to evaluation mode

# Run the trained model
run_trained_model(policy_net)