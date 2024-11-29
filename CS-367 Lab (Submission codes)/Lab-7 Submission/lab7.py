import numpy as np
import matplotlib.pyplot as plt

# Non-stationary 10-armed bandit function
def bandit_nonstat(action, true_rewards):
    """
    Simulates the reward for a selected action in a non-stationary 10-armed bandit.
    The true rewards for each arm evolve over time with a random walk.
    
    Parameters:
        action (int): The selected action (arm).
        true_rewards (numpy array): The true rewards of each arm, evolving over time.

    Returns:
        reward (float): The reward for the selected action with noise.
        true_rewards (numpy array): Updated true rewards after a random walk.
    """
    # Random walk for non-stationary rewards
    true_rewards += np.random.normal(0, 0.01, 10)  # Normal random walk
    reward = true_rewards[action] + np.random.normal(0, 0.1)  # Add noise to the reward
    return reward, true_rewards

def epsilon_greedy_nonstat(epsilon, Q, N, true_rewards):
    """
    Epsilon-greedy algorithm for selecting an action in the non-stationary bandit problem.
    This function decides whether to explore or exploit and updates the action-value estimates.

    Parameters:
        epsilon (float): The exploration rate (probability of exploring).
        Q (numpy array): The action-value estimates (initially zero).
        N (numpy array): The count of times each action has been selected.
        true_rewards (numpy array): The true rewards of each arm.

    Returns:
        Q (numpy array): Updated action-value estimates.
        N (numpy array): Updated counts for each action.
        action (int): The selected action.
        reward (float): The reward for the selected action.
        true_rewards (numpy array): Updated true rewards after the random walk.
    """
    # Decide whether to explore or exploit
    if np.random.rand() < epsilon:
        # Exploration: choose a random action
        action = np.random.randint(0, 10)
    else:
        # Exploitation: choose the action with the highest estimated reward
        action = np.argmax(Q)
    
    # Get the reward for the selected action from the non-stationary bandit
    reward, true_rewards = bandit_nonstat(action, true_rewards)
    
    # Update the action-value estimate using the incremental mean update
    N[action] += 1
    Q[action] += (reward - Q[action]) / N[action]  # Incremental mean update
    
    return Q, N, action, reward, true_rewards

# Main function to simulate the epsilon-greedy agent
def run_bandit_simulation(epsilon=0.1, num_steps=10000):
    """
    Run the epsilon-greedy agent in a non-stationary 10-armed bandit for a given number of time steps.

    Parameters:
        epsilon (float): The exploration rate.
        num_steps (int): The number of time steps to run the simulation.
        
    Returns:
        Q (numpy array): The final action-value estimates after the simulation.
        cumulative_rewards (list): The cumulative rewards over time.
    """
    # Initialize parameters
    Q = np.zeros(10)  # Action-value estimates (initialized to zero)
    N = np.zeros(10)  # Action counts (initialized to zero)
    true_rewards = np.random.randn(10)  # Initial random true rewards (mean=0, std=1)
    cumulative_rewards = []  # List to store cumulative rewards over time
    total_reward = 0  # Initialize total reward for cumulative calculation

    # Run the agent for num_steps time steps
    for t in range(num_steps):
        Q, N, action, reward, true_rewards = epsilon_greedy_nonstat(epsilon, Q, N, true_rewards)
        total_reward += reward
        cumulative_rewards.append(total_reward)

    return Q, cumulative_rewards

# Running the simulation
epsilon = 0.1  # Exploration rate
num_steps = 10000  # Number of time steps
Q, cumulative_rewards = run_bandit_simulation(epsilon, num_steps)

# Plot the final estimated action values after 10,000 steps
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(Q)
plt.title('Estimated Action Values (After 10,000 Steps)')
plt.xlabel('Action')
plt.ylabel('Estimated Value')

# Plot the cumulative reward over time
plt.subplot(1, 2, 2)
plt.plot(cumulative_rewards)
plt.title('Cumulative Reward Over Time')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Reward')

# Show the plots
plt.tight_layout()
plt.show()
