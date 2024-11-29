import numpy as np

# Define constants for the environment size and terminal states
ROWS, COLS = 4, 3
TERMINAL_POSITIVE = (0, 2)
TERMINAL_NEGATIVE = (3, 2)

# Define actions and their corresponding effects on the agent's position
ACTIONS = ['Up', 'Down', 'Left', 'Right']
ACTION_EFFECTS = {
    'Up': [(-1, 0), (0, -1), (0, 1)],  # Intended move (Up), Left, Right
    'Down': [(1, 0), (0, -1), (0, 1)],  # Intended move (Down), Left, Right
    'Left': [(0, -1), (-1, 0), (1, 0)],  # Intended move (Left), Up, Down
    'Right': [(0, 1), (-1, 0), (1, 0)],  # Intended move (Right), Up, Down
}

# Transition function to handle stochastic movement with correct probabilities
def transition(state, action):
    row, col = state
    # Possible moves for each action
    possible_moves = ACTION_EFFECTS[action]
    
    # Probabilities: 80% for the intended direction, 10% for each perpendicular direction
    probabilities = [0.8, 0.1, 0.1]
    
    next_states = []
    
    # Loop over the possible moves, assigning probabilities to each
    for i, move in enumerate(possible_moves):
        new_row, new_col = row + move[0], col + move[1]
        # Ensure the new position is within grid bounds
        if 0 <= new_row < ROWS and 0 <= new_col < COLS:
            next_states.append((new_row, new_col))
        else:
            next_states.append(state)  # Stay in place if out of bounds
    
    return next_states, probabilities

# Reward function for different reward types (r1 to r4)
def reward_function(s, reward_type):
    if s == TERMINAL_POSITIVE:
        return 1  # Positive terminal state
    elif s == TERMINAL_NEGATIVE:
        return -1  # Negative terminal state
    else:
        if reward_type == 'r1':
            return -2  # Reward for r1 scenario
        elif reward_type == 'r2':
            return 0.1  # Reward for r2 scenario
        elif reward_type == 'r3':
            return 0.02  # Reward for r3 scenario
        elif reward_type == 'r4':
            return 1  # Reward for r4 scenario
        else:
            return -0.04  # Default reward for non-terminal states

# Value iteration algorithm
def value_iteration(reward_type, gamma=1.0, threshold=1e-6, max_iterations=1000):
    # Initialize value function
    V = np.zeros((ROWS, COLS))
    
    # Loop until convergence or max iterations
    for iteration in range(max_iterations):
        delta = 0
        new_V = np.copy(V)
        
        for row in range(ROWS):
            for col in range(COLS):
                state = (row, col)
                if state == TERMINAL_POSITIVE or state == TERMINAL_NEGATIVE:
                    continue  # Skip terminal states
                
                # Calculate the value for each action and choose the max
                action_values = []
                for action in ACTIONS:
                    expected_value = 0
                    next_states, probabilities = transition(state, action)
                    # Sum over the possible next states with their probabilities
                    for next_state, prob in zip(next_states, probabilities):
                        expected_value += prob * (reward_function(next_state, reward_type) + gamma * V[next_state])
                    action_values.append(expected_value)
                
                # Take the best action
                new_V[row, col] = np.max(action_values)
                
                # Track the largest change in value function
                delta = max(delta, np.abs(new_V[row, col] - V[row, col]))
        
        V = np.copy(new_V)
        
        # Check for convergence
        if delta < threshold:
            break
    
    return V

# Test the value iteration for different reward functions
rewards = ['r1', 'r2', 'r3', 'r4']
for reward_type in rewards:
    print(f"Value Iteration for reward function {reward_type}:")
    V = value_iteration(reward_type)
    print(V)
    print("\n")
