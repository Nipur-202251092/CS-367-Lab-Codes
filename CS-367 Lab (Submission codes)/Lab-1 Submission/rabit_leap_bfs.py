from collections import deque

def bfs(initial_state, goal_state):
    queue = deque([(initial_state, [])])  
    visited = set()  
    visited.add(tuple(initial_state))
    
    while queue:
        state, path = queue.popleft()
        
        if state == goal_state:
            return path + [state]
               
        for next_state in get_next_states(state):     # Generate next possible states
            if tuple(next_state) not in visited:
                visited.add(tuple(next_state))
                queue.append((next_state, path + [state]))

    return None  

def get_next_states(state):
    next_states = []
    empty_index = state.index('_')
    
    # Move right-bound rabbit
    if empty_index > 0 and state[empty_index - 1] == 'R':  # R moves one step right
        next_state = state[:]
        next_state[empty_index], next_state[empty_index - 1] = next_state[empty_index - 1], next_state[empty_index]
        next_states.append(next_state)
    if empty_index > 1 and state[empty_index - 2] == 'R':  # R jumps over one L
        next_state = state[:]
        next_state[empty_index], next_state[empty_index - 2] = next_state[empty_index - 2], next_state[empty_index]
        next_states.append(next_state)
    
    # Move left-bound rabbit
    if empty_index < len(state) - 1 and state[empty_index + 1] == 'L':  # L moves one step left
        next_state = state[:]
        next_state[empty_index], next_state[empty_index + 1] = next_state[empty_index + 1], next_state[empty_index]
        next_states.append(next_state)
    if empty_index < len(state) - 2 and state[empty_index + 2] == 'L':  # L jumps over one R
        next_state = state[:]
        next_state[empty_index], next_state[empty_index + 2] = next_state[empty_index + 2], next_state[empty_index]
        next_states.append(next_state)
    
    return next_states

# Initial and goal states
initial_state = ['R', 'R', 'R', '_', 'L', 'L', 'L']
goal_state = ['L', 'L', 'L', '_', 'R', 'R', 'R']


result = bfs(initial_state, goal_state)
if result:
    print("\nSolution Found\n\n\nPath :\n")
    for step in result:
        print(step)
else:
    print("No solution found.")
