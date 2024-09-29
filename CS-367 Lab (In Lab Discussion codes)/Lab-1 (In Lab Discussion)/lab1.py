from collections import deque

def get_successors(state):
    successors = []
    for i in range(7):
        if state[i] == 1:
            if(i+1<7):
                if(state[i+1] == 0):
                    new_state = list(state)
                    new_state[i] = 0
                    new_state[i+1] = 1
                    successors.append(tuple(new_state))
            if(i+2<7):
                if(state[i+2] == 0):
                    new_state = list(state)
                    new_state[i] = 0
                    new_state[i+2] = 1
                    successors.append(tuple(new_state))
        if state[i] == -1:
            if(i-1>=0):
                if(state[i-1] == 0):
                    new_state = list(state)
                    new_state[i] = 0
                    new_state[i-1] = -1
                    successors.append(tuple(new_state))
            if(i-2>=0):
                if(state[i-2] == 0):
                    new_state = list(state)
                    new_state[i] = 0
                    new_state[i-2] = -1
                    successors.append(tuple(new_state))
    return successors

def dfs(start_state, goal_state,dfs_comp):
    stack = list([(start_state, [])])
    visited = set()
    while stack:
        (state, path) = stack.pop()
        dfs_comp += 1
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if state == goal_state:
            print("Total nodes explored", dfs_comp)
            return path
        for successor in get_successors(state):
            stack.append((successor, path))
    return None

def bfs(start_state, goal_state,bfs_comp):
    queue = deque([(start_state, [])])
    visited = set()
    while queue:
        (state, path) = queue.popleft()
        bfs_comp += 1
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if state == goal_state:
            print("Total nodes explored", bfs_comp)
            return path
        for successor in get_successors(state):
            queue.append((successor, path))
    return None

start_state = (1,1,1,0,-1,-1,-1)
goal_state = (-1,-1,-1,0,1,1,1)
dfs_comp = 0
bfs_comp = 0
print("DFS Solution")
solution = dfs(start_state, goal_state,dfs_comp)
dfs_steps = 0
if solution:
    print("Solution found:")
    print("Number of steps:", len(solution)-1)
    for step in solution:
        print(step)
else:
    print("No solution found.")

print("BFS Solution")
solution = bfs(start_state, goal_state,bfs_comp)
bfs_steps = 0
if solution:
    print("Solution found:")
    print("Number of steps:", len(solution)-1)
    for step in solution:
        print(step)
else:
    print("No solution found.")