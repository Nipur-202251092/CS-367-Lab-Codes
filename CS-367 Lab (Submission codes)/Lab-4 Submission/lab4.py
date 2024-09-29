import numpy as np
import matplotlib.pyplot as plt
import random

def load_puzzle(filename):
    matrix = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    is_matrix = False
    for line in lines:
        if is_matrix:
            values = [int(x) for x in line.strip().split() if x.isdigit()]
            if values:
                matrix.extend(values)

        if line.startswith("# ndims: 2"):
            is_matrix = True

    if len(matrix) != 512 * 512:
        raise ValueError(f"Matrix data has incorrect length: {len(matrix)} elements, expected 512x512 = {512*512} elements.")

    matrix = np.array(matrix, dtype=np.uint8).reshape((512, 512))
    return matrix

# Split the image into 4x4 pieces (each piece is 128x128)
def split_image_to_pieces(image, piece_size=128):
    pieces = []
    for i in range(0, image.shape[0], piece_size):
        for j in range(0, image.shape[1], piece_size):
            piece = image[i:i + piece_size, j:j + piece_size]
            pieces.append(piece)
    return pieces

# Combine pieces back into the original image
def combine_pieces(pieces, grid_size=4, piece_size=128):
    image = np.zeros((grid_size * piece_size, grid_size * piece_size), dtype=np.uint8)
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            image[i * piece_size:(i + 1) * piece_size, j * piece_size:(j + 1) * piece_size] = pieces[idx]
            idx += 1
    return image

def generate_neighbor_pieces(state):
    neighbor = state.copy()
    idx1, idx2 = random.sample(range(len(state)), 2)
    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
    return neighbor


def improved_cost_function(state, piece_size=128):
    cost = 0
    grid_size = int(np.sqrt(len(state)))
    edge_match_weight = 10

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            
            # Current piece edges
            top_edge = state[idx][0, :]      # Top edge
            bottom_edge = state[idx][-1, :]  # Bottom edge
            left_edge = state[idx][:, 0]     # Left edge
            right_edge = state[idx][:, -1]   # Right edge

            # Check right neighbor
            if j < grid_size - 1:  # Right neighbor exists
                right_idx = i * grid_size + (j + 1)
                neighbor_left_edge = state[right_idx][:, 0]  # Left edge of the right piece
                cost += np.sum(right_edge != neighbor_left_edge) * edge_match_weight
            
            # Check bottom neighbor
            if i < grid_size - 1:  # Bottom neighbor exists
                bottom_idx = (i + 1) * grid_size + j
                neighbor_top_edge = state[bottom_idx][0, :]  # Top edge of the bottom piece
                cost += np.sum(bottom_edge != neighbor_top_edge) * edge_match_weight

    return cost


def simulated_annealing_pieces(initial_state, initial_temp, cooling_rate, max_iterations, restart_limit=10):
    best_state = initial_state
    best_cost = improved_cost_function(best_state)

    for restart in range(restart_limit):
        current_state = initial_state
        current_cost = improved_cost_function(current_state)
        temperature = initial_temp

        for iteration in range(max_iterations):
            neighbor = generate_neighbor_pieces(current_state)
            neighbor_cost = improved_cost_function(neighbor)

            delta_cost = neighbor_cost - current_cost

            if delta_cost < 0 or random.uniform(0, 1) < np.exp(-delta_cost / temperature):
                current_state = neighbor
                current_cost = neighbor_cost


                if current_cost == 0:  
                    print(f"Puzzle solved at iteration {iteration} in restart {restart + 1}!")
                    return current_state 

            temperature *= cooling_rate

            if iteration % 500 == 0:
                print(f"Restart {restart + 1}, Iteration {iteration}, Cost: {current_cost:.2f}, Temp: {temperature:.4f}")

        if current_cost < best_cost:
            best_cost = current_cost
            best_state = current_state

    print("Best cost after all restarts:", best_cost)
    return best_state

def visualize_puzzle(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":

    scrambled_puzzle = load_puzzle('scrambled_lena.mat')
    puzzle_pieces = split_image_to_pieces(scrambled_puzzle)

    initial_temperature = 1.0
    cooling_rate = 0.95
    max_iterations = 1000 
    restart_limit = 20 
    random.seed(42)

    solved_puzzle_pieces = simulated_annealing_pieces(puzzle_pieces, initial_temperature, cooling_rate, max_iterations, restart_limit)

    solved_puzzle = combine_pieces(solved_puzzle_pieces)

    visualize_puzzle(solved_puzzle)
