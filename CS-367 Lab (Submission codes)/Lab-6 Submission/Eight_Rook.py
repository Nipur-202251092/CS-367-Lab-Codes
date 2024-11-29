import numpy as np

# Define the energy function for the Eight-Rooks problem
def energy_function(board):
    """Calculate the energy of the current board configuration."""
    energy = 0
    # Penalize for multiple rooks in the same row
    for row in range(8):
        energy += sum(board[row]) * (sum(board[row]) - 1)  # Rook conflicts in the same row
    
    # Penalize for multiple rooks in the same column
    for col in range(8):
        column_sum = sum(board[row][col] for row in range(8))
        energy += column_sum * (column_sum - 1)  # Rook conflicts in the same column
    
    return energy

# Define the Hopfield Network for the Eight-Rooks problem
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size * size, size * size))  # Initialize weights to zero
        self.board = np.zeros((size, size))  # Initialize a zero board
    
    def train(self):
        """Train the Hopfield network to minimize energy."""
        for row in range(self.size):
            for col in range(self.size):
                for row2 in range(self.size):
                    for col2 in range(self.size):
                        idx1 = row * self.size + col
                        idx2 = row2 * self.size + col2
                        
                        # Penalize for placing rooks in the same row or column
                        if row != row2 or col != col2:
                            if row == row2 or col == col2:
                                self.weights[idx1, idx2] = -2
                            else:
                                self.weights[idx1, idx2] = 0
    
    def update(self):
        """Update the network's state based on the current weights and input."""
        prev_board = self.board.copy()
        for i in range(self.size):
            for j in range(self.size):
                idx = i * self.size + j
                # Calculate the net input (weighted sum of other neurons)
                net_input = np.sum(self.weights[idx, :] * self.board.flatten())  
                # Update state based on net input
                self.board[i, j] = 1 if net_input >= 0 else 0
        
        # Ensure that we always have exactly 8 rooks (1's) on the board
        total_rooks = np.sum(self.board)
        if total_rooks != 8:
            # Re-set the board to a state with exactly 8 rooks randomly placed
            self.board = np.zeros((8, 8))
            rook_positions = np.random.choice(64, 8, replace=False)
            for pos in rook_positions:
                row, col = divmod(pos, 8)
                self.board[row, col] = 1

    def run(self, steps=100):
        """Run the Hopfield network for a number of steps."""
        for _ in range(steps):
            self.update()
            if energy_function(self.board) == 0:  # If no conflicts, stop
                break

# Solve the Eight-Rooks problem using the Hopfield Network
def solve_eight_rooks():
    # Initialize the Hopfield network
    hopfield = HopfieldNetwork(size=8)

    # Initialize board with random placement of rooks
    # We will randomly place 8 rooks on the board (random `1`s, no constraint on row/column uniqueness)
    num_rooks = 8  # Number of rooks to place
    positions = np.random.choice(8 * 8, num_rooks, replace=False)  # Random positions for 8 rooks

    # Place rooks (1s) randomly
    hopfield.board = np.zeros((8, 8))
    for pos in positions:
        row, col = divmod(pos, 8)
        hopfield.board[row, col] = 1

    # Print initial configuration
    print("Initial Configuration (Before Running Hopfield Network):")
    for row in hopfield.board:
        print(" ".join(str(int(cell)) for cell in row))

    # Train the network
    hopfield.train()
    
    # Run the network until it converges to a solution
    hopfield.run()
    
    # Print final configuration
    print("\nFinal Configuration (After Running Hopfield Network):")
    for row in hopfield.board:
        print(" ".join(str(int(cell)) for cell in row))
    
    return hopfield.board

# Main function to display the solution
if __name__ == "__main__":
    solve_eight_rooks()
