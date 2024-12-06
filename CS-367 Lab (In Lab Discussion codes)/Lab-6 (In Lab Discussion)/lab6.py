import numpy as np



class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        """
        Train the Hopfield network using Hebbian learning.
        :param patterns: List of binary patterns to store (-1 and 1 values).
        """
        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)  # Ensure it's a column vector
            self.weights += np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)  # No self-connections
        self.weights /= self.n_neurons  # Scale by the number of neurons

    def recall(self, pattern, max_iter=10):
        """
        Recall a pattern using the Hopfield network.
        :param pattern: Initial pattern (binary, -1 and 1 values).
        :param max_iter: Number of asynchronous update iterations.
        :return: Recalled pattern.
        """
        state = pattern.copy()
        for _ in range(max_iter):
            for i in range(self.n_neurons):
                net_input = np.dot(self.weights[i], state)
                state[i] = 1 if net_input >= 0 else -1
        return state

    def compute_capacity(self):
        """
        Estimate the capacity of the Hopfield network based on the theoretical limit.
        :return: Maximum number of patterns the network can store reliably.
        """
        return int(0.15 * self.n_neurons)

    def retrieval_accuracy(self, patterns, noise_level=0.1, max_iter=10):
        """
        Calculate retrieval accuracy by testing the network on noisy patterns.
        :param patterns: List of stored patterns.
        :param noise_level: Fraction of bits to flip in each pattern for testing.
        :param max_iter: Number of update iterations during recall.
        :return: Retrieval accuracy as a percentage.
        """
        correct_retrievals = 0
        for pattern in patterns:
            noisy_pattern = pattern.copy()
            noise_indices = np.random.choice(
                range(self.n_neurons), size=int(noise_level * self.n_neurons), replace=False
            )
            noisy_pattern[noise_indices] *= -1  # Introduce noise
            recalled_pattern = self.recall(noisy_pattern, max_iter)
            if np.array_equal(recalled_pattern, pattern):
                correct_retrievals += 1
        accuracy = (correct_retrievals / len(patterns)) * 100
        return accuracy

# Example usage
if __name__ == "__main__":
    # Define the Hopfield network with 10x10 neurons (N=100)
    n_neurons = 10 * 10
    hopfield_net = HopfieldNetwork(n_neurons)

    # Generate random binary patterns (-1 and 1)
    num_patterns = 10  # Number of patterns to store (less than capacity)
    patterns = [np.random.choice([-1, 1], n_neurons) for _ in range(num_patterns)]

    # Train the Hopfield network
    hopfield_net.train(patterns)

    # Test recall with a noisy version of a pattern
    test_pattern = patterns[0].copy()
    noise_indices = np.random.choice(range(n_neurons), size=10, replace=False)
    test_pattern[noise_indices] *= -1  # Flip bits to introduce noise
    recalled_pattern = hopfield_net.recall(test_pattern)

    # Print results
    print("Original Pattern:")
    print(patterns[0].reshape(10, 10))
    print("\nNoisy Input Pattern:")
    print(test_pattern.reshape(10, 10))
    print("\nRecalled Pattern:")
    print(recalled_pattern.reshape(10, 10))

    # Compute capacity
    capacity = hopfield_net.compute_capacity()
    print(f"\nTheoretical Capacity of the Hopfield Network: {capacity} patterns")

    # Compute retrieval accuracy
    accuracy = hopfield_net.retrieval_accuracy(patterns, noise_level=0.1)
    print(f"Retrieval Accuracy with 10% Noise: {accuracy:.2f}%")
