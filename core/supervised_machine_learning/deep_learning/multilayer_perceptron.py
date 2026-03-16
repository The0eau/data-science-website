import numpy as np

# ==========================================
# 1. ACTIVATION FUNCTIONS & METRICS
# ==========================================

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function for backpropgation."""
    return x * (1 - x)

def mse_loss(y_true, y_pred):
    """Mean Squared Error loss function."""
    return np.mean(np.power(y_true - y_pred, 2))

# ==========================================
# 2. NEURAL NETWORK CLASS
# ==========================================

class NeuralNetworkScratch:
    """
    A simple 2-layer Neural Network (Multi-Layer Perceptron).
    
    Attributes:
        input_size (int): Number of input features.
        hidden_size (int): Number of neurons in the hidden layer.
        output_size (int): Number of output neurons.
        lr (float): Learning rate.
    """
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.lr = lr
        # Weight initialization (Random)
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """Perform forward propagation."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        """Perform backward propagation and update weights."""
        # Error in output
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        # Error in hidden layer
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.a1)

        # Update weights and biases using Gradient Descent
        self.W2 += self.a1.T.dot(output_delta) * self.lr
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * self.lr
        self.W1 += X.T.dot(hidden_delta) * self.lr
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.lr

    def train(self, X, y, epochs=1000):
        """Train the neural network for a fixed number of epochs."""
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                loss = mse_loss(y, output)
                print(f"Epoch {epoch} - Loss: {loss:.4f}")

# ==========================================
# 3. TEST AND EVALUATION
# ==========================================

if __name__ == "__main__":
    print("--- Neural Network Test (XOR Problem) ---")
    # XOR is a classic nonlinear problem
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialize: 2 inputs, 3 hidden neurons, 1 output
    nn = NeuralNetworkScratch(input_size=2, hidden_size=3, output_size=1, lr=0.5)
    nn.train(X, y, epochs=2000)

    # Predictions
    final_output = nn.forward(X)
    predictions = (final_output > 0.5).astype(int)
    
    print("\nFinal Probabilities:\n", final_output)
    print("Predictions:\n", predictions)
    print(f"Final Accuracy: {np.mean(predictions == y) * 100:.2f}%")