import numpy as np

class Autoencoder:
    """
    Simple Shallow Autoencoder for unsupervised feature learning.

    This neural network compresses input data into a lower-dimensional 
    latent representation (encoding) and then attempts to reconstruct 
    the original input (decoding). It is trained using Mean Squared Error (MSE) 
    and backpropagation.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    encoding_dim : int
        Size of the bottleneck (latent space).
    learning_rate : float, default=0.01
        Step size for weight updates during gradient descent.

    Attributes
    ----------
    W_enc : ndarray
        Weights for the encoder layer.
    b_enc : ndarray
        Bias for the encoder layer.
    W_dec : ndarray
        Weights for the decoder layer.
    b_dec : ndarray
        Bias for the decoder layer.
    """

    def __init__(self, input_dim, encoding_dim, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values to break symmetry
        self.W_enc = np.random.randn(input_dim, encoding_dim) * 0.1
        self.b_enc = np.zeros(encoding_dim)
        
        self.W_dec = np.random.randn(encoding_dim, input_dim) * 0.1
        self.b_dec = np.zeros(input_dim)

    def sigmoid(self, x):
        """Compute the sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # Clip to avoid overflow
    
    def _sigmoid_derivative(self, sigmoid_output):
        """Compute derivative using the output of the sigmoid function."""
        return sigmoid_output * (1 - sigmoid_output)
    
    def fit(self, X, epochs=100):
        """
        Train the autoencoder using the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        epochs : int, default=100
            Number of iterations over the training set.
        """
        for epoch in range(epochs):
            # Forward pass
            z_enc = X @ self.W_enc + self.b_enc
            encoded = self.sigmoid(z_enc)
            
            z_dec = encoded @ self.W_dec + self.b_dec
            decoded = self.sigmoid(z_dec)
            
            # Compute loss (Mean Squared Error)
            loss = np.mean((X - decoded) ** 2)
            
            # Backpropagation - Output Layer
            error_dec = (decoded - X) / X.shape[0] # Normalizing by batch size
            d_z_dec = error_dec * self._sigmoid_derivative(decoded)
            
            # Backpropagation - Hidden Layer
            error_enc = d_z_dec @ self.W_dec.T
            d_z_enc = error_enc * self._sigmoid_derivative(encoded)
            
            # Update weights and biases (Gradient Descent)
            self.W_dec -= (encoded.T @ d_z_dec) * self.learning_rate
            self.b_dec -= np.sum(d_z_dec, axis=0) * self.learning_rate
            
            self.W_enc -= (X.T @ d_z_enc) * self.learning_rate
            self.b_enc -= np.sum(d_z_enc, axis=0) * self.learning_rate
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}')
        return self
    
    def transform(self, X):
        """
        Encode the data into the latent space.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to compress.

        Returns
        -------
        encoded : ndarray of shape (n_samples, encoding_dim)
        """
        z_enc = X @ self.W_enc + self.b_enc
        return self.sigmoid(z_enc)
    
    def fit_transform(self, X, epochs=100):
        """Fit the model and return the compressed representation."""
        self.fit(X, epochs)
        return self.transform(X)

    def reconstruct(self, X):
        """
        Pass data through the full Autoencoder (Encode then Decode).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to reconstruct.

        Returns
        -------
        reconstructed : ndarray of shape (n_samples, n_features)
        """
        encoded = self.transform(X)
        z_dec = encoded @ self.W_dec + self.b_dec
        return self.sigmoid(z_dec)