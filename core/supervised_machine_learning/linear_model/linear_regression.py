import numpy as np


def mean_squared_error(y_true, y_pred):
    """Calculate the Mean Squared Error between true and predicted values."""
    return np.mean((y_true - y_pred)**2)

class LinearRegressionScratch:
    """
    Linear Regression model trained using Gradient Descent.
    
    Attributes:
        lr (float): Learning rate for gradient descent.
        iterations (int): Number of passes over the training dataset.
        weights (numpy.ndarray): Weights assigned to features after training.
        bias (float): Bias (intercept) term after training.
    """
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Args:
            X (numpy.ndarray): Training features of shape (n_samples, n_features).
            y (numpy.ndarray): Target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """Predict target values for given features."""
        return np.dot(X, self.weights) + self.bias



class RegularizedRegression:
    """
    Linear Regression with L1 (Lasso) and L2 (Ridge) regularization.
    
    Args:
        lr (float): Learning rate.
        iterations (int): Iterations.
        l1_pen (float): L1 penalty coefficient (Lambda for Lasso).
        l2_pen (float): L2 penalty coefficient (Lambda for Ridge).
    """
    def __init__(self, lr=0.01, iterations=1000, l1_pen=0, l2_pen=0):
        self.lr = lr
        self.iterations = iterations
        self.l1_pen = l1_pen
        self.l2_pen = l2_pen
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Gradients of penalties
            l1_derivative = self.l1_pen * np.sign(self.weights)
            l2_derivative = self.l2_pen * 2 * self.weights
            
            dw = (1/n_samples) * (np.dot(X.T, (y_pred - y)) + l1_derivative + l2_derivative)
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    print("--- Testing Linear Regression ---")
    X_lin = np.array([[1], [2], [3], [4], [5]])
    y_lin = np.array([2.1, 3.9, 6.1, 8.0, 10.2]) # Close to y = 2x
    lin_model = LinearRegressionScratch(learning_rate=0.01, iterations=1000)
    lin_model.fit(X_lin, y_lin)
    preds_lin = lin_model.predict(X_lin)
    print(f"MSE: {mean_squared_error(y_lin, preds_lin):.4f}")

    print("\n--- Testing Regularized Regression (Ridge) ---")
    # Using L2 penalty (Ridge)
    reg_model = RegularizedRegression(lr=0.01, iterations=1000, l2_pen=5.0)
    reg_model.fit(X_lin, y_lin)
    preds_reg = reg_model.predict(X_lin)
    print(f"MSE with Ridge: {mean_squared_error(y_lin, preds_reg):.4f}")
    print(f"Weights (Linear vs Ridge): {lin_model.weights[0]:.2f} vs {reg_model.weights[0]:.2f}")