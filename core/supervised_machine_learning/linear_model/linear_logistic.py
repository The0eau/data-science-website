import numpy as np

def accuracy_score(y_true, y_pred):
    """Calculate the accuracy percentage for classification."""
    return np.sum(y_true == y_pred) / len(y_true)

class LogisticRegressionScratch:
    """
    Logistic Regression for binary classification using the Sigmoid function.
    
    Attributes:
        lr (float): Learning rate.
        iterations (int): Gradient descent iterations.
        weights (numpy.ndarray): Model coefficients.
        bias (float): Model intercept.
    """
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Fit the classifier to the training data (y must be 0 or 1)."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """Predict class labels (0 or 1)."""
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return np.array([1 if i > 0.5 else 0 for i in y_predicted])

if __name__ == "__main__":
    print("\n--- Testing Logistic Regression ---")
    X_log = np.array([[1], [2], [3], [7], [8], [9]])
    y_log = np.array([0, 0, 0, 1, 1, 1])
    log_model = LogisticRegressionScratch(learning_rate=0.1, iterations=1000)
    log_model.fit(X_log, y_log)
    preds_log = log_model.predict(X_log)
    print(f"Accuracy: {accuracy_score(y_log, preds_log) * 100:.2f}%")