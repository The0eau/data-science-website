import numpy as np

class OneClassSVM:
    """
    One-Class Support Vector Machine for Anomaly Detection.
    
    This implementation uses Stochastic Gradient Descent to find a 
    hyperplane that separates the data from the origin in a high-dimensional space.
    
    Parameters
    ----------
    nu : float, default=0.1
        An upper bound on the fraction of training errors and a 
        lower bound of the fraction of support vectors.
    learning_rate : float, default=0.01
    n_epochs : int, default=100
    """
    def __init__(self, nu=0.1, learning_rate=0.01, n_epochs=100):
        self.nu = nu
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.rho = 0.0

    def fit(self, X):
        """
        Fits the model to the data by finding the optimal hyperplane.
        """
        n_samples, n_features = X.shape
        # Initialize weights
        self.w = np.zeros(n_features)
        self.rho = 0.0
        
        for _ in range(self.n_epochs):
            for i in range(n_samples):
                # Decision function: <w, x> - rho
                score = np.dot(self.w, X[i]) - self.rho
                
                # Update rule (Hinge Loss for One-Class)
                if score < 0:
                    # Update weights to push the point inside the boundary
                    self.w = (1 - self.lr) * self.w + (self.lr / (self.nu * n_samples)) * X[i]
                    self.rho = self.rho - self.lr * (1 - 1/self.nu)
                else:
                    # Regularization step
                    self.w = (1 - self.lr) * self.w
                    self.rho = self.rho - self.lr
        return self

    def decision_function(self, X):
        """Returns the distance of each sample from the boundary."""
        return np.dot(X, self.w) - self.rho

    def predict(self, X):
        """Returns 1 for inliers and -1 for outliers."""
        return np.where(self.decision_function(X) >= 0, 1, -1)

# --- MAIN TESTING BLOCK ---
if __name__ == "__main__":
    # Generate normal cluster
    X_train = np.random.normal(0, 1, (100, 2))
    
    # Test with a normal point and an outlier
    X_test = np.array([[0.1, 0.1], [8.0, 8.0]])
    
    ocsvm = OneClassSVM(nu=0.1, n_epochs=200)
    ocsvm.fit(X_train)
    
    scores = ocsvm.decision_function(X_test)
    preds = ocsvm.predict(X_test)
    
    print("--- One-Class SVM Test Results ---")
    print(f"Normal point score: {scores[0]:.4f} | Prediction: {preds[0]}")
    print(f"Outlier point score: {scores[1]:.4f} | Prediction: {preds[1]}")