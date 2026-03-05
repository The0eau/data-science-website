import numpy as np
from collections import Counter

def accuracy_score(y_true, y_pred):
    """Calculate the accuracy percentage for classification."""
    return np.sum(y_true == y_pred) / len(y_true)

class KNNScratch:
    """
    K-Nearest Neighbors classifier from scratch.
    
    Attributes:
        k (int): Number of nearest neighbors to consider.
    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """
        In KNN, fitting just means storing the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict labels for a set of samples."""
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """Helper to predict a single sample."""
        # 1. Calculate Euclidean distances between x and all samples in X_train
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        
        # 2. Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Get the labels of these k neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Return the most common label (Majority Vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# ==========================================
# UPDATED MAIN TEST LOOP
# ==========================================

if __name__ == "__main__":
    # (Previous tests for Linear/Logistic here...)

    print("\n--- Testing k-Nearest Neighbors ---")
    # Simple dataset: group A (low values), group B (high values)
    X_train_knn = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    y_train_knn = np.array([0, 0, 1, 1, 0, 1]) # 0 = small, 1 = large
    
    knn = KNNScratch(k=3)
    knn.fit(X_train_knn, y_train_knn)
    
    # Testing with a new point [2, 2] - should be class 0
    X_test_knn = np.array([[2, 2], [7, 7]])
    preds_knn = knn.predict(X_test_knn)
    
    print(f"Predictions for {X_test_knn.tolist()}: {preds_knn}")
    # We can use the accuracy_score defined previously
    # If we knew the true labels were [0, 1]:
    print(f"KNN Accuracy: {accuracy_score(np.array([0, 1]), preds_knn) * 100}%")