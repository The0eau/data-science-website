import numpy as np

def accuracy_score(y_true, y_pred):
    """Calculate the accuracy percentage for classification."""
    return np.sum(y_true == y_pred) / len(y_true)

class CentroidPartitionClassifier:
    """
    Classification by spatial partitioning using class centroids.
    
    This model partitions the space into regions defined by the mean 
    position of each class.
    """
    def __init__(self):
        self.centroids = {}
        self.classes = None

    def fit(self, X, y):
        """
        Calculate the centroid (mean vector) for each class partition.
        """
        self.classes = np.unique(y)
        for cls in self.classes:
            # Partition the data by class and calculate the mean (centroid)
            X_cls = X[y == cls]
            self.centroids[cls] = np.mean(X_cls, axis=0)
        print(f"Space partitioned into {len(self.classes)} regions.")

    def predict(self, X):
        """
        Assign each point to the partition of the nearest centroid.
        """
        predictions = [self._assign_partition(x) for x in X]
        return np.array(predictions)

    def _assign_partition(self, x):
        """Find the closest centroid using Euclidean distance."""
        distances = {cls: np.sqrt(np.sum((x - c)**2)) for cls, c in self.centroids.items()}
        # Return the class of the nearest partition center
        return min(distances, key=distances.get)

# ==========================================
# MAIN TEST LOOP
# ==========================================

if __name__ == "__main__":
    # Sample spatial data
    X = np.array([[1, 1], [2, 1], [8, 8], [9, 8]])
    y = np.array([0, 0, 1, 1]) # Class 0 (Bottom-left), Class 1 (Top-right)

    model = CentroidPartitionClassifier()
    model.fit(X, y)

    # Test a point in the middle
    test_point = np.array([[3, 3], [7, 7]])
    preds = model.predict(test_point)

    print(f"Test Points: {test_point.tolist()}")
    print(f"Assigned Partitions: {preds}")
    
    # Using the accuracy metric from our previous script
    print(f"Partition Accuracy: {accuracy_score(np.array([0, 1]), preds) * 100}%")