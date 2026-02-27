import numpy as np


class IsolationTree:
    """A single tree for the Isolation Forest."""
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.size = 0
        self.node = None # Stores (split_feature, split_value, left_tree, right_tree)

    def fit(self, X, current_height=0):
        self.size = X.shape[0]
        
        # Termination conditions: height limit reached or only 1 sample left
        if current_height >= self.height_limit or self.size <= 1:
            return self

        # Randomly select a feature and a split value
        n_features = X.shape[1]
        split_feature = np.random.randint(0, n_features)
        
        min_val, max_val = X[:, split_feature].min(), X[:, split_feature].max()
        if min_val == max_val:
            return self # Cannot split further
            
        split_value = np.random.uniform(min_val, max_val)

        # Split the data
        left_mask = X[:, split_feature] < split_value
        right_mask = ~left_mask

        # Recursively build sub-trees
        self.node = (
            split_feature, 
            split_value,
            IsolationTree(self.height_limit).fit(X[left_mask], current_height + 1),
            IsolationTree(self.height_limit).fit(X[right_mask], current_height + 1)
        )
        return self

    def path_length(self, x, current_height=0):
        if self.node is None:
            # Correction factor for leaf nodes (c(n) logic)
            return current_height + (self._c(self.size) if self.size > 1 else 0)

        split_feature, split_value, left_tree, right_tree = self.node
        if x[split_feature] < split_value:
            return left_tree.path_length(x, current_height + 1)
        else:
            return right_tree.path_length(x, current_height + 1)

    def _c(self, n):
        """Normalization factor for path length."""
        return 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)


class IsolationForest:
    """
    Isolation Forest for Anomaly Detection.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    sample_size : int, default=256
        The number of samples to draw to train each tree.
    """
    def __init__(self, n_estimators=100, sample_size=256):
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.trees = []
        self.limit = int(np.ceil(np.log2(sample_size))) # Max tree height

    def fit(self, X):
        """Fit the forest to the data."""
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Subsample the data
            idx = np.random.choice(n_samples, min(n_samples, self.sample_size), replace=False)
            tree = IsolationTree(self.limit)
            tree.fit(X[idx])
            self.trees.append(tree)
        return self

    def decision_function(self, X):
        """Compute the anomaly score for each sample."""
        # Average path length across all trees
        avg_paths = np.array([
            np.mean([tree.path_length(x) for tree in self.trees]) 
            for x in X
        ])
        
        # Normalize scores (2^-E/c)
        c_n = self.trees[0]._c(self.sample_size)
        scores = 2 ** (-avg_paths / c_n)
        return scores

    def predict(self, X, threshold=0.5):
        """
        Predict if a sample is an anomaly.
        Returns -1 for anomalies and 1 for normal points.
        """
        scores = self.decision_function(X)
        return np.where(scores > threshold, -1, 1)


# --- TESTING ON TOY DATASET ---
if __name__ == "__main__":
    # 1. Generate normal data (clustered around 0,0)
    np.random.seed(42)
    X_normal = np.random.normal(0, 1, (100, 2))
    
    # 2. Add an anomaly (far away from the cluster)
    X_outlier = np.array([[10, 10]])
    X = np.vstack([X_normal, X_outlier])
    
    # 3. Fit the model
    clf = IsolationForest(n_estimators=50, sample_size=64)
    clf.fit(X)
    
    # 4. Predict
    scores = clf.decision_function(X)
    predictions = clf.predict(X, threshold=0.6)
    
    print("--- Isolation Forest Test Results ---")
    print(f"Total points: {len(X)}")
    print(f"Anomaly Score for the outlier [10, 10]: {scores[-1]:.4f}")
    print(f"Prediction for the outlier (Expected -1): {predictions[-1]}")
    print(f"Average Score for normal points: {np.mean(scores[:-1]):.4f}")
        