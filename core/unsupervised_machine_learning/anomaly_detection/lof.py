import numpy as np

class LOF:
    """
    Local Outlier Factor (LOF) for anomaly detection.
    
    Parameters
    ----------
    k : int, default=20
        Number of neighbors to consider (k-distance).
    """
    def __init__(self, k=20):
        self.k = k

    def _get_distances(self, X):
        """Compute pairwise Euclidean distances."""
        sum_X = np.sum(np.square(X), axis=1)
        dist_sq = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return np.sqrt(np.maximum(dist_sq, 0))

    def fit_predict(self, X):
        """
        Calculates LOF scores and returns them.
        Higher scores indicate a higher probability of being an outlier.
        """
        n_samples = X.shape[0]
        distances = self._get_distances(X)
        
        # 1. Find k-nearest neighbors and k-distances
        # We sort and take indices 1 to k+1 because index 0 is the point itself
        neighbors_idx = np.argsort(distances, axis=1)[:, 1:self.k+1]
        k_distances = np.zeros(n_samples)
        for i in range(n_samples):
            k_distances[i] = distances[i, neighbors_idx[i, -1]]

        # 2. Local Reachability Density (LRD)
        lrd = np.zeros(n_samples)
        for i in range(n_samples):
            # reach-dist(i, o) = max(k-dist(o), dist(i, o))
            reach_dists = np.maximum(k_distances[neighbors_idx[i]], distances[i, neighbors_idx[i]])
            lrd[i] = 1.0 / (np.mean(reach_dists) + 1e-10)

        # 3. Local Outlier Factor (LOF)
        lof_scores = np.zeros(n_samples)
        for i in range(n_samples):
            # Ratio of average LRD of neighbors to LRD of point i
            neighbors_lrd = lrd[neighbors_idx[i]]
            lof_scores[i] = np.mean(neighbors_lrd) / lrd[i]
            
        return lof_scores

# --- TESTING ON TOY DATASET ---
if __name__ == "__main__":
    # Create a dense cluster and one outlier
    np.random.seed(42)
    cluster = np.random.normal(0, 1, (50, 2))
    outlier = np.array([[10, 10]])
    X = np.vstack([cluster, outlier])

    detector = LOF(k=10)
    scores = detector.fit_predict(X)

    print("--- LOF Test Results ---")
    print(f"LOF Score for normal point (avg): {np.mean(scores[:-1]):.4f}")
    print(f"LOF Score for outlier [10, 10]: {scores[-1]:.4f}")