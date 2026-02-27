import polars as pl
import numpy as np

class KMeans:
    """A simple implementation of the K-Means clustering algorithm.

    This class partitions a dataset into a specified number of clusters by
    iteratively assigning each data point to the nearest centroid and then
    recalculating the centroids.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iters : int, default=100
        Maximum number of iterations of the k-means algorithm for a
        single run.
    tol : float, default=1e-4
        Relative tolerance with regards to the Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    random_state : int, default=None
        Determines random number generation for centroid initialization.
        Use an int to make the randomness deterministic.

    Attributes
    ----------
    centroids : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers, learned during fitting.
    """
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        """Compute k-means clustering and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Labels of each point.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 1. Initialize centroids randomly from the data points
        initial_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[initial_indices].astype(float)

        for _ in range(self.max_iters):
            # 2. Assign clusters
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            closest_centroids = np.argmin(distances, axis=1)

            # 3. Update centroids
            new_centroids = np.array([X[closest_centroids == i].mean(axis=0) for i in range(self.n_clusters)])

            # 4. Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        return closest_centroids
    
    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        This uses the centroids found by the `fit` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.centroids is None:
            raise RuntimeError("The model has not been fitted yet. Call 'fit' before 'predict'.")
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        closest_centroids = np.argmin(distances, axis=1)
        return closest_centroids
    

class KMedoids:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.medoids = None

    def _get_distances(self, X, medoids):
        """Calculate L1 distance (Manhattan) between X and medoids."""
        # K-Medoids often uses Manhattan distance for better robustness
        return np.sum(np.abs(X[:, np.newaxis] - medoids), axis=2)
    
    def fit(self, X):
        """Compute k-medoids clustering and return cluster labels."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 1. Initialize medoids randomly from the data points
        initial_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.medoids = X[initial_indices].astype(float)

        for _ in range(self.max_iters):
            # 2. Assign clusters based on L1 distance
            distances = self._get_distances(X, self.medoids)
            closest_medoids = np.argmin(distances, axis=1)

            # 3. Update medoids
            new_medoids = np.copy(self.medoids)
            for i in range(self.n_clusters):
                cluster_points = X[closest_medoids == i]
                if len(cluster_points) > 0:
                    # Find the point in the cluster that minimizes the total distance to other points
                    medoid_index = np.argmin(np.sum(np.abs(cluster_points[:, np.newaxis] - cluster_points), axis=2).sum(axis=1))
                    new_medoids[i] = cluster_points[medoid_index]

            # 4. Check for convergence
            if np.linalg.norm(new_medoids - self.medoids) < self.tol:
                break

            self.medoids = new_medoids

        return closest_medoids
    
    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        This uses the medoids found by the `fit` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.medoids is None:
            raise RuntimeError("The model has not been fitted yet. Call 'fit' before 'predict'.")
        distances = self._get_distances(X, self.medoids)
        closest_medoids = np.argmin(distances, axis=1)
        return closest_medoids
    

