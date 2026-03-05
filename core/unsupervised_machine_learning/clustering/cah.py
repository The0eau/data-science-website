import polars as pl
import numpy as np

class AHC:

    """Agglomerative Hierarchical Clustering (AHC).

    This implementation groups data points into a specified number of clusters
    by successively merging the closest pair of clusters.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters to find.
    linkage : {'ward', 'average'}, default='ward'
        The linkage criterion to use. This determines which distance to use
        between sets of observation.
        - 'ward': minimizes the variance of the clusters being merged.
        - 'average': uses the average of the distances of each observation of
          the two sets.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
    centroids_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers, computed after fitting.
    """

    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.centroids_ = None

    def _compute_distance(self, X):
        """Compute the pairwise Euclidean distance matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        ndarray of shape (n_samples, n_samples)
            The matrix of pairwise Euclidean distances.
        """
        n_samples = X.shape[0]
        # Euclidean distance matrix
        return np.sqrt(np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2))
    
    def fit(self, X):
        """Compute hierarchical clustering from features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        distance_matrix = self._compute_distance(X)
        # Initialize clusters as individual points
        clusters = [[i] for i in range(X.shape[0])]

        while len(clusters) > self.n_clusters:
            # Find the two closest clusters
            min_distance = np.inf
            to_merge = (0, 0)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if self.linkage == 'ward':
                        # Ward's method: minimize the variance within clusters
                        c1 = X[clusters[i]]
                        c2 = X[clusters[j]]
                        n1, n2 = len(c1), len(c2)
                        dist = (n1 * n2 / (n1 + n2)) * np.sum((c1.mean(axis=0) - c2.mean(axis=0)) ** 2)
                    elif self.linkage == 'average':
                        # Average linkage: average distance between points in clusters
                        dist = np.mean(distance_matrix[np.ix_(clusters[i], clusters[j])])
                    else:
                        raise ValueError(f"Unknown linkage type: {self.linkage}. "
                                         "Supported types are 'ward' and 'average'.")
                    
                    if dist < min_distance:
                        min_distance = dist
                        to_merge = (i, j)

            # Merge the closest clusters
            i, j = to_merge
            new_cluster = clusters[i] + clusters[j]
            # Remove old clusters and add the new one
            # Important to remove the one with the larger index first
            clusters.pop(max(i, j))
            clusters.pop(min(i, j))
            clusters.append(new_cluster)

        # Assign labels based on final clusters
        self.labels_ = np.zeros(X.shape[0], dtype=int)
        self.centroids_ = np.zeros((self.n_clusters, X.shape[1]))
        for idx, cluster in enumerate(clusters):
            self.labels_[cluster] = idx
            self.centroids_[idx] = X[cluster].mean(axis=0)

        return self
    
    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        This method is not part of the standard AHC algorithm but is provided
        for convenience. It assigns new points to the cluster with the
        closest centroid, as found by the `fit` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.centroids_ is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        
        # Compute Euclidean distance from each point in X to each centroid
        distances = np.sqrt(np.sum((X[:, np.newaxis] - self.centroids_) ** 2, axis=2))
        
        # Return the index of the closest centroid
        return np.argmin(distances, axis=1)
    