import polars as pl
import numpy as np

class DBSCAN:
    """A simple implementation of the DBSCAN clustering algorithm.

    This class identifies clusters based on the density of data points, allowing
    for the discovery of clusters of arbitrary shape and handling of noise.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for them to be considered as
        in the same neighborhood.
    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point to
        be considered as a core point. This includes the point itself.
    """
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.X_train_ = None

    def _get_neighbors(self, X, point_idx):
        """Find all points in `X` within `eps` distance of a given point.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The full dataset.
        point_idx : int
            The index of the point for which to find neighbors.

        Returns
        -------
        ndarray
            An array of indices of the neighboring points.
        """
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def fit(self, X):
        """Perform DBSCAN clustering from features or distance matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_train_ = X
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -2)  # -2 means 'unvisited'
        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != -2:
                continue
            
            neighbors = self._get_neighbors(X, i)
            
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # Mark as noise
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1
        return self

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        """Expand a cluster from a core point by exploring its neighbors.

        This method is called when a core point is found. It recursively
        finds all density-reachable points and assigns them to the current
        cluster.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The full dataset.
        point_idx : int
            The index of the core point to start expansion from.
        neighbors : list of int
            The list of neighbors of the core point.
        cluster_id : int
            The ID to assign to the new cluster.
        """
        self.labels_[point_idx] = cluster_id
        seeds = list(neighbors)
        
        i = 0
        while i < len(seeds):
            current_p = seeds[i]
            
            # If the point was marked as noise, it becomes a border point
            if self.labels_[current_p] == -1:
                self.labels_[current_p] = cluster_id
            
            # If the point is unvisited
            elif self.labels_[current_p] == -2:
                self.labels_[current_p] = cluster_id
                current_neighbors = self._get_neighbors(X, current_p)
                
                if len(current_neighbors) >= self.min_samples:
                    # Add new neighbors to the seeds list to explore
                    for n in current_neighbors:
                        if n not in seeds:
                            seeds.append(n)
            i += 1

    def predict(self):
        """Return cluster labels for the data passed to fit().

        Note: DBSCAN is a transductive algorithm, meaning it does not have a
        separate `predict` method for new, unseen data in the same way as
        algorithms like K-Means. This method simply returns the labels
        assigned during the `fit` process.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point in the dataset given to fit().
            Noise points are labeled -1.
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted before calling predict().")
        return self.labels_


class OPTICS:
    """A simple implementation of the OPTICS clustering algorithm.

    This class identifies clusters based on the density of data points, similar
    to DBSCAN, but can handle varying densities and does not require a fixed
    distance threshold.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for them to be considered as
        in the same neighborhood.
    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point to
        be considered as a core point. This includes the point itself.
    """
    def __init__(self, eps=np.inf, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.ordered_list_ = []
        self.reachability_ = None
        self.core_distances_ = None

    def fit(self, X):
        """Perform OPTICS clustering from features.

        Analyzes the data and produces an ordering of the points and their
        reachability distances, which can be used to extract clusters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        n_samples = X.shape[0]
        self.reachability_ = np.full(n_samples, np.inf)
        self.core_distances_ = np.full(n_samples, np.inf)
        self.ordered_list_ = []
        visited = np.zeros(n_samples, dtype=bool)

        # 1. Pre-calculate distances
        dist_matrix = np.sqrt(np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2))

        # 2. Compute core distances
        for i in range(n_samples):
            sorted_dists = np.sort(dist_matrix[i])
            if len(sorted_dists) >= self.min_samples:
                self.core_distances_[i] = sorted_dists[self.min_samples - 1]

        # 3. Process points
        for i in range(n_samples):
            if not visited[i]:
                visited[i] = True
                self.ordered_list_.append(i)
                
                if self.core_distances_[i] != np.inf:
                    seeds = []
                    self._update_seeds(i, visited, seeds, dist_matrix)
                    
                    while seeds:
                        # Priority Queue logic: pick point with smallest reachability
                        seeds.sort(key=lambda idx: self.reachability_[idx])
                        next_p = seeds.pop(0)
                        visited[next_p] = True
                        self.ordered_list_.append(next_p)
                        
                        if self.core_distances_[next_p] != np.inf:
                            self._update_seeds(next_p, visited, seeds, dist_matrix)
        return self

    def _update_seeds(self, p_idx, visited, seeds, dist_matrix):
        """Update the reachability distances of neighbors for a given point.

        For a given core point, this method iterates through its neighbors,
        calculates their reachability distance, and updates them in the
        priority queue (seeds) if a shorter path is found.

        Parameters
        ----------
        p_idx : int
            The index of the current core point.
        visited : ndarray of shape (n_samples,)
            Array indicating whether a point has been visited.
        seeds : list
            The priority queue of points to visit, stored as a list of indices.
        dist_matrix : ndarray of shape (n_samples, n_samples)
            Pre-computed distance matrix for the dataset.
        """
        core_dist = self.core_distances_[p_idx]
        neighbors = np.where(dist_matrix[p_idx] <= self.eps)[0]
        
        for o in neighbors:
            if not visited[o]:
                # Reachability = max(Core-dist of source, Distance to source)
                new_reach = max(core_dist, dist_matrix[p_idx, o])
                
                if self.reachability_[o] == np.inf:
                    self.reachability_[o] = new_reach
                    seeds.append(o)
                elif new_reach < self.reachability_[o]:
                    self.reachability_[o] = new_reach

    def get_reachability_plot(self):
        """Get the reachability distances for the reachability plot.

        The reachability plot is a bar chart that shows the reachability
        distance for each point in the ordered list. Valleys in the plot
        correspond to clusters.

        Returns
        -------
        ndarray
            An array of reachability distances, ordered according to the
            OPTICS algorithm's processing order.
        """
        return self.reachability_[self.ordered_list_]
