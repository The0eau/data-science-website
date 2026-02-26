import numpy as np

class TSNE:
    """
    t-distributed Stochastic Neighbor Embedding (t-SNE).

    A non-linear dimensionality reduction technique that maps high-dimensional 
    data to a lower-dimensional space (2D or 3D) by preserving local pairwise 
    similarities using a t-distribution.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors used.
    learning_rate : float, default=200.0
        The step size for gradient descent.
    n_iter : int, default=1000
        Maximum number of iterations for the optimization.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the low-dimensional representation after fitting.
    """

    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.embedding_ = None
    
    def _compute_pairwise_affinities(self, X):
        """Compute high-dimensional affinities using a Gaussian kernel."""
        n_samples = X.shape[0]
        # Optimized squared distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        sum_X = np.sum(np.square(X), axis=1)
        dist_sq = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        dist_sq = np.maximum(dist_sq, 0)
        
        affinities = np.exp(-dist_sq / (2 * self.perplexity ** 2))
        np.fill_diagonal(affinities, 0)
        # Symmetrize and normalize probabilities
        P = affinities / (np.sum(affinities, axis=1, keepdims=True) + 1e-12)
        return (P + P.T) / (2 * n_samples)
    
    def _get_probabilities(self, D_sq, sigma):
        """Compute low-dimensional affinities using Student's t-distribution."""
        # Standard t-SNE uses 1 / (1 + d^2) for low-dim space
        inv_dist = 1.0 / (1.0 + D_sq / (sigma**2))
        np.fill_diagonal(inv_dist, 0)
        return inv_dist / (np.sum(inv_dist) + 1e-12)
    
    def fit(self, X):
        """
        Fit the model with X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        n_samples = X.shape[0]
        P = np.maximum(self._compute_pairwise_affinities(X), 1e-12)
        
        # Initialize small random values
        Y = np.random.randn(n_samples, self.n_components) * 1e-4
        
        for i in range(self.n_iter):
            # Compute low-dim squared distances
            sum_Y = np.sum(np.square(Y), axis=1)
            dist_sq_Y = np.maximum(np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y), 0)
            
            Q = np.maximum(self._get_probabilities(dist_sq_Y, sigma=1.0), 1e-12)
            
            # Gradient calculation
            PQ_diff = P - Q
            grad = np.zeros((n_samples, self.n_components))
            for j in range(n_samples):
                # Standard gradient formula for t-SNE
                grad[j] = 4 * np.dot(PQ_diff[j, :], Y[j] - Y)
            
            Y -= self.learning_rate * grad
        
        self.embedding_ = Y
        return self
    
    def transform(self, X=None):
        """
        Returns the learned embedding.

        Parameters
        ----------
        X : ndarray, optional
            Ignored. Present for API consistency.

        Returns
        -------
        Y : ndarray of shape (n_samples, n_components)
        """
        if self.embedding_ is None:
            raise ValueError("The model has not been fitted yet.")
        return self.embedding_
    
    def fit_transform(self, X):
        """Fit the model and return the projected data."""
        self.fit(X)
        return self.embedding_
    
class UMAP:
    """
    Uniform Manifold Approximation and Projection (UMAP).

    A dimensionality reduction technique based on Riemannian geometry and 
    algebraic topology, optimized to preserve both local and global structures.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    n_neighbors : int, default=15
        Number of neighbors for local manifold approximation.
    learning_rate : float, default=1.0
        The step size for the layout optimization.
    n_epochs : int, default=200
        Number of iterations for the optimization.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the coordinates in the reduced space.
    """

    def __init__(self, n_components=2, n_neighbors=15, learning_rate=1.0, n_epochs=200):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.embedding_ = None

    def _compute_membership_strengths(self, X):
        """Construct the fuzzy simplicial complex (high-dim graph)."""
        n_samples = X.shape[0]
        sum_X = np.sum(np.square(X), axis=1)
        dist_sq = np.maximum(np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X), 0)
        distances = np.sqrt(dist_sq)
        
        membership_strengths = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            # Find k-nearest neighbors
            nn_indices = np.argsort(distances[i])[1:self.n_neighbors + 1]
            rho_i = distances[i, nn_indices[0]] # Local connectivity parameter
            
            for j in nn_indices:
                membership_strengths[i, j] = np.exp(-max(0, distances[i, j] - rho_i))
        
        # Symmetrize: fuzzy union of the graph
        return membership_strengths + membership_strengths.T - (membership_strengths * membership_strengths.T)
    
    def fit(self, X):
        """
        Fit the model with X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
        """
        n_samples = X.shape[0]
        P = self._compute_membership_strengths(X)
        
        # Initialize randomly
        Y = np.random.randn(n_samples, self.n_components)
        
        for epoch in range(self.n_epochs):
            sum_Y = np.sum(np.square(Y), axis=1)
            dist_sq_Y = np.maximum(np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y), 0)
            
            # Cross-Entropy Optimization
            # Attractive forces (P) and Repulsive forces (1-P)
            grad = np.zeros((n_samples, self.n_components))
            for i in range(n_samples):
                # Attractive force + Repulsive force logic
                attr = P[i, :] * (1 / (dist_sq_Y[i, :] + 1e-3))
                repul = (1 - P[i, :]) * (1 / (dist_sq_Y[i, :]**2 + 1e-3))
                force = attr - repul
                grad[i] = np.dot(force, Y[i] - Y)
            
            Y -= self.learning_rate * grad
        
        self.embedding_ = Y
        return self
    
    def transform(self, X=None):
        """Returns the pre-computed embedding."""
        if self.embedding_ is None:
            raise ValueError("The model has not been fitted yet.")
        return self.embedding_
    
    def fit_transform(self, X):
        """Fit the model and return the projected data."""
        self.fit(X)
        return self.embedding_