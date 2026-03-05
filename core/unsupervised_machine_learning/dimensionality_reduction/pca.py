import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.

    This class reduces the dimensionality of datasets while preserving 
    as much variance as possible by projecting data onto a new set 
    of orthogonal axes called principal components.

    Parameters
    ----------
    n_components : int, default=None
        The number of principal components to compute. 
        If None, all components are kept (n_components = n_features).

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions 
        of maximum variance.
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """
        Fit the model with X by computing the principal components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data used to extract the components.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Compute the Covariance Matrix
        # rowvar=False because variables are in columns
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # 3. Eigen Decomposition
        # eigh is optimized for symmetric matrices like covariance matrices
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # 4. Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Handle the case where n_components is not specified
        if self.n_components is None:
            self.n_components = X.shape[1]

        # 5. Store results
        # We transpose eigenvectors to follow the (n_components, n_features) convention
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Calculate the ratio of variance explained
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted 
        from a training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Projected data in the reduced space.
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
            
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)
    