import numpy as np

class LDA:
    """
    Linear Discriminant Analysis (LDA).

    A supervised dimensionality reduction technique that finds the projection 
    axes that maximize the distance between class means while minimizing 
    the variance within each class.

    Parameters
    ----------
    n_components : int, default=None
        Number of components for dimensionality reduction. 
        If None, it will be set to min(n_features, n_classes - 1).

    Attributes
    ----------
    scalings_ : ndarray of shape (n_features, n_components)
        The linear discriminants (eigenvectors) used to project the data.
    means_ : list of ndarray
        The mean vector for each class.
    overall_mean_ : ndarray of shape (n_features,)
        The global mean of all input samples.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.scalings_ = None
        self.overall_mean_ = None

    def fit(self, X, y):
        """
        Fit the LDA model according to the given training data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (class labels).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        n_samples, n_features = X.shape
        class_labels = np.unique(y)
        n_classes = len(class_labels)

        # Set n_components to the mathematical limit if not specified
        if self.n_components is None:
            self.n_components = min(n_features, n_classes - 1)
        else:
            self.n_components = min(self.n_components, n_classes - 1)

        # 1. Compute overall mean
        self.overall_mean_ = np.mean(X, axis=0)

        # 2. Compute Within-class scatter matrix (Sw) and Between-class scatter matrix (Sb)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            
            # Within-class scatter: Sw = sum((X_c - mean_c)^T * (X_c - mean_c))
            z_c = X_c - mean_c
            S_W += z_c.T @ z_c
            
            # Between-class scatter: Sb = n_c * (mean_c - mean_overall) * (mean_c - mean_overall)^T
            n_c = X_c.shape[0]
            mean_diff = (mean_c - self.overall_mean_).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        # 3. Solve the generalized eigenvalue problem: (Sw^-1 * Sb)v = lambda * v
        # We use pseudo-inverse for numerical stability against singular matrices
        A = np.linalg.pinv(S_W) @ S_B
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # 4. Sort eigenvectors by eigenvalues in descending order
        # We take the real part to handle tiny imaginary components from numerical noise
        idx = np.argsort(np.real(eigenvalues))[::-1]
        eigenvectors = np.real(eigenvectors[:, idx])

        # Store the top n_components
        self.scalings_ = eigenvectors[:, :self.n_components]

        return self

    def transform(self, X):
        """
        Project data into the learned LDA space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        if self.scalings_ is None:
            raise ValueError("Model has not been fitted yet.")
            
        return np.dot(X, self.scalings_)

    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
        """
        return self.fit(X, y).transform(X)