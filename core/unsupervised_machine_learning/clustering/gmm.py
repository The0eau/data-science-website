import polars as pl
import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    """
    Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of mixture components (clusters) to fit.
    max_iters : int, default=100
        Maximum number of iterations of the EM algorithm.
    tol : float, default=1e-4
        The convergence threshold based on log-likelihood improvement.
    random_state : int, optional
        Seed used for reproducibility.
    """

    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        
        # Model parameters
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None

    def fit(self, X):
        """
        Estimate model parameters with the EM algorithm.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape

        # 1. Initialization: Randomly pick points as starting means
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.means_ = X[indices]
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_clusters)])
        self.weights_ = np.full(self.n_clusters, 1 / self.n_clusters)
        
        prev_log_likelihood = -np.inf

        for i in range(self.max_iters):
            # --- E-STEP: Expectation ---
            # Calculate the weighted probability of each point belonging to each cluster
            weighted_pdfs = np.zeros((n_samples, self.n_clusters))
            
            for k in range(self.n_clusters):
                # Add a tiny value to the diagonal to prevent singular matrix errors
                cov_stable = self.covariances_[k] + np.eye(n_features) * 1e-6
                weighted_pdfs[:, k] = self.weights_[k] * multivariate_normal.pdf(
                    X, mean=self.means_[k], cov=cov_stable
                )
            
            # Total likelihood for each point (sum across clusters)
            sum_weighted_pdfs = weighted_pdfs.sum(axis=1)
            
            # Responsibilities (soft assignment): gamma(z_nk)
            responsibilities = weighted_pdfs / sum_weighted_pdfs[:, np.newaxis]

            # --- M-STEP: Maximization ---
            # Sum of responsibilities for each cluster
            Nk = responsibilities.sum(axis=0) + 1e-10 
            
            # Update Weights
            self.weights_ = Nk / n_samples
            
            # Update Means
            self.means_ = (responsibilities.T @ X) / Nk[:, np.newaxis]
            
            # Update Covariances
            for k in range(self.n_clusters):
                diff = X - self.means_[k]
                # Weighted covariance calculation
                self.covariances_[k] = (responsibilities[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]

            # --- Convergence Check ---
            # Compute total log-likelihood
            current_log_likelihood = np.sum(np.log(sum_weighted_pdfs))
            if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = current_log_likelihood
            
        return self

    def predict_proba(self, X):
        """
        Predict posterior probabilities for each component.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_clusters)
        """
        n_samples = X.shape[0]
        weighted_pdfs = np.zeros((n_samples, self.n_clusters))
        
        for k in range(self.n_clusters):
            cov_stable = self.covariances_[k] + np.eye(X.shape[1]) * 1e-6
            weighted_pdfs[:, k] = self.weights_[k] * multivariate_normal.pdf(
                X, mean=self.means_[k], cov=cov_stable
            )
            
        return weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)

    def predict(self, X):
        """
        Predict the cluster labels for X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        if self.means_ is None:
            raise ValueError("Model must be fitted before prediction.")
            
        return np.argmax(self.predict_proba(X), axis=1)