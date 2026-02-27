import numpy as np

def accuracy_score(y_true, y_pred):
    """Calculate the accuracy percentage."""
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix_scratch(y_true, y_pred):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    For binary classification, returns: [[TN, FP], [FN, TP]]
    """
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    
    # Map classes to indices if they aren't 0, 1 (useful for SVM -1, 1)
    class_to_idx = {val: i for i, val in enumerate(classes)}
    
    for gt, lp in zip(y_true, y_pred):
        matrix[class_to_idx[gt]][class_to_idx[lp]] += 1
    return matrix

class NaiveBayes:
    """
    Gaussian Naive Bayes classifier.
    
    Assumes that continuous features follow a normal distribution 
    and are independent of each other.
    """
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        """
        Calculate mean, variance, and prior probability for each class.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize parameters
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """Predict classes for multiple samples."""
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """Predict the class of a single sample using posterior probability."""
        posteriors = []

        for idx, c in enumerate(self.classes):
            # Calculate log-prior to avoid underflow
            prior = np.log(self.priors[idx])
            # Calculate log-likelihood based on Gaussian distribution
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        # Return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """Probability Density Function (Gaussian)."""
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# ==========================================
# MAIN TEST LOOP
# ==========================================

if __name__ == "__main__":
    # --- TESTING NAIVE BAYES ---
    print("\n" + "="*30)
    print("TESTING NAIVE BAYES")
    print("="*30)
    
    # Dataset: Heights (cm) and Weights (kg) for two groups
    X_nb = np.array([[180, 80], [175, 75], [190, 85], [150, 50], [160, 60], [155, 55]])
    y_nb = np.array([0, 0, 0, 1, 1, 1])
    
    nb = NaiveBayes()
    nb.fit(X_nb, y_nb)
    preds_nb = nb.predict(X_nb)
    
    print(f"Naive Bayes Accuracy: {accuracy_score(y_nb, preds_nb) * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix_scratch(y_nb, preds_nb))