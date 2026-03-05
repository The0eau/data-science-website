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

class SVMS:
    """
    Support Vector Machine classifier for binary classification.
    Uses Gradient Descent to maximize the margin.
    
    Attributes:
        lr (float): Learning rate.
        lambda_param (float): Regularization parameter (trade-off between margin and error).
        n_iters (int): Number of iterations.
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train the SVM. Note: y must be encoded as -1 and 1.
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Condition: y_i * (w * x_i + b) >= 1
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # Only regularization gradient
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # Regularization + Misclassification gradient
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y[idx])
                    db = -y[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        """Predict labels for samples."""
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

# ==========================================
# MAIN TEST LOOP
# ==========================================

if __name__ == "__main__":
    # --- TESTING SVM ---
    print("\n" + "="*30)
    print("TESTING SVM")
    print("="*30)
    
    # Dataset: y must be -1 or 1
    X_svm = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [6, 6], [7, 8], [8, 8], [9, 7]])
    y_svm = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    
    svm = SVMS(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_svm, y_svm)
    preds_svm = svm.predict(X_svm)
    
    print(f"SVM Accuracy: {accuracy_score(y_svm, preds_svm) * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix_scratch(y_svm, preds_svm))