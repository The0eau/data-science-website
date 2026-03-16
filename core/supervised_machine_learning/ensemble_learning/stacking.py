import numpy as np
from collections import Counter

# ==========================================
# 1. EVALUATION METRICS
# ==========================================

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy percentage.
    
    Args:
        y_true (np.array): Real target labels.
        y_pred (np.array): Predicted labels.
    """
    return np.sum(y_true == y_pred) / len(y_true)

def precision_recall_f1(y_true, y_pred):
    """
    Calculate Precision, Recall and F1-Score for binary classification.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# ==========================================
# 2. LAYER 0 MODELS (The Specialists)
# ==========================================

class NaiveBayesScratch:
    """Gaussian Naive Bayes Classifier."""
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = np.array([X[y==c].mean(axis=0) for c in self.classes])
        self.var = np.array([X[y==c].var(axis=0) for c in self.classes])
        self.priors = np.array([X[y==c].shape[0]/len(y) for c in self.classes])

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            likelihood = np.sum(np.log(self._pdf(i, x)))
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, idx, x):
        m, v = self.mean[idx], self.var[idx]
        return np.exp(-(x-m)**2 / (2*v)) / np.sqrt(2*np.pi*v)

class DecisionTreeScratch:
    """Basic Decision Tree for classification."""
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return Counter(y).most_common(1)[0][0]
        feat = np.argmax(np.var(X, axis=0)) # Heuristic split
        thresh = np.median(X[:, feat])
        left_idx = X[:, feat] <= thresh
        return {'feat': feat, 'thresh': thresh, 
                'left': self._grow_tree(X[left_idx], y[left_idx], depth+1),
                'right': self._grow_tree(X[~left_idx], y[~left_idx], depth+1)}

    def predict(self, X):
        return np.array([self._traverse(x, self.tree) for x in X])

    def _traverse(self, x, node):
        if not isinstance(node, dict): return node
        if x[node['feat']] <= node['thresh']:
            return self._traverse(x, node['left'])
        return self._traverse(x, node['right'])

# ==========================================
# 3. LAYER 1 MODEL (The Meta-Learner)
# ==========================================

class LogisticRegressionMeta:
    """Logistic Regression used as a Meta-Model to weigh Layer 0 predictions."""
    def __init__(self, lr=0.1, iters=500):
        self.lr, self.iters = lr, iters

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.iters):
            y_hat = 1 / (1 + np.exp(-(np.dot(X, self.w) + self.b)))
            self.w -= self.lr * (1/len(y)) * np.dot(X.T, (y_hat - y))
            self.b -= self.lr * (1/len(y)) * np.sum(y_hat - y)

    def predict(self, X):
        return (1 / (1 + np.exp(-(np.dot(X, self.w) + self.b))) > 0.5).astype(int)

# ==========================================
# 4. STACKING ENSEMBLE
# ==========================================

class StackingClassifierScratch:
    """
    Stacking Ensemble that learns to combine base models.
    
    Attributes:
        base_models (list): List of models for Layer 0.
        meta_model (object): Model for Layer 1.
    """
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        """Train base models and use their predictions to train the meta-model."""
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        # 1. Train workers
        for i, model in enumerate(self.base_models):
            model.fit(X, y)
            meta_features[:, i] = model.predict(X)
        
        # 2. Train manager on workers' output
        self.meta_model.fit(meta_features, y)

    def predict(self, X):
        """Final prediction via the meta-model."""
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            meta_features[:, i] = model.predict(X)
        return self.meta_model.predict(meta_features)

# ==========================================
# 5. TEST AND EVALUATION
# ==========================================

if __name__ == "__main__":
    # Synthetic dataset
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Architecture definition
    layer0 = [NaiveBayesScratch(), DecisionTreeScratch(max_depth=2)]
    layer1 = LogisticRegressionMeta()

    stack = StackingClassifierScratch(layer0, layer1)
    stack.fit(X, y)
    predictions = stack.predict(X)

    # Metrics
    acc = accuracy_score(y, predictions)
    p, r, f1 = precision_recall_f1(y, predictions)

    print("--- Stacking Performance ---")
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {p:.2f}")
    print(f"Recall    : {r:.2f}")
    print(f"F1-Score  : {f1:.2f}")