import numpy as np

# ==========================================
# 1. METRICS & UTILITIES
# ==========================================

def mean_squared_error(y_true, y_pred):
    """Mean Squared Error for Regression."""
    return np.mean((y_true - y_pred)**2)

def accuracy_score(y_true, y_pred):
    """Accuracy percentage for Classification."""
    return np.sum(y_true == y_pred) / len(y_true)

def r2_score_scratch(y_true, y_pred):
    """R-squared (Coefficient of Determination) for Regression."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# ==========================================
# 2. CORE XGBOOST ENGINE
# ==========================================

class XGBoostNode:
    """A node representing a split or a leaf in an XGBoost tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class XGBoostTree:
    """Individual tree that optimizes the structural score (Gain)."""
    def __init__(self, max_depth=3, lambda_reg=1, gamma=0):
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.root = None

    def _calc_leaf_value(self, g, h):
        return -np.sum(g) / (np.sum(h) + self.lambda_reg)

    def _calc_similarity(self, g, h):
        return (np.sum(g)**2) / (np.sum(h) + self.lambda_reg)

    def fit(self, X, g, h):
        self.root = self._build_tree(X, g, h, 0)

    def _build_tree(self, X, g, h, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < 2:
            return XGBoostNode(value=self._calc_leaf_value(g, h))

        best_gain, split_idx, split_thresh = 0, None, None
        root_sim = self._calc_similarity(g, h)

        for f_idx in range(n_features):
            thresholds = np.unique(X[:, f_idx])
            for threshold in thresholds:
                left_m = X[:, f_idx] <= threshold
                right_m = ~left_m
                if not np.any(left_m) or not np.any(right_m): continue

                gain = 0.5 * (self._calc_similarity(g[left_m], h[left_m]) + 
                             self._calc_similarity(g[right_m], h[right_m]) - 
                             root_sim) - self.gamma
                
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, f_idx, threshold

        if split_idx is None:
            return XGBoostNode(value=self._calc_leaf_value(g, h))

        left = self._build_tree(X[X[:, split_idx] <= split_thresh], g[X[:, split_idx] <= split_thresh], h[X[:, split_idx] <= split_thresh], depth + 1)
        right = self._build_tree(X[X[:, split_idx] > split_thresh], g[X[:, split_idx] > split_thresh], h[X[:, split_idx] > split_thresh], depth + 1)
        return XGBoostNode(split_idx, split_thresh, left, right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf(): return node.value
        return self._traverse(x, node.left) if x[node.feature] <= node.threshold else self._traverse(x, node.right)

# ==========================================
# 3. ENSEMBLE CLASSES
# ==========================================

class XGBoostRegressorScratch:
    """Extreme Gradient Boosting for Regression (MSE Loss)."""
    def __init__(self, n_estimators=10, lr=0.3, max_depth=3, lambda_reg=1):
        self.n_estimators, self.lr, self.max_depth, self.lambda_reg = n_estimators, lr, max_depth, lambda_reg
        self.trees = []

    def fit(self, X, y):
        self.base_pred = np.mean(y)
        y_hat = np.full(y.shape, self.base_pred)
        for i in range(self.n_estimators):
            g, h = -(y - y_hat), np.ones_like(y)
            tree = XGBoostTree(self.max_depth, self.lambda_reg)
            tree.fit(X, g, h)
            y_hat += self.lr * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        y_hat = np.full(X.shape[0], self.base_pred)
        for tree in self.trees: y_hat += self.lr * tree.predict(X)
        return y_hat

class XGBoostClassifierScratch:
    """Extreme Gradient Boosting for Binary Classification (Log-Loss)."""
    def __init__(self, n_estimators=10, lr=0.3, max_depth=3, lambda_reg=1):
        self.n_estimators, self.lr, self.max_depth, self.lambda_reg = n_estimators, lr, max_depth, lambda_reg
        self.trees = []

    def _sigmoid(self, x): return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        p_avg = np.mean(y)
        self.base_pred = np.log(p_avg / (1 - p_avg))
        y_hat = np.full(y.shape, self.base_pred)
        for i in range(self.n_estimators):
            p = self._sigmoid(y_hat)
            g, h = p - y, p * (1 - p)
            tree = XGBoostTree(self.max_depth, self.lambda_reg)
            tree.fit(X, g, h)
            y_hat += self.lr * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        y_hat = np.full(X.shape[0], self.base_pred)
        for tree in self.trees: y_hat += self.lr * tree.predict(X)
        return (self._sigmoid(y_hat) > 0.5).astype(int)

# ==========================================
# 4. TEST WITH METRICS
# ==========================================

if __name__ == "__main__":
    print("-" * 40)
    print("XGBOOST SCRATCH TEST SUITE")
    print("-" * 40)

    # REGRESSION TEST
    X_reg = np.array([[1], [2], [3], [4], [5], [6]])
    y_reg = np.array([1.1, 3.9, 9.2, 16.1, 24.8, 36.2]) # y ≈ x^2

    xgb_r = XGBoostRegressorScratch(n_estimators=15, lr=0.2)
    xgb_r.fit(X_reg, y_reg)
    preds_r = xgb_r.predict(X_reg)
    
    print(">>> REGRESSION RESULTS")
    print(f"MSE: {mean_squared_error(y_reg, preds_r):.4f}")
    print(f"R2 Score: {r2_score_scratch(y_reg, preds_r):.4f}")

    # CLASSIFICATION TEST
    # Features: [Study Hours, Sleep Hours] -> [Pass (1) / Fail (0)]
    X_clf = np.array([[1, 5], [2, 6], [8, 7], [9, 8], [1, 4], [10, 9]])
    y_clf = np.array([0, 0, 1, 1, 0, 1])

    xgb_c = XGBoostClassifierScratch(n_estimators=10, lr=0.3)
    xgb_c.fit(X_clf, y_clf)
    preds_c = xgb_c.predict(X_clf)
    
    print("\n>>> CLASSIFICATION RESULTS")
    print(f"Accuracy: {accuracy_score(y_clf, preds_c)*100:.2f}%")
    print(f"Predictions: {preds_c}")
    print(f"Actual:      {y_clf}")
    print("-" * 40)