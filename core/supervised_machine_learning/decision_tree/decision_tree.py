import numpy as np
from collections import Counter

# ==========================================
# 1. METRICS SECTION
# ==========================================

def mean_squared_error(y_true, y_pred):
    """MSE for Regression."""
    return np.mean((y_true - y_pred)**2)

def r2_score_scratch(y_true, y_pred):
    """R-squared score for Regression."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def accuracy_score(y_true, y_pred):
    """Accuracy for Classification."""
    return np.sum(y_true == y_pred) / len(y_true)

def precision_score_scratch(y_true, y_pred):
    """Precision (TP / TP + FP)."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_score_scratch(y_true, y_pred):
    """Recall (TP / TP + FN)."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def confusion_matrix_scratch(y_true, y_pred):
    """Confusion Matrix [[TN, FP], [FN, TP]]."""
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_idx = {val: i for i, val in enumerate(classes)}
    for gt, lp in zip(y_true, y_pred):
        matrix[class_to_idx[gt]][class_to_idx[lp]] += 1
    return matrix

# ==========================================
# 2. DECISION TREE COMPONENTS
# ==========================================

class Node:
    """Helper class for tree structure."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class BaseTree:
    """Shared logic between Classification and Regression trees."""
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def _split(self, X_column, split_thresh):
        left_idxs = np.where(X_column <= split_thresh)[0]
        right_idxs = np.where(X_column > split_thresh)[0]
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# ==========================================
# 3. CLASSIFICATION TREE
# ==========================================

class DecisionTreeClassifierScratch(BaseTree):
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return Node(value=Counter(y).most_common(1)[0][0])

        feat_idxs = np.random.choice(n_features, n_features, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            for threshold in np.unique(X_col):
                gain = self._info_gain(y, X_col, threshold)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, threshold
        return split_idx, split_thresh

    def _info_gain(self, y, X_col, thresh):
        parent_entropy = self._entropy(y)
        l_idx, r_idx = self._split(X_col, thresh)
        if len(l_idx) == 0 or len(r_idx) == 0: return 0
        n, n_l, n_r = len(y), len(l_idx), len(r_idx)
        child_entropy = (n_l/n)*self._entropy(y[l_idx]) + (n_r/n)*self._entropy(y[r_idx])
        return parent_entropy - child_entropy

    def _entropy(self, y):
        ps = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

# ==========================================
# 4. REGRESSION TREE
# ==========================================

class DecisionTreeRegressorScratch(BaseTree):
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feat = X.shape
        if (depth >= self.max_depth or n_samples < self.min_samples_split or np.var(y) == 0):
            return Node(value=np.mean(y))

        feat_idxs = np.random.choice(n_feat, n_feat, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        l_idx, r_idx = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[l_idx, :], y[l_idx], depth + 1)
        right = self._grow_tree(X[r_idx, :], y[r_idx], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_red = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            for threshold in np.unique(X_col):
                red = self._var_reduction(y, X_col, threshold)
                if red > best_red:
                    best_red, split_idx, split_thresh = red, feat_idx, threshold
        return split_idx, split_thresh

    def _var_reduction(self, y, X_col, thresh):
        parent_var = np.var(y)
        l_idx, r_idx = self._split(X_col, thresh)
        if len(l_idx) == 0 or len(r_idx) == 0: return 0
        n, n_l, n_r = len(y), len(l_idx), len(r_idx)
        return parent_var - ((n_l/n)*np.var(y[l_idx]) + (n_r/n)*np.var(y[r_idx]))

# ==========================================
# 5. MAIN TEST LOOP
# ==========================================

if __name__ == "__main__":
    print("--- 1. Testing CLASSIFICATION Tree ---")
    X_c = np.array([[20, 1], [25, 1], [30, 0], [45, 0], [50, 0]])
    y_c = np.array([1, 1, 0, 0, 0]) # 1: Young, 0: Old
    clf = DecisionTreeClassifierScratch(max_depth=3)
    clf.fit(X_c, y_c)
    preds_c = clf.predict(X_c)
    print(f"Accuracy: {accuracy_score(y_c, preds_c)*100:.1f}%")
    print(f"Confusion Matrix:\n{confusion_matrix_scratch(y_c, preds_c)}")

    print("\n--- 2. Testing REGRESSION Tree ---")
    X_r = np.array([[1], [2], [3], [4], [5]])
    y_r = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    reg = DecisionTreeRegressorScratch(max_depth=3)
    reg.fit(X_r, y_r)
    preds_r = reg.predict(X_r)
    print(f"MSE: {mean_squared_error(y_r, preds_r):.4f}")
    print(f"R2 Score: {r2_score_scratch(y_r, preds_r):.4f}")