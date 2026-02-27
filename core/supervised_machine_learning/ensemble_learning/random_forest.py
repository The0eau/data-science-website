import numpy as np
from collections import Counter

# ==========================================
# 1. METRICS SECTION (Classification)
# ==========================================

def accuracy_score(y_true, y_pred):
    """
    Calculate the percentage of correct predictions.
    
    Args:
        y_true (numpy.ndarray): Actual labels.
        y_pred (numpy.ndarray): Predicted labels.
    """
    return np.sum(y_true == y_pred) / len(y_true)

def precision_score_scratch(y_true, y_pred):
    """
    Calculate Precision: TP / (TP + FP).
    Measures the reliability of a positive prediction.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_score_scratch(y_true, y_pred):
    """
    Calculate Recall: TP / (TP + FN).
    Measures the ability to find all positive instances.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def confusion_matrix_scratch(y_true, y_pred):
    """
    Generate a 2x2 confusion matrix: [[TN, FP], [FN, TP]].
    """
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_idx = {val: i for i, val in enumerate(classes)}
    for gt, lp in zip(y_true, y_pred):
        matrix[class_to_idx[gt]][class_to_idx[lp]] += 1
    return matrix

# ==========================================
# 2. DECISION TREE COMPONENT
# ==========================================

class Node:
    """Helper class to store nodes of the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifierScratch:
    """
    Decision Tree Classifier using Information Gain and Entropy.
    """
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

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
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0: return 0
        n, n_l, n_r = len(y), len(left_idxs), len(right_idxs)
        child_entropy = (n_l / n) * self._entropy(y[left_idxs]) + (n_r / n) * self._entropy(y[right_idxs])
        return parent_entropy - child_entropy

    def _entropy(self, y):
        ps = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _split(self, X_column, split_thresh):
        left_idxs = np.where(X_column <= split_thresh)[0]
        right_idxs = np.where(X_column > split_thresh)[0]
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node(): return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# ==========================================
# 3. RANDOM FOREST COMPONENT
# ==========================================

class RandomForestClassifierScratch:
    """
    Random Forest Classifier implementing Bagging logic.
    Trains multiple decision trees on bootstrap samples.
    """
    def __init__(self, n_trees=5, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        """Train the forest by creating n_trees on bootstrap samples."""
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        """Aggregate tree predictions using majority voting."""
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        predictions = [Counter(sample_preds).most_common(1)[0][0] for sample_preds in tree_preds]
        return np.array(predictions)

# ==========================================
# 4. MAIN TEST LOOP
# ==========================================

if __name__ == "__main__":
    print("="*50)
    print("TESTING ENSEMBLE MODELS FROM SCRATCH")
    print("="*50)

    # Synthetic Dataset: Age & Income -> Target: Purchase (0 or 1)
    X = np.array([[22, 2000], [25, 2500], [45, 8000], [50, 7500], 
                  [23, 2100], [48, 8200], [52, 9000], [21, 1900], 
                  [35, 5000], [40, 5500], [60, 10000], [19, 1500]])
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0])

    # Initialize and Train Random Forest
    print("\n--- Training Random Forest (n_trees=5) ---")
    rf = RandomForestClassifierScratch(n_trees=5, max_depth=5)
    rf.fit(X, y)
    y_pred = rf.predict(X)

    # Print Metrics
    print(f"Accuracy  : {accuracy_score(y, y_pred) * 100:.2f}%")
    print(f"Precision : {precision_score_scratch(y, y_pred) * 100:.2f}%")
    print(f"Recall    : {recall_score_scratch(y, y_pred) * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix_scratch(y, y_pred))
    print("="*50)