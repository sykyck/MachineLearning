"""
K-Nearest Neighbors (KNN) - Complete Python Example
====================================================
Covers:
  1. KNN from scratch (Euclidean distance + majority vote)
  2. KNN with scikit-learn
  3. Choosing the best K via cross-validation
  4. Visualization of decision boundaries
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ──────────────────────────────────────────────
# 1. KNN FROM SCRATCH
# ──────────────────────────────────────────────

class KNNClassifier:
    """KNN classifier built from scratch."""

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """Store training data (KNN has no real training step)."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _predict_one(self, x):
        # Compute distances to all training points
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get indices of K nearest neighbors
        k_indices = np.argsort(distances)[: self.k]
        # Majority vote
        k_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == np.array(y))


# ──────────────────────────────────────────────
# 2. GENERATE DATASET
# ──────────────────────────────────────────────

np.random.seed(42)
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.2,
    random_state=42,
)

# Scale features — important for distance-based algorithms
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

print("=" * 55)
print("  K-NEAREST NEIGHBORS — Python Example")
print("=" * 55)
print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, 2 classes")
print(f"Train / Test split: {len(X_train)} / {len(X_test)}")


# ──────────────────────────────────────────────
# 3. SCRATCH vs SKLEARN COMPARISON
# ──────────────────────────────────────────────

K = 5

# Scratch model
scratch_model = KNNClassifier(k=K)
scratch_model.fit(X_train, y_train)
scratch_acc = scratch_model.score(X_test, y_test)

# Scikit-learn model
sklearn_model = KNeighborsClassifier(n_neighbors=K, metric="euclidean")
sklearn_model.fit(X_train, y_train)
sklearn_acc = sklearn_model.score(X_test, y_test)

print(f"\n── Accuracy at K={K} ──")
print(f"  From scratch : {scratch_acc:.4f}")
print(f"  scikit-learn : {sklearn_acc:.4f}")


# ──────────────────────────────────────────────
# 4. FIND BEST K VIA CROSS-VALIDATION
# ──────────────────────────────────────────────

k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print(f"\n── Cross-Validation (5-fold) ──")
print(f"  Best K  : {best_k}")
print(f"  Best CV accuracy : {max(cv_scores):.4f}")


# ──────────────────────────────────────────────
# 5. FINAL MODEL WITH BEST K
# ──────────────────────────────────────────────

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print(f"\n── Classification Report (K={best_k}) ──")
print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))


# ──────────────────────────────────────────────
# 6. VISUALIZATIONS
# ──────────────────────────────────────────────

COLORS = ["#3B8BD4", "#D85A30"]
LIGHT   = ["#E6F1FB", "#FAECE7"]

def plot_decision_boundary(ax, model, X, y, title, k):
    """Shade the decision boundary and overlay data points."""
    h = 0.03
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, colors=LIGHT, levels=[-0.5, 0.5, 1.5])
    ax.contour(xx, yy, Z, colors=["#888"], linewidths=0.7, levels=[0.5])

    for cls, color in enumerate(COLORS):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=color, edgecolors="white",
                   linewidths=0.8, s=45, label=f"Class {cls}", zorder=3)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Feature 1 (scaled)")
    ax.set_ylabel("Feature 2 (scaled)")
    ax.legend(fontsize=9, framealpha=0.7)


fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("K-Nearest Neighbors — Full Analysis", fontsize=14, fontweight="bold", y=1.01)

# Panel 1: Decision boundary (train data, K=1)
knn1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
plot_decision_boundary(axes[0, 0], knn1, X_train, y_train,
                       "Decision Boundary — K=1 (overfit)", k=1)

# Panel 2: Decision boundary (train data, best K)
knn_best = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
plot_decision_boundary(axes[0, 1], knn_best, X_train, y_train,
                       f"Decision Boundary — K={best_k} (best)", k=best_k)

# Panel 3: CV scores vs K
axes[1, 0].plot(list(k_values), cv_scores, "o-", color=COLORS[0],
                linewidth=1.8, markersize=5, markerfacecolor="white",
                markeredgewidth=1.5)
axes[1, 0].axvline(best_k, color=COLORS[1], linestyle="--", linewidth=1.5,
                    label=f"Best K = {best_k}")
axes[1, 0].set_xlabel("K (number of neighbors)")
axes[1, 0].set_ylabel("CV Accuracy")
axes[1, 0].set_title("Finding the Best K via Cross-Validation", fontsize=11, fontweight="bold")
axes[1, 0].legend(fontsize=9)
axes[1, 0].set_xticks(list(k_values))
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(min(cv_scores) - 0.02, 1.0)

# Panel 4: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", ax=axes[1, 1],
            cmap="Blues", cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"])
axes[1, 1].set_title(f"Confusion Matrix — K={best_k}", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("knn_analysis.png", dpi=150, bbox_inches="tight")
print("\n✓ Plot saved → knn_analysis.png")
plt.show()

print("\n✓ Done.")