import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import pandas as pd
from sklearn.decomposition import PCA

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=23
)

# Train model
clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:, 1]

# Accuracy & AUC
acc = accuracy_score(y_test, y_pred) * 100
auc = roc_auc_score(y_test, y_pred_prob)

# Compute concordance
concordant = discordant = tied = 0
for i, j in combinations(range(len(y_test)), 2):
    if y_test[i] != y_test[j]:
        if y_pred_prob[i] > y_pred_prob[j]:
            if y_test[i] == 1:
                concordant += 1
            else:
                discordant += 1
        elif y_pred_prob[i] < y_pred_prob[j]:
            if y_test[i] == 1:
                discordant += 1
            else:
                concordant += 1
        else:
            tied += 1

total_pairs = concordant + discordant + tied
concordance = concordant / total_pairs
discordance = discordant / total_pairs
ties = tied / total_pairs

# Print metrics
print(f"Binomial Logistic Regression Accuracy: {acc:.2f}%")
print(f"AUC: {auc:.3f}")
print(f"Concordance: {concordance:.3f}")
print(f"Discordance: {discordance:.3f}")
print(f"Tied: {ties:.3f}")

# ---------------- VISUALIZATIONS ---------------- #

plt.figure(figsize=(15,5))

# 1. Histogram of predicted probabilities
plt.subplot(1,3,1)
plt.hist(y_pred_prob[y_test==0], bins=20, alpha=0.6, label="Class 0")
plt.hist(y_pred_prob[y_test==1], bins=20, alpha=0.6, label="Class 1")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Histogram of Predicted Probabilities")
plt.legend()

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.subplot(1,3,2)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")

# 3. Concordance/Discordance/Tied
plt.subplot(1,3,3)
plt.bar(["Concordant", "Discordant", "Tied"],
        [concordance, discordance, ties],
        color=["green", "red", "gray"])
plt.title("Concordance Analysis")
plt.ylabel("Proportion")

plt.tight_layout()
plt.show()
