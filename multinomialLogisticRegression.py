import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics
import numpy as np

# Load dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Multinomial logistic regression
reg = linear_model.LogisticRegression(
    max_iter=10000, 
    random_state=0, 
    multi_class='multinomial', 
    solver='lbfgs'
)
reg.fit(X_train, y_train)

# Predictions
y_pred = reg.predict(X_test)
y_prob = reg.predict_proba(X_test)  # softmax probabilities

# Accuracy
print(f"Multinomial Logistic Regression Accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")

# Visualize probability distributions for a few test samples
num_samples = 5
fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
for i, ax in enumerate(axes):
    # Show the digit image
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"True: {y_test[i]}\nPred: {y_pred[i]}")
    ax.axis('off')

plt.show()

# Probability heatmap for first test sample
plt.figure(figsize=(8, 5))
sns.barplot(x=np.arange(10), y=y_prob[0])
plt.title(f"Class Probabilities for Sample 0 (Predicted: {y_pred[0]})")
plt.xlabel("Digit Class")
plt.ylabel("Predicted Probability")
plt.show()

# Optional: visualize the average probability distribution for all correctly classified samples
correct_idx = np.where(y_pred == y_test)[0]
mean_probs = y_prob[correct_idx].mean(axis=0)
plt.figure(figsize=(8, 5))
sns.barplot(x=np.arange(10), y=mean_probs)
plt.title("Average Predicted Probability Distribution (Correct Predictions)")
plt.xlabel("Digit Class")
plt.ylabel("Mean Probability")
plt.show()
