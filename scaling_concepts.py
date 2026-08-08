from sklearn.preprocessing import StandardScaler

import pandas as pd

fitTransform_data = pd.DataFrame({
    "Age": [20,30,40,50,60]
})

# Using Standard Scalar

scaler = StandardScaler()

# using fit_transform on data to learn the mean and standard deviation from the training data.

trained_data_using_fit_transform = scaler.fit_transform(fitTransform_data)

print(trained_data_using_fit_transform)

X_train = [
    [20, 20000],
    [30, 30000],
    [40, 40000],
    [50, 50000]
]

X_test = [
    [60, 60000]
]

# using fit_transform on data to learn the mean and standard deviation from the training data.
X_train_scaled = scaler.fit_transform(X_train)

# using transform to only scale
X_test_scaled = scaler.transform(X_test)

print(X_test_scaled)