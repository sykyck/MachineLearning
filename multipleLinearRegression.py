import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer  # Correct way to handle categorical encoding

# Step 1: Load dataset
data_set = pd.read_csv('synthetic_50_CompList.csv')

# Step 2: Explore the dataset
print(data_set.head())  # View first few rows
print(data_set.info())  # Check data types and missing values
print(data_set.describe())  # Summary statistics

# Step 3: Extracting Independent (X) and Dependent (y) Variables
X = data_set.iloc[:, :-1].values  # All columns except the last one
y = data_set.iloc[:, -1].values   # The last column (Profit)

# Step 4: Handling Categorical Data (State column)
# Use ColumnTransformer to apply OneHotEncoder to the 'State' column (index 3)
column_transformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first'), [3])],  # Drop first dummy variable
    remainder='passthrough'  # Keep other columns unchanged
)
X = column_transformer.fit_transform(X)

# Step 5: Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 6: Fitting the Multiple Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 7: Predicting the Test set results
y_pred = regressor.predict(X_test)

# Step 8: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print performance metrics
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# Step 9: Visualizing Actual vs Predicted Profit
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.show()

# Step 10: Print Model Coefficients
coefficients = pd.DataFrame(regressor.coef_, columns=['Coefficient'])
print(coefficients)
