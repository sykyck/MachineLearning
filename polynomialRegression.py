import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Step 1: Load data
df = pd.read_csv('poly_data.csv')
X = df[['x']].values         # Feature matrix
y = df['y'].values            # Target vector

# Step 2: Scatter plot of raw data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data Points')

# Step 3: Create polynomial features (degree = 2, change as needed)
degree = 4
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)

# Step 4: Fit linear regression on transformed features
model = LinearRegression()
model.fit(X_poly, y)

# Step 5: Predict
y_pred = model.predict(X_poly)

# Step 6: Plot polynomial curve
X_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
X_line_poly = poly.transform(X_line)
y_line = model.predict(X_line_poly)

plt.plot(X_line, y_line, color='red', linewidth=2,
         label=f'Polynomial Regression (degree={degree})')
plt.title('Polynomial Regression Fit', size=16)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Step 7 (Optional): Show coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
