import pandas as pd
import numpy as np  
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load CSV
df = pd.read_csv('synthetic_50_CompList.csv')

# Extracting Independent and Dependent Variables  
x = df.iloc[:, :-1].values  
y = df.iloc[:, 4].values

# Handle Categorical Data (e.g., 'State' at index 3)
column_transformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first'), [3])],
    remainder='passthrough'
)
x = column_transformer.fit_transform(x)

# Ensure dense array and float dtype
x = np.array(x, dtype=float)

# Add constant (intercept)
x = np.append(arr=np.ones((x.shape[0], 1)).astype(float), values=x, axis=1)

# Choose subset of variables if needed
x_opt = x[:, [2, 3, 4, 5]]  # Adjust columns as per your analysis

# Run OLS
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())
