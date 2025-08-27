# Importing libraries  
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
import statsmodels.api as smf  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression    

def backward_elimination(X, y, significance_level=0.05):
    numVars = X.shape[1]
    temp = np.zeros((X.shape[0], numVars))
    for i in range(numVars):
        regressor_OLS = smf.OLS(y, X).fit()
        max_p_val = max(regressor_OLS.pvalues)
        if max_p_val > significance_level:
            for j in range(X.shape[1]):
                if regressor_OLS.pvalues[j] == max_p_val:
                    X = np.delete(X, j, 1)
    return X

# Importing dataset  
data_set = pd.read_csv('synthetic_50_CompList.csv')  

# Extracting Independent and Dependent Variables  
X = data_set.iloc[:, :-1].values  # All columns except last (features)
y = data_set.iloc[:, -1].values  # Last column (target)

# Applying OneHotEncoder to the 'State' column (assumed to be at index 3)  
# Adjust index if needed
column_transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(drop='first'), [3])  # Drop first to avoid dummy variable trap
    ],
    remainder='passthrough'  # Keep other columns unchanged
)

X = column_transformer.fit_transform(X)

# Ensure dense array and float dtype
X = np.array(X, dtype=float)

# Add constant (intercept)
X = np.append(arr=np.ones((X.shape[0], 1)).astype(float), values=X, axis=1)

# Choose subset of variables if needed
# x_opt = X[:, [2, 3, 4, 5]]  
# Instead of doing this we are making it dynamic by calling backward_elimination
x_opt = backward_elimination(X, y)

# Splitting the dataset into training and test sets  
X_train, X_test, y_train, y_test = train_test_split(x_opt, y, test_size=0.2, random_state=0)  

# Fitting the Multiple Linear Regression model to the training set  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)  

# Predicting the Test set results  
y_pred = regressor.predict(X_test)  

# Evaluating the model  
print('Train Score:', regressor.score(X_train, y_train))  
print('Test Score:', regressor.score(X_test, y_test))  
