import numpy as nm  
import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Loading the dataset
data_set= pd.read_csv('Salary_data.csv')  

# Step 2: Exploratory Data Analysis
print(data_set.head())  # Preview the first few rows
print(data_set.describe())  # Summary statistics

#setting the x as experience and y as salary
x= data_set.iloc[:, :-1].values  
y= data_set.iloc[:, 1].values   

# Splitting the dataset into training and test set.   
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)  

# Step 4: Build and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(x_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 7: Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.scatter(x_test, y_test, color='orange', label='Test Data')
plt.plot(x, model.predict(x), color='red', label='Regression Line')
plt.title('Years of Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()