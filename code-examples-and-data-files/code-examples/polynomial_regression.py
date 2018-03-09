# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)


# Visualising the Polynomial Regression results

# Add curve to graph
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

# Create graph
plt.scatter(X, y, color="red")
plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color="blue")
plt.title("Position Salary Expectations (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


# Predicting a new result with Polynomial Regression
new_result = lin_reg.predict(poly_reg.fit_transform(6.5))
