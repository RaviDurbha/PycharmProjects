# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('D:\Data\ECG1.csv')
x = data.iloc[:, 0:1].values  # first two columns of data frame with all rows
y = data.iloc[:, 1].values

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor  # import the regression

regressor = RandomForestRegressor(n_estimators=100, random_state=0)  # create regression object
regressor.fit(x, y)  # fit the regression with x and y data
Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))  # test the output by changing values

# Visualising the Random Forest Regression results arrange for creating a range of values
# from min value of x to max value of x with a difference of 0.01 between two consecutive values
X_grid = np.arange(min(x), max(x), 1)

# reshape for reshaping the data into a len(X_grid)*1 array, i.e. to make a column out of the X_grid value
X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(x, y, color='blue')  # Scatter plot for original data
plt.plot(X_grid, regressor.predict(X_grid), color='green')  # plot predicted data
plt.title('Random Forest Regression')
plt.xlabel('Sample #')
plt.ylabel('Column 0')
plt.grid(True)
plt.show()
