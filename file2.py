import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#import joblib
#import neptune
#from neptune.types import File
import os

lr = LinearRegression()
# #############################################################################
# Load and split data
for _ in range(100):
    rng = np.random.RandomState(_)
    x = 10 * rng.rand(1000).reshape(-1,1)
    y = 2 * x - 5 + rng.randn(1000).reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    # #############################################################################
    # Fitting the model

    lr.fit(X_train, y_train)
    test_mse = mean_squared_error(y_test, lr.predict(X_test))
    average_mse = np.mean(test_mse)
    print(f'MSE Result: { test_mse}')
    print("Average Mean Squared Error:", average_mse)
    # Write metrics to file
    with open('metrics.txt', 'w') as outfile:
        outfile.write(f'\n Mean Squared Error = {average_mse}.')

