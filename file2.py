import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#import joblib
#import neptune
#from neptune.types import File
import os
import matplotlib.pyplot as plt



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
    y_preds = lr.predict(X_test)
    plt.figure(figsize=(6, 5))
  # Plot training data in blue
    plt.scatter(X_train, y_train, c="b", label="Training data")
  # Plot test data in green
    plt.scatter(X_test, y_test, c="g", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(X_test, y_preds, c="r", label="Predictions")
  # Show the legend
    plt.legend(shadow='True')
  # Set grids
    plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
  # Some text
    plt.title('Model Results', family='Arial', fontsize=14)
    plt.xlabel('X axis values', family='Arial', fontsize=11)
    plt.ylabel('Y axis values', family='Arial', fontsize=11)
  # Show
    plt.savefig('model_results.png', dpi=120)

    test_mse = mean_squared_error(y_test, y_preds )
    average_mse = np.mean(test_mse)
    print(f'MSE Result: { test_mse}')
    print("Average Mean Squared Error:", average_mse)
    with open('metrics.txt', 'w') as outfile:
        outfile.write(f'\n Mean Squared Error = {average_mse}.')
