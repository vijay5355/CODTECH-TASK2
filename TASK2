/* It implements a simple linear regression model using a dataset with continuous
target variables. Split the data into training and testing sets, train the model on
the training data, evaluate its performance using metrics like mean squared
error or R-squared, and make predictions on the test set. Visualize the
regression line and actual vs. predicted values to assess the model's accuracy */


import numpy as np
import pandas as pd
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
data = pd.DataFrame(np.hstack([X, y]), columns=['Feature', 'Target'])
data.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[['Feature']], data['Target'], test_size=0.2, random_state=42)



from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score
y_train_pred = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
print(f"Training Set - Mean Squared Error: {mse_train:.2f}, R-squared: {r2_train:.2f}")

y_test_pred = model.predict(X_test)

mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"Test Set - Mean Squared Error: {mse_test:.2f}, R-squared: {r2_test:.2f}")


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, y_train_pred, color='red', linewidth=2, label='Regression line')
plt.title('Training set')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='green', label='Testing data')
plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Regression line')
plt.title('Test set')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs. Predicted values (Test set)')
plt.show()
