import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 6, 9, 12, 15])

model = LinearRegression()
model.fit(X, y)

X_test = np.array([[6], [7]])
predictions = model.predict(X_test)

print("Predictions for", X_test.flatten(), "are", predictions)
