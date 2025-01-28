from sklearn.linear_model import LinearRegression
import numpy as np
import os

# Number of iterations (not needed for sklearn, but keeping it for environment parity)
epochs = int(os.getenv("EPOCHS", 10))

# Dummy data for regression (y = 2x + noise)
np.random.seed(0)
X = np.arange(1000).reshape(-1, 1) / 1000  # 1000 samples, 1 feature
y = 2 * X + np.random.randn(1000, 1) * 0.05

# Initialize the LinearRegression model
model = LinearRegression()

# Train the model
print("Training model")
model.fit(X, y)

# After fitting, we can check the coefficient and intercept
print("Model trained")
print(f"Coefficient: {model.coef_[0][0]}, Intercept: {model.intercept_[0]}")
print(f"The expected coeeficnet was 2.0")