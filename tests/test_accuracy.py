import unittest
from sklearn.linear_model import LinearRegression
import numpy as np

class TestLinearRegressionModel(unittest.TestCase):
    def setUp(self):
        # Create the same dummy data used in the main code
        np.random.seed(0)
        self.X = np.arange(1000).reshape(-1, 1) / 1000
        self.y = 2 * self.X + np.random.randn(1000, 1) * 0.05

    def test_coefficient_and_intercept(self):
        model = LinearRegression()
        model.fit(self.X, self.y)

        # The expected coefficient is 2.0, so we check if it is within a small tolerance
        expected_coefficient = 200.0
        self.assertAlmostEqual(model.coef_[0][0], expected_coefficient, delta=0.1)

        # The intercept should be close to 0.0 (since the data generation process has no large bias)
        self.assertAlmostEqual(model.intercept_[0], 0.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()
