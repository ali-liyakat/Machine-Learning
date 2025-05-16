import numpy as np
import pandas as pd

def gradient_descent(X, y, lr=0.1, num_iterations=1000):
    m , b = 0.0, 0.0  
    # Scale X and y using min-max scaling
    X_min, X_max = X.min(), X.max()
    y_min, y_max = y.min(), y.max()

    X_scaled = (X - X_min) / (X_max - X_min)
    y_scaled = (y - y_min) / (y_max - y_min)

    for epoch in range(num_iterations):
        y_pred = m * X_scaled + b
        error = y_scaled - y_pred
        cost = np.mean(error ** 2)

        m_gradient = -2 * np.mean(X_scaled * error)
        b_gradient = -2 * np.mean(error)

        m -= lr * m_gradient
        b -= lr * b_gradient
        if epoch % 100 == 0:
            print(f"M: {m}, B: {b}, Epoch: {epoch} Cost: {cost}")

    # Unscale m and b
    m_original = m * (y_max - y_min) / (X_max - X_min)
    b_original = b * (y_max - y_min) + y_min - m_original * X_min

    return m_original, b_original


if __name__ == "__main__":
    # Example usage
    # X = np.array([1,2,3,4,5])
    # y = np.array([5,7,9,11,13])
    df = pd.read_csv("Regression/home_prices.csv")
    X = df['area_sqr_ft'].to_numpy()
    y = df['price_lakhs'].to_numpy()
    b, m = gradient_descent(X,y)
    print(f"Final M: {m}, Final B: {b}")