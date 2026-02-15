import numpy as np
import catboost as cb

print(f"NumPy version: {np.__version__}")
print(f"CatBoost version: {cb.__version__}")

# Simple test
try:
    data = np.array([[1, 2, 3], [4, 5, 6]])
    print("NumPy array created successfully")

    model = cb.CatBoostClassifier(iterations=10, depth=2, learning_rate=1, loss_function='Logloss', verbose=False)
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    model.fit(X, y)
    print("CatBoost model trained successfully")
except Exception as e:
    print(f"Error: {e}")
    exit(1)
