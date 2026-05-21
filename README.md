# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1. Import Libraries

Import required Python libraries: numpy, pandas, sklearn.

#### 2. Load Dataset

Load or create a dataset with features (e.g., area, number of rooms, location index).

Include target variables (house price, number of occupants).

#### 3. Preprocess Data

Handle missing values if present.

Scale features using StandardScaler to speed up convergence.

#### 4. Split Dataset

Use train_test_split to split data into training and testing sets.

#### 5. Train Model with SGD Regressor

Initialize SGDRegressor with a learning rate and max iterations.

Use MultiOutputRegressor for predicting multiple outputs.

Train model using the training set.

#### 6. Make Predictions

Predict house price and number of occupants for the test set.

#### 7. Evaluate Model

Use metrics such as r2_score and mean_squared_error to check performance.

#### 8. Display Results

Print actual vs predicted values for better understanding.

## Program:
```py
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Muthu Ganesh R
RegisterNumber: 212225040267
*/

# Ex:No 4
#Manual Implementation using Numpy
import numpy as np

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
# Features: [Hours Studied, Attendance, Previous Marks]
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
], dtype=float)

# Target: Marks Scored
y = np.array([50, 45, 70, 80, 95], dtype=float)

# ------------------------------
# Step 2: Feature normalization
# ------------------------------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Add bias term (intercept)
X = np.c_[np.ones(X.shape[0]), X]  # shape becomes (n_samples, n_features + 1)

# ------------------------------
# Step 3: Initialize weights
# ------------------------------
n_features = X.shape[1]
weights = np.zeros(n_features)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# ------------------------------
# Step 4: Stochastic Gradient Descent
# ------------------------------
for epoch in range(epochs):
    for i in range(X.shape[0]):
        xi = X[i]
        yi = y[i]
        y_pred = np.dot(xi, weights)
        error = y_pred - yi
        # Update weights
        weights -= learning_rate * error * xi

print("Trained Weights (including intercept):", weights)

# ------------------------------
# Step 5: Make predictions
# ------------------------------
y_pred_all = np.dot(X, weights)
print("Predicted values:", y_pred_all)

```

## Output:
<img width="817" height="58" alt="Screenshot 2026-05-21 220704" src="https://github.com/user-attachments/assets/8dd89861-5983-4d26-a5e8-d6e71cc47b30" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
