import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load California Housing dataset
data = fetch_california_housing()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Features:", data.feature_names)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create K-NN Regression model
knn_model = KNeighborsRegressor(n_neighbors=5)

# Train model
knn_model.fit(X_train, y_train)

# Predict test data
y_pred = knn_model.predict(X_test)

# Calculate errors
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display Results
print("\nK-Nearest Neighbors (K-NN) Regression")
print("-------------------------------------")

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nK Value:", 5)

print("\nMean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)

print("\nActual Values:")
print(y_test[:10])

print("\nPredicted Values:")
print(y_pred[:10])

# Actual vs Predicted Graph
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("K-NN Regression: Actual vs Predicted")

plt.tight_layout()
plt.show()

# Error Graph
errors = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.hist(errors, bins=20)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("K-NN Regression Prediction Errors")

plt.tight_layout()
plt.show()