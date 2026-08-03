# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# Load dataset
print("Libraries imported")

# Read dataset
pima_df = pd.read_csv("diabetes.csv")

print("Dataset Loaded")

# Show first 5 rows
print("\nFirst 5 rows:")
print(pima_df.head())

# Select input and output
X = pima_df.drop("Outcome", axis=1)
y = pima_df["Outcome"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=7,
    stratify=y
)


# Scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData Transformed")


# Create Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

print("\nModel:")
print(model)


# Predict test data
predicted = model.predict(X_test)

print("\nPredicted Values:")
print(predicted)


# Check model accuracy
accuracy = accuracy_score(y_test, predicted)

print("\nAccuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(f"{accuracy * 100:.2f}%")


# Create confusion matrix
cm = confusion_matrix(y_test, predicted)

print("\nConfusion Matrix:")
print(cm)


# Show classification report
print("\nClassification Report:")
print(classification_report(y_test, predicted))


# Get prediction probability
y_predict_prob = model.predict_proba(X_test)[:, 1]


# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_predict_prob
)


# Calculate AUC score
roc_auc = auc(fpr, tpr)

print("\nROC AUC Score:")
print(roc_auc)


# Plot ROC curve
plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    color='darkorange',
    label='ROC curve (AUC = %0.2f)' % roc_auc
)

# Draw reference line
plt.plot(
    [0, 1],
    [0, 1],
    color='navy',
    linestyle='--'
)

# Add graph details
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gaussian Naive Bayes')
plt.legend(loc="lower right")
plt.grid()

# Show graph
plt.show()
