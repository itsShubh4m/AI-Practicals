import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Penguins dataset
data = sns.load_dataset("penguins")

# Remove missing values
data = data.dropna()

# Select features
X = data[[
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]]

# Binary target
# Adelie = 1, Other species = 0
y = (data["species"] == "Adelie").astype(int)

print("Dataset Shape:", X.shape)
print("Classes:", ["Other", "Adelie"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Individual Weak Classifier
weak_model = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)

weak_model.fit(X_train, y_train)
weak_pred = weak_model.predict(X_test)

weak_accuracy = accuracy_score(y_test, weak_pred)

# AdaBoost Ensemble
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

ada_model.fit(X_train, y_train)
ada_pred = ada_model.predict(X_test)

ada_accuracy = accuracy_score(y_test, ada_pred)

# Display Results
print("\nAdaBoost Ensemble Learning")
print("--------------------------")

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nWeak Classifier Accuracy:",
      weak_accuracy * 100, "%")

print("AdaBoost Accuracy:",
      ada_accuracy * 100, "%")

print("\nActual Values:")
print(y_test.values)

print("\nWeak Classifier Predictions:")
print(weak_pred)

print("\nAdaBoost Predictions:")
print(ada_pred)

print("\nAdaBoost Classification Report:")
print(classification_report(
    y_test,
    ada_pred,
    target_names=["Other", "Adelie"]
))

# Accuracy Comparison Graph
models = ["Weak Classifier", "AdaBoost"]
accuracies = [
    weak_accuracy * 100,
    ada_accuracy * 100
]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)

plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.title("Weak Classifier vs AdaBoost")
plt.ylim(0, 100)

for i, value in enumerate(accuracies):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()