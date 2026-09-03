import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Titanic dataset
data = sns.load_dataset("titanic")

# Select useful numerical features
data = data[
    ["survived", "pclass", "age", "sibsp", "parch", "fare"]
]

# Remove missing values
data = data.dropna()

# Features and target
X = data[
    ["pclass", "age", "sibsp", "parch", "fare"]
]

y = data["survived"]

print("Dataset Shape:", X.shape)
print("Classes:", ["Not Survived", "Survived"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create Naive Bayes model
nb_model = GaussianNB()

# Train model
nb_model.fit(X_train, y_train)

# Predict test data
y_pred = nb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate class probabilities
class_probabilities = nb_model.predict_proba(X_test)

# Display results
print("\nNaive Bayes Classifier")
print("----------------------")

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nAccuracy:", accuracy * 100, "%")

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

print("\nClass Probabilities for First 10 Test Samples:")
print(class_probabilities[:10])

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Survived", "Survived"]
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
plt.imshow(cm)

plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.colorbar()

plt.xticks(
    range(2),
    ["Not Survived", "Survived"]
)

plt.yticks(
    range(2),
    ["Not Survived", "Survived"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()

# Accuracy Graph
plt.figure(figsize=(6, 5))

plt.bar(
    ["Naive Bayes"],
    [accuracy * 100]
)

plt.ylabel("Accuracy (%)")
plt.title("Naive Bayes Classification Accuracy")
plt.ylim(0, 100)

plt.text(
    0,
    accuracy * 100 + 1,
    f"{accuracy * 100:.2f}%",
    ha="center"
)

plt.tight_layout()
plt.show()