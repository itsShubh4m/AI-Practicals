# AI Chatbot Using TensorFlow and NLP

import json
import random
import re

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

# 1. Set random seeds for reproducibility

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# 2. Load the intent dataset

with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 3. Prepare words, classes, and training examples

words = []
classes = []
documents = []

for intent in data["intents"]:
    tag = intent["tag"]

    if tag not in classes:
        classes.append(tag)

    for pattern in intent["patterns"]:
        tokens = re.findall(r"\b\w+\b", pattern.lower())

        words.extend(tokens)
        documents.append((tokens, tag))

words = sorted(set(words))
classes = sorted(set(classes))

print("Total training examples:", len(documents))
print("Total unique words:", len(words))
print("Total intent classes:", len(classes))

# 4. Convert text into a bag-of-words vector

def bag_of_words(tokens):
    token_set = set(tokens)

    return np.array(
        [
            1.0 if word in token_set else 0.0
            for word in words
        ],
        dtype=np.float32
    )

# 5. Create training data

X = []
y = []

for tokens, tag in documents:
    X.append(bag_of_words(tokens))
    y.append(classes.index(tag))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# 6. Build the TensorFlow neural network

model = Sequential([
    Input(shape=(len(words),)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(len(classes), activation="softmax")
])

# 7. Compile the model

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 8. Train the model

print("\nTraining the chatbot model...")

model.fit(
    X,
    y,
    epochs=250,
    batch_size=8,
    verbose=0
)

print("Model training completed successfully!")

# 9. Define the response function

def get_response(message):
    # Convert the user's message into tokens
    tokens = re.findall(r"\b\w+\b", message.lower())

    if not tokens:
        return "Please enter a question."

    # Check whether any input word is known to the model
    known_tokens = [word for word in tokens if word in words]

    if not known_tokens:
        return (
            "Sorry, I don't understand that question. "
            "Type 'help' to see the available topics."
        )

    # Predict the intent probabilities
    input_vector = bag_of_words(tokens).reshape(1, -1)

    probabilities = model.predict(input_vector, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    # Reject low-confidence predictions
    if confidence < 0.65:
        return (
            "I'm not sure I understand. "
            "Please rephrase your question or type 'help'."
        )

    predicted_tag = classes[predicted_index]

    # Find a suitable response for the predicted intent
    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "Sorry, I couldn't find a suitable response."

# 10. Run the chatbot

def main():
    print("\n" + "=" * 48)
    print("       TENSORFLOW AI CHATBOT")
    print("=" * 48)

    print("Ask me about AI, ML, Python, TensorFlow, and NLP.")
    print("Type 'help' to see topics.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            message = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break

        if message.lower() in ("exit", "quit"):
            print("Bot: Goodbye! Have a nice day.")
            break

        response = get_response(message)

        print("Bot:", response, "\n")


if __name__ == "__main__":
    main()
