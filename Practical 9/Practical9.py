import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Create market basket dataset
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Bread', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Bread', 'Butter', 'Eggs'],
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Eggs']
]

# Convert transactions into one-hot encoded format
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_array, columns=te.columns_)

print("Transaction Dataset:")
print(df)

# Find frequent itemsets
frequent_itemsets = apriori(
    df,
    min_support=0.3,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

rules = rules[
    ['antecedents', 'consequents', 'support', 'confidence', 'lift']
]

print("\nAssociation Rules:")
print(rules)

# Display rule interpretation
print("\nRule Interpretation:")

for _, rule in rules.iterrows():
    antecedent = ", ".join(rule['antecedents'])
    consequent = ", ".join(rule['consequents'])

    print(
        f"If customers buy [{antecedent}], "
        f"they are likely to buy [{consequent}] "
        f"(Support: {rule['support']:.2f}, "
        f"Confidence: {rule['confidence']:.2f}, "
        f"Lift: {rule['lift']:.2f})"
    )


# ---------------- GRAPH 1 ----------------

itemsets = frequent_itemsets.copy()

itemsets["items"] = itemsets["itemsets"].apply(
    lambda x: ", ".join(list(x))
)

plt.figure(figsize=(8, 5))

plt.bar(itemsets["items"], itemsets["support"])

plt.xlabel("Itemsets")
plt.ylabel("Support")
plt.title("Support of Frequent Itemsets")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ---------------- GRAPH 2 ----------------

rules["rule"] = rules.apply(
    lambda x: ", ".join(x["antecedents"]) + " → " +
              ", ".join(x["consequents"]),
    axis=1
)

plt.figure(figsize=(8, 5))

plt.bar(rules["rule"], rules["confidence"])

plt.xlabel("Association Rules")
plt.ylabel("Confidence")
plt.title("Confidence of Association Rules")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()