import pandas as pd

data = {
    "top_color": ["white", "black", "blue", "red", "white", "black"],
    "bottom_color": ["blue", "black", "black", "green", "black", "blue"],
    "shoes_color": ["white", "black", "white", "pink", "black", "white"],
    "occasion": ["casual", "formal", "casual", "casual", "formal", "casual"],
    "compatible": [1, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)
df_encoded = pd.get_dummies(
    df,
    columns=["top_color", "bottom_color", "shoes_color", "occasion"],
    dtype=int
)

print(df_encoded)
X = df_encoded.drop("compatible", axis=1)
y = df_encoded["compatible"]

print("\nFeatures:")
print(X)

print("\nTarget:")
print(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:")
print(X_train)

print("\nTesting data:")
print(X_test)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nModel trained successfully!")
# New outfit
new_outfit = pd.DataFrame({
    "top_color": ["white"],
    "bottom_color": ["black"],
    "shoes_color": ["white"],
    "occasion": ["casual"]
})

# Convert new outfit using the same encoding
new_outfit_encoded = pd.get_dummies(
    new_outfit,
    columns=["top_color", "bottom_color", "shoes_color", "occasion"],
    dtype=int
)


new_outfit_encoded = new_outfit_encoded.reindex(
    columns=X.columns,
    fill_value=0
)

prediction = model.predict(new_outfit_encoded)
print("New outfit prediction:", prediction)
print("Probability:", probability)
print("\nNew outfit prediction:", prediction)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)