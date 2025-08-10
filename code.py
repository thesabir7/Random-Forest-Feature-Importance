import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1️⃣ Load dataset (ensure 'train.csv' is in the same folder)
data = pd.read_csv("train.csv")

# 2️⃣ Preprocess data
# Select features (simplified)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Handle missing values
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# Convert categorical to numeric
data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)

X = data[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"]]
y = data["Survived"]

# 3️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Decision Tree for comparison
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
dt_preds = dt_clf.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)

# 5️⃣ Random Forest model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_preds = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

# 6️⃣ Print results
print("📊 Model Performance:")
print(f"Decision Tree Accuracy: {dt_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# 7️⃣ Feature importance
importances = rf_clf.feature_importances_
feature_names = X.columns

# Sort by importance
indices = importances.argsort()

plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()
