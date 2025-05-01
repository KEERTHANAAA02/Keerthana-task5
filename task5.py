import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("D:/INTERNSHIP/heart.csv")  

df = df.dropna()

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=["0", "1"])
plt.title("Decision Tree (Full Depth)")
plt.show()

dt_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_limited.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("=== Accuracy Comparison ===")
print(f"Decision Tree (Train):      {dt.score(X_train, y_train):.2f}")
print(f"Decision Tree (Test):       {dt.score(X_test, y_test):.2f}")
print(f"Limited Depth DT (Test):    {dt_limited.score(X_test, y_test):.2f}")
print(f"Random Forest (Train):      {rf.score(X_train, y_train):.2f}")
print(f"Random Forest (Test):       {rf.score(X_test, y_test):.2f}")

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', title="Random Forest Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

dt_cv = cross_val_score(dt, X, y, cv=5).mean()
rf_cv = cross_val_score(rf, X, y, cv=5).mean()
print("\n=== Cross-Validation Accuracy ===")
print(f"Decision Tree:  {dt_cv:.2f}")
print(f"Random Forest:  {rf_cv:.2f}")
