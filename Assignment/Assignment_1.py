import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("traffic.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ==========================
# Convert text to numbers
# ==========================
encoder = LabelEncoder()

df["Time"] = encoder.fit_transform(df["Time"])
df["Weather"] = encoder.fit_transform(df["Weather"])
df["Road Type"] = encoder.fit_transform(df["Road Type"])
df["Congestion"] = encoder.fit_transform(df["Congestion"])

# ==========================
# Features and Target
# ==========================
X = df[["Vehicle Density", "Time", "Weather", "Road Type"]]
y = df["Congestion"]

# ==========================
# Split Dataset
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Model
# ==========================
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================
y_pred = model.predict(X_test)

# ==========================
# Accuracy
# ==========================
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy =", round(accuracy * 100, 2), "%")

# ==========================
# Confusion Matrix
# ==========================
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ==========================
# Classification Report
# ==========================
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================
# Average Vehicle Density
# ==========================
print("\nAverage Vehicle Density =", np.mean(df["Vehicle Density"]))

# ==========================
# Visualization 1
# ==========================
plt.figure(figsize=(6,4))
df["Congestion"].value_counts().plot(kind="bar")
plt.title("Traffic Congestion")
plt.xlabel("Congestion")
plt.ylabel("Count")
plt.show()

# ==========================
# Visualization 2
# ==========================
plt.figure(figsize=(6,4))
plt.hist(df["Vehicle Density"], bins=10)
plt.title("Vehicle Density Distribution")
plt.xlabel("Vehicle Density")
plt.ylabel("Frequency")
plt.show()Assignment 1
