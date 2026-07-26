# Malnutrition Risk Classification Using Neural Network
# SDG 2 – Zero Hunger

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------
# Step 1: Create Dataset
# ---------------------------------------

data = {
    "Age": [2,4,1,3,5,2,4,1,3,5,2,4,1,3,5,2,4,1,3,5],
    "Weight": [8,12,6,10,15,7,13,5,9,14,8,11,6,10,15,7,12,5,9,14],
    "Height": [75,90,65,85,100,70,92,60,82,98,74,89,66,84,101,71,91,61,83,99],
    "MUAC": [11,13,10,12,14,11,13,9,12,14,11,13,10,12,14,11,13,9,12,14],
    "Malnutrition": [1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0]
}

df = pd.DataFrame(data)

print("\n========== DATASET ==========")
print(df)

# ---------------------------------------
# Step 2: Select Features and Target
# ---------------------------------------

X = df[["Age", "Weight", "Height", "MUAC"]]
y = df["Malnutrition"]

# ---------------------------------------
# Step 3: Split Dataset
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# ---------------------------------------
# Step 4: Create Neural Network
# ---------------------------------------

model = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=2000,
    random_state=42
)

# ---------------------------------------
# Step 5: Train Model
# ---------------------------------------

model.fit(X_train, y_train)

# ---------------------------------------
# Step 6: Prediction
# ---------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------
# Step 7: Accuracy
# ---------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n========== ACCURACY ==========")
print(round(accuracy * 100, 2), "%")

# ---------------------------------------
# Step 8: Confusion Matrix
# ---------------------------------------

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------
# Step 9: Classification Report
# ---------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred, zero_division=0))

# ---------------------------------------
# Step 10: Display Actual and Predicted
# ---------------------------------------

print("\n========== ACTUAL vs PREDICTED ==========")

result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(result)

# ---------------------------------------
# Step 11: Predict New Child
# ---------------------------------------

new_child = pd.DataFrame({
    "Age": [2],
    "Weight": [7],
    "Height": [70],
    "MUAC": [10]
})

prediction = model.predict(new_child)

print("\n========== NEW CHILD PREDICTION ==========")

if prediction[0] == 1:
    print("Child is at Risk of Malnutrition")
else:
    print("Child is Healthy")
