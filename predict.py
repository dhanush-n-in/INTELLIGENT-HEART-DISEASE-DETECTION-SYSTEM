import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("heart.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")



print("\n===== Enter Patient Details =====")

# Manual input
age = int(input("Age: "))
sex = int(input("Sex (0=Female, 1=Male): "))
cp = int(input("Chest Pain Type (cp): "))
trestbps = int(input("Resting Blood Pressure: "))
chol = int(input("Cholesterol: "))
fbs = int(input("Fasting Blood Sugar (>120 mg/dl, 1=True, 0=False): "))
restecg = int(input("Resting ECG: "))
thalach = int(input("Maximum Heart Rate Achieved: "))
exang = int(input("Exercise Induced Angina (1=Yes, 0=No): "))
oldpeak = float(input("Oldpeak: "))
slope = int(input("Slope: "))
ca = int(input("Number of Major Vessels (ca): "))
thal = int(input("Thal: "))

# Create input sample
sample = [[
    age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak,
    slope, ca, thal
]]

# Prediction
prediction = model.predict(sample)

print("\n===== Prediction Result =====")

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease Detected")