# train_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1) Load dataset
df = pd.read_csv('heart_disease_data.csv')

# 2) Split features and target
X = df.drop('target', axis=1)
y = df['target']

# 3) Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 5) Model (Random Forest tuned)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# 6) Train
model.fit(X_train, y_train)

# 7) Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy:  {test_acc:.4f}")
print("\nClassification report (test set):\n")
print(classification_report(y_test, model.predict(X_test)))

# 8) Save model and scaler
with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nSaved: heart_model.pkl, scaler.pkl")
