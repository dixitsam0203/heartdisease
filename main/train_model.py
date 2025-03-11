import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Dummy dataset (replace with real data)
X_train = np.random.rand(100, 13)  # 100 samples, 13 features
y_train = np.random.randint(2, size=100)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model inside the heartpredict folder
joblib.dump(model, "heartdisease/heart_disease_model.pkl")

print("✅ Model saved successfully at heartdisease/heart_disease_model.pkl")
