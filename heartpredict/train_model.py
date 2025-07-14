import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Set random seed for reproducibility
np.random.seed(42)

# Define dataset properties
NUM_SAMPLES = 100  # Number of training samples
NUM_FEATURES = 13  # Number of features per sample

# Generate a dummy dataset (Replace with real data)
X_train = np.random.rand(NUM_SAMPLES, NUM_FEATURES)
y_train = np.random.randint(2, size=NUM_SAMPLES)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # More trees for better accuracy
model.fit(X_train, y_train)

# Define model path
model_path = "heartdisease/heartpredict/heart_disease_model.pkl"

# Ensure the model directory exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Save the trained model
joblib.dump(model, model_path)

print(f"Model saved successfully at {model_path}")
