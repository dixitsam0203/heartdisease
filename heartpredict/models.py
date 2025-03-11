import joblib
import os

# Get the correct path for the model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")

# Load the model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Model file not found. Train and save the model first.")
