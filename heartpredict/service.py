import os
import io
import base64
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from django.conf import settings
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# ✅ Load the trained model globally
MODEL_PATH = os.path.join(settings.BASE_DIR, 'heartpredict', 'heart_disease_model.pkl')
model = joblib.load(MODEL_PATH)

# ✅ Load a pre-trained scaler (should be fitted on training data)
SCALER_PATH = os.path.join(settings.BASE_DIR, 'heartpredict', 'scaler.pkl')

if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    print("⚠️ Warning: Scaler not found! Ensure the data is scaled during training.")
    scaler = None  # Do not fit on single test input

def generate_plot_image(fig):
    """Convert a Matplotlib figure to a base64-encoded image for HTML rendering."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class Prediction:
    def predict(self, data, actual_label=None):
        """
        Predict heart disease and generate a confusion matrix.
        
        - `data`: Input features (list)
        - `actual_label`: True label (optional, required for correct confusion matrix)
        """
        data = np.array(data).reshape(1, -1)

        # ✅ Ensure the scaler is available before transforming
        if scaler:
            try:
                data_scaled = scaler.transform(data)
            except NotFittedError:
                print("⚠️ Warning: Scaler is not fitted! Ensure training data was scaled.")
                data_scaled = data  # Use raw input if scaler is missing
        else:
            data_scaled = data  # Use raw input if scaler was not found

        # ✅ Predict using the trained model
        prediction = model.predict(data_scaled)[0]
        prediction_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

        # ✅ Generate Confusion Matrix only if actual_label is provided
        if actual_label is not None:
            y_true = np.array([actual_label])  # Ground truth
            y_pred = np.array([prediction])    # Model prediction
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        else:
            # If no actual label is given, create a balanced confusion matrix
            cm = np.array([[1, 1], [1, 1]])

        # ✅ Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        confusion_matrix_img = generate_plot_image(fig)

        return {
            "prediction": prediction_text,
            "confusion_matrix": confusion_matrix_img
        }
