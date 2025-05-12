import joblib
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import JsonResponse
from .service import Prediction
from io import BytesIO
import base64

# Load the trained model
try:
    model = joblib.load("heart_disease_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

# Function to generate a bar chart for predictions
def generate_bar_chart(prediction):
    # Ensure prediction is a numeric value (0 or 1)
    prediction_value = 1 if prediction == "Heart Disease Detected" else 0
    
    # Create a bar chart
    categories = ['No Heart Disease', 'Heart Disease']
    values = [1 - prediction_value, prediction_value]
    
    fig, ax = plt.subplots()
    ax.bar(categories, values, color=['green', 'red'])
    ax.set_title("Heart Disease Prediction")
    ax.set_ylabel("Probability")
    
    # Save it to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    
    # Encode it as base64
    chart_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return chart_data


# Function to generate a risk factor chart
def generate_risk_factor_chart(input_data):
    # Define the feature labels (can be modified as per your model's features)
    feature_labels = [
        "Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol", 
        "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", 
        "Exercise Induced Angina", "Old Peak", "Slope", "CA", "Thal"
    ]
    
    # Convert input data into a list of numeric features
    values = input_data  # Make sure input_data is a list of numeric values
    
    # Create a bar chart for risk factors
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_labels, values, color='skyblue')
    ax.set_xlabel('Factor Value')
    ax.set_title('Risk Factors for Heart Disease')

    # Save it to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Encode it as base64
    chart_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return chart_data

def home(request):
    prediction_result = None
    confusion_matrix_img = None
    bar_chart_img = None
    risk_factor_chart_img = None  # New variable for risk factor chart

    if request.method == "POST":
        input_data = [
            float(request.POST["age"]),
            float(request.POST["sex"]),
            float(request.POST["cp"]),
            float(request.POST["trestbps"]),
            float(request.POST["chol"]),
            float(request.POST["fbs"]),
            float(request.POST["restecg"]),
            float(request.POST["thalach"]),
            float(request.POST["exang"]),
            float(request.POST["oldpeak"]),
            float(request.POST["slope"]),
            float(request.POST["ca"]),
            float(request.POST["thal"])
        ]

        predictor = Prediction()
        result = predictor.predict(input_data)

        prediction_result = result["prediction"]
        confusion_matrix_img = result["confusion_matrix"]
        
        # Generate the bar chart image based on prediction
        bar_chart_img = generate_bar_chart(prediction_result)
        
        # Generate the risk factor chart
        risk_factor_chart_img = generate_risk_factor_chart(input_data)

    return render(request, "index.html", {
        "prediction": prediction_result,
        "confusion_matrix": confusion_matrix_img,
        "bar_chart": bar_chart_img,  # Pass the bar chart image to the template
        "risk_factor_chart": risk_factor_chart_img  # Pass the risk factor chart image to the template
    })


def predict_heart_disease(request):
    if request.method == "POST":
        try:
            data = request.POST
            print("Received POST data:", data)

            # Extract features in correct order
            features = np.array([
                float(data.get("age")),
                float(data.get("sex")),
                float(data.get("cp")),
                float(data.get("trestbps")),
                float(data.get("chol")),
                float(data.get("fbs")),
                float(data.get("restecg")),
                float(data.get("thalach")),
                float(data.get("exang")),
                float(data.get("oldpeak")),
                float(data.get("slope")),
                float(data.get("ca")),
                float(data.get("thal"))
            ]).reshape(1, -1)
            print("Extracted Features:", features)

            # Ensure model prediction works
            prediction = model.predict(features)
            print("Raw Prediction Output:", prediction)

            result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
            print("Prediction Result:", result)  # Ensure this prints

            return JsonResponse({"prediction": result, "features": features.tolist()})

        except Exception as e:
            print("Error:", str(e))
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"message": "Send a POST request with required parameters."})
