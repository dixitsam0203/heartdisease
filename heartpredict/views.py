import joblib
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from .service import Prediction
# Load the trained model
try:
    model = joblib.load("heart_disease_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

# def home(request):
#     if request.method == "POST":
#         try:
#             # Extract input values from the form and convert them to float
#             age = float(request.POST.get("age", 0))
#             sex = float(request.POST.get("sex", 0))
#             cp = float(request.POST.get("cp", 0))
#             trestbps = float(request.POST.get("trestbps", 0))
#             chol = float(request.POST.get("chol", 0))
#             fbs = float(request.POST.get("fbs", 0))
#             restecg = float(request.POST.get("restecg", 0))
#             thalach = float(request.POST.get("thalach", 0))
#             exang = float(request.POST.get("exang", 0))
#             oldpeak = float(request.POST.get("oldpeak", 0))
#             slope = float(request.POST.get("slope", 0))
#             ca = float(request.POST.get("ca", 0))
#             thal = float(request.POST.get("thal", 0))

#             # Prepare input data as a list
#             input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

#             print(f"✅ Received Data: {input_data}")

#             # Make prediction
#             predictor = Prediction()
#             prediction = predictor.predict(input_data)

#             print(f"🎯 Prediction: {prediction}")

#             # Return response with prediction
#             return render(request, "index.html", {"prediction": prediction})

#         except Exception as e:
#             print(f"❌ Error: {e}")
#             return render(request, "index.html", {"error": str(e)})

#     # ✅ Ensure GET requests return the page without errors
#     return render(request, "index.html", {"prediction": None})


def home(request):
    prediction_result = None
    confusion_matrix_img = None

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

    return render(request, "index.html", {
        "prediction": prediction_result,
        "confusion_matrix": confusion_matrix_img
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
