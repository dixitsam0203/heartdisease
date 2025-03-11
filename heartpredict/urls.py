from django.urls import path
from .views import home, predict_heart_disease

urlpatterns = [
    path("", home, name="home"),
    path("predict/", predict_heart_disease, name="predict"),
]
