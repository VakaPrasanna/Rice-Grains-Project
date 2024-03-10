# myapp1/urls.py
from django.urls import path
from .views import classify_image

urlpatterns = [
    path('classify/', classify_image, name='classify_image'),
    # Add more paths as needed
]
