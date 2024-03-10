# views.py
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import ProcessedImage
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


def classify_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Save the uploaded image temporarily
            uploaded_image = request.FILES['image']
            temp_file_path = os.path.join(settings.BASE_DIR, 'temp_image.jpg')

            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(uploaded_image.read())

            # Check if the uploaded file is an image
            if not uploaded_image.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                raise ValueError("Invalid file type. Please upload a valid image.")

            # Load the trained CNN model
            model_path = os.path.join(settings.BASE_DIR, 'myapp1', 'mymodels', 'm1_model.h5')
            cnn_model = load_model(model_path)

            # Preprocess the uploaded image for prediction
            img = image.load_img(temp_file_path, target_size=(256, 256))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the pixel values

            # Make a prediction
            predictions = cnn_model.predict(img_array)
            class_labels = ['Arborio', 'Basmathi', 'Ispala', 'Jasmine', 'Karacadag']
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_labels[predicted_class_index]

            # Save the classification result to the model
            processed_image = ProcessedImage(image=uploaded_image, classification_result=predicted_class)
            processed_image.save()

            # Debugging: Print the processed image URL
            print(f"Processed Image URL: {processed_image.image.url}")
            print(f"Processed Image URL: {processed_image.image.url}")
            return render(request, 'myapp1/classification_result.html', {'processed_image': processed_image})

        except Exception as e:
            print(f"Error during image classification: {e}")
            return HttpResponse(f"Error during image classification. Details: {str(e)}")

    return render(request, 'myapp1/classify_image.html')
