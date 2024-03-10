# myapp1/models.py
"""from django.db import models

class ProcessedImage(models.Model):
    image = models.ImageField(upload_to='uploaded_images/')
    classification_result = models.CharField(max_length=100, blank=True)"""

from django.db import models

class ProcessedImage(models.Model):
    image = models.ImageField(upload_to='processed_images/')
    classification_result = models.CharField(max_length=255)
