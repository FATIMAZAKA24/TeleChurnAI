from django.db import models

class UploadedCSV(models.Model):
    file = models.FileField(upload_to='uploads/')  # Files will be stored in 'uploads/' directory
    uploaded_at = models.DateTimeField(auto_now_add=True)
