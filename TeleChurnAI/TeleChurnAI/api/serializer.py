from rest_framework import serializers
from .models import UploadedCSV

class UploadCSVSerializer(serializers.ModelSerializer)
    class Meta:
        model = UploadedCSV
        fields = '__all__'