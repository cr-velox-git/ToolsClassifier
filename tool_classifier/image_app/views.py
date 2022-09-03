from django.shortcuts import render, HttpResponse
from django.core.files.storage import default_storage
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse
from django.views.decorators.csrf import csrf_exempt #so we can use it to allow other domain to acces it easly

from tool_classifier.image_app import image_processing


# Create your views here.

@csrf_exempt
def ProfessFile(request):
   if request.method == 'POST':
        file = request.FILES['test_image']
        file_name = default_storage.save(file.name,file)
        return JsonResponse(image_processing.test_image(file),safe=False)
