from django.contrib import admin

from django.urls import path, include

#for image
from django.conf.urls.static import static
from django.conf import  settings

from tool_classifier.image_app import views

urlpatterns = [
    path(r'^image/upload',views.ProfessFile)
]+static(settings.MEDIA_URL,document_root = settings.MEDIA_ROOT)
