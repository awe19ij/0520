
from django.urls import path
from . import views

urlpatterns = [
    path('', views.ChatGPTView.as_view(), name="generate_text"),
]
