from django.urls import path
from .views import start_gaze_tracking, stop_gaze_tracking

urlpatterns = [
    path('start/', start_gaze_tracking, name='start-gaze-tracking'),
    path('stop/', stop_gaze_tracking, name='stop-gaze-tracking'),
]
