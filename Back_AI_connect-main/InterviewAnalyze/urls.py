# InterviewAnalyze/urls.py

from django.urls import path
from .views import ResponseAPIView, PronunciationAPIView, PitchAPIView

urlpatterns = [
    path('responses/<int:question_list_id>/', ResponseAPIView.as_view(), name='interview_responses'),
    path('pronunciation/', PronunciationAPIView.as_view(), name='pronunciation_analysis'),
    path('pitch/', PitchAPIView.as_view(), name='pitch_analysis'),
]
