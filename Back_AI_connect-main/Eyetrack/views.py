from django.http import JsonResponse
from .gaze_tracking import GazeTracking

gaze_tracker = GazeTracking()

def start_gaze_tracking(request):
    gaze_tracker.start_tracking()  # 시선 추적 시작
    return JsonResponse({"message": "Gaze tracking started"}, status=200)

def stop_gaze_tracking(request):
    gaze_tracker.stop_tracking()  # 시선 추적 중지
    return JsonResponse({"message": "Gaze tracking stopped"}, status=200)
