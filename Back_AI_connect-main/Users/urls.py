from django.urls import path
from .views import * 
from rest_framework_simplejwt.views import TokenRefreshView


urlpatterns = [
    # 회원가입
    path("signUp/", SignupAPIView.as_view(), name="signup"),
    
    # 로그인
    path("logIn/", LoginAPIView.as_view(), name="login"),

    # 토큰 재발급하기
    path('logIn/refresh/', TokenRefreshView.as_view()),

    # 로그아웃
    path('logOut/', LogoutAPIView.as_view(), name='logout'),
]