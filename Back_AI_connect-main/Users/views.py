from django.contrib.auth import authenticate
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
from .serializers import SignupSerializer, LoginSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.token_blacklist.models import BlacklistedToken, OutstandingToken
from rest_framework_simplejwt.exceptions import TokenError, InvalidToken


class SignupAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = SignupSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            return Response({
                "user": serializer.data,
                "message": "Signup Success",
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

User = get_user_model()

class LoginAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")
        user = authenticate(username=username, password=password)
        
        if user is not None:
            refresh = RefreshToken.for_user(user)
            return Response({
                "user": LoginSerializer(user).data,
                "message": "Login success",
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            }, status=status.HTTP_200_OK)
        else:
            
            # 존재하지 않는 아이디인지 검사 - 404
            if not User.objects.filter(username=username).exists():
                return Response({"error": "No user found with this username"}, status=status.HTTP_404_NOT_FOUND)
            # 비밀번호가 틀렸을 경우의 메시지 - 400
            return Response({"error": "Password is incorrect"}, status=status.HTTP_400_BAD_REQUEST)
        


class LogoutAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            # 요청에서 리프레시 토큰을 추출합니다.
            refresh_token = request.data.get("refresh_token")
            token = RefreshToken(refresh_token)
            # 토큰을 블랙리스트에 추가합니다.
            token.blacklist()

            # 로그잇 성공 시 - 205 
            return Response({"message": "Logout successful"}, status=status.HTTP_205_RESET_CONTENT)
        except TokenError as e:
            # 토큰 문제 있는 경우 - 400
            return Response({"error": "Token is invalid or expired"}, status=status.HTTP_400_BAD_REQUEST)
