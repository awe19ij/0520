import re
from django.core.exceptions import ValidationError
from rest_framework import serializers
from .models import User
from django.contrib.auth import get_user_model

# 회원가입 로직
class SignupSerializer(serializers.ModelSerializer):
    password2 = serializers.CharField(style={'input_type': 'password'}, write_only=True)

    class Meta:
        model = User
        fields = ['username', 'password', 'password2']
        extra_kwargs = {
            'password': {'write_only': True}
        }


    # 아이디 유효성 검사
    # 이미 존재하는 아이디 검증
    def validate_username(self, value):
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError("A user with that username already exists.")
        return value

    # 비밀번호 유효성 검사
    # 영어,숫자 포함 8자 이상
    def validate_password(self, value):
        # 비밀번호 길이가 8자 미만인 경우
        if len(value) < 8:
            raise serializers.ValidationError("Password must be at least 8 characters.")
        # 비밀번호에 한 글자 이상 숫자가 포함되지 않을 경우
        if not re.findall('\d', value):
            raise serializers.ValidationError("Password must contain at least one digit.")
        # 비밀번호에 한 글자 이상 영어가 포함되지 않을 경우
        if not re.findall('[A-Za-z]', value):
            raise serializers.ValidationError("Password must contain at least one letter.")
        return value

    # 비밀번호 확인 로직
    # 재확인을 위한 입력은 db에 저장될 필요 없으므로 여기서 처리
    def validate(self, data):
        if data['password'] != data['password2']:
            raise serializers.ValidationError({"password2": "Password fields didn't match."})
        return data

    # 검증 완료 후 User 생성 - id,pw 
    def create(self, validated_data):
        user = User.objects.create(
            username=validated_data['username']
        )
        user.set_password(validated_data['password'])
        user.save()
        return user

# 로그인 로직

# 로그인

# 현재 활성화된 사용자 모델 가져오기
# 사용자 모델 교체 시 유연성 제공
User = get_user_model()

class LoginSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'