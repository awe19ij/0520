from pathlib import Path
import os
import json

# JWT_AUTh 의 토큰 유효기간 설정
from datetime import timedelta
from django.core.exceptions import ImproperlyConfigured





# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Google Cloud 자격 증명 파일 경로

GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not GOOGLE_APPLICATION_CREDENTIALS:
    raise ImproperlyConfigured("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

# GOOGLE_APPLICATION_CREDENTIALS = os.path.join(BASE_DIR, 'sttapi.json')
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
# def get_env_variable(var_name):
#     """환경 변수를 가져오거나 명시적 예외를 반환한다."""
#     try:
#         return os.environ[var_name]
#     except KeyError:
#         error_msg = 'Set the {} environment variable'.format(var_name)
#         raise ImproperlyConfigured(error_msg)

# SECRET_KEY = get_env_variable('DJANGO_SECRET')

# secret_file = os.path.join(BASE_DIR, 'secrets.json')

# SECRET_KEY를 환경 변수에서 가져오기
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY')

# 환경 변수에서 가져온 SECRET_KEY가 없으면 예외 발생
if not SECRET_KEY:
    raise ImproperlyConfigured("Set the DJANGO_SECRET_KEY environment variable")


# with open(secret_file) as f:
#     secrets = json.loads(f.read())

# def get_secret(setting):
#     """비밀 변수를 가져오거나 명시적 예외를 반환한다."""
#     try:
#         return secrets[setting]
#     except KeyError:
#         error_msg = "Set the {} environment variable".format(setting)
#         raise ImproperlyConfigured(error_msg)


# SECRET_KEY = get_secret("SECRET_KEY")

# # JSON 파일 경로 설정
# secret_file = os.path.join(BASE_DIR, 'openapi.json')

# # JSON 파일에서 설정 로드
# with open(secret_file) as f:
#     secrets = json.load(f)

# def get_secret(setting, secrets=secrets):
#     """비밀 설정을 가져오거나 명시적 예외를 반환"""
#     try:
#         return secrets[setting]
#     except KeyError:
#         error_msg = f"Set the {setting} environment variable"
#         raise ImproperlyConfigured(error_msg)

# # API 키 설정
# OPENAI_API_KEY = get_secret("OPENAI_API_KEY")

# SECRET_KEY를 환경 변수에서 가져오기
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 환경 변수에서 가져온 SECRET_KEY가 없으면 예외 발생
if not OPENAI_API_KEY:
    raise ImproperlyConfigured("Set the OPENAI_API_KEY environment variable")


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!


# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']


CORS_ORIGIN_ALLOW_ALL = True
CORS_ALLOW_CREDENTIALS = True

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'QuestionList',
    'corsheaders',
    'Users',
    'rest_framework_simplejwt',
	'rest_framework_simplejwt.token_blacklist',
    'InterviewAnalyze',
    'Eyetrack',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',     # 추가
    'django.middleware.common.CommonMiddleware', # 추가
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
]

REST_FRAMEWORK = { # 추가
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',  #인증된 회원만 액세스 허용
        'rest_framework.permissions.AllowAny',         #모든 회원 액세스 허용
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': ( #api가 실행됬을 때 인증할 클래스 - Simple JWT 사용
         'rest_framework_simplejwt.authentication.JWTAuthentication', #이와 같이 추가
    ),
}

# 추가
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=30),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=3),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'VERIFYING_KEY': None,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
    'TOKEN_TYPE_CLAIM': 'token_type',
}

ROOT_URLCONF = 'ddok_back.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'ddok_back.wsgi.application'


# Database
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.0/howto/static-files/

STATIC_URL = 'static/'


# 미디어 파일을 저장할 경로
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Default primary key field type
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# settings.py
AUTH_USER_MODEL = 'Users.User'