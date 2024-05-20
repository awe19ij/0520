from django.shortcuts import get_object_or_404, render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from .models import InterviewAnalysis, QuestionLists
from google.cloud import speech
from google.cloud.speech import RecognitionConfig, RecognitionAudio
from google.oauth2 import service_account
import os
from django.conf import settings
import logging
from pydub import AudioSegment
import nltk
from nltk.tokenize import word_tokenize
import difflib
import parselmouth
import numpy as np
import base64
import io
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = os.path.join(settings.BASE_DIR, 'C:/Windows/Fonts/malgun.ttf')
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기본 폰트 설정

# matplotlib의 백엔드를 'Agg'로 설정
import matplotlib
matplotlib.use('Agg')
import pandas as pd

logger = logging.getLogger(__name__)

# Google Cloud 자격 증명 파일 경로 설정
credentials = service_account.Credentials.from_service_account_file(
    os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
)

client = speech.SpeechClient(credentials=credentials)

nltk.download('punkt')

 # responseAPI
class ResponseAPIView(APIView):
    # 파일 업로드를 위해 FileUploadParser 설정 > 단일 업로드가 아니라서 멀티로 바꿨어
    parser_classes = [MultiPartParser, FormParser]
    # 사용자 인증이 필요한 API 설정
    permission_classes = [IsAuthenticated]

    def post(self, request, question_list_id):
        # 질문 목록 ID를 기반으로 해당 객체를 찾고, 없을 경우 404 오류 반환
        question_list = get_object_or_404(QuestionLists, id=question_list_id)
        interview_response = InterviewAnalysis(question_list=question_list)
        
        # Google Cloud의 Speech-to-Text API를 활용해 음성 인식 클라이언트 설정
        client = speech.SpeechClient(credentials=credentials)
        config = RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=16000,
            language_code="ko-KR"
        )

        audio_file_path = None

        # 파일들을 순회하며 음성을 텍스트로 변환
        for i in range(1, 11):
            file_key = f'audio_{i}'
            if file_key not in request.FILES:
                continue

            audio_file = request.FILES[file_key]
            audio_file_path = os.path.join(settings.MEDIA_ROOT, audio_file.name)
            with open(audio_file_path, 'wb') as f:
                f.write(audio_file.read())

            audio = RecognitionAudio(content=audio_file.read())
            response = client.recognize(config=config, audio=audio)
            transcript = ' '.join([result.alternatives[0].transcript for result in response.results])
            setattr(interview_response, f'response_{i}', transcript)

        base_dir = settings.BASE_DIR
        redundant_expressions_path = os.path.join(base_dir, 'InterviewAnalyze', 'redundant_expressions.txt')
        inappropriate_terms_path = os.path.join(base_dir, 'InterviewAnalyze', 'inappropriate_terms.txt')

        try:
            with open(redundant_expressions_path, 'r') as file:
                redundant_expressions = file.read().splitlines()
            with open(inappropriate_terms_path, 'r') as file:
                inappropriate_terms = dict(line.strip().split(':') for line in file if ':' in line)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return Response({"error": "Required file not found"}, status=500)

        response_data = []
        for i in range(1, 11):
            question_key = f'question_{i}'
            response_key = f'response_{i}'
            question_text = getattr(question_list, question_key, None)
            response_text = getattr(interview_response, response_key, None)

            found_redundant = [expr for expr in redundant_expressions if expr in response_text]
            corrections = {}
            corrected_text = response_text
            for term, replacement in inappropriate_terms.items():
                if term in response_text:
                    corrections[term] = replacement
                    corrected_text = corrected_text.replace(term, replacement)

            response_data.append({
                'question': question_text,
                'response': response_text,
                'redundancies': found_redundant,
                'inappropriateness': list(corrections.keys()),
                'corrections': corrections,
                'corrected_response': corrected_text
            })

        # 발음 분석 및 피치 분석 수행
        pronunciation_result = None
        pitch_result = None
        intensity_result = None

        if audio_file_path:
            pronunciation_result = self.analyze_pronunciation(audio_file_path)
            pitch_result, intensity_result = self.analyze_pitch(audio_file_path)

            # 분석 결과를 인터뷰 응답 객체에 저장
            interview_response.pronunciation_similarity = pronunciation_result
            interview_response.pitch_analysis = pitch_result
            interview_response.intensity_analysis = intensity_result

        interview_response.save()

        return Response({
            'interview_id': interview_response.id,
            'responses': response_data,
            'pronunciation_similarity': pronunciation_result,
            'pitch_analysis': pitch_result,
            'intensity_analysis': intensity_result
        }, status=200)

    def analyze_pronunciation(self, audio_file_path):
        """음성 파일의 발음 분석을 수행합니다."""
        sound = AudioSegment.from_file(audio_file_path)
        sound = sound.set_channels(1)
        audio_content = sound.raw_data

        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000,
            language_code='ko-KR',
            enable_automatic_punctuation=True,
            max_alternatives=2  # 2개의 대안을 요청
        )

        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=90)

        # 첫 번째 대안은 가장 확신도가 높은 텍스트, 두 번째 대안은 가장 원시적인 텍스트로 사용
        highest_confidence_text = response.results[0].alternatives[0].transcript
        most_raw_text = response.results[0].alternatives[1].transcript if len(response.results[0].alternatives) > 1 else highest_confidence_text

        expected_sentences = re.split(r'[.!?]', most_raw_text)
        received_sentences = re.split(r'[.!?]', highest_confidence_text)

        pronunciation_result = []
        for expected_sentence, received_sentence in zip(expected_sentences, received_sentences):
            similarity = difflib.SequenceMatcher(None, expected_sentence.strip(), received_sentence.strip()).ratio()
            pronunciation_result.append({
                'expected': expected_sentence.strip(),
                'received': received_sentence.strip(),
                'similarity': similarity
            })

        return pronunciation_result

    def analyze_pitch(self, audio_file_path):
        """음성 파일의 피치 분석을 수행합니다."""
        sound = parselmouth.Sound(audio_file_path)

        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_times = pitch.xs()

        pitch_values_filtered = np.where((pitch_values >= 150) & (pitch_values <= 300), pitch_values, np.nan)

        intensity = sound.to_intensity()
        intensity_values = intensity.values.T
        intensity_times = intensity.xs()

        intensity_values_filtered = np.where((intensity_values >= 45) & (intensity_values <= 70), intensity_values, np.nan)
        
        # NaN을 None으로 변환
        pitch_values_filtered = np.where(np.isnan(pitch_values_filtered), None, pitch_values_filtered)
        intensity_values_filtered = np.where(np.isnan(intensity_values_filtered), None, intensity_values_filtered)

        pitch_result = {
            'times': pitch_times.tolist(),
            'values': pitch_values.tolist(),
            'filtered_values': pitch_values_filtered.tolist()
        }

        intensity_result = {
            'times': intensity_times.tolist(),
            'values': intensity_values.tolist(),
            'filtered_values': intensity_values_filtered.tolist()
        }

        return pitch_result, intensity_result

class PronunciationAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [AllowAny]

    def post(self, request):
        # 오디오 파일 확인
        if 'audio_file' not in request.FILES:
            return Response({"error": "Audio file not provided"}, status=400)

        audio_file = request.FILES['audio_file']
        audio_file_path = os.path.join(settings.MEDIA_ROOT, audio_file.name)
        with open(audio_file_path, 'wb') as f:
            f.write(audio_file.read())

        # 발음 분석 결과 가져오기
        pronunciation_result = self.analyze_pronunciation(audio_file_path)

        # JSON 형식의 결과 반환
        return Response({"pronunciation_similarity": pronunciation_result}, status=200)

    def analyze_pronunciation(self, audio_file_path):
        """음성 파일의 발음 분석을 수행합니다."""
        sound = AudioSegment.from_file(audio_file_path)
        sound = sound.set_channels(1)
        audio_content = sound.raw_data

        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000,
            language_code='ko-KR',
            enable_automatic_punctuation=True,
            max_alternatives=2  # 2개의 대안을 요청
        )

        client = speech.SpeechClient(credentials=credentials)
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=90)

        # 첫 번째 대안은 가장 확신도가 높은 텍스트, 두 번째 대안은 가장 원시적인 텍스트로 사용
        highest_confidence_text = response.results[0].alternatives[0].transcript
        most_raw_text = response.results[0].alternatives[1].transcript if len(response.results[0].alternatives) > 1 else highest_confidence_text

        expected_sentences = re.split(r'[.!?]', most_raw_text)
        received_sentences = re.split(r'[.!?]', highest_confidence_text)

        pronunciation_result = []
        for expected_sentence, received_sentence in zip(expected_sentences, received_sentences):
            similarity = difflib.SequenceMatcher(None, expected_sentence.strip(), received_sentence.strip()).ratio()
            pronunciation_result.append({
                'expected': expected_sentence.strip(),
                'received': received_sentence.strip(),
                'similarity': similarity
            })

        return pronunciation_result

class PitchAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        # 오디오 파일 확인
        audio_file = request.FILES.get('audio_file')
        if not audio_file:
            return Response({"error": "Audio file not provided"}, status=400)

        audio_file_path = os.path.join(settings.MEDIA_ROOT, audio_file.name)
        with open(audio_file_path, 'wb') as f:
            f.write(audio_file.read())

        # 피치 분석 결과 가져오기
        pitch_result, intensity_result, pitch_graph_base64, intensity_graph_base64 = self.analyze_pitch(audio_file_path)

        # JSON 형식의 결과 반환
        return Response({
            "pitch_analysis": pitch_result,
            "intensity_analysis": intensity_result,
            "pitch_graph": pitch_graph_base64,
            "intensity_graph": intensity_graph_base64
        }, status=200)

    def analyze_pitch(self, audio_file_path):
        """음성 파일의 피치 분석을 수행하고 그래프를 생성합니다."""
        sound = parselmouth.Sound(audio_file_path)

        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_times = pitch.xs()

        intensity = sound.to_intensity()
        intensity_values = intensity.values.T
        intensity_times = intensity.xs()

        # 피치 그래프 생성
        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(pitch_times, pitch_values, 'o', markersize=2, label='Pitch')
        ax1.plot(pitch_times, np.where((pitch_values >= 150) & (pitch_values <= 500), pitch_values, np.nan), 'o', markersize=2, color='blue', label='150-500 Hz')
        ax1.plot(pitch_times, np.where((pitch_values < 150) | (pitch_values > 500), pitch_values, np.nan), 'o', markersize=2, color='red', label='Out of Range')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_title('피치')
        ax1.set_xlim([0, max(pitch_times)])
        ax1.set_ylim([0, 500])
        ax1.grid(True)
        ax1.legend()
        ax1.set_xticks(np.arange(0, max(pitch_times), 1))

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pitch_graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        # 강도 그래프 생성
        fig, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(intensity_times, intensity_values, linewidth=1, label='Intensity')
        ax2.plot(intensity_times, np.where((intensity_values >= 35) & (intensity_values <= 65), intensity_values, np.nan), linewidth=1, color='blue', label='35-65 dB')
        ax2.plot(intensity_times, np.where((intensity_values < 35) | (intensity_values > 65), intensity_values, np.nan), linewidth=1, color='red', label='Out of Range')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Intensity (dB)')
        ax2.set_title('강도')
        ax2.set_xlim([0, max(intensity_times)])
        ax2.set_ylim([0, max(intensity_values)])
        ax2.grid(True)
        ax2.legend()
        ax2.set_xticks(np.arange(0, max(intensity_times), 1))

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        intensity_graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        pitch_result = {
            'times': pitch_times.tolist(),
            'values': pitch_values.tolist()
        }

        intensity_result = {
            'times': intensity_times.tolist(),
            'values': intensity_values.tolist()
        }

        return pitch_result, intensity_result, pitch_graph_base64, intensity_graph_base64
