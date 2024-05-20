from rest_framework import serializers
from .models import InterviewAnalysis

class InterviewResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = InterviewAnalysis
        fields = '__all__'


# class ResponseDetailSerializer(serializers.ModelSerializer):
#     question_text = serializers.SerializerMethodField()
#     response_text = serializers.SerializerMethodField()

#     class Meta:
#         model = ResponseDetail
#         fields = ['question_text', 'response_text', 'redundant_expressions', 'inappropriate_terms']

#     def get_question_text(self, obj):
#         # InterviewAnalysis에서 각 응답의 질문 텍스트를 가져오는 방법이 필요
#         # 예를 들어, InterviewAnalysis 모델에 question_text_1, ... question_text_10 필드가 있다고 가정
#         return getattr(obj.interview_analysis, f'question_text_{obj.response_number}', None)

#     def get_response_text(self, obj):
#         return getattr(obj.interview_analysis, f'response_{obj.response_number}', None)
