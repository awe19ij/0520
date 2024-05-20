from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from .models import QuestionLists, ProblemSolvingQuestion, CommunicationSkillQuestion, GrowthPotentialQuestion, PersonalityTraitQuestion
import random
import requests
import logging

logger = logging.getLogger(__name__)

class ChatGPTView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user_input_field = request.data.get('input_field', '')
        user_input_job = request.data.get('input_job', '')
        selected_categories = request.data.get('selected_categories', [])

        if not user_input_field or not user_input_job:
            return Response({"error": "분야와 직무 입력은 필수입니다."}, status=status.HTTP_400_BAD_REQUEST)

        modified_input = f"{user_input_field}분야의 {user_input_job}직무와 관련된 면접 질문 10가지 리스트업해줘 한국어로"
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                json={"model": "gpt-3.5-turbo-0125", "messages": [{"role": "user", "content": modified_input}]},
                timeout=10
            )
            response.raise_for_status()
            job_related_questions = response.json().get('choices')[0].get('message').get('content').splitlines()
            job_related_questions = [q.strip() for q in job_related_questions if q.strip()]
        except requests.exceptions.RequestException as e:
            return Response({"error": f"서버 오류: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        all_questions = job_related_questions[:10]  # Ensure only 10 questions are selected initially

        category_models = {
            'problem_solving': ProblemSolvingQuestion,
            'communication_skills': CommunicationSkillQuestion,
            'growth_potential': GrowthPotentialQuestion,
            'personality_traits': PersonalityTraitQuestion
        }

        for category in selected_categories:
            if category in category_models:
                model = category_models[category]
                questions = list(model.objects.all().values_list('question', flat=True))
                random_questions = random.sample(questions, min(len(questions), 2))
                all_questions.extend(random_questions)

        all_questions = random.sample(all_questions, 10)  # Shuffle and pick only 10 questions overall

        question_list = QuestionLists(user=request.user)
        for i, question in enumerate(all_questions, 1):
            setattr(question_list, f'question_{i}', question)
        question_list.save()

        return Response({"id": question_list.id, "questions": all_questions}, status=status.HTTP_201_CREATED)
