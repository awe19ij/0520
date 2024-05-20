# 인성 질문 DB 한 번에 넣기

from django.core.management.base import BaseCommand
from QuestionList.models import PersonalityTraitQuestion

class Command(BaseCommand):
    help = 'Populates the database with problem solving questions'

    def handle(self, *args, **options):
        questions = [
            "동료나 고객과의 갈등을 해결한 경험에 대해 말씀해 주십시오.",
            "직장 내에서 윤리적 딜레마에 직면했을 때 어떻게 처리했는지 설명해주세요.",
            "팀워크를 중요시하는 자신만의 원칙에 대해 설명해주세요.",
            "직장에서의 신뢰를 어떻게 구축하고 유지해 왔는지 구체적인 예를 들어 설명해주세요.",
            "어려운 상황에서도 긍정적인 태도를 유지하기 위해 어떤 노력을 했는지 말씀해 주십시오.",
            "다른 사람의 성공을 지원하고 독려하기 위해 취한 구체적인 조치에 대해 설명해주세요.",
            "개인적인 감정을 관리하면서 전문적인 태도를 유지한 경험에 대해 말씀해 주십시오.",
            "팀 또는 조직 내에서 신뢰를 쌓는 데 기여한 특별한 경험이 있나요? 그 과정을 설명해주세요.",
            "동료의 문제를 돕기 위해 개인적으로 희생한 경험에 대해 말씀해 주십시오.",
            "자신의 결점을 인정하고 개선하기 위해 취한 조치에 대해 설명해주세요."
        ]
        for q in questions:
            PersonalityTraitQuestion.objects.create(question=q)
        self.stdout.write(self.style.SUCCESS('Successfully populated problem solving questions.'))