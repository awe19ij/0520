# 성장 가능성 및 개인 발전 의지 질문 DB 한 번에 넣기

from django.core.management.base import BaseCommand
from QuestionList.models import GrowthPotentialQuestion

class Command(BaseCommand):
    help = 'Populates the database with problem solving questions'

    def handle(self, *args, **options):
        questions = [
            "자신의 업무 능력을 향상시키기 위해 최근에 참여한 교육이나 워크샵은 무엇인가요?",
            "개인적이거나 전문적인 발전을 위해 설정한 최근 목표와 그 목표 달성 과정을 설명해주세요.",
            "업무상의 도전을 해결하기 위해 새로운 기술이나 지식을 어떻게 습득했는지 예를 들어 설명해주세요.",
            "자신의 커리어 경로에서 중요한 전환점이 된 사건에 대해 말씀해 주십시오.",
            "장기적인 커리어 목표를 달성하기 위해 어떤 계획을 세우고 있는지 설명해주세요.",
            "실패 경험에서 배운 교훈과 그 이후 어떻게 대응했는지 구체적으로 설명해주세요.",
            "리더십 역량을 개발하기 위해 어떤 노력을 해왔는지 그 과정을 공유해주세요.",
            "다양한 프로젝트나 업무를 통해 얻은 가장 중요한 교훈은 무엇이었나요?",
            "자기계발을 위해 사용하는 도구나 자원에 대해 설명해주세요.",
            "업무 성과를 개선하기 위해 도입한 새로운 접근 방식이나 기술에 대해 말씀해 주십시오."
        ]
        for q in questions:
            GrowthPotentialQuestion.objects.create(question=q)
        self.stdout.write(self.style.SUCCESS('Successfully populated problem solving questions.'))