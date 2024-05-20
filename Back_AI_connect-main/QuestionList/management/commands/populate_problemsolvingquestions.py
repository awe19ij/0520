# 문제해결력 질문 DB 한 번에 넣기

from django.core.management.base import BaseCommand
from QuestionList.models import ProblemSolvingQuestion

class Command(BaseCommand):
    help = 'Populates the database with problem solving questions'

    def handle(self, *args, **options):
        questions = [
            "팀 프로젝트를 수행하면서 발생한 가장 복잡한 문제는 무엇이었고, 그 문제를 어떻게 해결했는지 설명해주세요.",
            "자신이 개발한 창의적인 해결책 중 하나를 예로 들고, 그 과정에서 어떤 도구나 기술을 사용했는지 설명해주세요.",
            "작업 중 우선 순위가 충돌할 때 어떻게 접근하고 결정하는지 설명해주세요.",
            "요구사항을 충족시키기 위해 기존 제품이나 서비스를 어떻게 개선하였는지 사례를 들어 설명해주세요.",
            "복잡한 데이터를 분석하여 중요한 비즈니스 결정을 내린 경험에 대해 설명해주세요.",
            "기술적 문제를 해결하기 위해 다른 부서와 협력한 경험에 대해 말씀해 주십시오.",
            "프로젝트 기한을 맞추기 위해 어떻게 시간 관리와 자원을 조정했는지 설명해주세요.",
            "잘못된 정보에 기초한 결정을 수정하기 위해 어떤 절차를 따랐는지 설명해주세요.",
            "팀 내 갈등을 해결하여 프로젝트의 성공을 이끌어낸 경험에 대해 말씀해 주십시오.",
            "신기술을 도입하여 업무 효율성을 향상시킨 경험에 대해 구체적으로 설명해주세요.",
        ]
        for q in questions:
            ProblemSolvingQuestion.objects.create(question=q)
        self.stdout.write(self.style.SUCCESS('Successfully populated problem solving questions.'))