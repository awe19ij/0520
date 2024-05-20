# 의사소통 능력 질문 DB 한 번에 넣기

from django.core.management.base import BaseCommand
from QuestionList.models import CommunicationSkillQuestion

class Command(BaseCommand):
    help = 'Populates the database with problem solving questions'

    def handle(self, *args, **options):
        questions = [
            "다양한 배경을 가진 팀원들과 효과적으로 의사소통하기 위해 어떤 노력을 했는지 설명해주세요.",
            "복잡한 기술적 내용을 비전문가에게 설명한 경험에 대해 말씀해 주십시오.",
            "의견 불일치가 있을 때 어떻게 중재하고 합의에 도달했는지 구체적인 예를 들어 설명해주세요.",
            "대규모 회의나 발표에서 효과적인 메시지 전달을 위해 사용한 전략에 대해 설명해주세요.",
            "고객의 불만을 해결하기 위해 사용한 의사소통 기술에 대해 설명해주세요.",
            "팀 내에서 정보를 효율적으로 공유하기 위해 개발한 시스템이나 방법에 대해 설명해주세요.",
            "프로젝트 업데이트를 정기적으로 어떻게 커뮤니케이션하는지 그 절차에 대해 설명해주세요.",
            "갈등 상황에서 상대방의 입장을 이해하고 감정을 관리하기 위해 사용한 기술을 공유해주세요.",
            "직장 내에서 동료들과의 관계를 개선하기 위해 취한 조치에 대해 설명해주세요.",
            "팀의 동기를 유지하고 진행 상황을 효과적으로 보고하기 위해 어떤 방법을 사용했는지 말씀해 주십시오.",
        ]
        for q in questions:
            CommunicationSkillQuestion.objects.create(question=q)
        self.stdout.write(self.style.SUCCESS('Successfully populated problem solving questions.'))