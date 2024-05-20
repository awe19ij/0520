from django.contrib import admin
from .models import QuestionLists, ProblemSolvingQuestion, CommunicationSkillQuestion, GrowthPotentialQuestion, PersonalityTraitQuestion

class QuestionListsAdmin(admin.ModelAdmin):
    # 관리자 페이지에서 표시할 필드 설정
    list_display = ('user', 'created_at', 'question_1', 'question_2', 'question_3', 'question_4', 'question_5', 
                    'question_6', 'question_7', 'question_8', 'question_9', 'question_10')

    # 필터 옵션 추가
    list_filter = ('user', 'created_at')

    # 검색 기능 추가
    search_fields = ('question_1', 'question_2', 'question_3', 'question_4', 'question_5',
                     'question_6', 'question_7', 'question_8', 'question_9', 'question_10')

    # 레코드 편집 시 표시할 필드
    fields = ('user', 'created_at', 'question_1', 'question_2', 'question_3', 'question_4', 'question_5', 
              'question_6', 'question_7', 'question_8', 'question_9', 'question_10')

    # created_at 필드가 자동으로 설정되도록 읽기 전용으로 설정
    readonly_fields = ('created_at',)

# QuestionLists 모델을 관리자 사이트에 등록
admin.site.register(QuestionLists, QuestionListsAdmin)

class ProblemSolvingQuestionAdmin(admin.ModelAdmin):
    list_display = ('id', 'question')  # 관리자 리스트에 표시될 필드
    search_fields = ('question',)  # 검색 기능에 사용될 필드
    list_per_page = 20  # 페이지 당 표시할 객체 수

# Django 관리자 사이트에 ProblemSolvingQuestion 모델을 등록
admin.site.register(ProblemSolvingQuestion, ProblemSolvingQuestionAdmin)


class CommunicationSkillQuestionAdmin(admin.ModelAdmin):
    list_display = ('id', 'question')  # 관리자 리스트에 표시될 필드
    search_fields = ('question',)  # 검색 기능에 사용될 필드
    list_per_page = 20  # 페이지 당 표시할 객체 수

# Django 관리자 사이트에 CommunicationSkillQuestion 모델을 등록
admin.site.register(CommunicationSkillQuestion, CommunicationSkillQuestionAdmin)


class GrowthPotentialQuestionAdmin(admin.ModelAdmin):
    list_display = ('id', 'question')  # 관리자 리스트에 표시될 필드
    search_fields = ('question',)  # 검색 기능에 사용될 필드
    list_per_page = 20  # 페이지 당 표시할 객체 수

# Django 관리자 사이트에 CommunicationSkillQuestion 모델을 등록
admin.site.register( GrowthPotentialQuestion, GrowthPotentialQuestionAdmin)


class PersonalityTraitQuestionAdmin(admin.ModelAdmin):
    list_display = ('id', 'question')  # 관리자 리스트에 표시될 필드
    search_fields = ('question',)  # 검색 기능에 사용될 필드
    list_per_page = 20  # 페이지 당 표시할 객체 수

# Django 관리자 사이트에 CommunicationSkillQuestion 모델을 등록
admin.site.register(PersonalityTraitQuestion, PersonalityTraitQuestionAdmin)

