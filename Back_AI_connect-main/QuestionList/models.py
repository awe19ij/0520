from django.db import models
from django.conf import settings


class QuestionLists(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    question_1 = models.TextField(blank=True, null=True)
    question_2 = models.TextField(blank=True, null=True)
    question_3 = models.TextField(blank=True, null=True)
    question_4 = models.TextField(blank=True, null=True)
    question_5 = models.TextField(blank=True, null=True)
    question_6 = models.TextField(blank=True, null=True)
    question_7 = models.TextField(blank=True, null=True)
    question_8 = models.TextField(blank=True, null=True)
    question_9 = models.TextField(blank=True, null=True)
    question_10 = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

class ProblemSolvingQuestion(models.Model):
    question = models.TextField()

class CommunicationSkillQuestion(models.Model):
    question = models.TextField()

class GrowthPotentialQuestion(models.Model):
    question = models.TextField()

class PersonalityTraitQuestion(models.Model):
    question = models.TextField()