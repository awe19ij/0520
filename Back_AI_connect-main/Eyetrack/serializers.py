from rest_framework import serializers

class GazeStatusSerializer(serializers.Serializer):
    status = serializers.CharField(max_length=200)
