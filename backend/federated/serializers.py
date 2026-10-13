from rest_framework import serializers
from .models import TrainingRound, LocalWeightSubmission, AuditLog
from users.models import User

class TrainingRoundSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingRound
        fields = '__all__'

class LocalWeightSubmissionSerializer(serializers.ModelSerializer):
    class Meta:
        model = LocalWeightSubmission
        fields = '__all__'
        read_only_fields = ['hospital', 'training_round', 'submitted_at']

class AuditLogSerializer(serializers.ModelSerializer):
    actor_username = serializers.CharField(source='actor.username', read_only=True)
    class Meta:
        model = AuditLog
        fields = '__all__'
