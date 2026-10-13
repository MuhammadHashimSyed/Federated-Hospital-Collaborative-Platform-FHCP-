from django.db import models
from django.conf import settings

class TrainingRound(models.Model):
    STATUS_CHOICES = (
        ('PENDING', 'Pending'),
        ('IN_PROGRESS', 'In Progress'),
        ('AGGREGATING', 'Aggregating'),
        ('COMPLETED', 'Completed'),
    )
    round_number = models.IntegerField(unique=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    global_model_file = models.FileField(upload_to='models/global/', null=True, blank=True)
    global_mean_auroc = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Round {self.round_number} - {self.get_status_display()}"

class LocalWeightSubmission(models.Model):
    hospital = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='submissions')
    training_round = models.ForeignKey(TrainingRound, on_delete=models.CASCADE, related_name='submissions')
    local_model_file = models.FileField(upload_to='models/local/')
    local_mean_auroc = models.FloatField()
    metrics_json = models.JSONField(help_text="Stores per-class AUROC for the 14 pathologies", null=True, blank=True)
    submitted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('hospital', 'training_round')

    def __str__(self):
        return f"{self.hospital.username} - Round {self.training_round.round_number}"

class AuditLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    action = models.CharField(max_length=255)
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    def __str__(self):
        return f"[{self.timestamp}] {self.actor} - {self.action}"
