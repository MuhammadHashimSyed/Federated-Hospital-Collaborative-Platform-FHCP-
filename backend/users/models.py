from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    ROLE_CHOICES = (
        ('SUPER_ADMIN', 'Super Admin'),
        ('HOSPITAL_CLIENT', 'Hospital Client'),
        ('DOCTOR', 'Doctor'),
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='HOSPITAL_CLIENT')
    hospital_name = models.CharField(max_length=255, blank=True, null=True)
    linked_hospital = models.ForeignKey(
        'self', on_delete=models.CASCADE,
        null=True, blank=True,
        related_name='doctors',
        limit_choices_to={'role': 'HOSPITAL_CLIENT'}
    )

    def __str__(self):
        return f"{self.username} - {self.get_role_display()}"
