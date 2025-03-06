from django.db import models

class AiModel(models.Model):
    model_name = models.CharField(max_length=100)
    model_version = models.CharField(max_length=50)
    model_path = models.CharField(max_length=255, blank=True, null=True)
    description = models.TextField(blank=True, null=True)

    class Meta:
        db_table = 'AiModel'

    def __str__(self):
        return f"{self.model_name} ({self.model_version})"


class Policy(models.Model):
    policy_name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)

    class Meta:
        db_table = 'ShiftPolicy'

    def __str__(self):
        return f"{self.policy_name}"


class ShiftPolicy(models.Model):
    policy = models.ForeignKey(
        Policy,
        on_delete=models.CASCADE,
        related_name='shift_details'
    )
    shift_index = models.PositiveIntegerField()
    start_time = models.TimeField()
    end_time = models.TimeField()

    class Meta:
        db_table = 'ShiftPolicyDetail'

    def __str__(self):
        return f"Policy: {self.policy.policy_name}, Shift #{self.shift_index}"
