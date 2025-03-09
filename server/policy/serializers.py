from rest_framework import serializers
from policy.models import AiModel, Policy, ShiftPolicy

class AiModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = AiModel
        fields = '__all__'


class PolicySerializer(serializers.ModelSerializer):
    class Meta:
        model = Policy
        fields = '__all__'


class ShiftPolicySerializer(serializers.ModelSerializer):
    # Nested or primary key representation for AI model
    
    # Optionally embed the shift details as nested
    shift_details = PolicySerializer(many=True, read_only=True)

    class Meta:
        model = ShiftPolicy
        fields = '__all__'