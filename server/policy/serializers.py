from rest_framework import serializers
from policy.models import AiModel, ShiftPolicy, ShiftPolicyDetail

class AiModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = AiModel
        fields = '__all__'


class ShiftPolicyDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShiftPolicyDetail
        fields = '__all__'


class ShiftPolicySerializer(serializers.ModelSerializer):
    # Nested or primary key representation for AI model
    ai_model = serializers.PrimaryKeyRelatedField(queryset=AiModel.objects.all())
    
    # Optionally embed the shift details as nested
    shift_details = ShiftPolicyDetailSerializer(many=True, read_only=True)

    class Meta:
        model = ShiftPolicy
        fields = ['id', 'policy_name', 'description', 'ai_model', 'shift_details']