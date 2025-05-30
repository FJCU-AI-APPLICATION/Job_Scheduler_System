from rest_framework import serializers
from policy.models import AiModel, Policy, ShiftPolicy

class AiModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = AiModel
        fields = '__all__'


class PolicySerializer(serializers.ModelSerializer):
    start_time = serializers.SerializerMethodField()
    end_time = serializers.SerializerMethodField()
    
    class Meta:
        model = Policy
        # fields = '__all__'
        fields = ['id', 'policy_name', 'description', 'start_time', 'end_time']
        
    def get_start_time(self, obj):
        shift = obj.shift_details.first()
        return shift.start_time if shift else None

    def get_end_time(self, obj):
        shift = obj.shift_details.first()
        return shift.end_time if shift else None


class ShiftPolicySerializer(serializers.ModelSerializer):
    # Nested or primary key representation for AI model
    
    # Optionally embed the shift details as nested
    shift_details = PolicySerializer(many=True, read_only=True)

    class Meta:
        model = ShiftPolicy
        fields = '__all__'