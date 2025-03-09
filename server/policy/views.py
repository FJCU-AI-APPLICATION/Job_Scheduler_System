# Create your views here.
from rest_framework import generics
from policy.models import AiModel, Policy, ShiftPolicy
from rest_framework import generics

from policy.serializers import (
    AiModelSerializer, 
    ShiftPolicySerializer, 
    PolicySerializer
)

class AiModelListCreateView(generics.ListCreateAPIView):
    queryset = AiModel.objects.all()
    serializer_class = AiModelSerializer

class AiModelDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = AiModel.objects.all()
    serializer_class = AiModelSerializer

class PolicyListCreateView(generics.ListCreateAPIView):
    queryset = Policy.objects.all()
    serializer_class = PolicySerializer

class PolicyDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Policy.objects.all()
    serializer_class = PolicySerializer


class ShiftPolicyListCreateView(generics.ListCreateAPIView):
    """
    GET: List all shift details for a given policy.
         Requires the query parameter 'policy_id'.
    POST: Create a new shift detail.
    """
    serializer_class = ShiftPolicySerializer

    def get_queryset(self):
        policy_id = self.request.query_params.get('policy_id')
        if not policy_id:
            # If you want to force policy_id, you could also raise an error:
            # from rest_framework.exceptions import ParseError
            # raise ParseError(detail="The 'policy_id' query parameter is required.")
            # For now, we return an empty queryset:
            return ShiftPolicy.objects.none()
        return ShiftPolicy.objects.filter(policy_id=policy_id).order_by("start_time")

class ShiftPolicyDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    GET: Retrieve a shift detail.
    PUT: Update a shift detail (e.g. change start_time, end_time).
    DELETE: Remove a shift detail.
    """
    queryset = ShiftPolicy.objects.all()
    serializer_class = ShiftPolicySerializer

