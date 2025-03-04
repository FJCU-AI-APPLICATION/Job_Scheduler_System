from django.shortcuts import render

# Create your views here.
from rest_framework import generics, viewsets
from policy.models import AiModel, ShiftPolicy, ShiftPolicyDetail
from policy.serializers import (
    AiModelSerializer, 
    ShiftPolicySerializer, 
    ShiftPolicyDetailSerializer
)

class AiModelListCreateView(generics.ListCreateAPIView):
    queryset = AiModel.objects.all()
    serializer_class = AiModelSerializer

class AiModelDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = AiModel.objects.all()
    serializer_class = AiModelSerializer

class ShiftPolicyListCreateView(generics.ListCreateAPIView):
    queryset = ShiftPolicy.objects.select_related('ai_model').all()
    serializer_class = ShiftPolicySerializer

class ShiftPolicyDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = ShiftPolicy.objects.select_related('ai_model').all()
    serializer_class = ShiftPolicySerializer

class ShiftPolicyDetailListCreateView(generics.ListCreateAPIView):
    queryset = ShiftPolicyDetail.objects.all()
    serializer_class = ShiftPolicyDetailSerializer

class ShiftPolicyDetailDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = ShiftPolicyDetail.objects.all()
    serializer_class = ShiftPolicyDetailSerializer
