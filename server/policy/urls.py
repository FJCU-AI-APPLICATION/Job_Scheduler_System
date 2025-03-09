from django.urls import path
from policy.views import (
    AiModelListCreateView,
    AiModelDetailView,
    PolicyListCreateView,
    PolicyDetailView,
    ShiftPolicyDetailView,
    ShiftPolicyListCreateView,
)

urlpatterns = [
    # AI Model
    path('aimodels/', AiModelListCreateView.as_view(), name='aimodel-list-create'),
    path('aimodels/<int:pk>/', AiModelDetailView.as_view(), name='aimodel-detail'),

    # Policy
    path('', PolicyListCreateView.as_view(), name='policy-list-create'),
    path('<int:pk>/', PolicyDetailView.as_view(), name='policy-detail'),

    # Shift Policy
    path('shiftpolicy/', ShiftPolicyListCreateView.as_view(), name='shiftpolicy-list-create'),
    path('shiftpolicy/<int:pk>/', ShiftPolicyDetailView.as_view(), name='shiftpolicy-detail'),
]
