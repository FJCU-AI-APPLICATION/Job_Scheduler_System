from django.urls import path
from policy.views import (
    AiModelListCreateView,
    AiModelDetailView,
    ShiftPolicyListCreateView,
    ShiftPolicyDetailView,
    ShiftPolicyDetailListCreateView,
    ShiftPolicyDetailDetailView
)

urlpatterns = [
    # AI Model
    path('aimodels/', AiModelListCreateView.as_view(), name='aimodel-list-create'),
    path('aimodels/<int:pk>/', AiModelDetailView.as_view(), name='aimodel-detail'),

    # Shift Policy
    path('', ShiftPolicyListCreateView.as_view(), name='shiftpolicy-list-create'),
    path('<int:pk>/', ShiftPolicyDetailView.as_view(), name='shiftpolicy-detail'),

    # Shift Policy Detail
    path('shiftpolicydetails/', ShiftPolicyDetailListCreateView.as_view(), name='shiftpolicydetail-list-create'),
    path('shiftpolicydetails/<int:pk>/', ShiftPolicyDetailDetailView.as_view(), name='shiftpolicydetail-detail'),
]
