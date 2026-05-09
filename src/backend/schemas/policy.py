from datetime import time
from typing import Optional

from pydantic import BaseModel, Field


class PolicyCreate(BaseModel):
    policy_name: str = Field(max_length=100)
    description: Optional[str] = None


class PolicyUpdate(BaseModel):
    policy_name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None


class PolicyResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    policy_name: str
    description: Optional[str]


class ShiftPolicyCreate(BaseModel):
    policy_id: int
    start_time: time
    end_time: time


class ShiftPolicyUpdate(BaseModel):
    start_time: Optional[time] = None
    end_time: Optional[time] = None


class ShiftPolicyResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    policy_id: int
    start_time: time
    end_time: time


class AiModelCreate(BaseModel):
    model_name: str = Field(max_length=100)
    model_version: str = Field(max_length=50)
    model_path: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None


class AiModelUpdate(BaseModel):
    model_name: Optional[str] = Field(None, max_length=100)
    model_version: Optional[str] = Field(None, max_length=50)
    model_path: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None


class AiModelResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    model_name: str
    model_version: str
    model_path: Optional[str]
    description: Optional[str]
