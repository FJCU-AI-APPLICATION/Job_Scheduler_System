from __future__ import annotations

from datetime import time
from typing import Optional

from sqlalchemy import ForeignKey, String, Text, Time
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


class Policy(Base):
    __tablename__ = "policy"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    policy_name: Mapped[str] = mapped_column(String(100))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    shift_details: Mapped[list[ShiftPolicy]] = relationship(
        back_populates="policy", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Policy(id={self.id}, name={self.policy_name})>"


class ShiftPolicy(Base):
    __tablename__ = "shift_policy_detail"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    policy_id: Mapped[int] = mapped_column(
        ForeignKey("policy.id", ondelete="CASCADE"), index=True
    )
    start_time: Mapped[time] = mapped_column(Time)
    end_time: Mapped[time] = mapped_column(Time)

    policy: Mapped[Policy] = relationship(back_populates="shift_details")

    def __repr__(self) -> str:
        return f"<ShiftPolicy(id={self.id}, {self.start_time}-{self.end_time})>"


class AiModel(Base):
    __tablename__ = "ai_model"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(50))
    model_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<AiModel(id={self.id}, name={self.model_name})>"
