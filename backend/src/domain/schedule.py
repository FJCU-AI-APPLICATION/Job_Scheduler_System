from __future__ import annotations

from datetime import datetime, time
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, String, Text, Time, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base

if TYPE_CHECKING:
    from domain.employee import Employee


class Schedule(Base):
    __tablename__ = "schedule"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    start_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    end_date: Mapped[datetime] = mapped_column(DateTime)
    start_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    end_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)

    schedule_assignments: Mapped[list[ScheduleEmployee]] = relationship(
        back_populates="schedule", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Schedule(id={self.id}, name={self.name})>"


class ScheduleEmployee(Base):
    __tablename__ = "schedule_employee"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    schedule_id: Mapped[int] = mapped_column(
        ForeignKey("schedule.id", ondelete="CASCADE"), index=True
    )
    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employee.id", ondelete="CASCADE"), index=True
    )
    assigned_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    schedule: Mapped[Schedule] = relationship(back_populates="schedule_assignments")
    employee: Mapped[Employee] = relationship(back_populates="schedule_assignments")

    __table_args__ = (
        UniqueConstraint("schedule_id", "employee_id", name="uq_schedule_employee"),
    )

    def __repr__(self) -> str:
        return f"<ScheduleEmployee(schedule={self.schedule_id}, employee={self.employee_id})>"
