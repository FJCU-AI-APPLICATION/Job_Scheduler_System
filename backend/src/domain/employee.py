from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Optional

from sqlalchemy import CheckConstraint, Date, ForeignKey, Integer, SmallInteger, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base

if TYPE_CHECKING:
    from domain.schedule import ScheduleEmployee


class Employee(Base):
    __tablename__ = "employee"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100))
    age: Mapped[int] = mapped_column(Integer)
    phone: Mapped[str] = mapped_column(String(20))
    identity: Mapped[str] = mapped_column(String(4), default="FULL")
    salary_type: Mapped[str] = mapped_column(String(5), default="MONTH")
    insert_date: Mapped[date] = mapped_column(Date, server_default=func.current_date())
    update_date: Mapped[date] = mapped_column(
        Date, server_default=func.current_date(), onupdate=func.current_date()
    )

    unavailabilities: Mapped[list[EmployeeUnavailability]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )
    schedule_assignments: Mapped[list[ScheduleEmployee]] = relationship(
        back_populates="employee"
    )

    def __repr__(self) -> str:
        return f"<Employee(id={self.id}, name={self.name})>"


class EmployeeUnavailability(Base):
    __tablename__ = "employee_unavailability"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employee.id", ondelete="CASCADE"), index=True
    )
    unavailability_type: Mapped[str] = mapped_column(String(20))
    day_of_week: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    start_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    end_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    reason: Mapped[str] = mapped_column(String(255), default="")

    employee: Mapped[Employee] = relationship(back_populates="unavailabilities")

    __table_args__ = (
        CheckConstraint(
            """
            (unavailability_type = 'DAY_OF_WEEK'
             AND day_of_week IS NOT NULL
             AND start_date IS NULL
             AND end_date IS NULL)
            OR
            (unavailability_type = 'DATE_RANGE'
             AND day_of_week IS NULL
             AND start_date IS NOT NULL
             AND end_date IS NOT NULL)
            """,
            name="ck_unavailability_type_fields",
        ),
    )

    def __repr__(self) -> str:
        return f"<EmployeeUnavailability(id={self.id}, type={self.unavailability_type})>"
