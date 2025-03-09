from django.db import models
from django.core.validators import MinValueValidator
from django.db.models import Q

class Employee(models.Model):
    name = models.CharField(max_length=100, verbose_name="姓名")
    age = models.PositiveIntegerField(validators=[MinValueValidator(0)], verbose_name="年齡")
    phone = models.CharField(max_length=20, verbose_name="電話")

    IDENTITY_CHOICES = (
        ('FULL', '正職'),
        ('PART', '兼職'),
    )
    identity = models.CharField(
        max_length=4,
        choices=IDENTITY_CHOICES,
        default='FULL',
        verbose_name="身份別"
    )

    SALARY_TYPE_CHOICES = (
        ('MONTH', '月薪'),
        ('HOUR', '時薪'),
    )
    salary_type = models.CharField(
        max_length=5,
        choices=SALARY_TYPE_CHOICES,
        default='MONTH',
        verbose_name="薪資類型"
    )

    insert_date = models.DateField(auto_now_add=True, verbose_name="建立日期")
    update_date = models.DateField(auto_now=True, verbose_name="更新日期")

    def __str__(self):
        return f"{self.name} ({self.get_identity_display()})"

    class Meta:
        db_table = "Employee"  # Explicitly set the database table name
        verbose_name = "員工"
        verbose_name_plural = "員工"

class EmployeeUnavailability(models.Model):
    UNAVAILABILITY_TYPE_CHOICES = [
        ('DAY_OF_WEEK', 'Day of Week'),
        ('DATE_RANGE', 'Date Range'),
    ]
    
    employee = models.ForeignKey(
        Employee,
        on_delete=models.CASCADE,
        related_name='unavailabilities'
    )
    unavailability_type = models.CharField(
        max_length=20,
        choices=UNAVAILABILITY_TYPE_CHOICES
    )
    day_of_week = models.PositiveSmallIntegerField(null=True, blank=True)
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    reason = models.CharField(max_length=255, blank=True)
    
    class Meta:
        db_table = "EmployeeUnavailability"
        constraints = [
            models.CheckConstraint(
                check=(
                    Q(unavailability_type='DAY_OF_WEEK') &
                    Q(day_of_week__isnull=False) &
                    Q(start_date__isnull=True) &
                    Q(end_date__isnull=True)
                ) | (
                    Q(unavailability_type='DATE_RANGE') &
                    Q(day_of_week__isnull=True) &
                    Q(start_date__isnull=False) &
                    Q(end_date__isnull=False)
                ),
                name="ck_employeeunavailability_type"
            )
        ]
    
    def __str__(self):
        return f"{self.employee} - {self.unavailability_type}"