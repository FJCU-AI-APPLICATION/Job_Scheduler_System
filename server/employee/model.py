from django.db import models
from django.core.validators import MinValueValidator

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
