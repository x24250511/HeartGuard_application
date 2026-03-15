from django.db import models
from django.contrib.auth.models import User


class ECGUpload(models.Model):
    STATUS_CHOICES = [
        ('UPLOADED', 'Uploaded'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    SEX_CHOICES = [(1, 'Male'), (0, 'Female')]
    CP_CHOICES = [
        (1, 'Typical Angina'), (2, 'Atypical Angina'),
        (3, 'Non-Anginal Pain'), (4, 'Asymptomatic'),
    ]
    RESTECG_CHOICES = [
        (0, 'Normal'), (1, 'ST-T Wave Abnormality'),
        (2, 'Left Ventricular Hypertrophy'),
    ]
    SLOPE_CHOICES = [(1, 'Upsloping'), (2, 'Flat'), (3, 'Downsloping')]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='ecg_uploads/')
    original_filename = models.CharField(max_length=255)
    patient_name = models.CharField(max_length=255, blank=True, default='')
    patient_age = models.PositiveIntegerField(null=True, blank=True)
    notes = models.TextField(blank=True, default='')
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default='UPLOADED')
    error_message = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # ── Clinical data (optional — enables tabular model) ──
    patient_sex = models.IntegerField(
        choices=SEX_CHOICES, null=True, blank=True)
    chest_pain_type = models.IntegerField(
        choices=CP_CHOICES, null=True, blank=True)
    resting_bp = models.IntegerField(null=True, blank=True)
    cholesterol = models.IntegerField(null=True, blank=True)
    fasting_bs = models.BooleanField(null=True, blank=True)
    resting_ecg = models.IntegerField(
        choices=RESTECG_CHOICES, null=True, blank=True)
    max_heart_rate = models.IntegerField(null=True, blank=True)
    exercise_angina = models.BooleanField(null=True, blank=True)
    oldpeak = models.FloatField(null=True, blank=True)
    st_slope = models.IntegerField(
        choices=SLOPE_CHOICES, null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"#{self.id} - {self.original_filename}"

    @property
    def is_image(self):
        ext = self.original_filename.rsplit(
            '.', 1)[-1].lower() if self.original_filename else ''
        return ext in ('jpg', 'jpeg', 'png', 'gif', 'webp')

    @property
    def status_color(self):
        return {'UPLOADED': 'secondary', 'PROCESSING': 'warning',
                'COMPLETED': 'success', 'FAILED': 'danger'}.get(self.status, 'secondary')

    @property
    def has_clinical_data(self):
        """True if enough clinical fields are filled to run the tabular model."""
        return all(v is not None for v in [self.resting_bp, self.cholesterol, self.max_heart_rate])

    def get_tabular_features(self):
        """Build feature dict for the tabular RF model."""
        return {
            'age': self.patient_age or 50,
            'sex': self.patient_sex if self.patient_sex is not None else 1,
            'cp': self.chest_pain_type or 1,
            'trestbps': self.resting_bp or 120,
            'chol': self.cholesterol or 200,
            'fbs': 1 if self.fasting_bs else 0,
            'restecg': self.resting_ecg or 0,
            'thalch': self.max_heart_rate or 150,
            'exang': 1 if self.exercise_angina else 0,
            'oldpeak': self.oldpeak or 0.0,
            'slope': self.st_slope or 1,
        }


class ECGResult(models.Model):
    upload = models.OneToOneField(
        ECGUpload, on_delete=models.CASCADE, related_name='result')
    diagnosis = models.CharField(max_length=150)
    confidence = models.FloatField()
    severity = models.CharField(max_length=20)
    heart_attack_risk = models.CharField(max_length=20)
    heart_attack_probability = models.FloatField(default=0.0)
    heart_rate = models.IntegerField(null=True, blank=True)
    pr_interval = models.FloatField(null=True, blank=True)
    qrs_duration = models.FloatField(null=True, blank=True)
    qt_interval = models.FloatField(null=True, blank=True)
    predictions_json = models.JSONField(default=dict)
    findings = models.TextField(blank=True, default='')
    recommendations = models.TextField(blank=True, default='')
    recommendation_steps = models.JSONField(default=list, blank=True)
    emergency_alert = models.BooleanField(default=False)
    processing_time = models.FloatField(default=0.0)
    tabular_used = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Result: {self.diagnosis}"

    @property
    def confidence_percent(self):
        return round(self.confidence * 100, 1)

    @property
    def heart_attack_percent(self):
        return round(self.heart_attack_probability * 100, 1)

    @property
    def severity_color(self):
        return {'normal': 'success', 'mild': 'info', 'moderate': 'warning',
                'critical': 'danger'}.get(self.severity, 'secondary')

    @property
    def risk_color(self):
        return {'low': 'success', 'moderate': 'warning', 'high': 'danger',
                'very_high': 'danger'}.get(self.heart_attack_risk, 'secondary')

    @property
    def heart_rate_status(self):
        if self.heart_rate is None:
            return 'unknown'
        if self.heart_rate < 60:
            return 'low'
        elif self.heart_rate > 100:
            return 'high'
        return 'normal'
