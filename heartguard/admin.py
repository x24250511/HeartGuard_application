from django.contrib import admin
from .models import ECGUpload, ECGResult

class ResultInline(admin.StackedInline):
    model = ECGResult
    extra = 0

@admin.register(ECGUpload)
class ECGUploadAdmin(admin.ModelAdmin):
    list_display = ('id', 'original_filename', 'patient_name', 'user', 'status', 'created_at')
    list_filter = ('status',)
    inlines = [ResultInline]

@admin.register(ECGResult)
class ECGResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'upload', 'diagnosis', 'severity', 'heart_attack_risk', 'emergency_alert')
    list_filter = ('severity', 'heart_attack_risk', 'emergency_alert')