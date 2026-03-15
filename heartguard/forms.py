from django import forms
from .models import ECGUpload

ALLOWED_EXT = ['jpg', 'jpeg', 'png', 'pdf']


class ECGUploadForm(forms.ModelForm):
    class Meta:
        model = ECGUpload
        fields = [
            'file', 'patient_name', 'patient_age', 'patient_sex',
            'chest_pain_type', 'resting_bp', 'cholesterol', 'fasting_bs',
            'resting_ecg', 'max_heart_rate', 'exercise_angina', 'oldpeak',
            'st_slope', 'notes',
        ]
        widgets = {
            'fasting_bs': forms.Select(choices=[('', '---'), (True, 'Yes (>120 mg/dL)'), (False, 'No')]),
            'exercise_angina': forms.Select(choices=[('', '---'), (True, 'Yes'), (False, 'No')]),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make file optional — user can submit clinical data only
        self.fields['file'].required = False

    def clean_file(self):
        f = self.cleaned_data.get('file')
        if f:
            ext = f.name.rsplit('.', 1)[-1].lower()
            if ext not in ALLOWED_EXT:
                raise forms.ValidationError(
                    f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXT)}'
                )
            if f.size > 10 * 1024 * 1024:
                raise forms.ValidationError('File too large. Max 10 MB.')
        return f

    def clean(self):
        cleaned = super().clean()
        has_file = bool(cleaned.get('file'))
        has_clinical = all([
            cleaned.get('resting_bp'),
            cleaned.get('cholesterol'),
            cleaned.get('max_heart_rate'),
        ])
        if not has_file and not has_clinical:
            raise forms.ValidationError(
                'Please upload an ECG file, provide clinical data (BP, cholesterol, max HR), or both.'
            )
        return cleaned


class ECGUpdateForm(forms.ModelForm):
    class Meta:
        model = ECGUpload
        fields = ['patient_name', 'patient_age', 'notes']
