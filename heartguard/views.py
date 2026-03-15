from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib import messages
from django.core.paginator import Paginator

from .models import ECGUpload
from .forms import ECGUploadForm, ECGUpdateForm
from .services import analyze_ecg


def home(request):
    return render(request, 'heartguard/home.html')


def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('upload_list')
    else:
        form = UserCreationForm()
    return render(request, 'heartguard/register.html', {'form': form})


@login_required
def upload_create(request):
    if request.method == 'POST':
        form = ECGUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.save(commit=False)
            upload.user = request.user

            # Handle file presence
            if 'file' in request.FILES:
                upload.original_filename = request.FILES['file'].name
            else:
                upload.original_filename = 'Clinical Data Only'

            upload.status = 'PROCESSING'
            upload.save()
            try:
                analyze_ecg(upload)
                upload.status = 'COMPLETED'
                upload.save()
                messages.success(request, 'Analysis completed successfully!')
            except Exception as e:
                upload.status = 'FAILED'
                upload.error_message = str(e)
                upload.save()
                messages.error(request, f'Analysis failed: {e}')
            return redirect('upload_detail', pk=upload.pk)
    else:
        form = ECGUploadForm()
    return render(request, 'heartguard/upload_create.html', {'form': form})


@login_required
def clinical_create(request):
    if request.method == 'POST':
        form = ECGUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.save(commit=False)
            upload.user = request.user
            if 'file' in request.FILES:
                upload.original_filename = request.FILES['file'].name
            else:
                upload.original_filename = 'Clinical Data Only'
            upload.status = 'PROCESSING'
            upload.save()
            try:
                analyze_ecg(upload)
                upload.status = 'COMPLETED'
                upload.save()
                messages.success(request, 'Clinical assessment completed!')
            except Exception as e:
                upload.status = 'FAILED'
                upload.error_message = str(e)
                upload.save()
                messages.error(request, f'Assessment failed: {e}')
            return redirect('upload_detail', pk=upload.pk)
    else:
        form = ECGUploadForm()
    return render(request, 'heartguard/clinical_create.html', {'form': form})


@login_required
def upload_list(request):
    qs = ECGUpload.objects.filter(user=request.user).select_related('result')
    paginator = Paginator(qs, 10)
    page = paginator.get_page(request.GET.get('page'))
    return render(request, 'heartguard/upload_list.html', {'page': page})


@login_required
def upload_detail(request, pk):
    upload = get_object_or_404(ECGUpload, pk=pk, user=request.user)
    result = getattr(upload, 'result', None)
    return render(request, 'heartguard/upload_detail.html', {'upload': upload, 'result': result})


@login_required
def upload_update(request, pk):
    upload = get_object_or_404(ECGUpload, pk=pk, user=request.user)
    if request.method == 'POST':
        form = ECGUpdateForm(request.POST, instance=upload)
        if form.is_valid():
            form.save()
            messages.success(request, 'Updated.')
            return redirect('upload_detail', pk=pk)
    else:
        form = ECGUpdateForm(instance=upload)
    return render(request, 'heartguard/upload_update.html', {'form': form, 'upload': upload})


@login_required
def upload_delete(request, pk):
    upload = get_object_or_404(ECGUpload, pk=pk, user=request.user)
    if request.method == 'POST':
        upload.file.delete(save=False)
        upload.delete()
        messages.success(request, 'Deleted.')
        return redirect('upload_list')
    return render(request, 'heartguard/upload_delete.html', {'upload': upload})
