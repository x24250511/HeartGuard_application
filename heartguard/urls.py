from django.urls import path
from django.contrib.auth.views import LoginView, LogoutView
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', LoginView.as_view(template_name='heartguard/login.html'), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),

    path('uploads/', views.upload_list, name='upload_list'),
    path('uploads/new/', views.upload_create, name='upload_create'),
    path('clinical/new/', views.clinical_create, name='clinical_create'),
    path('uploads/<int:pk>/', views.upload_detail, name='upload_detail'),
    path('uploads/<int:pk>/edit/', views.upload_update, name='upload_update'),
    path('uploads/<int:pk>/delete/', views.upload_delete, name='upload_delete'),
]
