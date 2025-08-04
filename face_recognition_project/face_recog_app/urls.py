from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('video_feed/', views.video_feed),
    path('enroll/', views.enroll, name='enroll'),
    path('capture/', views.capture_page, name='capture_page'),
    path('start_capture/', views.start_capture, name='start_capture'),
    path('get_progress/', views.get_progress, name='get_progress'),
]
