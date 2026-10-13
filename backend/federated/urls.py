from django.urls import path
from .views import (
    CurrentRoundView, UploadWeightsView,
    AnalyticsMetricsView, StartRoundView, LiveDiagnosticView,
    RegisteredHospitalsView, LocalHistoryView, ResetPlatformView,
    SystemLogsView, ClientLogView, RegisterHospitalView, DeleteHospitalView,
    ManualAggregateView, DownloadGlobalModelView,
    HospitalDoctorListView, HospitalDoctorDeleteView
)

urlpatterns = [
    path('rounds/current/', CurrentRoundView.as_view(), name='current-round'),
    path('rounds/start/', StartRoundView.as_view(), name='start-round'),
    path('rounds/reset/', ResetPlatformView.as_view(), name='reset-platform'),
    path('rounds/log/', ClientLogView.as_view(), name='client-log'),
    path('rounds/aggregate/', ManualAggregateView.as_view(), name='manual-aggregate'),
    path('weights/upload/', UploadWeightsView.as_view(), name='upload-weights'),
    path('weights/history/', LocalHistoryView.as_view(), name='local-history'),
    path('weights/download_global/', DownloadGlobalModelView.as_view(), name='download-global'),
    path('analytics/metrics/', AnalyticsMetricsView.as_view(), name='analytics-metrics'),
    path('analytics/hospitals/', RegisteredHospitalsView.as_view(), name='analytics-hospitals'),
    path('analytics/logs/', SystemLogsView.as_view(), name='analytics-logs'),
    path('hospitals/register/', RegisterHospitalView.as_view(), name='register-hospital'),
    path('hospitals/<int:hospital_id>/delete/', DeleteHospitalView.as_view(), name='delete-hospital'),
    path('doctors/', HospitalDoctorListView.as_view(), name='doctor-list'),
    path('doctors/<int:doctor_id>/delete/', HospitalDoctorDeleteView.as_view(), name='doctor-delete'),
    path('diagnostics/predict/', LiveDiagnosticView.as_view(), name='diagnostics-predict'),
]
