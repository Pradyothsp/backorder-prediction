from backorder.views import index, PredictView
from django.urls import path

urlpatterns = [
    path('', index, name='index'),
    path('predict/', PredictView.as_view(), name='predict')
]
