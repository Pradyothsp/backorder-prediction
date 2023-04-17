from backorder.views import index, PredictView1, PredictView2
from django.urls import path

urlpatterns = [
    path('', index, name='index'),
    path('predict/', PredictView2.as_view(), name='predict')
]
