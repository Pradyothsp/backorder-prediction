from backorder.views import index, predict
from django.urls import path

urlpatterns = [
    path('', index, name='index'),
    path('predict/', predict, name='predict')
]
