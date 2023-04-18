from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from backorder.views import index, PredictView

urlpatterns = [
                  path('', index, name='index'),
                  path('predict/', PredictView.as_view(), name='predict')
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
