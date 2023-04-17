import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from backorder.pipeline import BackorderPredictor

backorder_predictor = BackorderPredictor()


def index(request):
    return render(request, 'index.html')


@method_decorator(csrf_exempt, name='dispatch')
class PredictView(View):
    http_method_names = ['post']

    def post(self, request, *args, **kwargs) -> JsonResponse:
        df = pd.read_csv('Test_Dataset.csv')
        print(df.head())

        y_pred, y_actual = backorder_predictor.predict(df)
        return JsonResponse({'y_pred': y_pred, 'y_actual': y_actual})
