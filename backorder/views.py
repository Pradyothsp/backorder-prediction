import pandas as pd
from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from backorder.pipeline import BackorderPredictor

backorder_predictor = BackorderPredictor()
backorder_predictor._load_model_files()


def index(request):
    return render(request, 'index.html')


@method_decorator(csrf_exempt, name='dispatch')
class PredictView(View):
    http_method_names = ['post', 'get']

    def get(self, request, *args, **kwargs):
        return render(request, 'predict.html')

    def post(self, request, *args, **kwargs):
        file = request.FILES['file']
        df = pd.read_csv(file)

        y_pred, y_actual = backorder_predictor.predict(df)

        context = {
            'y_pred': y_pred,
            'y_actual': y_actual,
        }
        return render(request, 'result.html', context)
