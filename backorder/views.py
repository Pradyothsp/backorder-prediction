import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from backorder.pipeline import BackorderPredictor, final_fun_1

backorder_predictor = BackorderPredictor()


def index(request):
    return render(request, 'index.html')


@method_decorator(csrf_exempt, name='dispatch')
class PredictView1(View):
    http_method_names = ['post']

    def post(self, request, *args, **kwargs) -> JsonResponse:
        df = pd.read_csv('Test_Dataset.csv')
        # file = request.FILES.get('file')
        #
        # print(file)
        # if not file:
        #     return JsonResponse({'error': 'File not uploaded.'})
        # try:
        #     df = pd.read_csv(file)
        # except Exception as e:
        #     return JsonResponse({'error': f'Error reading CSV file: {str(e)}'})
        # print("Hello World")
        # print(df.head)
        y_pred, y_actual = final_fun_1(df, return_actual=True)
        # y_pred, y_actual = y_pred.tolist(), y_actual.tolist()

        print(y_pred, y_actual)
        return JsonResponse({'y_pred': y_pred, 'y_actual': y_actual})


@method_decorator(csrf_exempt, name='dispatch')
class PredictView2(View):
    http_method_names = ['post']

    def post(self, request, *args, **kwargs) -> JsonResponse:
        df = pd.read_csv('Test_Dataset.csv')
        print(df.head())

        y_pred, y_actual = backorder_predictor.predict(df)
        return JsonResponse({'y_pred': y_pred, 'y_actual': y_actual})
