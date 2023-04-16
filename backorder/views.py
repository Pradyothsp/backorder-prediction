import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from backorder.pipeline import final_fun_1


def index(request):
    return render(request, 'index.html')


@csrf_exempt
def predict(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'error': 'File not uploaded.'})
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return JsonResponse({'error': f'Error reading CSV file: {str(e)}'})
        print(df.head)
        y_pred, y_actual = final_fun_1(df, return_actual=True)

        print(y_pred, y_actual)
        return JsonResponse({'y_pred': y_pred, 'y_actual': y_actual})
    return JsonResponse({'error': 'Request method is not a POST'})
