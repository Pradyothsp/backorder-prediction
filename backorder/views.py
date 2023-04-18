import pandas as pd
from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from backorder.pipeline import BackorderPredictor
from backorder.utils import get_form_data

backorder_predictor = BackorderPredictor()
backorder_predictor._load_model_files()


def index(request):
    return render(request, 'backorder.html')


@method_decorator(csrf_exempt, name='dispatch')
class PredictView(View):
    http_method_names = ['post', 'get']

    def get(self, request, *args, **kwargs):
        return render(request, 'predict.html')

    def post(self, request, *args, **kwargs):
        file = self.request.FILES.get('file')
        # print(request.POST)
        if file:
            file = request.FILES['file']
            df = pd.read_csv(file)
            print(df.to_dict('dict'))
            y_pred, y_actual = backorder_predictor.predict(df)

            context = {
                'y_pred': y_pred,
                'y_actual': y_actual,
            }
            return render(request, 'result.html', context)

        elif all(key in request.POST for key in
                 ['sku', 'national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month', 'forecast_6_month',
                  'forecast_9_month', 'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank',
                  'pieces_past_due', 'perf_6_month_avg', 'perf_12_month_avg', 'local_bo_qty']):
            form_data = get_form_data(request)
            df = pd.DataFrame(form_data)
            print(df.to_dict('dict'))


            y_pred, y_actual = backorder_predictor.predict(df)

            context = {
                'y_pred': y_pred,
                'y_actual': y_actual,
            }
            return render(request, 'result.html', context)

        else:
            return render(request, 'predict.html')
