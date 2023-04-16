from itertools import combinations

import joblib
import numpy as np
import pandas as pd

num_col_list = ['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month', 'forecast_6_month',
                'forecast_9_month', 'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month',
                'min_bank', 'pieces_past_due', 'perf_6_month_avg', 'perf_12_month_avg', 'local_bo_qty']


class BackorderPredictor:
    def __init__(self):
        self.num_col_list = ['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month', 'forecast_6_month',
                             'forecast_9_month', 'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month',
                             'min_bank', 'pieces_past_due', 'perf_6_month_avg', 'perf_12_month_avg', 'local_bo_qty']
        self.best_feat = None
        self.sc = None
        self.model = None

    def _load_model_files(self):
        # Fetching the best features
        self.best_feat = joblib.load('models/test_best_feat.pkl').tolist()

        # Fetching the trained standardization object instance
        self.sc = joblib.load('models/sc.pkl')
        self.model = joblib.load('models/backorder_best_model.pkl')

    def _encode_bool_columns(self, df):
        dict_map_bool = {'Yes': 1.0, 'No': 0.0}

        df['deck_risk'] = df['deck_risk'].map(dict_map_bool)
        df['potential_issue'] = df['potential_issue'].map(dict_map_bool)
        df['oe_constraint'] = df['oe_constraint'].map(dict_map_bool)
        df['ppap_risk'] = df['ppap_risk'].map(dict_map_bool)
        df['stop_auto_buy'] = df['stop_auto_buy'].map(dict_map_bool)
        df['rev_stop'] = df['rev_stop'].map(dict_map_bool)
        df['went_on_backorder'] = df['went_on_backorder'].map(dict_map_bool)

        return df

    def _replace_perf_cols(self, df):
        df.perf_6_month_avg.replace({-99.0: np.nan}, inplace=True)
        df.perf_12_month_avg.replace({-99.0: np.nan}, inplace=True)
        return df

    def _drop_sku_col(self, df):
        return df.drop(columns=['sku'])

    def _handle_missing_values(self, df):
        df.lead_time.replace(to_replace=np.nan, value=8, inplace=True)
        df.perf_6_month_avg.replace(to_replace=-99, value=.85, inplace=True)
        df.perf_12_month_avg.replace(to_replace=-99, value=.83, inplace=True)
        return df

    def _perform_standardization(self, df):
        # TODO: .values is not added
        df_test_num_sc = self.sc.transform(df[self.num_col_list])
        df_test_num_sc = pd.DataFrame(df_test_num_sc, columns=self.num_col_list)
        return df_test_num_sc

    def _perform_feature_selection(self, df):
        df_test_num_sc = df[self.best_feat]
        return df_test_num_sc

    def _add_features(self, df):
        def add(df, num_cols):
            for i, j in combinations(num_cols, 2):
                df[f'{i}_{j}_add'] = df[i] + df[j]
            return df

        return add(df, self.num_col_list)

    def _multiply_features(self, df):
        def mult(df, num_cols):
            for i, j in combinations(num_cols, 2):
                df[f'{i}_{j}_mul'] = df[i] * df[j]
            return df

        return mult(df, self.num_col_list)

    def _inverse_features(self, df):
        def inv(df, num_cols):
            for i in num_cols:
                df[f'{i}_inv'] = 1 / (df[i] + 0.001)
            return df

        return inv(df, self.num_col_list)

    def _square_features(self, df):
        def square(df, num_cols):
            for i in num_cols:
                df[f'{i}_square'] = df[i] ** 2
            return df

        return square(df, self.num_col_list)

    def _square_root_features(self, df):
        def sqrt(df, num_cols):
            for i in num_cols:
                df[f'{i}_sqrt'] = np.sqrt(abs(df[i]))
            return df

        return sqrt(df, self.num_col_list)

    def _log_features(self, df):
        def log(df, num_cols):
            for i in num_cols:
                df[f'{i}_log'] = (np.log(abs(df[i]) + 1))
            return df

        return log(df, self.num_col_list)

    def perform_feature_engineering(self, df):
        df = self._add_features(df)
        df = self._multiply_features(df)
        df = self._inverse_features(df)
        df = self._square_features(df)
        df = self._square_root_features(df)
        df = self._log_features(df)
        return df

    def get_df_test_final(self, df):
        return df[self.best_feat]

    def get_ytest(self, df_test_trans):
        return df_test_trans['went_on_backorder'].values

    def preprocess(self, df):
        df = self._encode_bool_columns(df)
        df = self._replace_perf_cols(df)
        df = self._drop_sku_col(df)
        df = self._handle_missing_values(df)
        df = self._perform_standardization(df)
        df = self._perform_feature_selection(df)
        df = self.perform_feature_engineering(df)
        df_test_final = self.get_df_test_final(df)

        return df_test_final, self.get_ytest(df)

    def predict(self, df):
        df_test_final, y_test = self.preprocess(df)
        return y_test, self.model.predict(df_test_final)


def final_fun_1(df, return_actual=False):
    df_test = df.copy()

    # Fetching the best features
    best_feat = joblib.load('backorder/models/test_best_feat.pkl')
    best_feat = best_feat.tolist()

    # Fetching the trained standardization object instance
    sc = joblib.load('backorder/models/sc.pkl')
    df_test = df_test[df_test['went_on_backorder'].notna()]

    # Encode categorical columns with values Yes and No to 1 and 0 respectively
    dict_map_bool = {'Yes': 1.0, 'No': 0.0}

    df_test['deck_risk'] = df_test['deck_risk'].map(dict_map_bool)
    df_test['potential_issue'] = df_test['potential_issue'].map(dict_map_bool)
    df_test['oe_constraint'] = df_test['oe_constraint'].map(dict_map_bool)
    df_test['ppap_risk'] = df_test['ppap_risk'].map(dict_map_bool)
    df_test['stop_auto_buy'] = df_test['stop_auto_buy'].map(dict_map_bool)
    df_test['rev_stop'] = df_test['rev_stop'].map(dict_map_bool)
    df_test['went_on_backorder'] = df_test['went_on_backorder'].map(dict_map_bool)

    # Replacing -99 in perfomance columns with nan
    df_test.perf_6_month_avg.replace({-99.0: np.nan}, inplace=True)
    df_test.perf_12_month_avg.replace({-99.0: np.nan}, inplace=True)
    df_test = df_test.drop(columns=['sku'])

    # Handling missing values with median values
    df_test.lead_time.replace(to_replace=np.nan, value=8, inplace=True)
    df_test.perf_6_month_avg.replace(to_replace=-99, value=.85, inplace=True)
    df_test.perf_12_month_avg.replace(to_replace=-99, value=.83, inplace=True)

    # Performing Standardization
    df_test_num_sc = sc.transform(df_test[num_col_list].values)
    df_test_num_sc = pd.DataFrame(df_test_num_sc, index=df_test.index, columns=num_col_list)

    # Assigning numerical columns to original dataframe
    for i in num_col_list:
        df_test[i] = df_test_num_sc[i]

    # Function to perform addition of features
    def add(df, num_cols):
        for i in num_cols:
            for j in num_cols:
                if i != j:
                    df[i + '_' + j + '_add'] = df[i] + df[j]
        return df

    # Function to perform multiplication of features
    def mult(df, num_cols):
        for i in num_cols:
            for j in num_cols:
                if i != j:
                    df[i + '_' + j + '_mult'] = df[i] * df[j]
        return df

    # Function to perform inverse of features
    def inv(df, num_cols):
        for i in num_cols:
            df[i + '_' + 'inv'] = 1 / (df[i] + 0.001)
        return df

    # Function to perform square of features
    def square(df, num_cols):
        for i in num_cols:
            df[i + '_' + 'square'] = df[i] * df[i]
        return df

    # Function to perform square root of features
    def sqrt(df, num_cols):
        for i in num_cols:
            df[i + '_' + 'square_root'] = np.sqrt(abs(df[i]))
        return df

    # Function to perform log of features
    def log(df, num_cols):
        for i in num_cols:
            df[i + '_' + 'log'] = (np.log(abs(df[i]) + 1))
        return df

    # Applying tranformed functions
    df_test_trans = add(df_test, num_col_list)
    df_test_trans = mult(df_test_trans, num_col_list)
    df_test_trans = inv(df_test_trans, num_col_list)
    df_test_trans = square(df_test_trans, num_col_list)
    df_test_trans = sqrt(df_test_trans, num_col_list)
    df_test_trans = log(df_test_trans, num_col_list)

    df_test_final = df_test_trans[best_feat]
    y_test = df_test_trans['went_on_backorder'].values

    filename = 'backorder/models/backorder_best_model.pkl'
    model = joblib.load(filename)
    y_pred_test = model.predict(df_test_final)

    return y_test, y_pred_test
