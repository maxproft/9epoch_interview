import xgboost as xgb
import pandas as pd
from typing import List
import numpy as np
from article import ArticleFeatures, DailyArticles
from sklearn.metrics import mean_squared_error


class XGBootstrapModel:
    save_file_name = 'xgb.json'

    def __init__(self, max_depth=5, n_estimators=100, learning_rate=0.1, dropout=False, skip_drop=None, rate_drop=None,
                 objective=None):
        kwargs = {
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'objective': objective,
        }

        if dropout:
            kwargs['booster'] = 'dart'
            kwargs['rate_drop'] = rate_drop
            kwargs['skip_drop'] = skip_drop

        self.regressor = xgb.XGBRegressor(**kwargs)

    def fit(self, data: List[List[ArticleFeatures]], true_output: np.ndarray, num_bootstrap: int):
        input_x, y_true = DailyArticles.bootstrap_train_pd(data, true_output, num_bootstrap)
        self.regressor.fit(input_x, y_true)

    def feature_importance(self):
        return pd.DataFrame(self.regressor.feature_importances_.reshape(1, -1))

    def predict_one_day(self, data: List[ArticleFeatures], num_bootstrap: int):
        da = DailyArticles([d.input_vec() for d in data], None)
        input_data = da.bootstrap_to_pandas(num_bootstrap)
        predictions = self.regressor.predict(input_data)
        return np.mean(predictions)

    def predict_many_days(self, data: List[List[ArticleFeatures]], num_bootstrap: int):
        return np.array([self.predict_one_day(d, num_bootstrap) for d in data])

    def predict_series(self, data: List[ArticleFeatures], num_bootstrap: int):
        predictions = []
        for i in range(len(data)):
            subset = data[0:i + 1]
            predictions.append(self.predict_one_day(subset, num_bootstrap))
        return predictions

    def mse(self, true, pred):
        return mean_squared_error(true, pred)

    def save(self):
        self.regressor.save_model(self.save_file_name)

    @classmethod
    def load(cls):
        m = XGBootstrapModel()
        m.regressor.load_model(cls.save_file_name)
        return m
