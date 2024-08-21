import numpy as np

from models.short_models import ShortXGB
import xgboost

class XGB(ShortXGB):

    def predict(self, X_test):
        xgb_score = self.model.predict(xgboost.DMatrix(X_test))
        trend = X_test["his_rv_1month_futures"] > X_test["past_vol_2_futures"]
        return X_test["his_rv_1month_futures"] + xgb_score * trend * X_test["vol_std_20_futures"]
