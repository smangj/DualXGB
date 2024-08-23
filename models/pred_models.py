from models.models import Model
from models.short_models import ShortXGB
import xgboost

# from arch import arch_model


class XGB(ShortXGB):
    def predict(self, X_test):
        xgb_score = self.model.predict(xgboost.DMatrix(X_test))
        trend = X_test["his_rv_1month_futures"] > X_test["past_vol_2_futures"]
        return (
            X_test["his_rv_1month_futures"]
            + xgb_score * trend * X_test["vol_std_20_futures"]
        )


class Lstm(Model):
    def fit(self, ret):
        pass
        # garch_model = arch_model(ret, vol="Garch", p=1, q=1, dist="Normal")

        # 拟合模型
        # garch_fit = garch_model.fit(disp="off")
        # self.model = garch_fit

    def predict(self, *args, **kwargs):
        pass
        # horizon = 20
        # forecasts = self.model.forecast(horizon=horizon)


class Garch:
    pass
