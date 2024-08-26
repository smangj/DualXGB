import numpy as np
import pandas as pd

from models.models import Model
from models.short_models import ShortXGB
import xgboost
from tqdm import tqdm
from arch import arch_model


class XGB(ShortXGB):
    def predict(self, X_test):
        xgb_score = self.model.predict(xgboost.DMatrix(X_test))
        trend = X_test["his_rv_1month_futures"] > X_test["past_vol_2_futures"]
        return (
            X_test["his_rv_1month_futures"]
            + xgb_score * trend * X_test["vol_std_20_futures"]
        )


class Garch(Model):
    WindowSize = 120
    ForecastHorizon = 40
    def fit(self, ret, test_start_date=None):
        if test_start_date is not None:
            start = ret.index.get_loc(test_start_date)
        else:
            start = self.WindowSize

        end = len(ret[start:]) + start
        forecast_start_date = ret.index[start:end]
        forecast = []
        for i in tqdm(range(start, end)):
            window_data = ret.iloc[i - self.WindowSize:i]

            garch_model = arch_model(window_data, vol="Garch", p=1, q=1, mean="Constant", dist="Normal")

            try:
                # 拟合模型
                garch_fit = garch_model.fit(disp="off")
            except Exception as e:
                print("\n模型在索引{i}（{ret.index[i]}）失败:{e}")
                forecast.append([np.nan] * self.ForecastHorizon)
                continue

            forecast.append(garch_fit.forecast(horizon=self.ForecastHorizon).variance.iloc[-1].values)

        forecast_columns = [f'h{h}' for h in range(1, self.ForecastHorizon + 1)]
        forecasts_df = pd.DataFrame(forecast, index=forecast_start_date, columns=forecast_columns)

        forecasts_df['mean_forecast_var'] = forecasts_df.mean(axis=1)
        forecasts_df['mean_forecast_vol'] = np.sqrt(forecasts_df['mean_forecast_var'])
        self.result = forecasts_df

    def predict(self, *args, **kwargs):

        return self.result["mean_forecast_vol"]


class L:
    pass
