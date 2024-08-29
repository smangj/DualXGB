import numpy as np
import pandas as pd

from models.models import Model
from models.short_models import ShortXGB
import xgboost
from tqdm import tqdm
from arch import arch_model
import torch
import torch.nn as nn
import torch.optim as optim


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


class Lstm(Model):
    params = {
        "input_size": 49,
        "hidden_size": 30,
        "num_layers": 2,
        "output_size": 1,
        "learning_rate": 0.003,
        "num_epochs": 5
    }

    def __init__(self, GPU=0, batch_size=2000):
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.batch_size = batch_size
        self.model = LSTMModel(self.params["input_size"],
                          self.params["hidden_size"],
                          self.params["num_layers"],
                          self.params["output_size"])
        self.model.to(self.device)

    def fit(self,  X, y):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])

        x_train_values = X.values
        y_train_values = np.squeeze(y.values)

        self.model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for step in range(self.params["num_epochs"]):

            for i in range(len(indices))[:: self.batch_size]:

                feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
                label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

                pred = self.model(feature)
                loss = criterion(pred, label)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                optimizer.step()

                print(f'Epoch [{step + 1}/{self.params["num_epochs"]}], Loss: {loss}')

    def predict(self, x_test):

        index = x_test.index
        self.model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = self.model(x_batch).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.input_size, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()
