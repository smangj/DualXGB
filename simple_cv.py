#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2024/8/23 11:34
# @Author   : wsy
# @email    : 631535207@qq.com
import pandas as pd

from drawts import draw_ts
from evaluate import eva_regression
from models.pred_models import XGB, Garch, Lstm

pre_len = 40
data = pd.read_csv("data/豆粕.csv", index_col="date").iloc[:, 1:]
feature_column = data.columns
data["label"] = data["his_rv_1month_futures"].shift(-pre_len)
data = data.dropna()

test_start_date = "2023-07-03"

train_df = data.loc[:test_start_date]
test_df = data.loc[test_start_date:]

model_class = Lstm

model = model_class()

model.fit(
    train_df[feature_column],
    train_df["label"],
)
# model.fit(data["his_rv_1day_futures"], test_start_date)

pred = model.predict(test_df[feature_column])

e = eva_regression(test_df["label"], pred)

draw_ts(test_df["label"], pred)

print("haha")
