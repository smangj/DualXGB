#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2024/8/15 15:39
# @Author   : wsy
# @email    : 631535207@qq.com
from tscv import GapKFold
import pandas as pd
import numpy as np

from evaluate import eva, mean_evaresult
from models import LongXGB

# from sklearn.model_selection import train_test_split

pre_len = 40
data = pd.read_csv("data/豆粕.csv", index_col="date").iloc[:, 1:]
feature_column = data.columns
data["label"] = data["his_rv_1month_futures"].shift(-pre_len)
data = data.dropna()
# label = data["his_rv_1month_futures"].shift(-pre_len)
# X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=.2)

n_splits = 5
gap_size = 60
gkf = GapKFold(n_splits=n_splits, gap_before=gap_size, gap_after=gap_size)

model_list = [LongXGB]

for model_class in model_list:
    fold = []
    feature_importance = []
    for train_index, test_index in gkf.split(data):
        model = model_class()
        model.fit(data.iloc[train_index][feature_column], model.generate_label(data.iloc[train_index]["label"]))

        pred = model.predict(data.iloc[test_index][feature_column])
        fold.append(eva(model.generate_label(data.iloc[test_index]["label"]), pred))
        # try:
        #     fi = model.get_feature_importance()
        #     feature_importance.append(fi)
        # except:
        #     pass

    avg = mean_evaresult(fold)
    print(fold)


