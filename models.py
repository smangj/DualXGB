#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2024/8/16 15:46
# @Author   : wsy
# @email    : 631535207@qq.com
import abc

import pandas as pd
import xgboost
from xgboost import XGBClassifier


class Model(abc.ABC):

    def __int__(self):
        pass

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        pass


class LongXGB(Model):
    MEAN_VOL = 0.18
    params = {
        "eval_metric": "auc",
        "max_depth": 2,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic"
    }

    def __int__(self):
        self.model = None

    def fit(self, X, y):
        bst = XGBClassifier(eval_metric='auc', max_depth=2, colsample_bytree=0.8, objective='binary:logistic')

        bst.fit(X, y)
        self.model = bst

    def predict(self, X_test):
        return self.model.predict(X_test)

    def generate_label(self, y):
        return y >= self.MEAN_VOL

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance
        """
        return pd.Series(self.model.get_score(*args, **kwargs)).sort_values(ascending=False)


if __name__ == '__main__':
    pass
