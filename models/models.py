#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2024/8/16 15:46
# @Author   : wsy
# @email    : 631535207@qq.com
import abc
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class Model(abc.ABC):

    def __int__(self):
        pass

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        pass


class Long(Model, abc.ABC):
    MEAN_VOL = 0.18

    def generate_label(self, y):
        return y >= self.MEAN_VOL


class LongXGB(Long):

    params = {
        "eval_metric": "auc",
        "max_depth": 2,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic"
    }

    def __int__(self):
        self.model = None

    def fit(self, X, y):
        data = xgboost.DMatrix(X, label=y)
        num_rounds = 100
        bst = xgboost.train(self.params, data, num_rounds)
        self.model = bst

    def predict(self, X_test):
        return np.round(self.model.predict(xgboost.DMatrix(X_test)))

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance
        """
        return pd.Series(self.model.get_score(*args, **kwargs)).fillna(0)


class Logistic(Long):

    def fit(self, X_train, y_train):
        logistic_model = LogisticRegression(solver='liblinear', max_iter=1000)
        logistic_model.fit(X_train, y_train)

        self.model = logistic_model


    def predict(self, y_test):
        return self.model.predict(y_test)


class SVM(Long):

    def fit(self, X_train, y_train):
        model = LinearSVC(C=0.5)
        scaler = StandardScaler()
        X_train_scale = scaler.fit_transform(X_train)
        model.fit(X_train_scale, y_train)

        self.model = model
        self.scale = scaler


    def predict(self, y_test):
        y_test_scale = self.scale.transform(y_test)
        return self.model.predict(y_test_scale)


class Adaboost(Long):

    def fit(self, X_train, y_train):
        # 创建决策树基分类器
        base_clf = DecisionTreeClassifier(max_depth=1)

        # 创建AdaBoost分类器
        clf = AdaBoostClassifier(estimator=base_clf, n_estimators=50, learning_rate=1.0, random_state=1)

        # 训练AdaBoost分类器
        clf.fit(X_train, y_train)
        self.model = clf

    def predict(self, X_test):
        return self.model.predict(X_test)


class Rf(Long):

    def fit(self, X_train, y_train):
        random_forest = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

        # 训练AdaBoost分类器
        random_forest.fit(X_train, y_train)
        self.model = random_forest

    def predict(self, X_test):
        return self.model.predict(X_test)

if __name__ == '__main__':
    pass
