#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2024/8/16 15:46
# @Author   : wsy
# @email    : 631535207@qq.com
import abc

from models.models import Model, LongXGB, Logistic, SVM, Adaboost, Rf


class Short(Model, abc.ABC):
    MEAN_VOL = 0.1

    def generate_label(self, label, rv):
        return (label - rv).abs() / rv >= self.MEAN_VOL


class ShortXGB(Short, LongXGB):
    params = {
        "eval_metric": "auc",
        "max_depth": 5,
        "colsample_bytree": 0.8,
        "learning_rate": 0.1,
        "objective": "binary:logistic"
    }


class ShortLogistic(Short, Logistic):
    pass


class ShortSVM(Short, SVM):
    pass


class ShortAdaboost(Short, Adaboost):
    pass


class Rfshort(Short, Rf):
    pass