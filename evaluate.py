#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2024/8/19 10:25
# @Author   : wsy
# @email    : 631535207@qq.com
from dataclasses import dataclass, fields
import typing
import statistics

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


@dataclass
class EvaResult:
    AUC: float
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass
class EvaRegressionResult:
    MSE: float
    MAE: float
    HMSE: float
    HMAE: float


def eva(y_true, y_predict):
    result = EvaResult(
        AUC=roc_auc_score(y_true, y_predict),
        accuracy=accuracy_score(y_true, y_predict),
        precision=precision_score(y_true, y_predict),
        recall=recall_score(y_true, y_predict),
        f1=f1_score(y_true, y_predict),
    )
    return result


def eva_regression(y_true, y_predict):
    result = EvaRegressionResult(
        MSE=np.mean((y_true - y_predict) ** 2),
        MAE=np.mean(np.abs(y_true - y_predict)),
        HMSE=np.mean(((y_true - y_predict) ** 2) / (y_true**2)),
        HMAE=np.mean(np.abs((y_true - y_predict) / y_true)),
    )
    return result


def mean_evaresult(results: typing.List[EvaResult]) -> EvaResult:
    field_names = [f.name for f in fields(EvaResult)]
    mean_values = {
        name: statistics.mean(getattr(result, name) for result in results)
        for name in field_names
    }
    return EvaResult(**mean_values)


if __name__ == "__main__":
    pass
