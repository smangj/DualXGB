#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2024/8/15 15:39
# @Author   : wsy
# @email    : 631535207@qq.com
import pandas as pd
import numpy as np

data = pd.read_csv("data/豆粕.csv")


def time_series_gap_cv(data, n_splits, block_size, gap_size):
    """
    实现Time series gap cross-validation。

    :param data: 时间序列数据，可以是特征矩阵或目标向量
    :param n_splits: 折数
    :param block_size: 每个块的大小
    :param gap_size: 块之间的间隔大小
    :return: 生成器，产生训练/测试索引对
    """
    n_samples = len(data)

    for i in range(n_splits):
        # 计算当前折的测试块起始位置
        test_start = i * (block_size + gap_size) + block_size
        test_end = (i + 1) * (block_size + gap_size) + block_size

        # 确保测试块不会超出数据范围
        if test_end > n_samples:
            break

        # 划分训练和测试索引
        train_index = np.arange(test_start - block_size - gap_size)
        test_index = np.arange(test_start, min(test_end, n_samples))

        yield train_index, test_index


# 使用Blocked time series cross-validation
for train_index, test_index in time_series_gap_cv(
    data, n_splits=5, block_size=0, gap_size=1
):
    print("训练索引:", train_index, "测试索引:", test_index)
