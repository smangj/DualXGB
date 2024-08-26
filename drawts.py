#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2024/8/23 11:34
# @Author   : wsy
# @email    : 631535207@qq.com
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def draw_ts(true, pred):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题

    dates = pd.to_datetime(true.index)
    plt.figure(figsize=(10, 6))
    plt.plot(dates, true.values, 'k-', label='波动率实际值', linewidth=1.5)
    plt.plot(dates, pred.values, 'k--', label='波动率预测值', linewidth=1.5)
    plt.legend()
    # 设置标题和坐标轴标签
    # plt.title('豆粕实际波动率与预测值', fontsize=12)
    plt.xlabel('日期(年-月)', fontsize=10)
    plt.ylabel('波动率/%', fontsize=10)
    # 设置x轴格式
    plt.xticks(rotation=45)
    # 简化X轴日期显示，设置日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # 每2个月显示一次
    # plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())  # 每1个月作为次要刻度

    # 自动调整x轴标签，避免重叠
    plt.gcf().autofmt_xdate()

    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 显示图形
    plt.tight_layout()
    plt.savefig("aaa.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    df = pd.read_csv("aaa.csv")
