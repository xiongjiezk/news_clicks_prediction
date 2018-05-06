#!/usr/bin/env python
# -*- coding: utf-8 -*-  

# Copyright (c) 2017 - xiongjiezk <xiongjiezk@163.com>
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json


def plot_trend_curve():
    file_path = "E:/data/sohu_news/hot_trend/hot_trend_2.csv"
    df = pd.read_csv(file_path, header=None)
    startDate = datetime.datetime.strptime('2017-12-26', '%Y-%m-%d')
    date_list = [(startDate + datetime.timedelta(i)).strftime('%Y-%m-%d') for i in range(90)]
    df.columns = ['word', 'wid'] + date_list
    print(df.shape)

def plot_valid_curve(dataset='sohu', is_multi_class=False):
    result_dict = OrderedDict()
    lines = []
    with open("models/debug_log_plot_att", "r") as f:
        lines = f.read().strip().split("\n")
    N = 15
    for line in lines:
        res = json.loads(line)
        key = '_'.join([res['train_model'], res['dataset'], str(res['multi_class'])])
        history = res['history']
        # history = json.loads(history)
        val_acc = history['val_acc']
        if len(val_acc) < N:
            val_acc += [val_acc[-1]] * (N - len(val_acc))

        result_dict[key] = history['val_acc']

    res = []
    for key in result_dict:
        rr = [key]
        rr += [str(k) for k in result_dict[key] ]
        res.append(rr)
    df = pd.DataFrame(res)
    df.to_csv('../result/result_curve_3.csv', index=False)
    print(1/0)
    # plt.style.use("ggplot")
    # fig = plt.figure()
    for key in result_dict:
        plt.plot(np.arange(0, N), result_dict[key], label=key)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

# plot_valid_curve()

def plot_result():
    N = 15
    file_path = '../result/result_curve_2.csv'
    df = pd.read_csv(file_path)
    df.columns = ['model'] + ['epoch_' + str(i) for i in range(N)]
    model_dict = df.set_index('model').T.to_dict('list')

    fig = plt.figure()
    ax_sohu4c = fig.add_subplot(2, 2, 1)
    ax_sohu2c = fig.add_subplot(2, 2, 2)
    ax_tt4c = fig.add_subplot(2, 2, 3)
    ax_tt2c = fig.add_subplot(2, 2, 4)

    for model_key in model_dict:
        if 'sohu_4' in model_key:
            ax_sohu4c.plot(np.arange(0, N), model_dict[model_key], label=model_key.split('_')[0])
        if 'sohu_2' in model_key:
            ax_sohu2c.plot(np.arange(0, N), model_dict[model_key], label=model_key.split('_')[0])
        if 'toutiao_4' in model_key:
            ax_tt4c.plot(np.arange(0, N), model_dict[model_key], label=model_key.split('_')[0])
        if 'toutiao_2' in model_key:
            ax_tt2c.plot(np.arange(0, N), model_dict[model_key], label=model_key.split('_')[0])

    # ax_sohu4c.set_xlabel(r"Epoch #")
    ax_sohu4c.set_xlabel('(a) val_acc on Sohu-4Class', fontsize=15)
    ax_sohu4c.set_ylabel("val_acc", fontsize=15)
    # ax_sohu4c.set_title(r"(a) val_acc on Sohu-4Class", fontsize=15)
    ax_sohu4c.grid(True)
    ax_sohu4c.legend()
    # ax_sohu4c.tick_params(labelsize=10)

    # ax_sohu2c.set_xlabel(r"Epoch #")
    ax_sohu2c.set_xlabel('(b) val_acc on Sohu-2Class', fontsize=15)
    ax_sohu2c.set_ylabel("val_acc", fontsize=15)
    # ax_sohu2c.set_title(r"(b) val_acc on Sohu-2Class", fontsize=15)
    ax_sohu2c.grid(True)
    ax_sohu2c.legend()
    ax_sohu2c.tick_params(labelsize=10)

    ax_tt4c.set_xlabel(r"(c) val_acc on Toutiao-4Class", fontsize=15)
    ax_tt4c.set_ylabel("val_acc", fontsize=15)
    # ax_tt4c.set_title(r"(c) val_acc on Toutiao-4Class", fontsize=15)
    ax_tt4c.grid(True)
    ax_tt4c.legend()
    ax_tt4c.tick_params(labelsize=10)

    ax_tt2c.set_xlabel(r"(d) val_acc on Toutiao-2Class", fontsize=15)
    ax_tt2c.set_ylabel("val_acc", fontsize=15)
    # ax_tt2c.set_title(r"(d) val_acc on Toutiao-2Class", fontsize=15)
    ax_tt2c.grid(True)
    ax_tt2c.legend()
    ax_tt2c.tick_params(labelsize=10)

    plt.show()


plot_result()