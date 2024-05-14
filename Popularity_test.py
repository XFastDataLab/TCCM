#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import math
import sys
from utils import *
import numpy as np
import os
import json
import tensorflow as tf
from utils import *
import matplotlib.pyplot as plt

news_index = {}
user_index = {}
entity2index = {}
entity_values = []
news_publish_time = np.load('/home/ywy/PycharmProjects/News_Recommendation/data/popularity/news_publish_time.npy')
with tf.io.gfile.GFile('/home/ywy/PycharmProjects/News_Recommendation/data' + '/news.tsv', "r") as rd:
    for line in rd:
        nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split('\t')
        if nid in news_index:
            continue
        news_index[nid] = len(news_index) + 1

# news_popularity = np.zeros((len(news_index) + 1))
# news_click = np.zeros((len(news_index) + 1))
# news_clicks = np.zeros(int(np.ceil(len(news_click)/20)))
# news_all_click = 0
# news_exposure = np.zeros((len(news_index) + 1))
# news_exposures = np.zeros(int(np.ceil(len(news_exposure)/20)))
# with tf.io.gfile.GFile('/home/ywy/PycharmProjects/News_Recommendation/data' + '/behaviors.tsv', "r") as re:
#     for line in re:
#         uid, Time, history, impr = line.strip("\n").split('\t')[-4:]
#         for j in impr.split():
#             if int(j.split("-")[1]) == 1:
#                 news_click[news_index[j.split("-")[0]]] += 1
#                 news_all_click += 1
#                 news_exposure[news_index[j.split("-")[0]]] += 1
#             if int(j.split("-")[1]) == 0:
#                 news_exposure[news_index[j.split("-")[0]]] += 1
#         news_popularity = news_click
# for i in range(int(np.ceil(len(news_click)/20))):
#     news_clicks[i] = news_click[(i+20)]
# for j in range(int(np.ceil(len(news_exposure)/20))):
#     news_exposures[j] = news_exposure[(j+20)]
# plt.figure()
# # z = np.polyfit(news_popularity, news_exposure, 6)
# # p = np.poly1d(z)
# # plt.plot(news_popularity, p(news_popularity), "r-", linestyle='dashed')
# plt.scatter(news_click, news_exposure)
# plt.xlabel("x", fontsize=15)
# plt.ylabel("y", fontsize=15)
# plt.tick_params(labelsize=15)
# plt.legend()
# plt.show()

# 流行度对曝光的影响
day = 1
n = math.ceil(504 / day)
news_click = np.zeros((n + 1, len(news_index) + 1))
news_exposure = np.zeros((n + 1, len(news_index) + 1))
news_all_click = np.zeros((n + 1))
with tf.io.gfile.GFile('/home/ywy/PycharmProjects/News_Recommendation/data' + '/behaviors.tsv', "r") as re:
    for line in re:
        uid, Time, history, impr = line.strip("\n").split('\t')[-4:]
        Time = parse_time_bucket(Time)
        # Time = Time / 24
        Time = math.ceil(Time / day)
        for j in impr.split():
            if int(j.split("-")[1]) == 1:
                news_click[Time, news_index[j.split("-")[0]]] += 1
                # a = news_click[Time, self.news_index[j.split("-")[0]]]
                news_exposure[Time, news_index[j.split("-")[0]]] += 1
                news_all_click[Time] += 1
            if int(j.split("-")[1]) == 0:
                news_exposure[Time, news_index[j.split("-")[0]]] += 1
# news_clicks = np.concatenate(news_click[295:300, :])
# news_exposures = np.concatenate(news_exposure[295:300, :])
# news_plot = news_click[295:300, :]
# news_plot[0, :] = 1
# news_plot[1, :] = 2
# news_plot[2, :] = 3
# news_plot[3, :] = 5
# news_plot[4, :] = 5
# news_plot = np.concatenate(news_plot[0:5, :])
plt.figure(figsize=(8, 5))
# z = np.polyfit(news_popularity, news_exposure, 6)
# p = np.poly1d(z)
# plt.plot(news_popularity, p(news_popularity), "r-", linestyle='dashed')
# plt.scatter(news_clicks, news_exposures)
# plt.scatter(news_click[292], news_exposure[292])
# plt.scatter(news_click[293], news_exposure[293])
# plt.scatter(news_click[294], news_exposure[294])
plt.scatter(news_click[299], news_exposure[299], marker='o', c='w', edgecolors='g')
plt.scatter(news_click[295], news_exposure[295], marker='o', c='w', edgecolors='r')
plt.scatter(news_click[296], news_exposure[296], marker='o', c='w', edgecolors='#FFA500')
plt.scatter(news_click[297], news_exposure[297], marker='o', c='w', edgecolors='b')
plt.scatter(news_click[298], news_exposure[298], marker='o', c='w', edgecolors='#00FF00')

# plt.scatter(news_click[300], news_exposure[300])
# plt.scatter(news_click[301], news_exposure[301])
# plt.scatter(news_clicks, news_exposures, c=news_plot, alpha=0.5, cmap='jet')
plt.colorbar()
# plt.scatter(news_click[303], news_exposure[303])
# plt.scatter(news_click[304], news_exposure[304])
plt.xlabel("popularity", fontsize=15)
plt.ylabel("exposure", fontsize=15)
plt.tick_params(labelsize=15)
plt.savefig('/home/ywy/Pictures/exposure_and_popularity_Result_7.eps', format='eps', bbox_inches='tight',  dpi=400)
# plt.show()

# # 时间对曝光的影响
# day = 1
# n = math.ceil(504 / day)
# news_click = np.zeros((n + 1, len(news_index) + 1))
# news_exposure = np.zeros((n + 1, len(news_index) + 1))
# news_all_click = np.zeros((n + 1))
# with tf.io.gfile.GFile('/home/ywy/PycharmProjects/News_Recommendation/data' + '/behaviors.tsv', "r") as re:
#     for line in re:
#         uid, Time, history, impr = line.strip("\n").split('\t')[-4:]
#         news_current_time = trans2tsp(Time)
#         Time = parse_time_bucket(Time)
#         # Time = Time / 24
#         Time = math.ceil(Time / day)
#         for j in impr.split():
#             if int(j.split("-")[1]) == 1:
#                 news_click[Time, news_index[j.split("-")[0]]] = ((news_current_time - news_publish_time[news_index[j.split("-")[0]]]) // 3600 + 1)
#                 # a = news_click[Time, self.news_index[j.split("-")[0]]]
#                 news_exposure[Time, news_index[j.split("-")[0]]] += 1
#                 news_all_click[Time] += 1
#             if int(j.split("-")[1]) == 0:
#                 news_click[Time, news_index[j.split("-")[0]]] = ((news_current_time - news_publish_time[news_index[j.split("-")[0]]]) // 3600 + 1)
#                 news_exposure[Time, news_index[j.split("-")[0]]] += 1
# plt.figure(figsize=(8, 5))
# plt.scatter(news_click[299], news_exposure[299], marker='o', c='w', edgecolors='g')
# plt.scatter(news_click[295], news_exposure[295], marker='o', c='w', edgecolors='r')
# plt.scatter(news_click[296], news_exposure[296], marker='o', c='w', edgecolors='#FFA500')
# plt.scatter(news_click[297], news_exposure[297], marker='o', c='w', edgecolors='b')
# plt.scatter(news_click[298], news_exposure[298], marker='o', c='w', edgecolors='#00FF00')
# plt.colorbar()
# plt.xlabel("time", fontsize=15)
# plt.ylabel("exposure", fontsize=15)
# plt.tick_params(labelsize=15)
# plt.savefig('/home/ywy/Pictures/13.png', dpi=960)
# plt.show()
