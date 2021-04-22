import torch.optim as optim
from torch.utils.data import DataLoader
from utils.models import Classifier
from utils.dataset import ROIDataset, get_label
from utils.training import train_model, accuracy, compute_accuracy
from utils.create_train_val_set import clear_dir, split_train_test
import torch
import os
import numpy as np
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import random
import matplotlib.pyplot as plt
def gettest(path):
    data = dict()
    classes = (0, 1, 2, 3, 4, 5, 6)
    for label in classes:
        data[label] = []
    for i, file in enumerate(os.listdir(path)):
        with open(os.path.join(path, file)) as json_file:
            roi = json.load(json_file)
            roi['name'] = json_file.name
            if roi['label'] in classes:
                data[roi['label']].append(roi)
    test_name1 = []
    for j in range(len(data[1])):
        test_name1.append(data[1][j]['data'])
    test_name2 = []
    for j in range(len(data[2])):
        test_name2.append(data[2][j]['data'])
    test_name3 = []
    for j in range(len(data[3])):
        test_name3.append(data[3][j]['data'])
    test_name4 = []
    for j in range(len(data[4])):
        test_name4.append(data[4][j]['data'])
    test_name5 = []
    for j in range(len(data[5])):
        test_name5.append(data[5][j]['data'])
    test_name6 = []
    for j in range(len(data[6])):
        test_name6.append(data[6][j]['data'])
    test_name0 = []
    for j in range(len(data[0])):
        test_name0.append(data[0][j]['data'])
    x1=[]
    y1=[]
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    x4 = []
    y4 = []
    x5 = []
    y5 = []
    x6 = []
    y6 = []
    x0 = []
    y0 = []
    for j in range(len(test_name1)):
        x1.append(np.std(test_name1[j]))
        y1.append(np.max(test_name1[j]))
    for j in range(len(test_name2)):
        x2.append(np.std(test_name2[j]))
        y2.append(np.max(test_name2[j]))
    for j in range(len(test_name3)):
        x3.append(np.std(test_name3[j]))
        y3.append(np.max(test_name3[j]))
    for j in range(len(test_name4)):
        x4.append(np.std(test_name4[j]))
        y4.append(np.max(test_name4[j]))
    for j in range(len(test_name5)):
        x5.append(np.std(test_name5[j]))
        y5.append(np.max(test_name5[j]))
    for j in range(len(test_name6)):
        x6.append(np.std(test_name6[j]))
        y6.append(np.max(test_name6[j]))
    for j in range(len(test_name0)):
        x0.append(np.std(test_name0[j]))
        y0.append(np.max(test_name0[j]))
    plt.close('all')
    plt.figure(figsize=(12, 8), dpi=600)
    area = np.pi * 4 ** 2
    plt.scatter(x1, y1, s=area, c='#C4000B', alpha=0.4, label='A')
    plt.scatter(x2, y2, s=area, c='#0073B3', alpha=0.4, label='B')
    plt.scatter(x3, y3, s=area, c='#00ACB5', alpha=0.4, label='C')
    plt.scatter(x4, y4, s=area, c='#F7D800', alpha=0.4, label='D')
    plt.scatter(x5, y5, s=area, c='#FF00FF', alpha=0.4, label='E')
    plt.scatter(x6, y6, s=area, c='#0000FF', alpha=0.4, label='F')
    # plt.scatter(x0, y0, s=area, c='#008B00', alpha=0.4, label='G')
    # plt.plot([0, 9.5], [9.5, 0], linewidth='0.5', color='#000000')
    # plt.close('all')
    # plt.figure(figsize=(12, 8), dpi=600)
    # plt.plot(range(len(test_name[j])), test_name[j])
    # # plt.ylim(-0.125, 1.125)
    plt.legend()
    fileff = r"pic\sc_1\out2.png"
    plt.savefig(fileff)
    plt.close()
gettest('data/train')