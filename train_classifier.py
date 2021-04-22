import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.models import Classifier
from utils.dataset import ROIDataset, get_label
from utils.training import train_model, accuracy, compute_accuracy
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
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
    test_name = []
    for j in range(len(data[0])):
        test_name.append(data[0][j]['name'])
    for j in range(len(data[1])):
        test_name.append(data[1][j]['name'])
    for j in range(len(data[2])):
        test_name.append(data[2][j]['name'])
    for j in range(len(data[3])):
        test_name.append(data[3][j]['name'])
    for j in range(len(data[4])):
        test_name.append(data[4][j]['name'])
    for j in range(len(data[5])):
        test_name.append(data[5][j]['name'])
    for j in range(len(data[6])):
        test_name.append(data[6][j]['name'])
    random.shuffle(test_name)
    return test_name
def gettrain_val(path):
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
    random.shuffle(data[0])
    random.shuffle(data[1])
    random.shuffle(data[2])
    random.shuffle(data[3])
    random.shuffle(data[4])
    random.shuffle(data[5])
    random.shuffle(data[6])
    train_name=[]
    val_name=[]
    for j in range(len(data[0])):
        if j<int(len(data[0])*95/100):
            train_name.append(data[0][j]['name'])
        else:
            val_name.append(data[0][j]['name'])
    for j in range(len(data[1])):
        if j<int(len(data[1])*95/100):
            train_name.append(data[1][j]['name'])
        else:
            val_name.append(data[1][j]['name'])
    for j in range(len(data[2])):
        if j<int(len(data[2])*95/100):
            train_name.append(data[2][j]['name'])
        else:
            val_name.append(data[2][j]['name'])
    for j in range(len(data[3])):
        if j<int(len(data[3])*95/100):
            train_name.append(data[3][j]['name'])
        else:
            val_name.append(data[3][j]['name'])
    for j in range(len(data[4])):
        if j<int(len(data[4])*95/100):
            train_name.append(data[4][j]['name'])
        else:
            val_name.append(data[4][j]['name'])
    for j in range(len(data[5])):
        if j<int(len(data[5])*95/100):
            train_name.append(data[5][j]['name'])
        else:
            val_name.append(data[5][j]['name'])
    for j in range(len(data[6])):
        if j<int(len(data[6])*95/100):
            train_name.append(data[6][j]['name'])
        else:
            val_name.append(data[6][j]['name'])
    random.shuffle(train_name)
    random.shuffle(val_name)
    return train_name,val_name
def get_confux(y_test,y_test_predict_R):
    obj1 = confusion_matrix(y_test, y_test_predict_R)
    print('confusion_matrix\n', obj1)
    classes = list(set(y_test_predict_R))
    classes.sort()
    plt.figure(figsize=(12, 8), dpi=120)
    plt.imshow(obj1, cmap=plt.cm.Blues)

    indices = range(len(obj1))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('label2 of test set sample')
    plt.ylabel('label2 of predict')
    plt.title('Confusion Matrix', fontsize='10', fontproperties='arial')
    for first_index in range(len(obj1)):
        for second_index in range(len(obj1[first_index])):
            plt.text(first_index, second_index, obj1[first_index][second_index], fontsize=10, va='center',
                     ha='center')
    labels = ['0', '1', '2', '3', '4', '5', '6']
    cm = confusion_matrix(y_test, y_test_predict_R)
    tick_marks = np.array(range(len(labels))) + 0.5

    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylim(len(labels) - 0.5, -0.5)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        cm = confusion_matrix(y_test, y_test_predict_R)
        np.set_printoptions(precision=2)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    thresh = cm.max() / 2
    for x_vall, y_vall in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_vall][x_vall]
        if c > 0.0001:
            plt.text(x_vall, y_vall, "%0.4f" % (c,), color="white" if cm[x_vall, y_vall] > thresh else "black",
                     fontsize=10, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # show confusion matrix
    plt.savefig(r'confusion_matrix.png', format='png')
if __name__ == '__main__':
    for i in range(1):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        print('Current device is {}'.format(device))

        batch_size = 512
        train_name=[]
        val_name=[]
        train_name,val_name=gettrain_val(r'data/train')
        test_name=gettest(r'data/test')
        model = Classifier().to(device)
        # clear_dir(".\\data")
        # split_train_test(".\\data\\json\\ta", ".\\data", 0.2, 0.2)
        # split_train_test(".\\data\\json\\tb", ".\\data", 0.2, 0.2)
        # split_train_test(".\\data\\json\\mi", ".\\data", 0.2, 0.2)
        # split_train_test(".\\data\\json\\si", ".\\data", 0.2, 0.2)
        train_dataset = ROIDataset(path='data/train',load=train_name, key=get_label, mode='classification', gen_p=0)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = ROIDataset(path='data/train',load=val_name, key=get_label, mode='classification', gen_p=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()
        test_dataset = ROIDataset(path='data/test',load=test_name, key=get_label, mode='classification', gen_p=0)
        test_loader = DataLoader(test_dataset, batch_size=559)
        result = train_model(model, train_loader, val_loader,test_loader, criterion,
                             accuracy, optimizer, 2000, device, scheduler)
        ## model.load_state_dict(torch.load(os.path.join('data', model.__class__.__name__)))
        # model = Classifier().to(device)
        # model.load_state_dict(torch.load('checkpoint94.5.tar')['state_dict'])

        #print('test_accuracy: {:.4f}'.format(compute_accuracy(model, test_loader, device)))
        acc,true_label,pre_label,data=compute_accuracy(model, test_loader, device)
        get_confux(true_label,pre_label)
        p = precision_score(true_label, pre_label, average='macro')
        r = recall_score(true_label, pre_label, average='macro')
        f1score = f1_score(true_label, pre_label, average='macro')
        print(p)
        print(r)
        print(f1score)
        print('test_accuracy:',acc)
        if acc>0.940:
            torch.save({
                'state_dict': model.state_dict(),
                'best_prec1': acc,}, 'checkpoint4.15_10.48.tar')
            f1 = open(r'./output.txt', 'w+')
            f1.truncate()  # 清空文件
            for s in range(len(true_label)):

                f1.writelines(str(true_label[s]))
                f1.writelines(str(pre_label[s]))
                f1.writelines(data[s])
                f1.writelines('\n')

            f1.close()
            break

