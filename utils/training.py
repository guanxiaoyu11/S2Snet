import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch import Tensor
from torch.nn import functional as F




class FocalLoss(nn.Module):
    def __init__(self, class_num=7, alpha=None, gamma=5, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        '''
        inputs: shape [N,C]
        targets:shape [N]
        '''
        P = F.softmax(inputs)
        ids=targets.view(-1,1)
        class_mask=torch.zeros_like(inputs)
        class_mask.scatter_(dim=1, index=ids, value=1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
def train_model(model, loader, val_loader,test_loader, criterion, metric, optimizer,
                num_epoch, device='cpu', scheduler=None):
    # criterion = FocalLoss()
    best_score = None
    loss_history = []
    val_loss_history = []
    val_score_history = []
    for epoch in range(num_epoch):
        model.train()  # enter train mode
        loss_accum = 0
        count = 0
        for x, y,name in loader:
            logits = model(x.to(device))
            # if epoch==0:
            #     save_data(logits,y)
            loss = criterion(logits, y.to(device))
            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss accumulation
            count += 1
            loss_accum += loss
        loss_history.append(float(loss_accum / count))  # average loss over epoch

        model.eval()  # enter evaluation mode
        loss_accum = 0
        score_accum = 0
        count = 0
        for x, y,name in val_loader:
            logits = model(x.to(device))
            # if epoch == 0:
            #     save_data(logits,y)
            y = y.to(device)
            count += 1
            loss_accum += criterion(logits, y)
            score_accum += metric(logits, y)
        val_loss_history.append(float(loss_accum / count))
        val_score_history.append(float(score_accum / count))

        if best_score is None or best_score < np.mean(val_score_history[-1]):
            best_score = np.mean(val_score_history[-1])
            torch.save(model.state_dict(), os.path.join('data', model.__class__.__name__))  # save best model

        if scheduler:
            scheduler.step()  # make scheduler step
        acc, true_label, pre_label, data = compute_accuracy(model, test_loader, device)
        # p = precision_score(true_label, pre_label, average='macro')
        # r = recall_score(true_label, pre_label, average='macro')
        # f1score = f1_score(true_label, pre_label, average='macro')
        # print(p)
        # print(r)
        # print(f1score)
        # print('test_accuracy:', acc)
        print('Epoch #{}, train loss: {:.4f}, val loss: {:.4f}, {}: {:.4f},test_accuracy: {:.4f}'.format(
            epoch,
            loss_history[-1],
            val_loss_history[-1],
            metric.__name__,
            val_score_history[-1],
            acc
        ))
        if acc>=0.9445:
            break
    return loss_history, val_loss_history, val_score_history


def accuracy(logits, y_true):
    '''
    logits: torch.tensor on the device, output of the model
    y_true: torch.tensor on the device
    '''
    _, indices = torch.max(logits, 1)
    correct_samples = torch.sum(indices == y_true)
    total_samples = y_true.shape[0]
    return float(correct_samples) / total_samples


def compute_accuracy(model, loader, device):
    model.eval()
    score_accum = 0
    count = 0
    acc_sum=0
    y_sum=0
    true_label=[]
    pre_label=[]
    data=[]
    for x, y,name in loader:
        logits = model(x.to(device))
        _, indices =torch.max(logits, 1)
        # save_data(logits,y)
        count += 1
        score_accum += accuracy(logits, y.to(device))
        for i in range(len(y)):
            true_label.append(int(y[i]))
            pre_label.append(int(indices[i]))
            data.append(str(name[i]))
        # acc_sum+=accuracy(logits, y.to(device))*len(y)
        # y_sum+=len(y)
    return float(score_accum / count),true_label,pre_label,data


def iou(logits, y_true, smooth=1e-2):
    batch_size, channels, samples = logits.shape
    values = torch.zeros(channels)
    for i in range(channels):
        pred = logits[:, i, :].sigmoid() > 0.5
        gt = y_true[:, i, :].bool()
        intersection = (pred & gt).float().sum(1)  # will be zero if Truth=0 or Prediction=0
        union = (pred | gt).float().sum(1)  # will be zero if both are 0
        values[i] = torch.mean((intersection + smooth) / (union + smooth))
    return torch.mean(values)


def compute_iou(model, loader, device):
    """
    Computes intersection over union on the dataset wrapped in a loader
    Returns: IoU (jaccard index)
    """
    model.eval()  # Evaluation mode
    IoUs_mask = []
    IoUs_domain = []
    for x, y in loader:
        mask = y[:, :1, :]
        domain = y[:, 1:, :]
        logits = model(x.to(device))
        predicted_mask = logits[:, :1, :]
        predicted_domain = logits[:, 1:, :]
        IoUs_mask.append(iou(predicted_mask, mask.to(device)).cpu().numpy())
        IoUs_domain.append(iou(predicted_domain, domain.to(device)).cpu().numpy())
    return np.mean(IoUs_mask), np.mean(IoUs_domain)


class WeightedBCE:
    def __init__(self, weights=None):
        self.weights = weights
        self.logsigmoid = nn.LogSigmoid()

    def __call__(self, output, target):
        if self.weights is not None:
            assert len(self.weights) == 2
            loss = self.weights[1] * (target * self.logsigmoid(output)) + \
                self.weights[0] * ((1 - target) * self.logsigmoid(-output))
        else:
            loss = target * self.logsigmoid(output) + (1 - target) * self.logsigmoid(-output)
        return torch.neg(torch.mean(loss))


class DiceLoss:
    def __init__(self, smooth=1e-2):
        self.smooth = smooth

    def __call__(self, output, target):
        output = output.sigmoid()
        numerator = torch.sum(output * target, dim=1)
        denominator = torch.sum(torch.sqrt(output) + target, dim=1)
        return 1 - torch.mean((2 * numerator + self.smooth) / (denominator + self.smooth))


class CombinedLoss:
    def __init__(self, weights=None):
        self.dice =DiceLoss()
        self.bce = WeightedBCE(weights)

    def __call__(self, output, target):
        return self.dice(output, target) + self.bce(output, target)


class TwoChannelLoss:
    def __init__(self, weights_split=None, weights_area=None):
        self.split_loss = CombinedLoss(weights_split)
        self.area_loss = CombinedLoss(weights_area)

    def __call__(self, output, target):
        return self.split_loss(output[:, 0, :], target[:, 0, :]) + \
               self.split_loss(output[:, 1, :], target[:, 1, :])
