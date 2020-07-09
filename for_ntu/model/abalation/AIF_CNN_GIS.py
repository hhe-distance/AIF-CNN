# coding:utf-8
# author:黄宏�?

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

sys.path.append('../')
from utils import utils
from collections import defaultdict
import torchvision
import os
from initial_metrics import ntu_metrics
ntu_metrics = ntu_metrics.astype('float32')

class AIF_CNN(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''

    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=300,
                 num_class=60,
                 ):
        super(AIF_CNN, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        self.num_joint = num_joint
        self.out_channel = out_channel
        # position
        # metric1_numpy = np.eye(self.num_joint, 26).astype('float32') + 0.01 * np.random.randn(
        #     self.num_joint, 26).astype('float32')
        # self.adat_metrices1 = nn.Parameter(torch.from_numpy(metric1_numpy))

        self.adat_metrices1 = nn.Parameter(torch.from_numpy(ntu_metrics))
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel * 2, out_channels=out_channel // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=(3, 1), stride=1,
                      padding=(1, 0)),
            nn.MaxPool2d((2, 1))

        )

        metric_r1_numpy = np.eye(25).astype('float32') + 0.01 * np.random.randn(25).astype('float32')
        self.adat_metrices_r1 = nn.Parameter(torch.from_numpy(metric_r1_numpy))
        self.conv_resize1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=(3, 1), stride=1,
                      padding=(2, 0)),
            nn.MaxPool2d((2, 1))
        )

        metric_r2_numpy = np.eye(25).astype('float32') + 0.01 * np.random.randn(25).astype('float32')
        self.adat_metrices_r2 = nn.Parameter(torch.from_numpy(metric_r2_numpy))
        self.conv_resize2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=(3, 1), stride=1,
                      padding=(1, 0)),
            nn.MaxPool2d((2, 1))
        )

        metric3_numpy = np.eye(25, 26).astype('float32') + 0.01 * np.random.randn(25, 26).astype(
            'float32')
        self.adat_metrices3 = nn.Parameter(torch.from_numpy(metric3_numpy))
        # self.adat_metrices3_O = torch.from_numpy(metric3_numpy).cuda()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=3, stride=1,
                      padding=(2, 1)),
            nn.MaxPool2d(2)
        )

        metric4_numpy = np.eye(13, 14).astype('float32') + 0.01 * np.random.randn(13, 14).astype(
            'float32')
        self.adat_metrices4 = nn.Parameter(torch.from_numpy(metric4_numpy))
        # self.adat_metrices4_O = torch.from_numpy(metric4_numpy).cuda()

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        # motion
        # metric1_numpy_m = np.eye(self.num_joint, 26).astype('float32') + 0.01 * np.random.randn(
        #     self.num_joint, 26).astype('float32')
        # self.adat_metrices1_m = nn.Parameter(torch.from_numpy(metric1_numpy_m))

        self.adat_metrices1_m = nn.Parameter(torch.from_numpy(ntu_metrics))
        self.relu1_m = nn.ReLU()

        self.conv1_m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel * 2, out_channels=out_channel // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.conv2_m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=(3, 1), stride=1,
                      padding=(1, 0)),
            nn.MaxPool2d((2, 1))
        )

        metric_r1m_numpy = np.eye(25).astype('float32') + 0.01 * np.random.randn(25).astype('float32')
        self.adat_metrices_r1m = nn.Parameter(torch.from_numpy(metric_r1m_numpy))
        self.conv_resize1_m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=(3, 1), stride=1,
                      padding=(2, 0)),
            nn.MaxPool2d((2, 1))
        )

        metric_r2m_numpy = np.eye(25).astype('float32') + 0.01 * np.random.randn(25).astype('float32')
        self.adat_metrices_r2m = nn.Parameter(torch.from_numpy(metric_r2m_numpy))
        self.conv_resize2_m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=(3, 1), stride=1,
                      padding=(1, 0)),
            nn.MaxPool2d((2, 1))
        )

        metric3_numpy_m = np.eye(25, 26).astype('float32') + 0.01 * np.random.randn(25, 26).astype(
            'float32')
        self.adat_metrices3_m = nn.Parameter(torch.from_numpy(metric3_numpy_m))
        # self.adat_metrices3_m_O = torch.from_numpy(metric3_numpy_m).cuda()
        self.conv3_m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=3, stride=1,
                      padding=(2, 1)),
            nn.MaxPool2d(2)
        )

        metric4_numpy_m = np.eye(13, 14).astype('float32') + 0.01 * np.random.randn(13, 14).astype(
            'float32')
        self.adat_metrices4_m = nn.Parameter(torch.from_numpy(metric4_numpy_m))
        # self.adat_metrices4_m_o = torch.from_numpy(metric4_numpy_m).cuda()
        self.conv4_m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=3, stride=1, padding=(1, 1)),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        # concatenate motion & position
        metric5_numpy = np.eye(7, 8).astype('float32') + 0.01 * np.random.randn(7, 8).astype(
            'float32')
        self.adat_metrices5 = nn.Parameter(torch.from_numpy(metric5_numpy))
        # self.adat_metrices5_o = torch.from_numpy(metric5_numpy).cuda()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel * 4, kernel_size=3, stride=1,
                      padding=(2, 1)),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        metric6_numpy = np.eye(out_channel // 16).astype('float32') + 0.01 * np.random.randn(out_channel // 16).astype(
            'float32')
        self.adat_metrices6 = nn.Parameter(torch.from_numpy(metric6_numpy))
        # self.adat_metrices6_o = torch.from_numpy(metric6_numpy).cuda()
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel * 4, out_channels=out_channel * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7 = nn.Sequential(
            nn.Linear((out_channel * 8) * 3 * (out_channel // 32), 512),
            # 4*4 for window=64; 8*8 for window=128
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(512, num_class)
        utils.initial_model_weight(layers=list(self.children()))
        print('weight initial finished!')

    def forward(self, x, y, target=None, need_graph_optim=True):
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        if T != 64 and T != 128 and T != 256 and T != 300:
            raise ValueError('window size must be an int in [64, 128, 256, 300]!')
        motion = x[:, :, 1::, :, :] - x[:, :, 0:-1, :, :]
        motion = motion.permute(0, 1, 4, 2, 3).contiguous().view(N, C * M, T - 1, V)
        motion = F.interpolate(motion, size=(T, V), mode='bilinear',
                               align_corners=False).contiguous().view(N, C, M, T, V).permute(0, 1, 3, 4, 2)

        motion_y = y[:, :, 1::, :, :] - y[:, :, 0:-1, :, :]
        motion_y = motion_y.permute(0, 1, 4, 2, 3).contiguous().view(N, C * M, T - 1, V)
        motion_y = F.interpolate(motion_y, size=(T, V), mode='bilinear',
                                 align_corners=False).contiguous().view(N, C, M, T, V).permute(0, 1, 3, 4, 2)

        x = torch.cat((x, y), dim=1)
        motion = torch.cat((motion, motion_y), dim=1)

        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = x[:, :, :, :, i]
            adat_metrices1 = self.relu1(self.adat_metrices1)
            out = torch.matmul(out, adat_metrices1)
            out = self.conv1(out)
            out = self.conv2(out)
            out = torch.matmul(out, self.adat_metrices_r1)
            out = self.conv_resize1(out)
            out = torch.matmul(out, self.adat_metrices_r2)
            out = self.conv_resize2(out)
            out = torch.matmul(out, self.adat_metrices3)
            out = self.conv3(out)
            out = torch.matmul(out, self.adat_metrices4)
            out_p = self.conv4(out)

            # motion
            # N0,C1,T2,V3 point-level
            out = motion[:, :, :, :, i]
            adat_metrices1m = self.relu1_m(self.adat_metrices1_m)
            out = torch.matmul(out, adat_metrices1m)
            out = self.conv1_m(out)
            out = self.conv2_m(out)
            out = torch.matmul(out, self.adat_metrices_r1m)
            out = self.conv_resize1_m(out)
            out = torch.matmul(out, self.adat_metrices_r2m)
            out = self.conv_resize2_m(out)
            out = torch.matmul(out, self.adat_metrices3_m)
            out = self.conv3_m(out)
            out = torch.matmul(out, self.adat_metrices4_m)
            out_m = self.conv4_m(out)

            # concat
            out = torch.cat((out_p, out_m), dim=1)
            out = torch.matmul(out, self.adat_metrices5)
            out = self.conv5(out)
            out = torch.matmul(out, self.adat_metrices6)
            out = self.conv6(out)
            logits.append(out)

        # max out logits
        out = torch.max(logits[0], logits[1])
        out = out.view(out.size(0), -1)
        out = self.fc7(out)
        out = self.fc8(out)

        t = out
        assert not ((t != t).any())  # find out nan in tensor
        assert not (t.abs().sum() == 0)  # find out 0 tensor

        return out

def loss_fn(outputs, labels, current_epoch=None, params=None):
    """
    Compute the cross entropy loss given outputs and labels.

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    if params.loss_args["type"] == 'CE':
        CE = nn.CrossEntropyLoss()(outputs, labels)
        loss_all = CE
        loss_bag = {'ls_all': loss_all, 'ls_CE': CE}
    # elif: other losses

    return loss_bag

def accuracytop1(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def accuracytop2(output, target, topk=(2,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def accuracytop3(output, target, topk=(3,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def accuracytop5(output, target, topk=(5,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracytop1': accuracytop1,
    'accuracytop5': accuracytop5,
    # could add more metrics such as accuracy for each token type
}

if __name__ == '__main__':
    model = AIF_CNN()
    children = list(model.children())
    print(children)