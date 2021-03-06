#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-17-20 02:55
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # padding (28, 28) to (32, 32)
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # x = x.view(-1, self.num_flat_features(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        self.feature = x.detach()
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    """
    def train(self, train_loader, steps):
        LeNet5 train
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).data[0]  # sum up batch loss
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    """
