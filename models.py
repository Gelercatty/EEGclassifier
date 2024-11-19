import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from sklearn.model_selection import train_test_split
from utils import parse_args

import torchvision.models as models


class EMGClassifierV1(nn.Module):
    def __init__(self):
        super(EMGClassifierV1, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))

        # 全连接层
        self.fc1 = nn.Linear(128 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # 展平操作
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class EMGClassifierV2(nn.Module):
    def __init__(self):
        super(EMGClassifierV2, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))

        # 全连接层
        self.fc1 = nn.Linear(128 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # 展平操作
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x



class EMGClassifierV3(nn.Module):
    def __init__(self):
        super(EMGClassifierV3, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))

        # 全连接层
        self.fc1 = nn.Linear(128 * 8, 512)
        self.fc2 = nn.Linear(512, 6)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # 展平操作
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
    


class EMGClassifierV4(nn.Module):
    def __init__(self):
        super(EMGClassifierV4, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 8, kernel_size=(3, 3), padding=(1, 1))

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)  # Adjust the input features to match your final pooling layer output
        self.fc2 = nn.Linear(512, 6)     # Output layer for 6 classes

    def forward(self, x):
        # First convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Concatenate along the third dimension
        x = x.view(-1, 1024, 1024)  # Adjust the view parameters based on previous layers output

        # Downsample using pooling
        while x.shape[1] > 1:
            x = self.pool(x)

        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


class EMG_Resnet18(nn.Module):
    def __init__(self):
        super(EMG_Resnet18, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        # 修改第一层卷积层
        self.resnet18 = resnet18
        self.resnet18.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 修改最后一层全连接层
        num_ftrs = resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 6)
    def forward(self, x):
        return self.resnet18(x)