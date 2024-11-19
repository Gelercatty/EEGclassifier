import torch
import torch.nn as nn
import torch.optim as optim # 优化器
from torch.utils.data import  DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from utils import parse_args
import models


config = parse_args()


np.random.seed(config.rand_seed)  # 设置随机种子以保证结果可重复

#####loading data######
X_train = np.load('./data_processed/X_train_sub1000.npy')
y_train = np.load('./data_processed/y_train_sub1000.npy')
X_test = np.load('./data_processed/X_test_sub1000.npy')
y_test = np.load('./data_processed/y_test_sub1000.npy')

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model_class = getattr(models, config.models)
print(model_class)
model = model_class()#### 此处动态的加载模型


train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

batch_size = config.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # true 是打乱
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # false 不打乱

logger = SummaryWriter(f'./runs/{config.experiment_name}')

criterion = nn.CrossEntropyLoss() # 交叉熵损失函数

optimizer = optim.Adam(model.parameters(), lr=config.lr) # adam 优化器

num_epochs = config.epoches #训练轮数
eval_every = config.eval_every # 每多少轮进行一次评估
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)



for epoch in tqdm(range(num_epochs),desc='trainig'):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)# forward
        loss = criterion(outputs, labels) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新权重

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        _, labels_max = torch.max(labels, 1)

        total += labels.size(0)
        correct += (predicted == labels_max).sum().item()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    train_loss/=len(train_loader)
    train_accuracy = correct*100 / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
    
    logger.add_scalar('Loss/train', train_loss, epoch+1)
    logger.add_scalar('Accuracy/train', train_accuracy, epoch+1)
    
    # 每 eval_every 轮进行一次评估
    if (epoch+1) % eval_every ==0:
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                _, labels_max = torch.max(labels, 1)
                total += labels.size(0)
                correct += (predicted == labels_max).sum().item()
        val_loss /= len(test_loader)
        val_accuracy = correct*100 / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        logger.add_scalar('Loss/val', val_loss, epoch+1)
        logger.add_scalar('Accuracy/val', val_accuracy, epoch+1)

print('Finished Training')
# make sure there is a model folder
import os
if not os.path.exists(f'./model/{config.models+config.experiment_name}'):
    os.makedirs(f'./model/{config.models+config.experiment_name}')
torch.save(model.state_dict(), f'./model/{config.models+config.experiment_name}/model.pth')
print('Model Saved')
logger.close()