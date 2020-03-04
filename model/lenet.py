import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
from sklearn.metrics import accuracy_score,f1_score,roc_curve,precision_recall_curve,average_precision_score,auc
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,matthews_corrcoef,roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,TensorDataset,DataLoader
import pandas as pd
import numpy as np
import random

BATCH_SIZE = 128
ran = [i for i in range(30000)]
testlist = []
trainlist = random.sample(ran,27000)
for j in ran:
    if(j not in trainlist):
        testlist.append(j)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (2,2)))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TrainDataset(Dataset):
    def __init__(self):
        train_npy = np.load("train.npy")
        arr = np.array(train_npy[trainlist], dtype=np.float32)
        arr.resize((27000, 1, 28, 28))
        self.train_data = torch.from_numpy(arr)
        labelnp = np.loadtxt("train.csv", delimiter=",", skiprows=1, usecols=[1],dtype=np.longlong)
        self.train_label = torch.from_numpy(labelnp[trainlist])
        self.len = self.train_data.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self,idx):
        return self.train_data[idx], self.train_label[idx]

class TestDataset(Dataset):
    def __init__(self):
        test_npy = np.load("train.npy")
        arr = np.array(test_npy[testlist], dtype=np.float32)
        arr.resize((3000, 1, 28, 28))
        self.test_data = torch.from_numpy(arr)
        labelnp = np.loadtxt("train.csv", delimiter=",", skiprows=1, usecols=[1],dtype=np.longlong)
        self.test_label = torch.from_numpy(labelnp[testlist])
        self.len = self.test_data.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self,idx):
        return self.test_data[idx], self.test_label[idx]

model = Model().cuda() #实例化卷积层
loss = nn.CrossEntropyLoss() #损失函数选择，交叉熵函数
optimizer = optim.SGD(model.parameters(),lr = 0.01)
num_epochs = 25
losses = []
acces = []
eval_losses = []
eval_acces = []
train_dataset = TrainDataset()
test_dataset = TestDataset()
train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)

for echo in range(num_epochs):
    train_loss = 0  # 定义训练损失
    train_acc = 0  # 定义训练准确度
    model.train()  # 将网络转化为训练模式
    for i, (X, label) in enumerate(train_loader):  # 使用枚举函数遍历train_loader
        # X = X.view(-1,784)       #X:[64,1,28,28] -> [64,784]将X向量展平
        X = Variable(X).cuda()          #包装tensor用于自动求梯度
        #print(X.size())
        label = Variable(label).cuda()
        #print(X.size())
        #print(label)
        out = model(X)  # 正向传播
        #print(out)
        lossvalue = loss(out, label)  # 求损失值
        optimizer.zero_grad()  # 优化器梯度归零
        lossvalue.backward()  # 反向转播，刷新梯度值
        optimizer.step()  # 优化器运行一步，注意optimizer搜集的是model的参数

        # 计算损失
        train_loss += float(lossvalue)
        # 计算精确度
        _, pred = out.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print("echo:" + ' ' + str(echo))
    print("lose:" + ' ' + str(train_loss / len(train_loader)))
    print("accuracy:" + ' ' + str(train_acc / len(train_loader)))

# eval_loss = 0
# eval_acc = 0
label_all = None
pred_all = None
pred_pro_all = None
model.eval() #模型转化为评估模式
for X,label in test_loader:
    #X = X.view(-1,784)
    X = Variable(X).cuda()
    label = Variable(label).cuda()
    testout = model(X)
    testloss = loss(testout,label)
    #eval_loss += float(testloss)

    _, pred = testout.max(1)
    if label_all is None:
        label_all = label
    else:
        label_all = torch.cat([label_all,label])


    if pred_all is None:
        pred_all = torch.cat([pred])
    else:
        pred_all = torch.cat([pred_all,pred])

    if pred_pro_all is None:
        pred_pro_all = torch.cat([torch.sigmoid(testout)])
    else:
        pred_pro_all = torch.cat([pred_pro_all,torch.sigmoid(testout)])
#     num_correct = (pred == label).sum()
#     acc = int(num_correct) / X.shape[0]
#     eval_acc += acc

y_test = label_all.cpu().detach().numpy()
#print(y_test)
y_pred = pred_all.cpu().detach().numpy()
#print(y_pred)
y_pred_pro = pred_pro_all.cpu().detach().numpy()

print('ACC:%.7f' %accuracy_score(y_true=y_test, y_pred=y_pred))
print('Precision-macro:%.7f' %precision_score(y_true=y_test, y_pred=y_pred,average='macro'))
print('Recall-macro:%.7f' %recall_score(y_true=y_test, y_pred=y_pred,average='macro'))
print('F1-macro:%.7f' %f1_score(y_true=y_test, y_pred=y_pred,average='macro'))