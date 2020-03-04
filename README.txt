model文件夹三个网络模型：
数据来自train.npy文件，训练集(27000张图片)和验证集（3000张图片）

classify文件夹：
Resnet网络模型，Resnetclassify.py文件是30000张train.npy训练集图片训练后直接对5000张test.npy测试集图片预测

Resnet.ipynb是包括训练集（27000张），验证集（3000张）和测试集（5000张）的总程序
最终保存的模型文件Resnet18.pkl