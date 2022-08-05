import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

'''导入数据集'''
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


'''初始化模型参数'''
# 每个图像由28*28=784个灰度像素值组成
# 图像共分为10个类别
# 实现256个隐藏层
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)  # W1 ： 784 * 256
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))  # b1 ： 1 * 256
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)  # W2 ： 256 * 10
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))  # b2 ： 1 * 10

params = [W1, b1, W2, b2]


'''激活函数'''
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


'''模型'''
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)


'''损失函数'''
loss = nn.CrossEntropyLoss(reduction='none')


'''训练'''
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
plt.show()


'''预测'''
d2l.predict_ch3(net, test_iter)
plt.show()
