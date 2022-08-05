# %matplotlib inline
import matplotlib.pyplot as plt
import random
import torch
from d2l import torch as d2l


# 生成数据集
def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 以正态分布添加噪音
    return X, y.reshape((-1, 1))


# 生成 y = 2X - 3.4 +噪音的数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# features中的每一行都包含一个二维数据样本， labels中的每一行都包含一维标签值（一个标量）
# print('features:', features[0], '\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);

plt.show()


# 读取数据集
# 我们定义一个data_iter函数
# 该函数接收批量大小、特征矩阵和标签向量作为输入
# 生成大小为batch_size的小批量。
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# 模拟读取数据集
batch_size = 10
#
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break
# tensor([[ 0.7645, -1.3892],
#         [ 1.6107,  0.0810],
#         [ 0.3487,  2.5856],
#         [ 0.9868,  0.5390],
#         [-0.3331,  1.8682],
#         [-1.2828, -0.3002],
#         [-1.0663, -0.1783],
#         [ 1.6962,  2.0506],
#         [ 1.1200,  0.4016],
#         [-1.0953, -0.4782]])
#  tensor([[10.4522],
#         [ 7.1304],
#         [-3.8837],
#         [ 4.3318],
#         [-2.8197],
#         [ 2.6508],
#         [ 2.6691],
#         [ 0.6243],
#         [ 5.0516],
#         [ 3.6547]])


# 定义模型
# X 特征值
# w 权重
# b 偏置值
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b


# 定义损失函数
# 平方损失函数
# y_hat 预测值
# y 实际值
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 初始化模型 w从标准差0.01正太分布中取，b取0
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练模型

lr = 0.03  # 学习率
num_epochs = 3  # 迭代周期数
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    # 每个周期从数据集读取数据
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
