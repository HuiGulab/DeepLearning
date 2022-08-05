import torch
from torch import nn


'''
一、填充
'''
# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


# 核函数为3*3  kernel_size=3
# 每边都填充了1行或1列  padding=1
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
# print(comp_conv2d(conv2d, X).shape)
# torch.Size([8, 8])

# 核函数为5*3  kernel_size=(5, 3)
# 每边都填充了2行或1列  padding=(2, 1)
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
# print(comp_conv2d(conv2d, X).shape)
# torch.Size([8, 8])


'''
二、步幅
'''
# 将高度和宽度的步幅设置为2  stride=2
# 输入的高度和宽度减半
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)
# torch.Size([4, 4])


# 稍微复杂的例子
# 输入8*8
# 核函数3*5  左右填充各一行  步幅3*4
# 0 1 2 3 4|5 6 7 8 0|
# 0 2 2 3 4|5 6 7 8 0|
# 0 3 2 3 4|5 6 7 8 0|
# --------------------
# 0 4 2 3 4|5 6 7 8 0|
# 0 5 2 3 4|5 6 7 8 0|
# 0 6 2 3 4|5 6 7 8 0|
# --------------------
# 0 7
# 0 8
# 输出为 2*2
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
# print(comp_conv2d(conv2d, X).shape)
# torch.Size([2, 2])
