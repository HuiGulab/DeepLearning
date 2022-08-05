import torch
from d2l import torch as d2l


'''一、多输入通道'''
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


X = torch.tensor([[[0.0, 1.0, 2.0],
                   [3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0]],

                  [[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]]])

K = torch.tensor([[[0.0, 1.0],
                   [2.0, 3.0]],

                  [[1.0, 2.0],
                   [3.0, 4.0]]])

print(corr2d_multi_in(X, K))
# tensor([[ 56.,  72.],
#         [104., 120.]])


'''二、多输出通道'''
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


# 通过将核张量K与K+1（K中每个元素加）和K+2连接起来，构造了一个具有3个输出通道的卷积核。
K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)
# torch.Size([3, 2, 2, 2])

print(corr2d_multi_in_out(X, K))
# tensor([[[ 56.,  72.],
#          [104., 120.]],
#
#         [[ 76., 100.],
#          [148., 172.]],
#
#         [[ 96., 128.],
#          [192., 224.]]])

