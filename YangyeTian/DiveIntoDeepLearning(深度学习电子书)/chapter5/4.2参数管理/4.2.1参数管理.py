import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
# net(X)


'''参数访问'''
# 每次定义网络，参数随机生成
# print(net[2].state_dict())
# # OrderedDict([('weight', tensor([[ 0.2000,  0.0378, -0.1202,  0.1493, -0.3114,  0.2072,  0.3459, -0.2496]])),
# #                                                                                ('bias', tensor([0.0181]))])

# print(type(net[2].bias))
# # <class 'torch.nn.parameter.Parameter'>
# print(net[2].bias)
# # tensor([-0.1059], requires_grad=True)
# print(net[2].bias.data)
# # tensor([-0.1059])

# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# # ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
# print(*[(name, param.shape) for name, param in net.named_parameters()])
# # ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8]))
# # ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))


# 从嵌套种取数据
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
# print(rgnet)
# Sequential(
#   (0): Sequential(
#     (block 0): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block 1): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block 2): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block 3): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#   )
#   (1): Linear(in_features=4, out_features=1, bias=True)
# )


'''初始化参数'''


# 将所有权重参数初始化为标准差为0.01的高斯随机变量
# 将偏置参数设置为0
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


# net.apply(init_normal)
# print(net[0].weight.data[0], net[0].bias.data[0])
# tensor([ 0.0115, -0.0007,  0.0039, -0.0060]) tensor(0.)


# 将权重参数全置一
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


# net.apply(init_constant)
# print(net[0].weight.data[0], net[0].bias.data[0])
# (tensor([1., 1., 1., 1.]), tensor(0.))


# 我们使用Xavier初始化方法初始化第一个神经网络层
# 然后将第三个神经网络层初始化为常量值42。
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

# net[0].apply(xavier)
# net[2].apply(init_42)
# print(net[0].weight.data[0])
# print(net[2].weight.data)
# tensor([ 0.5158, -0.6288, -0.6625,  0.3727])
# tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])


'''参数绑定'''
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
print(net)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
