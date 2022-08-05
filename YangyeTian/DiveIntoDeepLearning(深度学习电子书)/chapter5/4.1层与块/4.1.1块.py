import torch
from torch import nn
from torch.nn import functional as F


'''
一、自定义块
基本功能：
1.将输入数据作为其前向传播函数的参数。
2.通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。
例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
3.计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
4.存储和访问前向传播计算所需的参数。
5.根据需要初始化模型参数。
'''
class MLP(nn.Module):  # 继承 nn.Module
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))

# net = MLP()


'''
二、顺序块
Sequential的设计是为了把其他模块串起来：
1.一种将块逐个追加到列表中的函数。
2.一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。
'''
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

# net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))


'''三、在前向传播函数中执行代码'''
# 我们需要一个计算函数f(X,w)=c·wT·X的层
# 其中X是输入，w是参数，c是某个在优化过程中没有更新的指定常量
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

# net = FixedHiddenMLP()


# 可以混搭嵌套块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

# chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
