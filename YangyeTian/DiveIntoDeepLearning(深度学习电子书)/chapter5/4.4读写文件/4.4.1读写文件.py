import torch
from torch import nn
from torch.nn import functional as F

'''加载、保存张量'''
# 直接调用load和save函数分别读写它们
# 都需要提供一个名称
# x = torch.arange(4)
# # torch.save(x, 'x-file')
# #
# # x2 = torch.load('x-file')
# # print(x2)
#
# y = torch.zeros(4)
# torch.save([x, y],'x-files')
# x2, y2 = torch.load('x-files')
# print(x2, y2)


'''加载、保存模型参数'''


# 先定义三层的多层感知机
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
# 将模型的参数存储在一个叫做“mlp.params”的文件中
torch.save(net.state_dict(), 'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
Y_clone = clone(X)
print(Y_clone == Y)

