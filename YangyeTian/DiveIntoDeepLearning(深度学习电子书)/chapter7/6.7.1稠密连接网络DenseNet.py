import torch
from torch import nn
from d2l import torch as d2l


# 定义卷积块
# 一个稠密块由多个卷积块组成
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


# 定义稠密块
# 每个卷积块使用相同数量的输出通道
# num_convs  卷积块数量
# input_channels  输入通道数
# num_channels  每个卷积块输出通道数
# 例如（2，3，10）
# 输入3
#          \    BN-ReLU-Conv（3，10） \
#                                       BN-ReLU-Conv（13，10）
# 最终输出通道为3+10+10，即2*10+3，即num_convs*num_channels+input_channels
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        #     见注释最后一行
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出，类似GoogLeNet中的Inception块连接
            X = torch.cat((X, Y), dim=1)
        return X


blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)
# torch.Size([4, 23, 8, 8])


# 定义过度层
# 由于每个稠密块都会带来通道数的增加，使用过多则会过于复杂化模型。 而过渡层可以用来控制模型复杂度。
# 通过1*1卷积层来减小通道数，并使用步幅为2的平均汇聚层减半高和宽，从而进一步降低模型复杂度。
# input_channels  输入通道数
# num_channels    输出通道数
# 宽高会减半
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        # 前面费劲使通道数连接在一起，若使用最大汇聚层，则失去原有效果了，故采用效率低一些的平均汇聚层
        nn.AvgPool2d(kernel_size=2, stride=2))


blk = transition_block(23, 10)
print(blk(Y).shape)
# torch.Size([4, 10, 4, 4])


# 定义DenseNet模型
# 第一模块类似VGG，7*7卷积+批量规范+池化
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# num_channels为当前的通道数，初始输入通道为64
# 增长率为32，对应稠密块内的num_channels，块内每次卷积增加32个输出通道，共四次卷积，每块增加128个通道
num_channels, growth_rate = 64, 32
# 设置共有四个稠密块，每个块包含四个卷积块
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 计算下一个稠密块的输入通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加一个转换层，使通道数量减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2  # //表整除

net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    # 最后采用全局平均汇聚层
    # 将宽高变为1
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))


lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()
# 运行49-35=14min
# loss 0.142, train acc 0.949, test acc 0.805
# 836.8 examples/sec on cuda:0

