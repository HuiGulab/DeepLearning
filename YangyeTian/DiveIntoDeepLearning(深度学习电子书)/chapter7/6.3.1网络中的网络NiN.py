import torch
from torch import nn
from d2l import torch as d2l


# NiN块
# 每层卷积层后面跟两个1*1的卷积层
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())


# NiN模型
net = nn.Sequential(
    # 通道、形状类似AlexNet
    # 输入（1，1，224，224）
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    # （1，96，54，54） (224-11+4)/4=217/4=54.25
    nn.MaxPool2d(3, stride=2),
    # （1，96，26，26） (54-3+2)/2=26.5
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    # （1，256，26，26）
    nn.MaxPool2d(3, stride=2),
    # (1,256,12,12)  (24-3+2)/2=12.5
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    # (1,384,12,12) (12-3+1+1+1)=12
    nn.MaxPool2d(3, stride=2),
    # (1,384,5,5) (12-3+2)/2=5.5
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()
# loss 0.591, train acc 0.776, test acc 0.785
# 运行30min
