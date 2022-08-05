import torch
from torch import nn
from d2l import torch as d2l


# 定义了一个名为vgg_block的函数来实现一个VGG块
# 带有3*3卷积核、填充为1（保持高度和宽度）的卷积层
# 带有2*2汇聚窗口、步幅为2（每个块后的分辨率减半）的最大汇聚层。
# 超参数变量conv_arch:该变量指定了每个VGG块里卷积层个数和输出通道数。全连接模块则与AlexNet中的相同。
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# 其中前两个块各有一个卷积层，后三个块各包含两个卷积层。
# 第一个参数表示一个VGG块内有几个卷积层
# 第二个参数表示通道数量
# 第一个模块有64个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到512。
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
# 卷积层共五个块共8层
# 卷积层3*3卷积核、填充为1
# 池化层2*2汇聚窗口、步幅为2
# 输入(1, 1, 224, 224)
# 第一模块：卷积层（1，64，224，224）池化层（1，64，122，122）
# 第二模块：卷积层（1，128，122，122）池化层（1，128，56，56）
# 第三模块：卷积层（1，256，56，56）卷积层（1，256，56，56）池化层（1，256，28，28）
# 第四模块：卷积层（1，512，28，28）卷积层（1，512，28，28）池化层（1，512，14，14）
# 第五模块：卷积层（1，512，14，14）卷积层（1，512，14，14）池化层（1，512，7，7）


# 定义VGG网络
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks,
        # 卷积层输出（1，512，7，7）
        nn.Flatten(),
        # 张量化，输入为512*7*7
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))


net = vgg(conv_arch)

# VGG计算量大，因此构建一个通道数较少的网络
ratio = 4
# conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
# //表示整除运算，向下取整
# small_conv_arch = ((1, 16), (1, 32), (2, 64), (2, 128), (2, 128))
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()
# loss 0.180, train acc 0.934, test acc 0.923
# 运行35min
