import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


'''读取数据集'''
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


'''独热编码'''
# one_hot函数：将每个索引转换为不同的单位向量
# 如（0，2）转化为[[1,0,0],[0,0,1]]向量形状由后一个参数决定
# F.one_hot(torch.tensor([0, 2]), len(vocab))

# 每次采样的小批量数据形状是二维张量：（批量大小，时间步数）
# 我们经常转换输入的维度，以便获得形状为（时间步数，批量大小，词表大小）的输出
# X = torch.arange(10).reshape((2, 5))
# F.one_hot(X.T, 28).shape


'''初始化模型参数'''
# 隐藏单元数num_hiddens是一个可调的超参数。
# 当训练语言模型时，输入和输出来自相同的词表。 因此，它们具有相同的维度，即词表的大小。
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


'''循环神经网络RNN模型'''
# 初始化时返回隐状态
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


# 定义RNN网络
def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # 沿第一维度即时间维度去取X，获取0时刻的X
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh)
                       + torch.mm(H, W_hh)
                       + b_h)
        # Y是当前时刻的预测
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    # 网络返回输出Y以及隐状态H
    return torch.cat(outputs, dim=0), (H,)


#
class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
# state = net.begin_state(X.shape[0], d2l.try_gpu())


'''预测'''
# prefix:根据prefix生成词
# num_preds:需要生成多少个词
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix后面生成新字符"""
    # 生成初始隐藏状态
    # batch_size=1:表示进队一个字符串做预测,即prefix
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    # 把最新预测的值,当作下一次的输入
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)  # 通过已有的词prefix,初始化状态
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)  # 预测,并更新状态
        outputs.append(int(y.argmax(dim=1).reshape(1)))  # 将最大的坐标转成标量,放进输出
    return ''.join([vocab.idx_to_token[i] for i in outputs])  # 将outputs通过字典,反向解码


# 梯度裁剪
def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    # 实现
    # g <- min(1,θ/||g||)g
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    # 将所有梯度的平方和开根号 去和theta比
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    # 如果大则需要进行裁剪
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


'''训练模型'''
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            # 对1进行梯度裁剪
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    # 做指数,用困惑度进行度量
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    # 需要预测的类型
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


num_epochs, lr = 500, 1

# train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
# d2l.plt.show()
# 困惑度 1.0, 78122.5 词元/秒 cuda:0
# time travelleryou can show black is white by argument said filby
# travelleryou can show black is white by argument said filby

# 采用随机采样
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
d2l.plt.show()
