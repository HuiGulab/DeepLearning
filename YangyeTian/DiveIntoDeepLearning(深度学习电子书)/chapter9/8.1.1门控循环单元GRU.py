import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


'''初始化模型参数'''
# vocab_size：词典大小
# num_hiddens：隐藏层单元数量
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    # 从标准差为0.01的高斯分布中提取权重
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# 返回一个形状为（批量大小，隐藏单元个数）的张量
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


# 根据定义将模型构建
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        # @是矩阵乘法
        # *表示点乘
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)  # 更新门
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)  # 重置门
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)  # 候选隐状态
        H = Z * H + (1 - Z) * H_tilda  #隐状态
        Y = H @ W_hq + b_q
        outputs.append(Y)
    # 网络返回输出Y以及隐状态H
    return torch.cat(outputs, dim=0), (H,)


vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()
# perplexity 1.1, 23808.1 tokens/sec on cuda:0
# time travelleryou can show black is white by argument said filby
# travelleryou can show black is white by argument said filby
