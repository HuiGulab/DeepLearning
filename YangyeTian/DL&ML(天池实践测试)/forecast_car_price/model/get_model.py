import torch
from torch import nn
from sklearn import metrics
import d2l.torch as d2l
import numpy as np
import pandas as pd


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# 简单定义了一个线性网络
def get_linear_net(in_features):
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


# 评价标准 Mean Absolute Error(MAE)
# 详见数据解释文档 值越小越好
# 输入两个为数组
def get_mae(y_pred, y_true):
    # 将tensor转成np
    y_t = np.exp(y_true.detach().numpy())
    y_p = np.exp(y_pred.detach().numpy())
    return metrics.mean_absolute_error(y_t, y_p)
    # return metrics.mean_absolute_error(y_true.detach().numpy(), y_pred.detach().numpy())


# 训练
# 返回训练结果以及测试结果
def train(net, train_features, train_labels, test_features, test_labels, loss,
          num_epochs, learning_rate, weight_decay, batch_size):
    # 保存训练结果以及测试结果
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X).squeeze(-1), y)
            l.backward()
            optimizer.step()
        train_ls.append(get_mae(net(train_features), train_labels))
        if test_labels is not None:
            test_ls.append(get_mae(net(test_features), test_labels))
    return train_ls, test_ls


# 获取第i折的数据
# k:一共进行k折
# i：当前进行到第i折
# X：特征
# y：标签
def get_k_fold_data(k, i, X, y):
    assert k > 1  # k不大于0则跳出报错
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


# 定义一个k折交叉验证
def k_fold(k, X_train, y_train, net, loss, num_epochs, learning_rate, weight_decay, batch_size, device):
    train_l_sum, valid_l_sum = 0, 0
    # net.to(device)
    # X_train, y_train = X_train.to(device), y_train.to(device)
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        # 获取训练集、测试集在这一折
        train_ls, valid_ls = train(net, *data, loss, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='MAE', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练MAE：{float(train_ls[-1]):f}, '
              f'验证MAE：{float(valid_ls[-1]):f}')
    d2l.plt.show()
    return train_l_sum / k, valid_l_sum / k


def train_model(net, train_features, train_labels, loss,
                num_epochs, learning_rate, weight_decay, batch_size):
    # 保存训练结果以及测试结果
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    # 训练模型
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X).squeeze(-1), y)
            l.backward()
            optimizer.step()
    return net


def predict_price(net, test_features):
    np_pred = net(test_features).detach().numpy()
    true_pred = np.round(np.exp(np_pred))
    price = pd.DataFrame(true_pred)
    return price
