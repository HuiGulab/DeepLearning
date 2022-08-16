import os
import torch
from torch import nn
from d2l import torch as d2l


"""读取数据集"""
#@save
d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
print("============获取数据集路径完毕============")
print("data_dir = ", data_dir)


#@save
def read_imdb(data_dir, is_train):
    """读取IMDb评论数据集文本序列和标签"""
    data, labels = [], []
    for label in ('pos', 'neg'):  # 分别从pos和neg中读取数据
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


train_data = read_imdb(data_dir, True)
test_data = read_imdb(data_dir, False)
print("=============读取数据集完毕=============")


# 创建词表
train_tokens = d2l.tokenize(train_data[0], token='word')
test_tokens = d2l.tokenize(test_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5)
print("=============创建词表完毕==============")

# 我们通过截断和填充将每个评论的长度设置为500
num_steps = 500  # 序列长度
train_features = torch.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
test_features = torch.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
print("=============文本预处理完毕=============")
print(train_features.shape)

# 装载数据集
batch_size = 64
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), batch_size)
test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])), batch_size, is_train=False)


"""创建网络"""


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 将bidirectional设置为True以获取双向循环神经网络
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        # 因为长短期记忆网络要求其输入的第一个维度是时间维，
        # 所以在获得词元表示之前，输入会被转置。
        # 输出形状为（时间步数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 返回上一个隐藏层在不同时间步的隐状态，
        # outputs的形状是（时间步数，批量大小，2*隐藏单元数）
        outputs, _ = self.encoder(embeddings)
        # 连结初始和最终时间步的隐状态，作为全连接层的输入，
        # 其形状为（批量大小，4*隐藏单元数）
        # 双向长短期记忆网络在初始和最终时间步的隐状态（在最后一层）被连结起来作为文本序列的表示
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs


# # 构造一个具有两个隐藏层的双向循环神经网络来表示单个文本以进行情感分析。
# embed_size, num_hiddens, num_layers = 100, 100, 2
# net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
# # 初始化权重
# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.xavier_uniform_(m.weight)
#     if type(m) == nn.LSTM:
#         for param in m._flat_weights_names:
#             if "weight" in param:
#                 nn.init.xavier_uniform_(m._parameters[param])
# net.apply(init_weights)

class MyNet(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(MyNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # 编码层
        self.fc = nn.Linear(embed_size, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


embed_size = 256
net = MyNet(len(vocab), embed_size)

devices = d2l.try_all_gpus()
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
d2l.plt.show()
