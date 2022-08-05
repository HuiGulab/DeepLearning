import os
import torch
from d2l import torch as d2l

# 下载和预处理数据集
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')


#@save
def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()


#@save
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    # 如果结果是符号，且符号前无空格，则将其分割出来
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1])  # 如果i>0，并且该字符是符号且前面无空格，则在前面添加空格
           else char  # 否则不添加空格
           for i, char in enumerate(text)]
    return ''.join(out)


#@save
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    # 英语是源语言（source language）， 法语是目标语言（target language）
    source, target = [], []
    # 划分行
    for i, line in enumerate(text.split('\n')):
        # num_examples 表示对文本前num_examples行进行词元化
        if num_examples and i > num_examples:
            # 超过num_examples，停止词元化
            break

        # 以分隔符划分英语-法语
        parts = line.split('\t')
        if len(parts) == 2:
            # 将划分后的两段，再根据空格划分为词元
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


# # 读取数据集
# raw_text = read_data_nmt()
# # print(raw_text[:75])
# # 数据集预处理
# text = preprocess_nmt(raw_text)
# # print(text[:80])
# # 词元化
# source, target = tokenize_nmt(text)
# # print(source[:6], target[:6])


#@save
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 超过num_steps则截断
    return line + [padding_token] * (num_steps - len(line))  # 不足num_steps则填充，用padding_token进行填充


#@save
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]  # 将每一行通过字典转码
    lines = [l + [vocab['<eos>']] for l in lines]  # 每行以eos为结尾
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])  # 设定截断/填充的长度，设定pad为填充字符，并转为张量
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)  # 统计长度
    return array, valid_len


#@save
# 打包数据集读取及预处理
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    # 读取数据集，并对符号空格等进行简单预处理
    text = preprocess_nmt(read_data_nmt())
    # 将text根据分隔符划分为英语部分和法语部分
    source, target = tokenize_nmt(text, num_examples)
    # 将英语和法语进行分别字典化
    # 填充词元（“<pad>”），以及序列的开始词元（“<bos>”）和结束词元（“<eos>”）
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 对英语、法语进行小批量处理，对每一行进行同一长度的截断/填充,并转化为张量
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 打包成dataset，通过dataloader读取
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


# 尝试读取数据
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break
