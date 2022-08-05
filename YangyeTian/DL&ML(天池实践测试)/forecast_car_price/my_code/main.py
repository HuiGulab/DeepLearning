from feature import get_data
from torch import nn
from model import get_model

# 读取数据集
train_features, train_labels, test_features, test_ids = get_data.get_data()

# 定义损失函数与网络
loss = nn.MSELoss()
in_features = train_features.shape[1]
net = get_model.get_linear_net(in_features)

# 定义相关参数
device = get_model.get_device()
k, num_epochs, lr, weight_decay, batch_size = 3, 15, 0.0001, 0, 64

# 测试数据
# train_l, valid_l = get_model.k_fold(k, train_features, train_labels, net, loss, num_epochs, lr,
#                                     weight_decay, batch_size, device)
# print(f'{k}-折验证: 平均训练MAE: {float(train_l):f}, '
#       f'平均验证MAE: {float(valid_l):f}')


# 训练预测
net = get_model.train_model(net, train_features, train_labels, loss,
                            num_epochs, lr, weight_decay, batch_size)
price_pred = get_model.predict_price(net, test_features)
test_ids['price'] = price_pred.astype(int)
test_ids.to_csv("../prediction_result/test.csv", sep=',', index=False)


# all_files = pd.read_csv(basepath + "file_types.csv")  # 读取分类的结果
# c_files = pd.Dataframe(all_files[label == 'C'])  # 从全部文件中抽取出标签为C类的文件
# c_files_path = c_files['file_path']  # 读取C类文件的路径
# # 使用shuffle函数打乱数据，n_samples表示随机抽取出的数量，设置为C类文件的10%    需要修改则改变下面那个0.1
# random_file_path = shuffle(c_files_path, n_samples=0.1*len(c_files_path.shape[0]))
