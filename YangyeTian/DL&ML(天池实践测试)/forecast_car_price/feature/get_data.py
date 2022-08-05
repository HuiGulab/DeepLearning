import pandas as pd
import numpy as np
import torch
from datetime import datetime

'''载入训练集和测试集'''


# 数据预处理
def preprocess_data(df):
    # 'name'有部分重复值，做一个简单统计
    df['name_count'] = df.groupby(['name'])['SaleID'].transform('count')
    del df['name']
    del df['offerType']
    del df['seller']

    # 对'price'做对数变换
    df['price'] = np.log1p(df['price'])

    # 用众数填充缺失值
    df['fuelType'] = df['fuelType'].fillna(0)
    df['gearbox'] = df['gearbox'].fillna(0)
    df['bodyType'] = df['bodyType'].fillna(0)
    df['model'] = df['model'].fillna(0)

    # 处理异常值
    df['power'] = df['power'].map(lambda x: 600 if x > 600 else x)  # 限定power<=600
    df['notRepairedDamage'] = df['notRepairedDamage'].astype('str').apply(lambda x: x if x != '-' else None).astype(
        'float32')  # 类型转换

    df = df.fillna(0)

    # 对可分类的连续特征进行分桶，kilometer是已经分桶了
    bin = [i * 10 for i in range(31)]
    df['power_bin'] = pd.cut(df['power'], bin, labels=False)
    df = df.fillna(int(df["power_bin"].mean()))
    bin = [i * 10 for i in range(24)]
    df['model_bin'] = pd.cut(df['model'], bin, labels=False)

    df = df.fillna(0)
    # del df['power']
    # del df['model']

    return df


# 特征工程
def engineer_feature(df):
    # 时间处理，提取出年月日
    def date_process(x):
        year = int(str(x)[:4])
        month = int(str(x)[4:6])
        day = int(str(x)[6:8])
        # 观察到部分数据月份为0，修改为1
        if month < 1:
            month = 1
        date = datetime(year, month, day)
        return date

    df['regDate'] = df['regDate'].apply(date_process)
    df['creatDate'] = df['creatDate'].apply(date_process)
    df['car_age_day'] = (df['creatDate'] - df['regDate']).dt.days  # 二手车使用天数
    df['car_age_year'] = round(df['car_age_day'] / 365, 1)  # 二手车使用年数

    del df['regDate']
    del df['creatDate']

    # num_cols = [0, 3, 8, 12]
    # for i in num_cols:
    #     for j in num_cols:
    #         df['new' + str(i) + '*' + str(j)] = df['v_' + str(i)] * df['v_' + str(j)]
    # 
    # for i in num_cols:
    #     for j in num_cols:
    #         df['new' + str(i) + '+' + str(j)] = df['v_' + str(i)] + df['v_' + str(j)]
    #
    # for i in num_cols:
    #     for j in num_cols:
    #         df['new' + str(i) + '-' + str(j)] = df['v_' + str(i)] - df['v_' + str(j)]

    return df


# 读取csv文件
def read_csv():
    path = '../data/'
    Train_data = pd.read_csv(path + 'used_car_train_20200313.csv', sep=' ')
    Test_data = pd.read_csv(path + 'used_car_testB_20200421.csv', sep=' ')
    df = pd.concat([Train_data, Test_data], ignore_index=True)
    df = preprocess_data(df)
    df = engineer_feature(df)

    Train_data = df.iloc[:150000]
    Test_data = df.iloc[150000:]
    return Train_data, Test_data


# 返回数据集
def get_data():
    Train_data, Test_data = read_csv()
    # 特征值中，去除ID，价格  axis=1表示按列删除  inplace=False表示不覆盖Train_data的数据
    train_features = Train_data.drop(['SaleID', 'price'], axis=1, inplace=False)
    train_label = Train_data['price']
    test_features = Test_data.drop(['SaleID', 'price'], axis=1, inplace=False)

    # 转化为张量
    train_features = torch.tensor(data=train_features.values, dtype=torch.float32)
    train_labels = torch.tensor(data=train_label.values, dtype=torch.float32)
    test_features = torch.tensor(test_features.values, dtype=torch.float32)
    test_ids = Test_data[['SaleID', 'price']]
    test_ids.index = range(len(test_ids))  # 序号重排
    return train_features, train_labels, test_features, test_ids


