import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopy.distance
from dipy.segment.metric import Metric
from dipy.segment.clustering import ResampleFeature
from dipy.segment.clustering import QuickBundles
from gmplot import gmplot
import random
from math import sin, asin, cos, radians, fabs, sqrt
from hmmlearn import hmm
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import Callback
import keras.backend as KTF
import tensorflow as tf
import keras.callbacks
import transbigdata as tbd
import geopandas as gpd
import geohash
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
KTF.set_session(session)


# 数据清洗 Data Cleansing


Path1 = "D:\\Geolife Trajectories 1.3\\Data\\"  # 目标文件夹

# 筛选出带有label文件的文件夹
label_list = []
for i in range(0, 180):
    if i < 10:
        i = '00' + str(i)
    elif i < 100:
        i = '0' + str(i)
    Files = os.listdir(Path1 + str(i))
    for k in range(len(Files)):
        Files[k] = os.path.splitext(Files[k])[1]  # 提取文件夹内所有文件的后缀
    Str = '.txt'  # 判断有无.txt文件
    if Str in Files:
        label_list.append(str(i))

# 筛选含有5次以上出租车轨迹数据的用户
taxi_list = []
for i in label_list:
    freq = 0
    for line in open(Path1 + str(i) + "\\labels.txt"):
        Str1 = 'taxi'
        if Str1 in line:
            freq = freq + 1
            if freq > 5:
                taxi_list.append(i)
taxi_list = list(set(taxi_list))

# 提取用户的出租车轨迹数据
taxi_count = 0
geo_data_user = []
geohash_data_user = []
tr_number = 0
total_tr = 0

for i in taxi_list:
    for line in open(Path1 + str(i) + "\\labels.txt"):
        Str1 = 'taxi'
        if Str1 in line:
            taxi_date = line[0:10].replace('/', '')  # 获取出租车乘坐日期，用于筛选plt文件
            start_time = line[0:19].replace(' ', ',').replace('/', '-')  # 获取用户开始乘坐出租车的时间
            end_time = line[20:40].replace(' ', ',').replace('/', '-')  # 获取用户结束乘坐出租车的时间
            Path2 = (Path1 + str(i) + "\\Trajectory")
            filelist = os.listdir(Path2)
            tr_data_df = pd.DataFrame(columns=['lat', 'lon'])  # 每一条轨迹的位置点经纬度集合
            geohash_tr_df = pd.DataFrame(columns=['geohash'])
            geohash_tr_list = []

            for k in range(len(filelist)):
                if taxi_date in filelist[k]:  # 遍历包含taxi日期的plt文件
                    for each_line in open(Path2 + "\\" + filelist[k]):  # 遍历plt文件的每行
                        if end_time >= each_line[-20:-1] >= start_time:  # 筛选在出租车乘坐时间段内的数据
                            geo_data = list(map(float, each_line.split(",")[0:2]))  # 获得当前位置经纬度坐标
                            if 117.4 > geo_data[1] > 115.7 and 41.6 > geo_data[0] > 39.4:  # 通过经纬度筛选出位于北京市的位置点
                                taxi_count = taxi_count + 1
                                geo_df = pd.DataFrame(geo_data).T
                                geo_df.columns = ['lat', 'lon']
                                tr_data_df = pd.concat([tr_data_df, geo_df])  # 将当前点的位置数据加入轨迹位置集合中
                                geohash_data = geohash.encode(geo_data[1], geo_data[0], precision=8)  # geohash数据
                                geohash_tr_list.append(geohash_data)  # 一条轨迹的geohash数据list

            # 经纬度表示的轨迹数据集
            if not tr_data_df.empty:  # 过滤掉空轨迹数据
                lat_lon_data = np.c_[tr_data_df['lat'].values, tr_data_df['lon'].values]
                if lat_lon_data.shape[0] >= 10:
                    geo_data_user.append(lat_lon_data)

            # Geohash编码表示的轨迹数据集
            if len(geohash_tr_list) > 0:
                tr_number = tr_number + 1
                geohash_tr_df = pd.DataFrame(geohash_tr_list, columns=['geohash'])
                geohash_tr_df['time'] = geohash_tr_df.index  # 添加时间列，对每条轨迹构建时空数据集合
                geohash_tr_df['time'] = geohash_tr_df['time'].map(lambda x: str(tr_number) + '_' + str(x))
                geohash_lat_lon_data = np.c_[
                    geohash_tr_df['geohash'].values, geohash_tr_df['time'].values]  # 一个用户的所有轨迹的Geohash
                if geohash_lat_lon_data.shape[0] >= 10:  # 由于需要一定量的数据进行预测，这里只取包含10条数据以上的轨迹
                    total_tr = total_tr + 1
                    geohash_data_user.append(geohash_lat_lon_data)


# 聚类 Clustering


class Distance(Metric):  # 定义距离类
    def __init__(self):
        super(Distance, self).__init__(feature=ResampleFeature(nb_points=256))  # 对轨迹重新采样，使得轨迹点数量相同

    def are_compatible(self, shape1, shape2):
        return len(shape1) == len(shape2)

    def dist(self, v1, v2):  # 定义距离
        x = [geopy.distance.distance([p[0][0], p[0][1]],  # 利用geopy包中的距离测量
                                     [p[1][0], p[1][1]]).kilometers for p in list(zip(v1, v2))]
        curr_dist = np.mean(x)
        return curr_dist


THRESHOLD = 1  # 聚类阈值（km）
dist_metric = Distance()
qb = QuickBundles(threshold=THRESHOLD, metric=dist_metric)  # 采用QuickBundles聚类算法

clusters = qb.cluster(geo_data_user)
print("聚类数量:", len(clusters))


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


# 绘制聚类图


gmap = gmplot.GoogleMapPlotter(geo_data_user[0][0, 0], geo_data_user[0][0, 1], 12)  # 地图中心以及放大倍数

for clustersIndex in range(len(clusters)):
    color = randomcolor()
    for i in clusters[clustersIndex].indices:
        gmap.plot(geo_data_user[i][:, 0], geo_data_user[i][:, 1], color, edge_width=1)  # 描绘轨迹点

gmap.draw("D:\\cluster.html")


# 数据预处理 Data Preprocessing


# 经纬度数据集处理
def create_dataset(data, n_predictions, n_next):  # 数据，步长，预测长度
    dim = data.shape[1]
    print('dim=' + str(dim))
    print(data.shape[0])
    train_X, train_Y = [], []
    test_X, test_Y = [], []

    for i in range(data.shape[0] - n_predictions - n_next - 1):
        a = data[i:(i + n_predictions), :]
        # 训练集
        train_X.append(a)
        tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
        b = []
        for j in range(len(tempb)):
            for k in range(dim):
                b.append(tempb[j, k])
        train_Y.append(b)

        # 测试集
        test_X = train_X
        test_Y = train_Y
    train_X = np.array(train_X, dtype='float64')
    train_Y = np.array(train_Y, dtype='float64')
    test_X = np.array(test_X, dtype='float64')
    test_Y = np.array(test_Y, dtype='float64')

    return train_X, train_Y, test_X, test_Y


# 归一化
def Normalize(data, set_range):
    norm = np.arange(2 * data.shape[1], dtype='float64')
    norm = norm.reshape(data.shape[1], 2)
    for i in range(0, data.shape[1]):
        if set_range:
            list = data[:, i]
            listlow, listhigh = np.percentile(list, [0, 100])  # 取百分比
        else:
            if i == 0:  # 纬度
                listlow = -90
                listhigh = 90
            else:  # 经度
                listlow = -180
                listhigh = 180
        norm[i, 0] = listlow
        norm[i, 1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta
    return data, norm


# 模型训练 Model Training


def trainModel(train_X, train_Y):
    model = Sequential()
    model.add(LSTM(
        120,  # 隐藏层神经元个数
        input_shape=(train_X.shape[1], train_X.shape[2]),
        return_sequences=True))  # 使LSTM层的输入数据维度相同
    model.add(Dropout(0.3))  # 神经元的随机失活率

    model.add(LSTM(
        120,  # 隐藏层神经元个数
        return_sequences=False))
    model.add(Dropout(0.3))  # 神经元的随机失活率

    model.add(Dense(
        train_Y.shape[1]))  # 输出层维度
    model.add(Activation("relu"))  # Relu激活函数

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])  # 损失函数；优化器；评价函数
    model.fit(train_X, train_Y, epochs=100, batch_size=64, verbose=1)  # 训练：输入数据；迭代次数；样本数；日志参数
    model.summary()
    return model


for i in range(len(geo_data_user)):
    # 读入文件数据
    data = geo_data_user[i]

    print("输入data：", data)
    print("样本数：{0}，维度：{1}".format(data.shape[0], data.shape[1]))

    # 归一化
    set_range = True
    data, normalize = Normalize(data, set_range)
    # print("归一化后data：", data)

    # 生成训练数据
    train_num = 6  # 设置步长值
    per_num = 1  # 预测数
    train_X, train_Y, test_X, test_Y = create_dataset(data, train_num, per_num)
    # print("x\n", train_X.shape)
    # print("y\n", train_Y.shape)

    # 训练模型
    model = trainModel(train_X, train_Y)
    loss, mae = model.evaluate(train_X, train_Y, verbose=2)
    print('Loss : {}, Mae: {}'.format(loss, mae))

    # 保存模型
    np.save("D:\\model_trueNorm.npy", normalize)
    model.save("D:\\traj_model.h5")


# 模型预测 Model Prediction


# 均方误差
def rmse(predictions, targets):
    return ((predictions - targets) ** 2).mean().sqrt()


# reshape
def reshape_y_prd(y_prd, dim):
    re_y = []
    i = 0
    while i < len(y_prd):
        tmp = []
        for j in range(dim):
            tmp.append(y_prd[i + j])
        i = i + dim
        re_y.append(tmp)
    re_y = np.array(re_y, dtype='float64')
    return re_y


# 数据切分
def data_set(dataset, test_num):  # 创建时间序列数据样本
    dataX, dataY = [], []
    for i in range(len(dataset) - test_num - 1):
        a = dataset[i:(i + test_num)]
        dataX.append(a)
        dataY.append(dataset[i + test_num])
    return np.array(dataX), np.array(dataY)


# 反归一化
def FNormalize(data, normalize):
    data = np.array(data, dtype='float64')
    # 列
    for i in range(0, data.shape[1]):
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        # 行
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = data[j, i] * delta + listlow
    return data


# 使用之前归一化的训练数据
def NormalizeUseData(data, normalize):
    for i in range(0, data.shape[1]):

        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow

        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta
    return data


EARTH_RADIUS = 6371  # 地球平均半径，6371km


# 计算两个经纬度之间的直线距离
def hav(theta):
    s = sin(theta / 2)
    return s * s


# 用haversine公式计算球面两点间的距离
def get_distance_hav(lat0, lng0, lat1, lng1):
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
    return distance


# 选择预测数据
test_num = 6
per_num = 1
orig_tr = pd.read_csv('D:\\20070429083432.txt', sep=',').iloc[:, 0:2].values  # 原始数据
exam_data = pd.read_csv('D:\\20070429083432.txt', sep=',').iloc[:, 0:2].values  # 预测数据

# 输入数据
data, dataY = data_set(exam_data, test_num)
data.dtype = 'float64'
y = dataY

# 归一化
normalize = np.load("D:\\model_trueNorm.npy")
data_guiyi = []
for i in range(len(data)):
    data[i] = list(NormalizeUseData(data[i], normalize))
    data_guiyi.append(data[i])

# 加载经过训练的模型
model = load_model("D:\\traj_model.h5")
y_prd = []  # 预测轨迹数据列表
for i in range(len(data)):
    test_X = data_guiyi[i].reshape(1, data_guiyi[i].shape[0], data_guiyi[i].shape[1])
    prd = model.predict(test_X)  # 预测
    prd = prd.reshape(prd.shape[1])
    prd = reshape_y_prd(prd, 2)
    prd = FNormalize(prd, normalize)
    prd = prd.tolist()
    y_prd.append(prd[0])  # 添加每个预测点到列表
y_prd = np.array(y_prd)
print("predict: {0}\ntrue：{1}".format(y_prd, y))
print('预测均方根误差：', rmse(y_prd, y))
print('预测直线距离：{:.4f} KM'.format(get_distance_hav(y_prd[0, 0], y_prd[0, 1], y[0, 0], y[0, 1])))

# 画预测轨迹对比图
plt.rcParams['font.sans-serif'] = ['simhei']  # 用来正常显示中文标签
p1 = plt.scatter(orig_tr[:, 1], orig_tr[:, 0], c='r', marker='o', label='True')  # 原始轨迹
p2 = plt.scatter(y_prd[:, 1], y_prd[:, 0], c='b', marker='o', label='Predict')  # 预测轨迹
plt.legend(loc='upper left')
plt.grid()
plt.show()

