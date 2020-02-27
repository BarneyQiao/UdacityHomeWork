"""
测试各种小功能模块用的


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from BikeNN import BikeNetWork

# ==========1 观察数据==========


data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)  # 给一个csv的路径即可
# rides[:24*10].plot(x='dteday',y='cnt')
# plt.show()
# 转换成one-hot编码的变量，把非连续变量转换成01的编码
dummy_fields = ['season','weathersit','mnth','hr','weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each],prefix=each,drop_first=False)
    rides = pd.concat([rides,dummies],axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
              'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop,axis=1)
print(data.head())

# 标准化每个连续变量，均值为0，标准差为1
quant_features = ['casual','registered','cnt','temp','hum','windspeed']
scaled_features = {}
for each in quant_features:
    mean,std = data[each].mean(),data[each].std()
    scaled_features[each] = [mean,std]
    data.loc[:,each] = (data[each]-mean)/std

# 拆分训练集、测试集(最后21天)、验证集
test_data = data[-21*24:]
data = data[:-21*24]

# 数据拆分成特征features和targets
targets_fields = ['cnt','casual','registered'] #target fields [租赁自行车给的总用户数，临时用户数，注册用户数]
features,targets = data.drop(targets_fields,axis=1),data[targets_fields]
test_features,test_targets = test_data.drop(targets_fields,axis=1),test_data[targets_fields]

# 再将训练集中的数据拆分成训练集和验证集，因为数据是有时间序列的，所以用历史数据训练，尝试预测未来的验证集
train_features,train_targets = features[:-60*24], targets[:-60*24]
val_features,val_targets = features[-60*24:], targets[-60*24:]

# print(train_features.head())
# print(train_targets.head())

# ==========2 构建网络==========
# 见BikeNN

# ==========3 训练网络==========

def MSE(y, Y):
    return np.mean((y-Y)**2)

iterations = 2000
learning_rate = 0.8
hidden_nodes = 12
output_nodes = 1

N_i = train_features.shape[1]
network = BikeNetWork(input_nodes=N_i,hidden_nodes=hidden_nodes,
                      output_nodes=output_nodes,learning_rate=learning_rate)
losses = {'train':[],'validation':[]}
for ii in range(iterations):
    batch = np.random.choice(train_features.index,size=128)
    X,y = train_features.ix[batch].values,train_targets.ix[batch]['cnt']

    network.train(X,y)

    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii / float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)


plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()
plt.show()