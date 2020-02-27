"""
构建用于分类的模型：
    神经网络结构： Input --hidden*1 --output
"""
import unittest
import numpy as np


class BikeNetWork(object):

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
            初始化网络结构
        """
        # 设置每一层的神经元个数
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # 初始化权重参数
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                        (self.hidden_nodes, self.output_nodes))

        # 初始化学习率
        self.lr = learning_rate

        # 初始化激活函数
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        self.activation_function = sigmoid

    def train(self, features, targets):
        """
            定义训练函数
        """
        n_records = features.shape[0]  # 总共的数据条数
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)  # 初始化梯度0
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)  # 初始化第二个梯度

        for X, y in zip(features, targets):
            # 开始进行前向传播
            z1 = np.dot(X,self.weights_input_to_hidden)
            a1 = self.activation_function(z1)  # 完成第一层的传播

            z2 = np.dot(a1, self.weights_hidden_to_output)
            a2 = z2  # 无激活函数，因为是输出层

            # 得到output之后，准备计算loss,这里不需要输出loss，从后面的公式也可以看出，使用的是逻辑回归的损失函数

            # 得到loss之后，准备进行反向传播计算梯度
            error = y - a2  # Output layer error is the difference between desired target and actual output.
            output_error_term = error
            hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
            hidden_error_term = hidden_error * a1 * (1 - a1)

            # Weight step (input to hidden)
            delta_weights_i_h += hidden_error_term * X[:, None]
            # Weight step (hidden to output)
            delta_weights_h_o += (y-a2) * a1[:, None]


        # 更新梯度
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        """
       验证集测试用的
       :param features:
       :return: 验证测试结果
       """
        # 开始进行前向传播
        hidden_z = np.dot(features, self.weights_input_to_hidden)
        hidden_a = self.activation_function(hidden_z)  # 完成第一层的传播

        output_z = np.dot(hidden_a, self.weights_hidden_to_output)
        output_a = output_z  # 无激活函数，因为是输出层

        return output_a



