# @Time    : 2017/10/25 17:10
# @Author  : Jalin Hu
# @File    : logistic regression.py
# @Software: PyCharm

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random


def load_data_set(file_path):
    data_list = []
    label_list = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            line_arr = line.strip().split()
            data_list.append([1.0,float(line_arr[0]), float(line_arr[1])])
            label_list.append(int(line_arr[2]))
    return data_list, label_list


def sigmoid(int_x):
    return 1/(1 + np.exp(-int_x))


'''
函数说明:随机梯度上升算法

Parameters:
    data_list - 数据集
    label_list - 数据标签
Returns:
    weights - 求得的权重数组(最优参数)
    weights_array - 每次更新的回归系数
Author:
    Jalin Hu
Modify:
    2017-10-26
'''
def grad_ascent(data_list, label_list):
    data_mat = np.mat(data_list)
    label_mat = np.mat(label_list).transpose()
    m, n = np.shape(data_mat)  # Row:m , Column:n
    alpha = 0.001
    max_cycle = 15000
    weigths = np.ones((n, 1))
    weights_array = np.array([])
    # weigths = np.mat(weigths)
    for k in range(max_cycle):
        h = sigmoid(data_mat * weigths)
        error = label_mat - h
        weigths += alpha * data_mat.transpose() *error
        weights_array = np.append(weights_array, weigths)
    return weigths, weights_array


'''
函数说明:改进的随机梯度上升算法

Parameters:
    data_list - 数据集
    label_list - 数据标签
Returns:
    weights - 求得的权重数组(最优参数)
    weights_array - 每次更新的回归系数
Author:
    Jalin Hu
Modify:
    2017-10-26
'''
def super_grad_ascent(data_list, label_list):
    data_mat = np.mat(data_list)
    label_mat = np.mat(label_list).transpose()
    m, n = np.shape(data_mat)  # Row:m , Column:n
    # alpha = 0.001
    max_cycle = 150
    weights_arr = np.array([])
    weigths = np.ones((n, 1))

    for j in range(max_cycle):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            random_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(data_mat[random_index] * weigths)
            error = label_mat[random_index] - h
            weigths += alpha * data_mat[random_index].transpose() * error
            weights_arr = np.append(weights_arr, weigths)
    return weigths, weights_arr


def plot(data_list, label_list, weights):
    data_arr = np.array(data_list)
    num = len(label_list)  # the number of total data_list
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(num):
        if int(label_list[i]) == 1:
            x1.append(data_arr[i, 1])
            y1.append(data_arr[i, 2])
        else:
            x2.append(data_arr[i, 1])
            y2.append(data_arr[i, 2])
    plt.figure()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)  # 添加subplot
    plt.scatter(x1, y1, s=15, c='red', edgecolor='none',marker='s', alpha=0.5)  # 绘制正样本
    plt.scatter(x2, y2, s=15, alpha=0.5)  # 绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] -weights[1] * x)/weights[2]
    plt.plot(x, y)
    plt.title('DataSet')  # 绘制title
    plt.xlabel('x')
    plt.ylabel('y')  # 绘制label
    plt.show()


def plot_weights(weights_arr1, weights_arr2):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    fig, axs = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(20,15))
    x1 = np.arange(0,len(weights_array1[0::3]), 1)
    axs[0, 0].plot(x1, weights_array1[0::3])
    axs0_title = axs[0, 0].set_title(u'改进梯度上升算法：回归次数与迭代次数关系', FontProperties=font)
    axs0_label = axs[0, 0].set_ylabel('WO')
    plt.setp(axs0_title,size=8, color='black')
    plt.setp(axs0_label, size=20, color='black')

    axs[1, 0].plot(x1, weights_array1[1::3])
    axs0_title = axs[1, 0].set_title(u'改进梯度上升算法：回归次数与迭代次数关系', FontProperties=font)
    axs0_label = axs[1, 0].set_ylabel('W1')
    plt.setp(axs0_title,size=8, color='black')
    plt.setp(axs0_label, size=20, color='black')

    axs[2, 0].plot(x1, weights_array1[2::3])
    axs0_title = axs[2, 0].set_title(u'改进梯度上升算法：回归次数与迭代次数关系', FontProperties=font)
    axs0_label = axs[2, 0].set_ylabel('W2')
    plt.setp(axs0_title,size=8, color='black')
    plt.setp(axs0_label, size=20, color='black')

    x2 = np.arange(0, len(weights_array2[0::3]), 1)
    axs[0, 1].plot(x2, weights_array2[0::3])
    axs0_title = axs[0, 1].set_title(u'梯度上升算法：回归次数与迭代次数关系', FontProperties=font)
    axs0_label = axs[0, 1].set_ylabel('WO')
    plt.setp(axs0_title,size=8, color='black')
    plt.setp(axs0_label, size=20, color='black')

    axs[1, 1].plot(x2, weights_array2[1::3])
    axs0_title = axs[1, 1].set_title(u'梯度上升算法：回归次数与迭代次数关系', FontProperties=font)
    axs0_label = axs[1, 1].set_ylabel('W1')
    plt.setp(axs0_title,size=8, color='black')
    plt.setp(axs0_label, size=20, color='black')

    axs[2, 1].plot(x2, weights_array2[2::3])
    axs0_title = axs[2, 1].set_title(u'梯度上升算法：回归次数与迭代次数关系', FontProperties=font)
    axs0_label = axs[2, 1].set_ylabel('W2')
    plt.setp(axs0_title,size=8, color='black')
    plt.setp(axs0_label, size=20, color='black')
    plt.show()


if __name__ == '__main__':
    data_list, label_list = load_data_set('./testSet.txt')
    # plot(data_list, label_list)
    weights1, weights_array1 = super_grad_ascent(data_list, label_list)
    weights2, weights_array2 = grad_ascent(data_list, label_list)
    # print(np.shape(weights_array1), np.shape(weights1))
    plot(data_list, label_list, weights1)
    plot_weights(weights_array1,weights_array2)



