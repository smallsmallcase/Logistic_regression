# @Time    : 2017/10/25 17:10
# @Author  : Jalin Hu
# @File    : logistic regression.py
# @Software: PyCharm


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


# def grad_ascent(data_list, label_list):
#     data_mat = np.mat(data_list)
#     label_mat = np.mat(label_list).transpose()
#     m, n = np.shape(data_mat)  # Row:m , Column:n
#     alpha = 0.001
#     max_cycle = 500
#     weigths = np.ones((n, 1))
#     # weigths = np.mat(weigths)
#     for k in range(max_cycle):
#         h = sigmoid(data_mat * weigths)
#         error = label_mat - h
#         weigths += alpha * data_mat.transpose() *error
#     return weigths

'''
改进后，随机梯度上升算法
'''
def grad_ascent(data_list, label_list):
    data_mat = np.mat(data_list)
    label_mat = np.mat(label_list).transpose()
    m, n = np.shape(data_mat)  # Row:m , Column:n
    # alpha = 0.001
    max_cycle = 150
    weigths = np.ones((n, 1))
    for j in range(max_cycle):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            random_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(data_mat[random_index] * weigths)
            error = label_mat[random_index] - h
            weigths += alpha * data_mat[random_index].transpose() * error
    return weigths


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


if __name__ == '__main__':
    data_list, label_list = load_data_set('./testSet.txt')
    # plot(data_list, label_list)
    result = grad_ascent(data_list, label_list)
    plot(data_list, label_list, result)


