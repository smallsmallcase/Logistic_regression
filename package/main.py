# @Time    : 2017/10/26 19:41
# @Author  : Jalin Hu
# @File    : main.py
# @Software: PyCharm
from package.logistic_regression import super_grad_ascent
from package.logistic_regression import grad_ascent
from package.logistic_regression import sigmoid
import numpy as np
from sklearn.linear_model import LogisticRegression

def classifyVector(in_x, weights):
    # r = sum(in_x * weights)
    prob = sigmoid(in_x*weights)
    if prob >0.5:
        return 1
    else:
        return 0


def get_data(file_path):
    train_data = []
    label_data = []
    with open(file_path,'r') as f:
        for line in f.readlines():
            line_arr = []
            current_line_list = line.strip().split('\t')
            for i in range(len(current_line_list)-1):
                line_arr.append(float(current_line_list[i]))
            train_data.append(line_arr)
            label_data.append(float(current_line_list[-1]))
    return train_data, label_data

if __name__ == '__main__':
    train_data_file = './train_data.txt'
    test_data_file = './test_data.txt'
    train_data, train_label = get_data(train_data_file)
    train_weights, weight_arr= super_grad_ascent(train_data,train_label)
    train_weights2, weight_arr2 = grad_ascent(train_data, train_label)

    test_data, test_label = get_data(test_data_file)

    err_count = 0
    num_test = len(test_label)

    for i in range(num_test):
        if int(classifyVector(np.mat(test_data[i]), np.mat(train_weights2))) != int(test_label[i]):
            err_count += 1

    err_rate = float(err_count/num_test) *100
    print('错误率：%.2f%%' % err_rate)


    clf = LogisticRegression(solver='liblinear', max_iter=10)
    clf.fit(train_data, train_label)
    accuricy = clf.score(test_data, test_label) *100
    print('sklearn库算得的错误率是：%.2f%%' % (100.00-accuricy))





