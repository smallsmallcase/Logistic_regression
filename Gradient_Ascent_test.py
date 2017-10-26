# @Time    : 2017/10/24 18:25
# @Author  : Jalin Hu
# @File    : Gradient_Ascent_test.py
# @Software: PyCharm
def Gradient_Ascent_test():
    def f_prime(x_old):
        return -2 * x_old + 4
    x_old = -1
    x_new = 0
    alpha = 0.01
    persision = 0.00000001
    while(x_new - x_old) > persision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
    print(x_new)

if __name__ == '__main__':
    Gradient_Ascent_test()
