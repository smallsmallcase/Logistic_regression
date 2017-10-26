# @Time    : 2017/10/24 16:19
# @Author  : Jalin Hu
# @File    : note.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import math
x = []
y = []
for i in range(-20, 21):
    x.append(i)
f = lambda x: 1 / (1 + math.e ** (-2 * x))
for i in map(f, x):
    print(i)
    y.append(i)
plt.figure()
plt.plot(x, y)
plt.show()
