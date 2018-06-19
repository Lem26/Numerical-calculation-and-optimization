# -*-encoding:utf-8-*-
'''
梯度下降和最速下降法
10142510168 刘恩铭
'''
N = 1000
ACC = 1e-6
from numpy import *

'所求函数的梯度'


def g_f(A, x, B):
    return (A.T) * (A * x) - (A.T) * B


'梯度下降法'


def GD(t, A, B):
    count = 0
    x_k = 2 * ones((N, 1))
    while (count <= 1000):
        x_k1 = x_k - t * g_f(A, x_k, B)
        x_k = x_k1
        count = count + 1
    return x_k, count


'最速下降法'


def AGD(A, B):
    count = 0

    x_k = 2 * ones((N, 1))
    tidu = g_f(A, x_k, B)
    '1000x1'
    t = (tidu.T * tidu) / (tidu.T * A.T * A * tidu)
    while (count <= 1000):
        x_k1 = x_k - t[0, 0] * tidu
        x_k = x_k1
        tidu = g_f(A, x_k, B)
        t = (tidu.T * tidu) / (tidu.T * A.T * A * tidu)
        count = count + 1
    return x_k, count


if __name__ == '__main__':
    A = matrix(random.randint(0, 100, size=(N, N)))
    x_star = ones((N, 1))
    B = A * x_star

    print("----------------最优解为(梯度下降法)：--------------")
    x, count = GD(0.0000000001,A, B)
    print(x)
    print("count=", count)
    print("----------------最优解为(最速下降法)：--------------")
    x, count = AGD(A, B)
    print(x)
    print("count=", count)
