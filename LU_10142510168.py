# -*-encoding:utf-8-*-
'''
LU分解法解线性方程组
参考了课本的公式
10142510168 刘恩铭
'''
from numpy import *
N=100

def calc_LU(A):
    # 计算L（下三角矩阵）和U（上三角矩阵），采用杜利特尔分解法
    L=matrix(eye(N))
    U=matrix(zeros((N,N)))

    #   L 是主对角线全部为1的矩阵
    for k in range(N):
        for i in range(k,N):
            U[k,i]=A[k,i]-sum(L[k,j]*U[j,i] for j in range(k))
        for i in range(k+1,N):
            L[i,k]=(A[i,k]-sum(L[i,j]*U[j,k] for j in range(k)))/U[k,k]
    return L,U
'''
A=mat([[2,1,5],[4,1,12],[-2,-4,5]])

测试分解是否正确

L,U=calc_LU(A)

print("L=",L)
print("U=",U)

//////////////////
E:\Python\python.exe C:/Users/26404/Desktop/数值计算/LU_10142510168.py
L= [[ 1.  0.  0.]
 [ 2.  1.  0.]
 [-1.  3.  1.]]
U= [[ 2.  1.  5.]
 [ 0. -1.  2.]
 [ 0.  0.  4.]]

Process finished with exit code 0
分解结果正确
'''

def solve(A,B):
    # 计算X向量 利用LU分解
    #   LY=B,UX=Y
    L,U=calc_LU(A)

    # 1)计算LY=B ，得到Y

    Y=matrix(zeros((N, 1)))
    for k in range(N):
        Y[k,0]=B[k,0]-sum(L[k,j]*Y[j,0] for j in range(k))

    # 2)计算UX=Y ，得到X
    X=matrix(zeros((N,1)))
    for k in range(N-1,1,-1):
        X[k,0]=(Y[k,0]-sum(U[k,j]*X[j,0] for j in range(k+1,N)))/U[k,k]

    return X

#main program
M=matrix(random.randint(0, 100, size=(N, N)))
A=M+matrix(eye(N))
print("--------系数矩阵A为：--------\n",A)

X=matrix([[i] for i in range(1,101)])
print("--------矩阵X为：--------\n",X.T,"T")

# Generate the vector b as b = Ax~
# 作业上这个步骤没怎么理解。。x~没搞懂是个什么矩阵，看书上写的是对角元为正的下三角矩阵
# 但感觉如果这样理解就没办法做这题了，就只能认为X=[[1][2]..[100]]T
B=A*X
print("--------矩阵B为：--------\n",B.T,"T")
print("--------解出来的矩阵x为：--------\n",solve(A,B).T,"T")
