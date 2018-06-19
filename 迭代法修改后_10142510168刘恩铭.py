# -*-encoding:utf-8-*-
'''
迭代法解线性方程组
参考了课本的公式
10142510168 刘恩铭
'''

N=100
ACC=1e-6
from numpy import *

def gene_A():
    M=2*matrix(eye(N))
    M1=-matrix(eye(N,k=1))
    M2=-matrix(eye(N,k=-1))
    return  M+M1+M2

def gene_B():
    B=matrix([[1] for i in range(1,N+1)])
    return B

def calc_DLU(A):
    D=matrix(zeros((N,N)))
    L=matrix(zeros((N,N)))
    U=matrix(zeros((N,N)))

    for i in range(N):
        D[i,i]=A[i,i]
        for j in range(i+1,N):
            U[i,j]=-A[i,j]

    L=-(A-D+U)
    return D,L,U

# 雅可比迭代法
def Jacobi_solve(A,B):
    count=0 # 记录迭代的次数
    #   X=gene_X()  # get x~
    X_k=matrix([[0] for i in range(1,N+1)]) # 迭代法得到的X的解向量

    D,L,U=calc_DLU(A)

    bj=D.I*(L+U)
    fj=D.I*B
    # 迭代开始,精度控制在ACC内
    # 如果迭代矩阵某个算子范数>=1，则发散，无法用迭代方法解
    lamda,hU=linalg.eig(bj)
    lamda=abs(lamda)
    print("迭代矩阵的特征值的绝对值:",lamda)
    print("迭代矩阵的谱半径:",max(lamda))
    if(max(lamda)>=1):
        print("迭代矩阵发散，无法用迭代法计算")
        return X_k,0
    while(True):
        count+=1
        X_k=bj*X_k+fj
        if((linalg.norm((A*X_k-B))/linalg.norm(B))<ACC):
            break

    return X_k,count

# 高斯-赛德尔迭代法
def Gauss_Seidel_solve(A,B):
    count=0 # 记录迭代的次数
    X_k=matrix([[0] for i in range(1,N+1)]) # 迭代法得到的X的解向量

    D,L,U=calc_DLU(A)
    bg=((D-L).I)*U
    fg=((D-L).I)*B
    # 迭代开始,精度控制在ACC内
    # 如果迭代矩阵某个算子范数>=1，则发散，无法用迭代方法解
    lamda,hU=linalg.eig(bg)
    lamda=abs(lamda)
    print("迭代矩阵的特征值的绝对值:",lamda)
    print("迭代矩阵的谱半径:",max(lamda))
    if(max(lamda)>=1):
        print("迭代矩阵发散，无法用迭代法计算")
        return X_k,0
    while(True):
        count+=1
        X_k=bg*X_k+fg
        if((linalg.norm((A*X_k-B))/linalg.norm(B))<ACC):
            break

    return X_k,count

# 逐次超松弛迭代法,默认w=1的话为高斯-赛德尔迭代法
def SOR_solve(A,B,w=1):
    count=0 # 记录迭代的次数
    X_k=matrix([[0] for i in range(1,N+1)]) # 迭代法得到的X的解向量

    D,L,U=calc_DLU(A)
    # 超松弛迭代矩阵
    bw=((D-w*L).I)*((1-w)*D+w*U)
    fw=w*((D-w*L).I)*B

    # 迭代开始,精度控制在ACC内
    # 如果迭代矩阵某个算子范数>=1，则发散，无法用迭代方法解
    lamda,hU=linalg.eig(bw)
    lamda=abs(lamda)
    print("迭代矩阵的特征值的绝对值:",lamda)
    print("迭代矩阵的谱半径:",max(lamda))
    if(max(lamda)>=1):
        print("迭代矩阵发散，无法用迭代法计算")
        return X_k,0

    while(True):
        count+=1
        X_k=bw*X_k+fw
        if((linalg.norm((A*X_k-B))/linalg.norm(B))<ACC):
            break

    return X_k,count


if __name__=='__main__':
    '''
    A=matrix([[10,-2,-1],[-2,10,-1],[-1,-2,5]])
    B=matrix([[3],[15],[10]])

    '''
    A=gene_A()
    B=gene_B()

    print("---------------雅可比迭代法-----------------\n")
    X_k,count=Jacobi_solve(A,B)
    print('A\n',A)
    print('B.T\n',B.T)
    print('X_K.T\n',X_k.T)
    print('count=',count)
    print('\n')
    print("---------------高斯-赛德尔迭代法-----------------\n")
    X_k,count=Gauss_Seidel_solve(A,B)
    print('X_K.T\n',X_k.T)
    print('count=',count)
    print('\n')
    print("---------------逐次超松弛迭代法-----------------\n")
    w=1.46
    print("松弛因子w=",w)
    X_k,count=SOR_solve(A,B,w)
    print('X_K.T\n',X_k.T)
    print('count=',count)
    print('\n')



