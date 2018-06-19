# -*-encoding:utf-8-*-
'''
共轭梯度法、QR法解线性方程组
参考了PPT
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

#   共轭梯度法
#   X（k+1）=X（K）+a（k）*p（k）
def CG_method(A,B):
    #   generate X0
    X_k=matrix([[0] for i in range(1,N+1)])
    #   generate r0
    r_k=B-A*X_k
    #   d0=r0
    pk=r_k
    for k in range(1,N+1):

        a_k=(r_k.T*(r_k))/((pk.T*(A*pk)))
        X_k=X_k+a_k[0,0]*pk
        if((linalg.norm(A*X_k-B)/linalg.norm(B))<ACC):
            break
        r_k=r_k-a_k[0,0]*A*pk
        b_k=-(r_k.T*(A*pk))/(pk.T*(A*pk))
        pk=r_k+b_k[0,0]*pk

    return X_k,k
#   QR方法
def QR_method(A,B):

    Q=matrix(zeros((N,N)))
    R=matrix(zeros((N,N)))
    X=matrix([[0] for i in range(N)])
    Q[:,0]=A[:,0]/(linalg.norm(A[:,0]))
    for j in range(1,N):
        #   Q~j
        for i in range(0,j):
            q_j=A[:,(j)]-(Q[:,(i)])*(sum(((A[:,(j)]).T)*(Q[:,(i)])))
        Q[:,j]=q_j/(linalg.norm(q_j))
    for i in range(N):
        for j in range(i,N):
            R[i,j]=((A[:,j]).T)*(Q[:,i])
    #   十分抱歉，这个Q和R的矩阵按照ppt的算法，我检查了很多遍，觉得代码应该是没有问题的，但是算出来的Q就是不对
    #   Q.T*Q算出来不是单位矩阵,R是一个上三角矩阵没有问题
    #   这个QR还是有点问题，导致算出来的X非常大
    X,count=CG_method(R,Q.T*B)
    return   Q.T*Q,R,X


if __name__=='__main__':

    A=gene_A()
    B=gene_B()

    print("---------------共轭梯度法-----------------\n")
    X_k,count=CG_method(A,B)
    print('A\n',A)
    print('B.T\n',B.T)
    print('X_K.T\n',X_k.T)
    print('count=',count)
    print('\n')
    print("---------------QR法-----------------\n")
    C,R,X=QR_method(A,B)
    print('Q.T*Q\n',C)
    print('R\n',R)
    print('X.T\n',X.T)