
import numpy as np

import pandas as pd


def init_data():
    pdData=pd.read_csv('HTRU_2_train.csv',header=None,names=['x','y','labels'])
    pdData.insert(0,'1',1)
    orig_data=pdData.values
    columns=orig_data.shape[1]
    dataMatIn=orig_data[:,0:columns-1]
    classLabels=orig_data[:,columns-1:columns]
    return dataMatIn,classLabels

#sigmoid函数，将线性回归函数改为逻辑回归函数 g(z) = 1/(1 + exp(-z))
def sigmoid(z):
    
    
    return 1/(1+np.exp(-z))
   
#梯度下降
#dataMatrix*weights:θ向量的转置的乘积x向量得出线性回归方程
#h:关于逻辑回归函数对θ求偏导
#weights:初始化θ，将其设置为与x行列数相等的矩阵
    
def grad_descent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)#将dataMatrix从列表类型变为矩阵
    labelMat = np.mat(classLabels).transpose()#将classLabels从列表类型变为矩阵，并转置
    m, n = np.shape(dataMatrix)#m,n为dataMatrix的行数和列数
    weights = np.ones((n, 1))#生成一个n行1列的数值全为1的矩阵
    learning_rate = 0.1#learning_rate：初始化学习率为0.001
    maxCycle = 500#maxCycle：初始化迭代次数为500
    for i in range(maxCycle):
        h = sigmoid(dataMatrix*weights)
        weights = weights - learning_rate*dataMatrix.transpose()*(h - labelMat)
    #迭代500次，得到最终的θ矩阵
    return weights
def testResult():
    trainData=pd.read_csv('HTRU_2_train.csv',header=None,names=['x','y'])
    trainData.insert(0,'1',1)
    orig_data=trainData.values
    rows=orig_data.shape[0]
    result=[]
    for i in range(rows):
        r=sigmoid(orig_data[i]*weights)
        if r>0.5:
            result.append(1)
        else:
            result.append(0)
    num=np.arange(1,701)
    p=pd.DataFrame({'id':num,'y':result})
    p.to_csv('a.csv',index=False)
if __name__=='__main__':
    dataMatIn,classLabels=init_data()
    weights=grad_descent(dataMatIn, classLabels)
    testResult()
        
    
                                                                     
